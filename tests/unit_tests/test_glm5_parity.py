# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Composable single-device GLM-5 parity tests.

The module separates five concerns:

* ``ParityDataFactory`` creates repeatable inputs and masks.
* ``ParityModelPair`` owns the HF/TorchTitan model pair and layer lookup.
* ``ParityRecorder`` stores scalar and discrete comparison results.
* ``LayerTrace`` captures reusable layer, indexer, and router observations.
* ``RecursiveModuleTrace`` captures every tensor leaf and its gradient under selected layers and top-level modules.
* ``TestGlm5Parity`` selects implementation x precision endpoints at runtime.
* The single test class declares the component, layers, precision, and data.

The reference model is optional. Paired execution uses the accelerator selected
by PyTorch, while offline artifacts allow endpoints on different backends and
servers. The native router precision tests remain CPU-safe. Reports use
canonical paths, compact numeric summaries, and explicit discrete selections.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import tempfile
import traceback
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from html import escape
from importlib.util import find_spec
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Iterator

import torch
import torch.nn.functional as F

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import TokenChoiceTopKRouter
from torchtitan.models.glm5 import (
    build_glm5_layers,
    Glm5StateDictAdapter,
    glm5_configs,
)
from torchtitan.ops.scatter_add import deterministic_scatter_add
from tests.glm5_2_parity.artifacts import (
    EndpointIdentity,
    json_digest,
    ObservationMetadata,
    ParityArtifactError,
    ParityArtifactReader,
    ParityArtifactWriter,
    runtime_metadata,
    tensor_digest,
)

_TRANSFORMERS_IMPORT_ERROR: Exception | None = None
try:
    from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM
except Exception as exc:  # pragma: no cover - environment dependent
    _TRANSFORMERS_IMPORT_ERROR = exc
    GlmMoeDsaConfig = None
    GlmMoeDsaForCausalLM = None


@dataclass(frozen=True)
class PrecisionPolicy:
    """Numerical policy for one parity run."""

    name: str
    dtype: torch.dtype
    rtol: float
    atol: float
    exact_discrete: bool = True


# rtol, atol
# abs(actual - expected) <= atol + rtol * abs(expected)
FP32 = PrecisionPolicy("fp32", torch.float32, 1e-4, 1e-5)
BF16 = PrecisionPolicy("bf16", torch.bfloat16, 5e-2, 5e-2)
TOP_LEVEL_COMPONENTS = {"tok_embeddings", "norm", "lm_head"}
GLM5_PARITY_SUITE_VERSION = 3


@dataclass
class ParityBatch:
    """All inputs shared by a component or end-to-end comparison."""

    hidden_states: torch.Tensor
    positions: torch.Tensor
    causal_mask: torch.Tensor
    hf_position_embeddings: tuple[torch.Tensor, torch.Tensor]
    tokens: torch.Tensor | None = None


@dataclass(frozen=True)
class ModelEndpoint:
    """One model implementation at one numerical precision."""

    implementation: str
    precision: PrecisionPolicy

    @property
    def label(self) -> str:
        return f"{self.implementation}:{self.precision.name}"


@dataclass(frozen=True)
class ComparisonSpec:
    """The two endpoints and tolerance used by one comparison."""

    actual: ModelEndpoint
    expected: ModelEndpoint
    scope: str
    component: str
    rtol: float
    atol: float

    @property
    def label(self) -> str:
        return f"{self.actual.label} vs {self.expected.label}"


@dataclass(frozen=True)
class ParityCase:
    """One stable, independently seeded case in a parity suite."""

    ordinal: int
    case_id: str
    section_id: str
    component: str
    seed: int


def _load_requested_device_backend() -> None:
    """Import an explicitly requested out-of-tree device backend."""
    requested = os.environ.get("GLM5_PARITY_DEVICE", "auto").lower()
    if requested == "npu":
        try:
            __import__("torchtitanturbo")
        except ImportError as error:
            raise RuntimeError(
                "GLM5_PARITY_DEVICE=npu requires source-installed "
                "torchtitanturbo and torch_npu"
            ) from error


def _parity_device() -> tuple[torch.device, Any, str]:
    """Return the available accelerator through PyTorch's device registry."""
    _load_requested_device_backend()
    from torch._utils import _get_available_device_type, _get_device_module

    requested = os.environ.get("GLM5_PARITY_DEVICE", "auto").lower()
    device_type = _get_available_device_type()
    if requested != "auto":
        if device_type != requested:
            raise RuntimeError(
                f"requested parity device {requested!r}, but PyTorch selected "
                f"{device_type!r}"
            )
    if device_type is None:
        raise RuntimeError("GLM-5 parity requires an available accelerator")
    device_module = _get_device_module(device_type)
    device = torch.device(device_type, 0)
    device_module.set_device(device)
    return device, device_module, device_type


def _git_metadata_for_root(root: Path, prefix: str = "") -> dict[str, str]:
    values: dict[str, str] = {}
    commands = {
        "git_commit": ["rev-parse", "HEAD"],
        "git_branch": ["branch", "--show-current"],
        "git_status": [
            "status",
            "--short",
            "--untracked-files=no",
        ],
        "git_untracked": [
            "status",
            "--short",
            "--untracked-files=normal",
        ],
        "git_untracked_source": [
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "*.py",
            "*.toml",
        ],
    }
    for name, command in commands.items():
        key = f"{prefix}{name}"
        try:
            values[key] = subprocess.run(
                ["git", "-C", str(root), *command],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            values[key] = f"unavailable: {error}"
    return values


def _package_source_root(package: str) -> Path | None:
    spec = find_spec(package)
    if spec is None:
        return None
    if spec.submodule_search_locations:
        source = Path(next(iter(spec.submodule_search_locations))).resolve()
    elif spec.origin:
        source = Path(spec.origin).resolve().parent
    else:
        return None
    for candidate in (source, *source.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _git_metadata() -> dict[str, str]:
    """Record test and installed source identities for reproducible artifacts."""
    values = _git_metadata_for_root(Path.cwd())
    for package in ("torchtitan", "torchtitanturbo"):
        prefix = f"{package}_"
        source_root = _package_source_root(package)
        values[f"{prefix}source_root"] = (
            str(source_root) if source_root is not None else "unavailable"
        )
        if source_root is not None:
            values.update(_git_metadata_for_root(source_root, prefix))
    return values


class _TeeStream:
    """Mirror Python text output to the console and an artifact log."""

    def __init__(self, primary: Any, log: Any) -> None:
        self.primary = primary
        self.log = log
        self.encoding = getattr(primary, "encoding", "utf-8")

    def write(self, value: str) -> int:
        self.primary.write(value)
        self.log.write(value)
        return len(value)

    def flush(self) -> None:
        self.primary.flush()
        self.log.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.primary, "isatty", lambda: False)())

    def __getattr__(self, name: str) -> Any:
        return getattr(self.primary, name)


# 测试数据构造类
class ParityDataFactory:
    """Build repeatable batches without embedding data assumptions in tests."""

    @staticmethod
    def make(
        *,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int = 2,
        sequence_length: int = 16,
        seed: int = 41,
        vocab_size: int = 2048,
        hidden_size: int = 256,
        data_case: str = "random",
        make_tokens: bool = False,
    ) -> ParityBatch:
        data_case = data_case.lower()
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        # Generate deterministic synthetic data for repeatable comparisons.
        if data_case == "zeros":
            hidden_states = torch.zeros(
                batch_size, sequence_length, hidden_size,
                device=device, dtype=dtype,
            )
        elif data_case == "ones":
            hidden_states = torch.ones(
                batch_size, sequence_length, hidden_size,
                device=device, dtype=dtype,
            )
        elif data_case == "extreme":
            hidden_states = 20.0 * torch.randn(
                batch_size, sequence_length, hidden_size,
                device=device, dtype=dtype, generator=generator,
            )
        elif data_case == "alternating":
            values = torch.tensor(
                (-1.0, 1.0), device=device, dtype=dtype
            )
            pattern = torch.arange(
                batch_size * sequence_length * hidden_size,
                device=device,
            ).reshape(batch_size, sequence_length, hidden_size)
            hidden_states = values[(pattern % 2).long()]
        elif data_case == "random":
            hidden_states = torch.randn(
                batch_size, sequence_length, hidden_size,
                device=device, dtype=dtype, generator=generator,
            )
        else:
            raise ValueError(f"unsupported parity data case: {data_case}")
        positions = torch.arange(
            sequence_length, device=device, dtype=torch.long
        ).expand(batch_size, -1)
        causal_mask = _causal_mask(positions, dtype)
        if make_tokens:
            if data_case == "zeros":
                tokens = torch.zeros(
                    batch_size, sequence_length, device=device, dtype=torch.long
                )
            elif data_case == "ones":
                tokens = torch.full(
                    (batch_size, sequence_length), vocab_size - 1,
                    device=device, dtype=torch.long,
                )
            elif data_case == "alternating":
                tokens = torch.arange(
                    batch_size * sequence_length, device=device
                ).reshape(batch_size, sequence_length) % vocab_size
            else:
                tokens = torch.randint(
                    vocab_size,
                    (batch_size, sequence_length),
                    device=device,
                    dtype=torch.long,
                    generator=generator,
                )
        else:
            tokens = None
        return ParityBatch(
            hidden_states=hidden_states,
            positions=positions,
            causal_mask=causal_mask,
            hf_position_embeddings=(
                torch.empty(0, device=device),
                torch.empty(0, device=device),
            ),
            tokens=tokens,
        )

    @staticmethod
    def attach_hf_position_embeddings(
        batch: ParityBatch, hf_model: torch.nn.Module
    ) -> ParityBatch:
        batch.hf_position_embeddings = hf_model.model.rotary_emb(
            batch.hidden_states, position_ids=batch.positions
        )
        return batch

    @staticmethod
    def cast(batch: ParityBatch, *, model: torch.nn.Module, dtype: torch.dtype) -> ParityBatch:
        """Cast numeric inputs for one endpoint and recompute its RoPE cache."""
        hidden_states = batch.hidden_states.to(dtype=dtype)
        positions = batch.positions
        cast_batch = ParityBatch(
            hidden_states=hidden_states,
            positions=positions,
            causal_mask=batch.causal_mask.to(dtype=dtype),
            hf_position_embeddings=(
                torch.empty(0, device=hidden_states.device),
                torch.empty(0, device=hidden_states.device),
            ),
            tokens=batch.tokens,
        )
        if hasattr(model, "model"):
            cast_batch.hf_position_embeddings = model.model.rotary_emb(
                hidden_states, position_ids=positions
            )
        return cast_batch


@dataclass
class ParityModelPair:
    """A weight-identical HF/TorchTitan model pair."""

    hf: torch.nn.Module
    titan: torch.nn.Module
    device: torch.device
    precision: PrecisionPolicy

    def hf_layer(self, layer_index: int) -> torch.nn.Module:
        return self.hf.model.layers[layer_index]

    def titan_layer(self, layer_index: int) -> torch.nn.Module:
        return self.titan.layers[str(layer_index)]

    def convert(self, precision: PrecisionPolicy) -> None:
        self.hf.to(device=self.device, dtype=precision.dtype)
        self.titan.to(device=self.device, dtype=precision.dtype)
        for module in self.hf.modules():
            indexer = getattr(module, "indexer", None)
            if indexer is not None:
                indexer.weights_proj.float()
        for model in (self.hf, self.titan):
            indexer_weights = [
                (name, parameter)
                for name, parameter in model.named_parameters()
                if name.endswith("indexer.weights_proj.weight")
            ]
            expected = len(getattr(model, "model", model).layers)
            if len(indexer_weights) != expected:
                raise AssertionError(
                    "unexpected number of GLM-5 indexer weights: "
                    f"expected {expected}, got {len(indexer_weights)}"
                )
            for name, parameter in indexer_weights:
                if parameter.dtype is not torch.float32:
                    raise AssertionError(
                        f"{name} must remain FP32, got {parameter.dtype}"
                    )
        self.precision = precision


# 统一模型配置入口
@dataclass(frozen=True)
class ParityModelSize:
    """One size definition shared by HF and TorchTitan parity models."""

    # Balanced single-A100 profile. Core head/indexer geometry follows the
    # published GLM-5.2 config, while width and expert count are reduced.
    vocab_size: int = 154880
    dim: int = 2048 # 6144
    num_layers: int = 11 # 78
    num_dense_layers: int = 1 # 3
    num_attention_heads: int = 32 # 64
    q_lora_rank: int = 768 # 2044
    kv_lora_rank: int = 192 # 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    dense_hidden_dim: int = 4096 # 12288
    moe_hidden_dim: int = 768 # 2048
    num_experts: int = 32 # 256
    num_shared_experts: int = 1
    router_top_k: int = 8
    router_num_expert_groups: int = 1
    router_num_limited_groups: int = 1
    router_route_scale: float = 2.5
    index_num_heads: int = 32
    index_head_dim: int = 128
    index_top_k: int = 2048
    max_position_embeddings: int = 1048576
    rope_theta: float = 8_000_000.0
    rope_cache_max_seq_len: int = 128

    @classmethod
    def router_unit_profile(cls) -> "ParityModelSize":
        """Return a tiny profile for CPU-only router implementation tests."""
        return cls(
            vocab_size=2048,
            dim=256,
            num_layers=2,
            num_dense_layers=1,
            num_attention_heads=8,
            q_lora_rank=128,
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=32,
            v_head_dim=64,
            dense_hidden_dim=1024,
            moe_hidden_dim=256,
            num_experts=8,
            num_shared_experts=1,
            router_top_k=2,
            router_num_expert_groups=1,
            router_num_limited_groups=1,
            router_route_scale=2.5,
            index_num_heads=4,
            index_head_dim=64,
            index_top_k=8,
            max_position_embeddings=128,
            rope_theta=1_000_000.0,
            rope_cache_max_seq_len=128,
        )

    @classmethod
    def from_env(cls) -> "ParityModelSize":
        defaults = cls()

        def integer(name: str, value: int) -> int:
            return int(os.environ.get(f"GLM5_PARITY_MODEL_{name}", str(value)))

        return cls(
            vocab_size=integer("VOCAB_SIZE", defaults.vocab_size),
            dim=integer("DIM", defaults.dim),
            num_layers=integer("LAYERS", defaults.num_layers),
            num_dense_layers=integer(
                "DENSE_LAYERS", defaults.num_dense_layers
            ),
            num_attention_heads=integer(
                "ATTENTION_HEADS", defaults.num_attention_heads
            ),
            q_lora_rank=integer("Q_LORA_RANK", defaults.q_lora_rank),
            kv_lora_rank=integer("KV_LORA_RANK", defaults.kv_lora_rank),
            qk_nope_head_dim=integer(
                "QK_NOPE_HEAD_DIM", defaults.qk_nope_head_dim
            ),
            qk_rope_head_dim=integer(
                "QK_ROPE_HEAD_DIM", defaults.qk_rope_head_dim
            ),
            v_head_dim=integer("V_HEAD_DIM", defaults.v_head_dim),
            dense_hidden_dim=integer(
                "DENSE_HIDDEN_DIM", defaults.dense_hidden_dim
            ),
            moe_hidden_dim=integer(
                "MOE_HIDDEN_DIM", defaults.moe_hidden_dim
            ),
            num_experts=integer("EXPERTS", defaults.num_experts),
            num_shared_experts=integer(
                "SHARED_EXPERTS", defaults.num_shared_experts
            ),
            router_top_k=integer("ROUTER_TOP_K", defaults.router_top_k),
            router_num_expert_groups=integer(
                "EXPERT_GROUPS", defaults.router_num_expert_groups
            ),
            router_num_limited_groups=integer(
                "LIMITED_GROUPS", defaults.router_num_limited_groups
            ),
            router_route_scale=float(
                os.environ.get(
                    "GLM5_PARITY_MODEL_ROUTE_SCALE",
                    str(defaults.router_route_scale),
                )
            ),
            index_num_heads=integer(
                "INDEX_HEADS", defaults.index_num_heads
            ),
            index_head_dim=integer(
                "INDEX_HEAD_DIM", defaults.index_head_dim
            ),
            index_top_k=integer("INDEX_TOP_K", defaults.index_top_k),
            max_position_embeddings=integer(
                "MAX_POSITION_EMBEDDINGS",
                defaults.max_position_embeddings,
            ),
            rope_theta=float(
                os.environ.get(
                    "GLM5_PARITY_MODEL_ROPE_THETA",
                    str(defaults.rope_theta),
                )
            ),
            rope_cache_max_seq_len=integer(
                "ROPE_CACHE_MAX_SEQ_LEN",
                defaults.rope_cache_max_seq_len,
            ),
        )

    def validate(self, *, sequence_length: int) -> None:
        if self.num_layers <= 0:
            raise ValueError("GLM5 parity model must have at least one layer")
        if not 0 <= self.num_dense_layers <= self.num_layers:
            raise ValueError("dense layers must be in [0, num_layers]")
        if self.dim <= 0 or self.vocab_size <= 0:
            raise ValueError("model dim and vocabulary size must be positive")
        if self.max_position_embeddings < sequence_length:
            raise ValueError(
                "max position embeddings must cover the parity sequence length"
            )
        if self.rope_cache_max_seq_len < sequence_length:
            raise ValueError(
                "RoPE cache length must cover the parity sequence length"
            )

    @property
    def label(self) -> str:
        return (
            f"l{self.num_layers}-d{self.dim}-e{self.num_experts}"
            f"-f{self.moe_hidden_dim}"
        )

    @property
    def estimated_num_parameters(self) -> int:
        attention = (
            self.dim * self.q_lora_rank
            + self.q_lora_rank
            + self.q_lora_rank
            * self.num_attention_heads
            * (self.qk_nope_head_dim + self.qk_rope_head_dim)
            + self.dim * (self.kv_lora_rank + self.qk_rope_head_dim)
            + self.kv_lora_rank
            + self.kv_lora_rank
            * self.num_attention_heads
            * (self.qk_nope_head_dim + self.v_head_dim)
            + self.num_attention_heads * self.v_head_dim * self.dim
            + self.q_lora_rank * self.index_num_heads * self.index_head_dim
            + self.dim * self.index_head_dim
            + 2 * self.index_head_dim
            + self.dim * self.index_num_heads
            + 2 * self.dim
        )
        dense_ffn = 3 * self.dim * self.dense_hidden_dim
        moe_ffn = (
            self.dim * self.num_experts
            + 3 * self.num_experts * self.dim * self.moe_hidden_dim
            + 3
            * self.dim
            * self.moe_hidden_dim
            * self.num_shared_experts
        )
        model_io = 2 * self.vocab_size * self.dim + self.dim
        return (
            model_io
            + self.num_layers * attention
            + self.num_dense_layers * dense_ffn
            + (self.num_layers - self.num_dense_layers) * moe_ffn
        )

    @property
    def estimated_fp32_size_gb(self) -> float:
        return self.estimated_num_parameters * 4 / 1_000_000_000


def _hf_config(model_size: ParityModelSize) -> Any:
    assert GlmMoeDsaConfig is not None
    return GlmMoeDsaConfig(
        vocab_size=model_size.vocab_size,
        hidden_size=model_size.dim,
        intermediate_size=model_size.dense_hidden_dim,
        moe_intermediate_size=model_size.moe_hidden_dim,
        num_hidden_layers=model_size.num_layers,
        num_attention_heads=model_size.num_attention_heads,
        num_key_value_heads=model_size.num_attention_heads,
        n_shared_experts=model_size.num_shared_experts,
        n_routed_experts=model_size.num_experts,
        routed_scaling_factor=model_size.router_route_scale,
        kv_lora_rank=model_size.kv_lora_rank,
        q_lora_rank=model_size.q_lora_rank,
        qk_rope_head_dim=model_size.qk_rope_head_dim,
        qk_nope_head_dim=model_size.qk_nope_head_dim,
        v_head_dim=model_size.v_head_dim,
        n_group=model_size.router_num_expert_groups,
        topk_group=model_size.router_num_limited_groups,
        num_experts_per_tok=model_size.router_top_k,
        norm_topk_prob=True,
        max_position_embeddings=model_size.max_position_embeddings,
        rms_norm_eps=1e-5,
        attention_dropout=0.0,
        index_topk=model_size.index_top_k,
        index_head_dim=model_size.index_head_dim,
        index_n_heads=model_size.index_num_heads,
        first_k_dense_replace=model_size.num_dense_layers,
        indexer_types=["full"] * model_size.num_layers,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": model_size.rope_theta,
        },
        use_cache=False,
        _attn_implementation="eager",
    )


def _titan_config(model_size: ParityModelSize) -> Any:
    base = glm5_configs["debugmodel"]()
    rope = replace(
        base.layers[0].attention.rope,
        dim=model_size.qk_rope_head_dim,
        max_seq_len=model_size.rope_cache_max_seq_len,
        theta=model_size.rope_theta,
    )
    return replace(
        base,
        vocab_size=model_size.vocab_size,
        dim=model_size.dim,
        tok_embeddings=replace(
            base.tok_embeddings,
            num_embeddings=model_size.vocab_size,
            embedding_dim=model_size.dim,
        ),
        norm=replace(base.norm, normalized_shape=model_size.dim),
        lm_head=replace(
            base.lm_head,
            in_features=model_size.dim,
            out_features=model_size.vocab_size,
        ),
        layers=build_glm5_layers(
            n_layers=model_size.num_layers,
            n_dense_layers=model_size.num_dense_layers,
            dim=model_size.dim,
            n_heads=model_size.num_attention_heads,
            q_lora_rank=model_size.q_lora_rank,
            kv_lora_rank=model_size.kv_lora_rank,
            qk_nope_head_dim=model_size.qk_nope_head_dim,
            qk_rope_head_dim=model_size.qk_rope_head_dim,
            v_head_dim=model_size.v_head_dim,
            dense_hidden_dim=model_size.dense_hidden_dim,
            moe_hidden_dim=model_size.moe_hidden_dim,
            num_experts=model_size.num_experts,
            num_shared_experts=model_size.num_shared_experts,
            router_top_k=model_size.router_top_k,
            router_num_expert_groups=model_size.router_num_expert_groups,
            router_num_limited_groups=model_size.router_num_limited_groups,
            router_route_scale=model_size.router_route_scale,
            index_n_heads=model_size.index_num_heads,
            index_head_dim=model_size.index_head_dim,
            index_topk=model_size.index_top_k,
            attention_dropout=0.0,
            rope=rope,
        ),
    )


def _build_pair(
    device: torch.device,
    *,
    model_size: ParityModelSize | None = None,
    precision: PrecisionPolicy = FP32,
    seed: int = 41,
) -> ParityModelPair:
    assert GlmMoeDsaForCausalLM is not None
    model_size = model_size or ParityModelSize.from_env()
    torch.manual_seed(seed)
    # Build both implementations from one HF state dict so every comparison
    # tests execution differences rather than unrelated random weights.
    hf_model = GlmMoeDsaForCausalLM(_hf_config(model_size)).float()
    titan_config = _titan_config(model_size)
    titan_model = titan_config.build()
    titan_model.init_states()  # Initialize decoder runtime state.
    adapter = Glm5StateDictAdapter(titan_config, hf_assets_path=None)
    # The adapter is the single source of truth for HF -> TorchTitan names.
    titan_model.load_state_dict(
        adapter.from_hf(hf_model.state_dict()), strict=True
    )
    pair = ParityModelPair(
        hf=hf_model.to(device).eval(),
        titan=titan_model.to(device).eval(),
        device=device,
        precision=FP32,
    )
    if precision.dtype is not torch.float32:
        pair.convert(precision)
    return pair


def _set_hf_routed_expert_compute_dtype(
    hf_model: torch.nn.Module,
    compute_dtype: torch.dtype,
    *,
    use_grouped_mm: bool,
) -> None:
    """Run HF routed experts with TorchTitan's explicit dtype boundaries."""
    if compute_dtype is not torch.bfloat16:
        raise ValueError(
            "the routed-expert reference currently supports only torch.bfloat16"
        )

    num_wrapped = 0
    for layer in hf_model.model.layers:
        experts = getattr(layer.mlp, "experts", None)
        if experts is None:
            continue

        def forward_with_explicit_dtype(
            self,
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
            _compute_dtype=compute_dtype,
            _use_grouped_mm=use_grouped_mm,
        ) -> torch.Tensor:
            if not _use_grouped_mm:
                final_hidden_states = torch.zeros_like(hidden_states)
                expert_mask = F.one_hot(
                    top_k_index, num_classes=self.num_experts
                ).permute(2, 1, 0)
                expert_hit = torch.greater(
                    expert_mask.sum(dim=(-1, -2)), 0
                ).nonzero()
                for expert_index_tensor in expert_hit:
                    expert_index = expert_index_tensor[0]
                    top_k_position, token_index = torch.where(
                        expert_mask[expert_index]
                    )
                    current_state = hidden_states[token_index].to(_compute_dtype)
                    gate_weight, up_weight = self.gate_up_proj[
                        expert_index
                    ].chunk(2, dim=0)
                    gate = F.linear(
                        current_state, gate_weight.to(_compute_dtype)
                    )
                    up = F.linear(
                        current_state, up_weight.to(_compute_dtype)
                    )
                    current_hidden_states = self.act_fn(gate) * up
                    current_hidden_states = F.linear(
                        current_hidden_states,
                        self.down_proj[expert_index].to(_compute_dtype),
                    ).to(hidden_states.dtype)
                    routed_weights = top_k_weights[
                        token_index, top_k_position, None
                    ].to(current_hidden_states.dtype)
                    final_hidden_states.index_add_(
                        0,
                        token_index,
                        current_hidden_states * routed_weights,
                    )
                return final_hidden_states

            # Mirror LocalTokenDispatcher._local_reorder exactly.  Stable
            # expert sorting determines both grouped-MM rows and combine order.
            top_k = top_k_index.shape[-1]
            expert_order = torch.argsort(top_k_index.reshape(-1), stable=True)
            token_indices = expert_order // top_k
            routed_input = hidden_states[token_indices]
            routed_scores = top_k_weights.reshape(-1)[expert_order]

            routing_map = torch.zeros(
                hidden_states.shape[0],
                self.num_experts,
                dtype=torch.bool,
                device=hidden_states.device,
            ).scatter_(1, top_k_index, True)
            tokens_per_expert = routing_map.sum(dim=0)
            offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

            gate_weight, up_weight = self.gate_up_proj.chunk(2, dim=1)
            gate = torch._grouped_mm(
                routed_input.to(_compute_dtype),
                gate_weight.to(_compute_dtype).transpose(-2, -1),
                offs=offsets,
            )
            up = torch._grouped_mm(
                routed_input.to(_compute_dtype),
                up_weight.to(_compute_dtype).transpose(-2, -1),
                offs=offsets,
            )
            routed_output = torch._grouped_mm(
                self.act_fn(gate) * up,
                self.down_proj.to(_compute_dtype).transpose(-2, -1),
                offs=offsets,
            ).to(hidden_states.dtype)

            # Mirror LocalTokenDispatcher.combine: scores multiply in FP32,
            # then deterministic scatter-add restores original token order.
            routed_output = (
                routed_output.to(torch.float32) * routed_scores.reshape(-1, 1)
            ).to(hidden_states.dtype)
            final_hidden_states = torch.zeros_like(hidden_states)
            return deterministic_scatter_add(
                final_hidden_states,
                token_indices.reshape(-1, 1).expand_as(routed_output),
                routed_output,
            )

        experts.forward = MethodType(forward_with_explicit_dtype, experts)
        experts._parity_compute_dtype = compute_dtype
        experts._parity_uses_grouped_mm = use_grouped_mm
        num_wrapped += 1
    if num_wrapped == 0:
        raise ValueError("HF model has no routed-expert modules to wrap")


def _set_titan_routed_expert_compute_dtype(
    titan_model: torch.nn.Module,
    compute_dtype: torch.dtype,
) -> None:
    """Run test-model TorchTitan inner experts with eager FP32 linears."""
    if compute_dtype is not torch.float32:
        raise ValueError(
            "the TorchTitan routed-expert experiment supports only FP32"
        )

    num_wrapped = 0
    for layer in titan_model.layers.values():
        if not getattr(layer, "moe_enabled", False):
            continue
        experts = layer.moe.routed_experts.inner_experts

        def forward_with_fp32(
            self,
            x_RD: torch.Tensor,
            num_tokens_per_expert_E: torch.Tensor,
            _compute_dtype=compute_dtype,
        ) -> torch.Tensor:
            def local_tensor(value: torch.Tensor) -> torch.Tensor:
                return value.to_local() if hasattr(value, "to_local") else value

            w1 = local_tensor(self.w1_EFD)
            w2 = local_tensor(self.w2_EDF)
            w3 = local_tensor(self.w3_EFD)
            token_counts = [
                int(value)
                for value in num_tokens_per_expert_E.detach().cpu().tolist()
            ]
            if sum(token_counts) != x_RD.shape[0]:
                raise ValueError(
                    "expert token counts do not match the dispatched input: "
                    f"counts={sum(token_counts)}, rows={x_RD.shape[0]}"
                )

            outputs: list[torch.Tensor] = []
            start = 0
            for expert_index, token_count in enumerate(token_counts):
                end = start + token_count
                if token_count:
                    current = x_RD[start:end].to(_compute_dtype)
                    hidden = F.silu(
                        F.linear(current, w1[expert_index].to(_compute_dtype))
                    )
                    hidden = hidden * F.linear(
                        current, w3[expert_index].to(_compute_dtype)
                    )
                    outputs.append(
                        F.linear(
                            hidden,
                            w2[expert_index].to(_compute_dtype),
                        )
                    )
                start = end

            if not outputs:
                return x_RD.new_empty((0, x_RD.shape[-1]))
            return torch.cat(outputs, dim=0).to(x_RD.dtype)

        experts.forward = MethodType(forward_with_fp32, experts)
        experts._parity_compute_dtype = compute_dtype
        experts._parity_uses_grouped_mm = False
        num_wrapped += 1

    if num_wrapped == 0:
        raise ValueError("TorchTitan model has no routed-expert modules to wrap")


def _annotate_known_compute_dtypes(pair: ParityModelPair) -> None:
    """Expose explicit mixed-precision contracts to module trace hooks."""
    for module in pair.hf.modules():
        if module.__class__.__name__ == "GlmMoeDsaRMSNorm":
            module._parity_compute_dtype = torch.float32
    for layer_index in range(len(pair.hf.model.layers)):
        hf_attention = pair.hf_layer(layer_index).self_attn
        titan_attention = pair.titan_layer(layer_index).attention
        hf_attention._parity_compute_dtype = "mixed(fp32_softmax)"
        titan_attention._parity_compute_dtype = "mixed(fp32_softmax)"
        hf_attention.indexer._parity_compute_dtype = "mixed(fp32_scores)"
        titan_attention.indexer._parity_compute_dtype = "mixed(fp32_scores)"
        titan_layer = pair.titan_layer(layer_index)
        if getattr(titan_layer, "moe_enabled", False):
            titan_layer.moe.router.gate._parity_compute_dtype = torch.float32
            experts = titan_layer.moe.routed_experts.inner_experts
            if not hasattr(experts, "_parity_compute_dtype"):
                experts._parity_compute_dtype = torch.bfloat16


def _annotate_endpoint_compute_dtypes(
    endpoint: ModelEndpoint,
    model: torch.nn.Module,
) -> None:
    """Apply the same dtype trace annotations to one captured endpoint."""
    if endpoint.implementation == "hf":
        for module in model.modules():
            if module.__class__.__name__ == "GlmMoeDsaRMSNorm":
                module._parity_compute_dtype = torch.float32
        for layer in model.model.layers:
            layer.self_attn._parity_compute_dtype = "mixed(fp32_softmax)"
            layer.self_attn.indexer._parity_compute_dtype = "mixed(fp32_scores)"
        return
    for layer in model.layers.values():
        layer.attention._parity_compute_dtype = "mixed(fp32_softmax)"
        layer.attention.indexer._parity_compute_dtype = "mixed(fp32_scores)"
        if getattr(layer, "moe_enabled", False):
            layer.moe.router.gate._parity_compute_dtype = torch.float32
            experts = layer.moe.routed_experts.inner_experts
            if not hasattr(experts, "_parity_compute_dtype"):
                experts._parity_compute_dtype = torch.bfloat16


def _build_models(
    device: torch.device, *, seed: int = 41
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Backward-compatible FP32 constructor used by older parity callers."""
    pair = _build_pair(device, precision=FP32, seed=seed)
    return pair.hf, pair.titan


def _convert_models_to_bfloat16(
    hf_model: torch.nn.Module,
    titan_model: torch.nn.Module,
    device: torch.device,
) -> None:
    """Backward-compatible conversion that preserves FP32 indexer weights."""
    pair = ParityModelPair(hf_model, titan_model, device, FP32)
    pair.convert(BF16)


def _selection_mismatch_coordinates(
    actual_BLK: torch.Tensor,
    expected_BLK: torch.Tensor,
    positions_BL: torch.Tensor | None = None,
) -> list[tuple[int, int]]:
    """Return batch and query coordinates whose selected sets differ."""
    if actual_BLK.shape != expected_BLK.shape:
        raise ValueError(
            "selection shapes differ: "
            f"actual={tuple(actual_BLK.shape)}, expected={tuple(expected_BLK.shape)}"
        )
    if actual_BLK.ndim != 3:
        raise ValueError(
            f"selections must have shape [B, L, K], got {actual_BLK.shape}"
        )
    if positions_BL is not None and positions_BL.shape != actual_BLK.shape[:2]:
        raise ValueError(
            "positions must match the selection batch and sequence dimensions: "
            f"positions={tuple(positions_BL.shape)}, selections={tuple(actual_BLK.shape)}"
        )
    actual_cpu = actual_BLK.detach().cpu()
    expected_cpu = expected_BLK.detach().cpu()
    positions_cpu = None if positions_BL is None else positions_BL.detach().cpu()
    mismatched_coordinates: list[tuple[int, int]] = []
    for batch_index in range(actual_cpu.shape[0]):
        for sequence_index in range(actual_cpu.shape[1]):
            causal_position = (
                None
                if positions_cpu is None
                else int(positions_cpu[batch_index, sequence_index])
            )
            actual = {
                int(index)
                for index in actual_cpu[batch_index, sequence_index]
                if causal_position is None or int(index) <= causal_position
            }
            expected = {
                int(index)
                for index in expected_cpu[batch_index, sequence_index]
                if causal_position is None or int(index) <= causal_position
            }
            if actual != expected:
                mismatched_coordinates.append((batch_index, sequence_index))
    return mismatched_coordinates


def _ordered_mismatch_coordinates(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> list[tuple[int, int]]:
    """Return batch and query coordinates with different ordered values."""
    if actual.shape != expected.shape or actual.ndim != 3:
        return []
    coordinates = []
    for batch_index in range(actual.shape[0]):
        for position in range(actual.shape[1]):
            if not torch.equal(
                actual[batch_index, position],
                expected[batch_index, position],
            ):
                coordinates.append((batch_index, position))
    return coordinates


def _replay_titan_indexer_scores(
    q_projection: torch.Tensor,
    q_rotated: torch.Tensor,
    k_normalized: torch.Tensor,
    k_rotated: torch.Tensor,
    projected_weights: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
) -> torch.Tensor:
    """Reconstruct TorchTitan index scores from captured indexer tensors."""
    batch_size, sequence_length = q_projection.shape[:2]
    q_full = q_projection.reshape(
        batch_size, sequence_length, n_heads, head_dim
    )
    q_pass = q_full[..., rope_head_dim:]
    k_pass = k_normalized[..., rope_head_dim:]
    q = torch.cat((q_rotated, q_pass), dim=-1).float()
    k = torch.cat((k_rotated.squeeze(2), k_pass), dim=-1).float()
    scores = torch.matmul(
        q.transpose(1, 2),
        k.transpose(1, 2).unsqueeze(1),
    ) * (head_dim**-0.5)
    scores = F.relu(scores)
    weights = projected_weights.float() * (n_heads**-0.5)
    index_scores = torch.matmul(
        weights.unsqueeze(-2), scores.transpose(1, 2)
    ).squeeze(-2)
    if attention_mask.ndim == 4:
        attention_mask = attention_mask[:, 0]
    if attention_mask.shape != index_scores.shape:
        raise ValueError(
            "captured attention mask does not match replayed index scores: "
            f"mask={tuple(attention_mask.shape)}, "
            f"scores={tuple(index_scores.shape)}"
        )
    return index_scores + attention_mask.float()


@torch.no_grad()
def _capture_titan_indexer_scores(
    indexer: torch.nn.Module,
    inputs: tuple[Any, ...],
) -> torch.Tensor:
    """Evaluate the indexer score path without applying top-k."""
    if len(inputs) != 4:
        raise ValueError(
            f"TorchTitan indexer expected four inputs, got {len(inputs)}"
        )
    hidden_states, q_residual, positions, attention_mask = inputs
    batch_size, sequence_length = hidden_states.shape[:2]
    q_projection = indexer.wq_b(q_residual).view(
        batch_size,
        sequence_length,
        indexer.n_heads,
        indexer.head_dim,
    )
    q_rotated, _q_pass = torch.split(
        q_projection,
        [indexer.qk_rope_head_dim, indexer.head_dim - indexer.qk_rope_head_dim],
        dim=-1,
    )
    k_normalized = indexer.k_norm(indexer.wk(hidden_states))
    k_rotated, _k_pass = torch.split(
        k_normalized.unsqueeze(2),
        [indexer.qk_rope_head_dim, indexer.head_dim - indexer.qk_rope_head_dim],
        dim=-1,
    )
    q_rotated, k_rotated = indexer.rope(
        q_rotated, k_rotated, positions
    )
    projected_weights = indexer.weights_proj(
        hidden_states.to(indexer.weights_proj.weight.dtype)
    )
    return _replay_titan_indexer_scores(
        q_projection,
        q_rotated,
        k_normalized,
        k_rotated,
        projected_weights,
        attention_mask,
        n_heads=indexer.n_heads,
        head_dim=indexer.head_dim,
        rope_head_dim=indexer.qk_rope_head_dim,
    )


def _causal_mask(positions_BL: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    batch_size, sequence_length = positions_BL.shape
    token_indices = torch.arange(sequence_length, device=positions_BL.device)
    allowed = token_indices[None, :, None] >= token_indices[None, None, :]
    return torch.zeros(
        batch_size,
        1,
        sequence_length,
        sequence_length,
        dtype=dtype,
        device=positions_BL.device,
    ).masked_fill(~allowed.unsqueeze(1), torch.finfo(dtype).min)


@dataclass
class ComparisonResult:
    scope: str
    component: str
    precision: str
    layer: int | str
    samples: int
    passed: bool
    max_abs: float | None = None
    max_rel: float | None = None
    mean_abs: float | None = None
    mean_rel: float | None = None
    relative_l2: float | None = None
    cosine_similarity: float | None = None
    rmse: float | None = None
    mismatch_count: int = 0
    mismatch_positions: list[int] = field(default_factory=list)
    peak_position: int | None = None
    module_path: str = ""
    detail: str = ""
    parent_path: str = ""
    level: int = 0
    node_kind: str = "checkpoint"
    checkpoint: bool = True
    actual_summary: str = ""
    expected_summary: str = ""
    actual_dtype: str = ""
    expected_dtype: str = ""
    explosion: bool = False
    growth_ratio: float | None = None


class ParityRecorder:
    """Collect comparable scalar rows and render a compact report."""

    def __init__(
        self,
        precision: PrecisionPolicy,
        *,
        precision_label: str | None = None,
        title: str | None = None,
    ):
        self.precision = precision
        self.precision_label = precision_label or precision.name
        self.title = title or self.precision_label
        self.results: list[ComparisonResult] = []

    @staticmethod
    def _cpu(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().float().cpu()

    @staticmethod
    def _numeric_summary(tensor: torch.Tensor) -> str:
        value = tensor.detach().cpu()
        if value.numel() <= 8:
            values = value.reshape(-1).tolist()
            formatted = []
            for item in values:
                formatted.append(
                    str(int(item)) if isinstance(item, int) else f"{float(item):.6g}"
                )
            return "[" + ", ".join(formatted) + "]"
        return f"shape={tuple(value.shape)}"

    @staticmethod
    def _discrete_summary(
        tensor: torch.Tensor,
        positions: torch.Tensor | None,
        mismatch_positions: list[int],
        value_name: str,
    ) -> str:
        value = tensor.detach().cpu()
        if value.ndim == 3:
            sequence_length = value.shape[1]
            selected = sorted(set(mismatch_positions)) or [0]
            entries = []
            for position in selected[:8]:
                if position >= sequence_length:
                    continue
                entries.append(
                    f"p{position}: {value_name}="
                    f"{value[0, position].reshape(-1).tolist()}"
                )
            suffix = "" if len(selected) <= 8 else f" ... (+{len(selected) - 8})"
            return "; ".join(entries) + suffix
        return f"{value_name}={value.reshape(-1).tolist()}"

    @staticmethod
    def _discrete_mismatch_detail(
        actual: torch.Tensor,
        expected: torch.Tensor,
        coordinates: list[tuple[int, int]],
        value_name: str,
    ) -> str:
        entries = []
        for batch_index, position in coordinates:
            actual_values = actual[batch_index, position].reshape(-1).tolist()
            expected_values = expected[batch_index, position].reshape(-1).tolist()
            entries.append(
                f"b{batch_index},p{position} actual {value_name}="
                f"{actual_values} expected {value_name}={expected_values}"
            )
        return "; ".join(entries)

    @staticmethod
    def _canonical_module_path(module_path: str) -> str:
        raw = module_path.split("<->", 1)[0].strip()
        if ":" in raw:
            raw = raw.rsplit(":", 1)[-1]
        replacements = (
            ("model.layers.", "layers."),
            (".self_attn.indexer", ".attention.indexer"),
            (".self_attn", ".attention"),
            (".input_layernorm", ".attention_norm"),
            (".post_attention_layernorm", ".ffn_norm"),
            (".q_a_proj", ".wq_a"),
            (".q_a_layernorm", ".q_norm"),
            (".q_b_proj", ".wq_b"),
            (".kv_a_proj_with_mqa", ".wkv_a"),
            (".kv_a_layernorm", ".kv_norm"),
            (".kv_b_proj", ".wkv_b"),
            (".o_proj", ".wo"),
            (".mlp.gate_proj", ".feed_forward.w1"),
            (".mlp.up_proj", ".feed_forward.w3"),
            (".mlp.down_proj", ".feed_forward.w2"),
            (".mlp.gate", ".moe.router"),
            ("model.embed_tokens", "tok_embeddings"),
            ("model.norm", "norm"),
        )
        for source, target in replacements:
            raw = raw.replace(source, target)
        return raw

    @classmethod
    def _display_path(cls, result: ComparisonResult) -> str:
        if result.component in {"logits", "loss"}:
            return result.component
        if result.module_path:
            canonical = cls._canonical_module_path(result.module_path)
            if canonical:
                if result.component == "router_indices":
                    return f"{canonical}.indices"
                if result.component == "router_weights":
                    return f"{canonical}.weights"
                return canonical
        if result.parent_path == "layers" and isinstance(result.layer, int):
            return f"layers.{result.layer}"
        if result.parent_path:
            return f"{result.parent_path}.{result.component}"
        if result.component:
            return result.component
        return cls._canonical_module_path(result.module_path)

    @staticmethod
    def _hf_module_path(result: ComparisonResult) -> str:
        for endpoint_path in result.module_path.split("<->"):
            value = endpoint_path.strip()
            if not value.startswith("hf:"):
                continue
            path = value.removeprefix("hf:")
            path = re.sub(
                r"^(?:fp32|bf16|bfloat16)@[^:]+(?:\[[^]]+\])?:",
                "",
                path,
            )
            for precision in ("fp32:", "bf16:", "bfloat16:"):
                if path.startswith(precision):
                    path = path.removeprefix(precision)
                    break
            if path.startswith("layers."):
                return "-"
            return path
        return "-"

    @classmethod
    def _path_key(cls, result: ComparisonResult) -> tuple[tuple[int, object], ...]:
        path = cls._display_path(result)
        parts = [part for part in re.split(r"(\d+)", path) if part]
        return tuple(
            (0, int(part)) if part.isdigit() else (1, part)
            for part in parts
        )

    @staticmethod
    def _phase(result: ComparisonResult) -> int:
        """Keep forward evidence ahead of outputs, backward, and parameters."""
        if result.component in {"logits", "loss"}:
            return 1
        if result.scope == "activation_gradient":
            return 2
        if result.scope == "gradient":
            return 3
        if result.scope == "parameters":
            return 4
        return 0

    @classmethod
    def _result_key(cls, result: ComparisonResult) -> tuple[Any, ...]:
        path = cls._display_path(result)
        layer_match = re.match(r"layers\.(\d+)(?:\.|$)", path)
        if layer_match:
            dependency_order = -1 if result.component == "q_residual" else 0
            return (
                cls._phase(result),
                0,
                int(layer_match.group(1)),
                dependency_order,
                cls._path_key(result),
            )
        return cls._phase(result), 1, 0, 0, cls._path_key(result)

    def ordered_results(self) -> list[ComparisonResult]:
        self._annotate_explosions()
        return sorted(self.results, key=self._result_key)

    def _annotate_explosions(self) -> None:
        for result in self.results:
            result.explosion = False
            result.growth_ratio = None
        threshold = float(
            os.environ.get("GLM5_PARITY_EXPLOSION_RATIO", "8.0")
        )
        numerical_floor = torch.finfo(self.precision.dtype).eps * 0.1
        rows: dict[str, ComparisonResult] = {}
        for result in self.results:
            if result.max_abs is None or not isinstance(result.layer, int):
                continue
            if self._phase(result) != 0:
                continue
            path = self._display_path(result)
            previous = rows.get(path)
            if previous is None or result.checkpoint:
                rows[path] = result

        previous_layer_error = numerical_floor
        layers = sorted(
            {
                result.layer
                for result in rows.values()
                if isinstance(result.layer, int)
            }
        )
        for layer in layers:
            prefix = f"layers.{layer}"
            sequence = (
                f"{prefix}.attention_norm",
                f"{prefix}.attention",
                f"{prefix}.ffn_norm",
                f"{prefix}.feed_forward",
                f"{prefix}.moe",
                prefix,
            )
            baseline = previous_layer_error
            for path in sequence:
                result = rows.get(path)
                if result is None or result.max_abs is None:
                    continue
                ratio = result.max_abs / max(baseline, numerical_floor)
                result.growth_ratio = ratio
                if ratio >= threshold and result.max_abs >= numerical_floor:
                    result.explosion = True
                baseline = max(result.max_abs, numerical_floor)
            layer_result = rows.get(prefix)
            if layer_result is not None and layer_result.max_abs is not None:
                previous_layer_error = max(
                    layer_result.max_abs, numerical_floor
                )

        # Component reports often have no decoder-level checkpoints.  In that
        # case, compare the same logical component across adjacent layers so a
        # sudden cross-layer increase is still highlighted.
        component_series: dict[str, dict[int, ComparisonResult]] = {}
        for result in self.results:
            if result.max_abs is None or not isinstance(result.layer, int):
                continue
            if self._phase(result) != 0:
                continue
            path = self._display_path(result)
            if not path.startswith(f"layers.{result.layer}."):
                continue
            series_name = re.sub(r"^layers\.\d+\.", "", path)
            by_layer = component_series.setdefault(series_name, {})
            existing = by_layer.get(result.layer)
            if existing is None or result.checkpoint:
                by_layer[result.layer] = result
        for by_layer in component_series.values():
            previous_error: float | None = None
            for layer in sorted(by_layer):
                result = by_layer[layer]
                if previous_error is not None and result.max_abs is not None:
                    ratio = result.max_abs / max(
                        previous_error, numerical_floor
                    )
                    if result.growth_ratio is None or ratio > result.growth_ratio:
                        result.growth_ratio = ratio
                    if ratio >= threshold and result.max_abs >= numerical_floor:
                        result.explosion = True
                previous_error = max(result.max_abs or 0.0, numerical_floor)

    def summary(self) -> str:
        checkpoint_total = sum(result.checkpoint for result in self.results)
        trace_total = len(self.results) - checkpoint_total
        passed = checkpoint_total - len(self.failed)
        rate = (
            100.0
            if checkpoint_total == 0
            else 100.0 * passed / checkpoint_total
        )
        component_names = sorted({result.component for result in self.results})
        return (
            f"rows={len(self.results)} checkpoints={checkpoint_total} "
            f"trace_rows={trace_total} passed={passed} "
            f"failed={len(self.failed)} pass_rate={rate:.1f}% "
            f"components={','.join(component_names)}"
        )

    def tensor(
        self,
        *,
        scope: str,
        component: str,
        layer: int | str,
        actual: torch.Tensor,
        expected: torch.Tensor,
        rtol: float | None = None,
        atol: float | None = None,
        module_path: str = "",
        parent_path: str = "",
        level: int = 0,
        node_kind: str = "checkpoint",
        checkpoint: bool = True,
        actual_dtype: str | None = None,
        expected_dtype: str | None = None,
    ) -> ComparisonResult:
        actual_dtype = actual_dtype or f"out={str(actual.dtype).removeprefix('torch.')}"
        expected_dtype = expected_dtype or f"out={str(expected.dtype).removeprefix('torch.')}"
        actual_cpu = self._cpu(actual)
        expected_cpu = self._cpu(expected)
        rtol = self.precision.rtol if rtol is None else rtol
        atol = self.precision.atol if atol is None else atol
        if actual_cpu.shape != expected_cpu.shape:
            result = ComparisonResult(
                scope, component, self.precision_label, layer, 0, False,
                module_path=module_path,
                detail=f"shape {tuple(actual_cpu.shape)} != {tuple(expected_cpu.shape)}",
                parent_path=parent_path,
                level=level,
                node_kind=node_kind,
                checkpoint=checkpoint,
                actual_summary=f"shape={tuple(actual_cpu.shape)}",
                expected_summary=f"shape={tuple(expected_cpu.shape)}",
                actual_dtype=actual_dtype,
                expected_dtype=expected_dtype,
            )
        else:
            diff = (actual_cpu - expected_cpu).abs()
            scale = expected_cpu.abs().clamp_min(torch.finfo(torch.float32).tiny)
            symmetric_scale = (actual_cpu.abs() + expected_cpu.abs()).clamp_min(
                torch.finfo(torch.float32).eps
            )
            diff_l2 = torch.linalg.vector_norm(diff.reshape(-1))
            expected_l2 = torch.linalg.vector_norm(expected_cpu.reshape(-1))
            cosine_denominator = (
                torch.linalg.vector_norm(actual_cpu.reshape(-1)) * expected_l2
            )
            if cosine_denominator == 0:
                cosine_similarity = 1.0 if torch.equal(actual_cpu, expected_cpu) else 0.0
            else:
                cosine_similarity = float(
                    torch.dot(actual_cpu.reshape(-1), expected_cpu.reshape(-1))
                    / cosine_denominator
                )
            peak_position = None
            mismatch_positions: list[int] = []
            detail = ""
            if diff.ndim >= 3:
                position_max = diff.amax(dim=tuple([0] + list(range(2, diff.ndim))))
                peak_position = int(position_max.argmax())
                mismatch_mask = diff > atol + rtol * expected_cpu.abs()
                position_mismatch = mismatch_mask.any(dim=tuple([0] + list(range(2, diff.ndim))))
                mismatch_positions = [
                    index for index, value in enumerate(position_mismatch.tolist()) if value
                ]
                detail = f"peak_position={peak_position}"
            mismatch_count = int(
                (diff > atol + rtol * expected_cpu.abs()).sum().item()
            )
            result = ComparisonResult(
                scope,
                component,
                self.precision_label,
                layer,
                actual_cpu.numel(),
                bool(torch.all(diff <= atol + rtol * expected_cpu.abs())),
                float(diff.max()) if diff.numel() else 0.0,
                float((diff / scale).max()) if diff.numel() else 0.0,
                mean_abs=float(diff.mean()) if diff.numel() else 0.0,
                mean_rel=(
                    float((2.0 * diff / symmetric_scale).mean())
                    if diff.numel()
                    else 0.0
                ),
                relative_l2=float(
                    diff_l2 / expected_l2.clamp_min(torch.finfo(torch.float32).eps)
                ),
                cosine_similarity=cosine_similarity,
                rmse=(
                    float(torch.sqrt(torch.mean(diff.square())))
                    if diff.numel()
                    else 0.0
                ),
                mismatch_count=mismatch_count,
                mismatch_positions=mismatch_positions,
                peak_position=peak_position,
                module_path=module_path,
                detail=detail,
                parent_path=parent_path,
                level=level,
                node_kind=node_kind,
                checkpoint=checkpoint,
                actual_summary=self._numeric_summary(actual_cpu),
                expected_summary=self._numeric_summary(expected_cpu),
                actual_dtype=actual_dtype,
                expected_dtype=expected_dtype,
            )
        self.results.append(result)
        return result

    def discrete(
        self,
        *,
        scope: str,
        component: str,
        layer: int | str,
        actual: torch.Tensor,
        expected: torch.Tensor,
        positions: torch.Tensor | None = None,
        module_path: str = "",
        parent_path: str = "",
        level: int = 0,
        node_kind: str = "checkpoint",
        checkpoint: bool = True,
        actual_dtype: str | None = None,
        expected_dtype: str | None = None,
    ) -> ComparisonResult:
        actual_cpu = actual.detach().cpu()
        expected_cpu = expected.detach().cpu()
        mismatch_positions: set[int] = set()
        mismatch_coordinates: list[tuple[int, int]] = []
        mismatch_count = 0
        detail = ""
        if actual_cpu.shape != expected_cpu.shape:
            passed = False
            detail = f"shape {tuple(actual_cpu.shape)} != {tuple(expected_cpu.shape)}"
        elif actual_cpu.ndim < 2:
            passed = bool(torch.equal(actual_cpu, expected_cpu))
            mismatch_count = 0 if passed else 1
        elif positions is not None and actual_cpu.ndim == 3:
            # Indexer selections are causal.  Differences in future slots are
            # not observable at that query and must not be reported as fails.
            mismatch_coordinates = _selection_mismatch_coordinates(
                actual_cpu, expected_cpu, positions
            )
            mismatch_positions = {
                sequence_index
                for _batch_index, sequence_index in mismatch_coordinates
            }
            mismatch_count = len(mismatch_coordinates)
            passed = mismatch_count == 0
            if mismatch_coordinates:
                detail = self._discrete_mismatch_detail(
                    actual_cpu,
                    expected_cpu,
                    mismatch_coordinates,
                    "topk" if component == "indexer" else "indices",
                )
        else:
            passed = True
            for batch_index in range(actual_cpu.shape[0]):
                for sequence_index in range(actual_cpu.shape[1]):
                    left = actual_cpu[batch_index, sequence_index].reshape(-1)
                    right = expected_cpu[batch_index, sequence_index].reshape(-1)
                    if not torch.equal(left, right):
                        passed = False
                        mismatch_count += 1
                        mismatch_positions.add(sequence_index)
                        mismatch_coordinates.append(
                            (batch_index, sequence_index)
                        )
            if mismatch_coordinates:
                detail = self._discrete_mismatch_detail(
                    actual_cpu,
                    expected_cpu,
                    mismatch_coordinates,
                    "topk" if component == "indexer" else "indices",
                )
        result = ComparisonResult(
            scope,
            component,
            self.precision_label,
            layer,
            int(actual_cpu.numel()),
            passed,
            mismatch_count=mismatch_count,
            mismatch_positions=sorted(mismatch_positions),
            module_path=module_path,
            detail=detail,
            parent_path=parent_path,
            level=level,
            node_kind=node_kind,
            checkpoint=checkpoint,
            actual_summary=self._discrete_summary(
                actual_cpu,
                positions,
                sorted(mismatch_positions),
                "topk" if component == "indexer" else "indices",
            ),
            expected_summary=self._discrete_summary(
                expected_cpu,
                positions,
                sorted(mismatch_positions),
                "topk" if component == "indexer" else "indices",
            ),
            actual_dtype=(
                actual_dtype
                or f"out={str(actual.dtype).removeprefix('torch.')}"
            ),
            expected_dtype=(
                expected_dtype
                or f"out={str(expected.dtype).removeprefix('torch.')}"
            ),
        )
        self.results.append(result)
        return result

    def missing(
        self,
        *,
        scope: str,
        component: str,
        layer: int | str,
        module_path: str,
        detail: str,
        parent_path: str = "",
        level: int = 0,
        node_kind: str = "activation",
        checkpoint: bool = True,
    ) -> ComparisonResult:
        """Record a missing hook/output instead of silently dropping a row."""
        result = ComparisonResult(
            scope=scope,
            component=component,
            precision=self.precision_label,
            layer=layer,
            samples=0,
            passed=not checkpoint,
            module_path=module_path,
            detail=detail,
            parent_path=parent_path,
            level=level,
            node_kind=node_kind,
            checkpoint=checkpoint,
            actual_summary="present" if "actual=True" in detail else "missing",
            expected_summary="present" if "expected=True" in detail else "missing",
        )
        self.results.append(result)
        return result

    @property
    def failed(self) -> list[ComparisonResult]:
        return [
            result for result in self.results
            if result.checkpoint and not result.passed
        ]

    @staticmethod
    def _status(result: ComparisonResult) -> str:
        if not result.checkpoint:
            return "TRACE"
        return "PASS" if result.passed else "FAIL"

    @staticmethod
    def _mismatch_display(result: ComparisonResult) -> str:
        if not result.passed and result.detail:
            return f"{result.mismatch_count}: {result.detail}"
        return str(result.mismatch_count)

    def table(self, *, color: bool = False) -> str:
        headers = (
            "component",
            "hf_path",
            "actual_dtype",
            "expected_dtype",
            "actual",
            "expected",
            "max_abs",
            "mean_abs",
            "rmse",
            "mean_rel",
            "rel_l2",
            "cosine",
            "mismatches",
            "trend",
            "status",
        )
        lines = [
            f"GLM-5 parity report: {self.title}",
            "",
            " | ".join(headers),
            "-|-|-|-|-|-|-|-|-|-|-|-|-|-|-| ",
        ]
        for result in self.ordered_results():
            status = self._status(result)
            if color:
                status = (
                    f"\033[32m{status}\033[0m"
                    if status == "PASS"
                    else f"\033[31m{status}\033[0m"
                    if status == "FAIL"
                    else f"\033[36m{status}\033[0m"
                )
            trend = (
                f"EXPLODE x{result.growth_ratio:.3g}"
                if result.explosion and result.growth_ratio is not None
                else "-"
            )
            lines.append(
                " | ".join(
                    (
                        self._display_path(result),
                        self._hf_module_path(result),
                        result.actual_dtype or "-",
                        result.expected_dtype or "-",
                        result.actual_summary or "-",
                        result.expected_summary or "-",
                        "-" if result.max_abs is None else f"{result.max_abs:.6g}",
                        "-" if result.mean_abs is None else f"{result.mean_abs:.6g}",
                        "-" if result.rmse is None else f"{result.rmse:.6g}",
                        "-" if result.mean_rel is None else f"{result.mean_rel:.6g}",
                        "-" if result.relative_l2 is None else f"{result.relative_l2:.6g}",
                        (
                            "-"
                            if result.cosine_similarity is None
                            else f"{result.cosine_similarity:.8g}"
                        ),
                        self._mismatch_display(result),
                        trend,
                        status,
                    )
                )
            )
        lines.append("")
        lines.append(f"summary: {self.summary()}")
        return "\n".join(lines)

    def html_table(self) -> str:
        headers = (
            ("component", ""),
            ("hf_path", "path"),
            ("actual_dtype", "dtype"),
            ("expected_dtype", "dtype"),
            ("actual", "value"),
            ("expected", "value"),
            ("max_abs", "metric"),
            ("mean_abs", "metric"),
            ("rmse", "metric"),
            ("mean_rel", "metric"),
            ("rel_l2", "metric"),
            ("cosine", "metric"),
            ("mismatches", "diagnostic"),
            ("trend", "diagnostic"),
            ("status", ""),
        )

        def cell(value: str, group: str = "", extra_class: str = "") -> str:
            classes = [name for name in (f"col-{group}" if group else "", extra_class) if name]
            class_attribute = f" class='{' '.join(classes)}'" if classes else ""
            return f"<td{class_attribute}>{escape(value)}</td>"

        rows = []
        for result in self.ordered_results():
            status = self._status(result)
            status_class = (
                "pass" if status == "PASS"
                else "fail" if status == "FAIL" else "trace"
            )
            row_class = " class='explosion-row'" if result.explosion else ""
            trend = (
                f"EXPLODE x{result.growth_ratio:.3g}"
                if result.explosion and result.growth_ratio is not None
                else "-"
            )
            rows.append(
                f"<tr{row_class}>"
                + cell(self._display_path(result))
                + cell(self._hf_module_path(result), "path")
                + cell(result.actual_dtype or "-", "dtype")
                + cell(result.expected_dtype or "-", "dtype")
                + cell(result.actual_summary or "-", "value")
                + cell(result.expected_summary or "-", "value")
                + cell(
                    "-" if result.max_abs is None else f"{result.max_abs:.6g}",
                    "metric",
                )
                + cell(
                    "-" if result.mean_abs is None else f"{result.mean_abs:.6g}",
                    "metric",
                )
                + cell("-" if result.rmse is None else f"{result.rmse:.6g}", "metric")
                + cell(
                    "-" if result.mean_rel is None else f"{result.mean_rel:.6g}",
                    "metric",
                )
                + cell(
                    "-"
                    if result.relative_l2 is None
                    else f"{result.relative_l2:.6g}",
                    "metric",
                )
                + cell(
                    "-"
                    if result.cosine_similarity is None
                    else f"{result.cosine_similarity:.8g}",
                    "metric",
                )
                + cell(self._mismatch_display(result), "diagnostic")
                + cell(trend, "diagnostic")
                + cell(status, extra_class=status_class)
                + "</tr>"
            )
        controls = "".join(
            f"<button type='button' data-column-group='{group}' "
            "onclick=\"this.closest('.parity-table-container').classList."
            f"toggle('hide-{group}')\">Toggle {label}</button>"
            for group, label in (
                ("path", "paths"),
                ("dtype", "dtypes"),
                ("value", "values"),
                ("metric", "metrics"),
                ("diagnostic", "diagnostics"),
            )
        )
        header_cells = []
        for name, group in headers:
            class_attribute = f" class='col-{group}'" if group else ""
            header_cells.append(
                f"<th{class_attribute}>{escape(name)}</th>"
            )
        header = "".join(header_cells)
        return (
            "<div class='parity-table-container'>"
            f"<div class='column-controls'>{controls}</div>"
            "<div class='parity-table-scroll'>"
            f"<table class='parity-table'><thead><tr>{header}</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div></div>"
        )

    def html_error_curve(self) -> str:
        """Render main forward errors by layer on a logarithmic SVG axis."""
        patterns = {
            "layer": re.compile(r"^layers\.(\d+)$"),
            "attention": re.compile(r"^layers\.(\d+)\.attention$"),
            "ffn_norm": re.compile(r"^layers\.(\d+)\.ffn_norm$"),
            "ffn_or_moe": re.compile(
                r"^layers\.(\d+)\.(?:feed_forward|moe)$"
            ),
        }
        series: dict[str, dict[int, ComparisonResult]] = {
            name: {} for name in patterns
        }
        for result in self.ordered_results():
            if result.max_abs is None or not isinstance(result.layer, int):
                continue
            path = self._display_path(result)
            for name, pattern in patterns.items():
                match = pattern.fullmatch(path)
                if match:
                    existing = series[name].get(result.layer)
                    if existing is None or result.checkpoint:
                        series[name][result.layer] = result
                    break
        populated = {name: values for name, values in series.items() if values}
        if not populated:
            # Generic component tests do not necessarily expose decoder-level
            # boundaries.  Group repeated logical paths across layers instead.
            generic: dict[str, dict[int, ComparisonResult]] = {}
            for result in self.ordered_results():
                if result.max_abs is None or not isinstance(result.layer, int):
                    continue
                path = self._display_path(result)
                if not path.startswith(f"layers.{result.layer}."):
                    continue
                name = re.sub(r"^layers\.\d+\.", "", path)
                values = generic.setdefault(name, {})
                existing = values.get(result.layer)
                if existing is None or result.checkpoint:
                    values[result.layer] = result
            # Keep the chart readable while the table retains every row.
            populated = dict(
                sorted(generic.items(), key=lambda item: item[0])[:12]
            )
        if not populated:
            return ""

        all_layers = sorted(
            {layer for values in populated.values() for layer in values}
        )
        floor = max(torch.finfo(self.precision.dtype).eps * 0.01, 1e-12)
        all_values = [
            max(result.max_abs or 0.0, floor)
            for values in populated.values()
            for result in values.values()
        ]
        log_min = math.floor(math.log10(min(all_values)))
        log_max = math.ceil(math.log10(max(all_values)))
        if log_min == log_max:
            log_max += 1

        width, height = 920, 300
        left, right, top, bottom = 72, 24, 24, 54
        plot_width = width - left - right
        plot_height = height - top - bottom

        def x_position(layer: int) -> float:
            if len(all_layers) == 1:
                return left + plot_width / 2
            return left + plot_width * (
                (layer - all_layers[0]) / (all_layers[-1] - all_layers[0])
            )

        def y_position(value: float) -> float:
            normalized = (math.log10(max(value, floor)) - log_min) / (
                log_max - log_min
            )
            return top + plot_height * (1.0 - normalized)

        palette = (
            "#1f77b4",
            "#9467bd",
            "#ff7f0e",
            "#2ca02c",
            "#17becf",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#d62728",
            "#393b79",
            "#637939",
        )
        colors = {
            "layer": "#1f77b4",
            "attention": "#9467bd",
            "ffn_norm": "#ff7f0e",
            "ffn_or_moe": "#2ca02c",
        }
        for index, name in enumerate(populated):
            colors.setdefault(name, palette[index % len(palette)])
        svg = [
            f"<svg viewBox='0 0 {width} {height}' role='img' "
            "aria-label='Layer error curves'>",
            f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_height}' class='axis'/>",
            f"<line x1='{left}' y1='{top + plot_height}' x2='{left + plot_width}' y2='{top + plot_height}' class='axis'/>",
        ]
        for exponent in range(log_min, log_max + 1):
            y_value = y_position(10.0**exponent)
            svg.append(
                f"<line x1='{left}' y1='{y_value:.2f}' "
                f"x2='{left + plot_width}' y2='{y_value:.2f}' class='grid'/>"
                f"<text x='{left - 8}' y='{y_value + 4:.2f}' "
                f"text-anchor='end'>1e{exponent}</text>"
            )
        for layer in all_layers:
            x_value = x_position(layer)
            svg.append(
                f"<text x='{x_value:.2f}' y='{top + plot_height + 22}' "
                f"text-anchor='middle'>{layer}</text>"
            )
        for name, values in populated.items():
            points = [
                (
                    layer,
                    x_position(layer),
                    y_position(values[layer].max_abs or 0.0),
                    values[layer],
                )
                for layer in sorted(values)
            ]
            polyline = " ".join(
                f"{x_value:.2f},{y_value:.2f}"
                for _, x_value, y_value, _ in points
            )
            svg.append(
                f"<polyline points='{polyline}' fill='none' "
                f"stroke='{colors[name]}' stroke-width='2'/>"
            )
            for layer, x_value, y_value, result in points:
                marker = "#d62728" if result.explosion else colors[name]
                svg.append(
                    f"<circle cx='{x_value:.2f}' cy='{y_value:.2f}' r='4' "
                    f"fill='{marker}'><title>{escape(name)} layer {layer}: "
                    f"max_abs={result.max_abs:.6g}</title></circle>"
                )
        legend_x = left
        for index, name in enumerate(populated):
            x_value = legend_x + index * 180
            svg.append(
                f"<line x1='{x_value}' y1='{height - 12}' x2='{x_value + 22}' "
                f"y2='{height - 12}' stroke='{colors[name]}' stroke-width='3'/>"
                f"<text x='{x_value + 28}' y='{height - 8}'>{escape(name)}</text>"
            )
        svg.append("</svg>")
        return (
            "<div class='error-chart'><h3>Max-abs error by layer</h3>"
            "<p>Log scale. Red markers identify configured explosion points.</p>"
            + "".join(svg)
            + "</div>"
        )

    def html_section(self, section_id: str) -> str:
        failed_class = "fail" if self.failed else "pass"
        return (
            f"<section id='{escape(section_id)}'>"
            f"<h2>{escape(self.title)}</h2>"
            f"<p class='{failed_class}'>{escape(self.summary())}</p>"
            "<p class='metric-note'>mean_rel is symmetric: "
            "2*|actual-expected|/(|actual|+|expected|). "
            "rel_l2 is ||actual-expected||2/||expected||2. "
            "Dtype flow reports observed input/parameter/output dtypes and "
            "explicit compute dtypes where the implementation declares one.</p>"
            f"{self.html_error_curve()}{self.html_table()}</section>"
        )

    def write(self, path: str | None = None) -> str:
        report = self.table(color=False)
        output_path = path or os.environ.get("GLM5_PARITY_REPORT")
        if output_path:
            if output_path.lower().endswith(".html"):
                html_report = (
                    "<!doctype html><html><head><meta charset='utf-8'>"
                    "<style>body{font-family:monospace}"
                    ".column-controls{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}"
                    ".column-controls button{cursor:pointer;padding:4px 10px}"
                    ".parity-table-scroll{max-height:72vh;overflow:auto;border:1px solid #bbb}"
                    ".parity-table{border-collapse:separate;border-spacing:0;width:100%}"
                    ".parity-table th,.parity-table td{border:0;border-right:1px solid #bbb;"
                    "border-bottom:1px solid #bbb;padding:4px 8px;text-align:left}"
                    ".parity-table thead th{position:sticky;top:0;z-index:2;background:#f5f5f5}"
                    ".parity-table td:first-child{white-space:nowrap}"
                    ".hide-path .col-path,.hide-dtype .col-dtype,"
                    ".hide-value .col-value,.hide-metric .col-metric,"
                    ".hide-diagnostic .col-diagnostic{display:none}"
                    ".pass{color:#087f23;font-weight:bold}.fail{color:#b00020;font-weight:bold}"
                    ".trace{color:#007c91;font-weight:bold}"
                    ".explosion-row{background:#fff1d6}.axis{stroke:#333}.grid{stroke:#ddd}"
                    ".error-chart svg{width:100%;max-width:920px}"
                    "</style></head><body>"
                    f"<h1>GLM-5 parity report</h1>{self.html_section('result')}"
                    "</body></html>\n"
                )
                with open(output_path, "w", encoding="utf-8") as file:
                    file.write(html_report)
            else:
                with open(output_path, "w", encoding="utf-8") as file:
                    file.write(report + "\n")
        return report

    def assert_all_passed(self) -> None:
        if self.failed:
            raise AssertionError(self.table(color=False))


@dataclass
class ParityReportSection:
    section_id: str
    recorder: ParityRecorder


class ParitySuiteReport:
    """Render independently executed parity tests into one HTML document."""

    def __init__(
        self,
        title: str,
        configuration: list[tuple[str, str, str]] | None = None,
    ):
        self.title = title
        self.sections: list[ParityReportSection] = []
        self.configuration = configuration or []

    def add(self, section_id: str, recorder: ParityRecorder) -> None:
        self.sections.append(ParityReportSection(section_id, recorder))

    @property
    def failed(self) -> list[ParityReportSection]:
        return [section for section in self.sections if section.recorder.failed]

    def write(self, path: str) -> None:
        if not path.lower().endswith(".html"):
            raise ValueError("GLM5_PARITY_REPORT must end with .html")
        anchored_sections = []
        for index, section in enumerate(self.sections):
            safe_id = re.sub(
                r"[^A-Za-z0-9_.-]+", "-", section.section_id
            ).strip("-")
            anchored_sections.append(
                (f"section-{index}-{safe_id or 'result'}", section)
            )
        links = "".join(
            f"<li><a href='#{escape(anchor)}'>"
            f"{escape(section.recorder.title)}</a></li>"
            for anchor, section in anchored_sections
        )
        sections = "".join(
            section.recorder.html_section(anchor)
            for anchor, section in anchored_sections
        )
        configuration_rows = "".join(
            "<tr>"
            f"<td>{escape(category)}</td>"
            f"<td>{escape(name)}</td>"
            f"<td>{escape(value)}</td>"
            "</tr>"
            for category, name, value in self.configuration
        )
        configuration = (
            "<details open><summary>Effective test configuration</summary>"
            "<table class='configuration'><thead><tr>"
            "<th>category</th><th>parameter</th><th>effective value</th>"
            "</tr></thead><tbody>"
            f"{configuration_rows}</tbody></table></details>"
        )
        document = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<style>"
            "body{font-family:ui-monospace,Consolas,monospace;margin:24px}"
            ".report-toc{padding:8px 16px;border:1px solid #bbb;background:#f8f8f8}"
            ".report-toc h2{margin:4px 0 8px}.report-toc ul{display:flex;gap:18px;flex-wrap:wrap;list-style:none;padding:0}"
            "section{margin:36px 0;scroll-margin-top:16px}table{border-collapse:collapse;width:100%}"
            "th,td{border:1px solid #bbb;padding:4px 8px;text-align:left}"
            "th{background:#f5f5f5}"
            "details{margin:20px 0}summary{font-weight:bold;cursor:pointer}"
            ".configuration{width:auto;min-width:720px;margin-top:10px}"
            ".configuration th{position:static}"
            ".column-controls{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}"
            ".column-controls button{cursor:pointer;padding:4px 10px}"
            ".parity-table-scroll{max-height:72vh;overflow:auto;border:1px solid #bbb}"
            ".parity-table{border-collapse:separate;border-spacing:0;width:100%}"
            ".parity-table th,.parity-table td{border:0;border-right:1px solid #bbb;"
            "border-bottom:1px solid #bbb;padding:4px 8px;text-align:left}"
            ".parity-table thead th{position:sticky;top:0;z-index:2;background:#f5f5f5}"
            ".parity-table td:first-child{white-space:nowrap}"
            ".hide-path .col-path,.hide-dtype .col-dtype,"
            ".hide-value .col-value,.hide-metric .col-metric,"
            ".hide-diagnostic .col-diagnostic{display:none}"
            ".pass{color:#087f23;font-weight:bold}"
            ".fail{color:#b00020;font-weight:bold}"
            ".trace{color:#007c91;font-weight:bold}"
            ".explosion-row{background:#fff1d6}"
            ".axis{stroke:#333}.grid{stroke:#ddd}"
            ".error-chart{overflow-x:auto}.error-chart svg{width:100%;min-width:700px;max-width:920px}"
            "</style></head><body>"
            f"<h1>{escape(self.title)}</h1>"
            "<nav id='report-contents' class='report-toc' aria-label='Report contents'>"
            f"<h2>Contents</h2><ul>{links}</ul></nav>{configuration}"
            f"{sections}</body></html>\n"
        )
        with open(path, "w", encoding="utf-8") as file:
            file.write(document)

    def failure_message(self) -> str:
        return "\n\n".join(
            section.recorder.table(color=False) for section in self.failed
        )


def _format_parity_diagnostics(
    records: dict[str, dict[int, torch.Tensor | None]],
    positions_BL: torch.Tensor,
) -> str:
    """Format layer-local block, indexer, and router parity evidence."""
    expected_record_names = (
        "hf_blocks", "titan_blocks", "hf_indexer", "titan_indexer",
        "hf_router", "titan_router",
    )
    missing_record_names = [
        name for name in expected_record_names if name not in records
    ]
    if missing_record_names:
        return f"missing record groups: {missing_record_names}"
    recorder = ParityRecorder(BF16)
    for layer in sorted(set(records.get("hf_blocks", {})) | set(records.get("titan_blocks", {}))):
        hf_block = records.get("hf_blocks", {}).get(layer)
        titan_block = records.get("titan_blocks", {}).get(layer)
        if hf_block is not None and titan_block is not None:
            recorder.tensor(
                scope="e2e", component="block", layer=layer,
                actual=titan_block, expected=hf_block,
            )
        for component in ("indexer", "router"):
            hf_value = records.get(f"hf_{component}", {}).get(layer)
            titan_value = records.get(f"titan_{component}", {}).get(layer)
            if hf_value is not None and titan_value is not None:
                recorder.discrete(
                    scope="e2e", component=component, layer=layer,
                    actual=titan_value, expected=hf_value,
                    positions=positions_BL,
                )
    lines = []
    layers = sorted(
        set(records.get("hf_blocks", {}))
        | set(records.get("titan_blocks", {}))
        | set(records.get("hf_indexer", {}))
        | set(records.get("titan_indexer", {}))
        | set(records.get("hf_router", {}))
        | set(records.get("titan_router", {}))
    )
    for layer in layers:
        hf_block = records.get("hf_blocks", {}).get(layer)
        titan_block = records.get("titan_blocks", {}).get(layer)
        if hf_block is None or titan_block is None:
            lines.append(
                f"layer {layer}: block_records=missing"
                f"(hf={hf_block is not None}, titan={titan_block is not None})"
            )
        elif hf_block.ndim != 3 or titan_block.ndim != 3:
            lines.append(
                f"layer {layer}: block_shape_error=expected [B, L, D] compatible with positions"
            )
        elif hf_block.shape != titan_block.shape:
            lines.append(
                f"layer {layer}: block_shape_mismatch="
                f"hf{tuple(hf_block.shape)}!=titan{tuple(titan_block.shape)}"
            )
        else:
            difference = (hf_block.float() - titan_block.float()).abs()
            position_max = difference.amax(dim=(0, 2)).detach().cpu()
            values = ", ".join(
                f"{position}:{float(value):.6g}"
                for position, value in enumerate(position_max)
            )
            lines.append(
                f"layer {layer}: block_max_abs={float(position_max.max()):.6g} "
                f"block_position_max_abs=[{values}]"
            )
    for result in recorder.results:
        if result.component == "block":
            continue
        else:
            lines.append(
                f"layer {result.layer}: {result.component}_mismatched_queries="
                f"{result.mismatch_count} positions={result.mismatch_positions}"
            )
    for component in ("indexer", "router"):
        hf_records = records.get(f"hf_{component}", {})
        titan_records = records.get(f"titan_{component}", {})
        for layer in sorted(set(hf_records) | set(titan_records)):
            if hf_records.get(layer) is None or titan_records.get(layer) is None:
                lines.append(
                    f"layer {layer}: {component}_records=missing"
                    f"(hf={hf_records.get(layer) is not None}, "
                    f"titan={titan_records.get(layer) is not None})"
                )
    failures = [
        result
        for result in recorder.results
        if not result.passed and result.component in {"indexer", "router"}
    ]
    first_failure = min(
        failures,
        key=lambda result: (
            int(result.layer) if isinstance(result.layer, int) else 10**9,
            result.mismatch_positions[0] if result.mismatch_positions else 10**9,
            result.component,
        ),
        default=None,
    )
    if first_failure is None:
        lines.append("first_discrete_mismatch=none")
    else:
        lines.append(
            f"first_discrete_mismatch=layer {first_failure.layer} "
            f"position {first_failure.mismatch_positions[0]} "
            f"source={first_failure.component}"
        )
    return "\n".join(lines)


def _first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for value in output:
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f"module output has no tensor: {type(output)!r}")


def _tensor_leaves(output: Any, prefix: str = "") -> list[tuple[str, torch.Tensor]]:
    """Return every tensor in a module output, including tuple branches."""
    if isinstance(output, torch.Tensor):
        return [(prefix, output)]
    if isinstance(output, (tuple, list)):
        leaves: list[tuple[str, torch.Tensor]] = []
        for index, value in enumerate(output):
            branch = f"{prefix}[{index}]"
            leaves.extend(_tensor_leaves(value, branch))
        return leaves
    if isinstance(output, dict):
        leaves = []
        for key, value in output.items():
            leaves.extend(_tensor_leaves(value, f"{prefix}.{key}"))
        return leaves
    return []


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_flow(
    module: torch.nn.Module,
    inputs: Any,
    output: torch.Tensor,
) -> str:
    input_dtypes = sorted(
        {_dtype_name(value.dtype) for _, value in _tensor_leaves(inputs)}
    )
    parameter_dtypes = sorted(
        {_dtype_name(value.dtype) for value in module.parameters(recurse=False)}
    )
    compute_dtype = getattr(module, "_parity_compute_dtype", None)
    parts = [f"in={','.join(input_dtypes) or '-'}"]
    if parameter_dtypes:
        parts.append(f"param={','.join(parameter_dtypes)}")
    if compute_dtype is not None:
        compute_name = (
            _dtype_name(compute_dtype)
            if isinstance(compute_dtype, torch.dtype)
            else str(compute_dtype)
        )
        parts.append(f"compute={compute_name}")
    parts.append(f"out={_dtype_name(output.dtype)}")
    return ";".join(parts)


@dataclass
class LayerTrace:
    blocks_hf: dict[int, torch.Tensor] = field(default_factory=dict)
    blocks_titan: dict[int, torch.Tensor] = field(default_factory=dict)
    indexer_hf: dict[int, torch.Tensor] = field(default_factory=dict)
    indexer_titan: dict[int, torch.Tensor] = field(default_factory=dict)
    router_hf: dict[int, torch.Tensor] = field(default_factory=dict)
    router_titan: dict[int, torch.Tensor] = field(default_factory=dict)

    @staticmethod
    @contextmanager
    def install(pair: ParityModelPair, layer_indices: list[int]) -> Iterator["LayerTrace"]:
        trace = LayerTrace()
        handles = []

        def save(
            target: dict[int, torch.Tensor],
            layer: int,
            output: Any,
            index: int | None = None,
        ) -> None:
            value = output[index] if index is not None else output
            target[layer] = _first_tensor(value).detach().cpu()

        for layer_index in layer_indices:
            hf_layer = pair.hf_layer(layer_index)
            titan_layer = pair.titan_layer(layer_index)
            handles.append(hf_layer.register_forward_hook(
                lambda _m, _i, o, layer=layer_index: save(trace.blocks_hf, layer, o, 0)
            ))
            handles.append(titan_layer.register_forward_hook(
                lambda _m, _i, o, layer=layer_index: save(trace.blocks_titan, layer, o)
            ))
            handles.append(hf_layer.self_attn.indexer.register_forward_hook(
                lambda _m, _i, o, layer=layer_index: save(trace.indexer_hf, layer, o)
            ))
            handles.append(titan_layer.attention.indexer.register_forward_hook(
                lambda _m, _i, o, layer=layer_index: save(trace.indexer_titan, layer, o)
            ))
            if getattr(titan_layer, "moe_enabled", False):
                def save_hf_router(
                    _module, inputs, output, *, layer=layer_index
                ) -> None:
                    batch_size, sequence_length = inputs[0].shape[:2]
                    value = output[2].view(batch_size, sequence_length, -1)
                    trace.router_hf[layer] = value.detach().cpu()

                handles.append(hf_layer.mlp.gate.register_forward_hook(save_hf_router))
                handles.append(titan_layer.moe.router.register_forward_hook(
                    lambda _m, _i, o, layer=layer_index: save(trace.router_titan, layer, o, 1)
                ))
        try:
            yield trace
        finally:
            for handle in handles:
                handle.remove()


@dataclass
class EndpointTrace:
    """Layer observations keyed by endpoint label for matrix comparisons."""

    blocks: dict[str, dict[int, torch.Tensor]] = field(default_factory=dict)
    indexer: dict[str, dict[int, torch.Tensor]] = field(default_factory=dict)
    indexer_scores: dict[str, dict[int, torch.Tensor]] = field(
        default_factory=dict
    )
    router: dict[str, dict[int, torch.Tensor]] = field(default_factory=dict)
    router_weights: dict[str, dict[int, torch.Tensor]] = field(default_factory=dict)
    router_scores: dict[str, dict[int, torch.Tensor]] = field(
        default_factory=dict
    )
    expert_load: dict[str, dict[int, torch.Tensor]] = field(default_factory=dict)
    moe_inputs: dict[str, dict[int, torch.Tensor]] = field(default_factory=dict)

    @staticmethod
    @contextmanager
    def install(
        endpoints: dict[ModelEndpoint, torch.nn.Module],
        layer_indices: list[int],
    ) -> Iterator["EndpointTrace"]:
        trace = EndpointTrace()
        handles = []

        for endpoint, model in endpoints.items():
            label = endpoint.label
            trace.blocks[label] = {}
            trace.indexer[label] = {}
            trace.indexer_scores[label] = {}
            trace.router[label] = {}
            trace.router_weights[label] = {}
            trace.router_scores[label] = {}
            trace.expert_load[label] = {}
            trace.moe_inputs[label] = {}
            layers = model.model.layers if endpoint.implementation == "hf" else model.layers
            for layer_index in layer_indices:
                layer = (
                    layers[layer_index]
                    if endpoint.implementation == "hf"
                    else layers[str(layer_index)]
                )

                def capture_block(
                    _module,
                    _inputs,
                    output,
                    *,
                    label=label,
                    layer_index=layer_index,
                    implementation=endpoint.implementation,
                ) -> None:
                    value = output[0] if implementation == "hf" else output
                    trace.blocks[label][layer_index] = _first_tensor(value).detach().cpu()

                def capture_indexer(
                    _module, _inputs, output, *, label=label, layer_index=layer_index
                ) -> None:
                    trace.indexer[label][layer_index] = _first_tensor(output).detach().cpu()

                handles.append(layer.register_forward_hook(capture_block))
                indexer = (
                    layer.self_attn.indexer
                    if endpoint.implementation == "hf"
                    else layer.attention.indexer
                )
                handles.append(indexer.register_forward_hook(capture_indexer))
                if endpoint.implementation == "titan":
                    def capture_indexer_scores(
                        module,
                        inputs,
                        *,
                        label=label,
                        layer_index=layer_index,
                    ) -> None:
                        trace.indexer_scores[label][layer_index] = (
                            _capture_titan_indexer_scores(module, inputs)
                            .detach()
                            .cpu()
                        )

                    handles.append(
                        indexer.register_forward_pre_hook(
                            capture_indexer_scores
                        )
                    )
                if endpoint.implementation == "hf" and hasattr(layer.mlp, "gate"):
                    def capture_moe_input(
                        _module, inputs, *, label=label, layer_index=layer_index
                    ) -> None:
                        trace.moe_inputs[label][layer_index] = inputs[0].detach()

                    handles.append(
                        layer.mlp.register_forward_pre_hook(capture_moe_input)
                    )

                    def capture_hf_router(
                        _module, inputs, output, *, label=label, layer_index=layer_index
                    ) -> None:
                        batch_size, sequence_length = inputs[0].shape[:2]
                        indices = output[2].view(batch_size, sequence_length, -1)
                        weights = output[1].view(batch_size, sequence_length, -1)
                        trace.router[label][layer_index] = indices.detach().cpu()
                        trace.router_weights[label][layer_index] = (
                            weights.detach().cpu()
                        )
                        trace.expert_load[label][layer_index] = F.one_hot(
                            indices,
                            num_classes=_module.num_experts,
                        ).sum(dim=(0, 1, 2)).detach().cpu()

                    handles.append(layer.mlp.gate.register_forward_hook(capture_hf_router))
                elif getattr(layer, "moe_enabled", False):
                    def capture_moe_input(
                        _module, inputs, *, label=label, layer_index=layer_index
                    ) -> None:
                        trace.moe_inputs[label][layer_index] = inputs[0].detach()

                    handles.append(
                        layer.moe.register_forward_pre_hook(capture_moe_input)
                    )

                    def capture_titan_router(
                        _module, _inputs, output, *, label=label, layer_index=layer_index
                    ) -> None:
                        indices = output[1]
                        scores_for_choice = output[2]
                        expert_bias = (
                            _inputs[1] if len(_inputs) > 1 else None
                        )
                        if expert_bias is not None:
                            scores_for_choice = scores_for_choice + expert_bias
                        if _module.num_expert_groups is not None:
                            scores_for_choice = (
                                _module._get_node_limited_routing_scores(
                                    scores_for_choice
                                )
                            )
                        trace.router[label][layer_index] = indices.detach().cpu()
                        trace.router_weights[label][layer_index] = (
                            output[0].detach().cpu()
                        )
                        trace.router_scores[label][layer_index] = (
                            scores_for_choice.detach().cpu()
                        )
                        trace.expert_load[label][layer_index] = F.one_hot(
                            indices,
                            num_classes=_module.num_experts,
                        ).sum(dim=(0, 1, 2)).detach().cpu()

                    handles.append(layer.moe.router.register_forward_hook(capture_titan_router))
        try:
            yield trace
        finally:
            for handle in handles:
                handle.remove()


@dataclass
class RecursiveModuleTrace:
    """Capture module outputs and tensor gradients under selected layers."""

    activations: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    gradients: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    dtype_flows: dict[str, dict[str, str]] = field(default_factory=dict)

    @staticmethod
    @contextmanager
    def install(
        endpoints: dict[ModelEndpoint, torch.nn.Module],
        layer_indices: list[int],
        *,
        include_model_modules: bool = False,
    ) -> Iterator["RecursiveModuleTrace"]:
        trace = RecursiveModuleTrace()
        handles = []
        for endpoint, model in endpoints.items():
            label = endpoint.label
            trace.activations[label] = {}
            trace.gradients[label] = {}
            trace.dtype_flows[label] = {}
            layers = model.model.layers if endpoint.implementation == "hf" else model.layers
            for layer_index in layer_indices:
                layer = (
                    layers[layer_index]
                    if endpoint.implementation == "hf"
                    else layers[str(layer_index)]
                )
                root = (
                    f"model.layers.{layer_index}"
                    if endpoint.implementation == "hf"
                    else f"layers.{layer_index}"
                )
                for suffix, module in layer.named_modules():
                    path = root if not suffix else f"{root}.{suffix}"

                    def capture(
                        _module,
                        _inputs,
                        output,
                        *,
                        label=label,
                        path=path,
                    ) -> None:
                        for suffix, value in _tensor_leaves(output):
                            output_path = path + suffix
                            trace.activations[label][output_path] = value.detach().cpu()
                            trace.dtype_flows[label][output_path] = _dtype_flow(
                                _module, _inputs, value
                            )
                            if value.requires_grad:
                                value.register_hook(
                                    lambda gradient,
                                    label=label,
                                    path=output_path: trace.gradients[label].__setitem__(
                                        path, gradient.detach().cpu()
                                    )
                                )

                    handles.append(module.register_forward_hook(capture))
            if include_model_modules:
                layer_prefix = (
                    "model.layers."
                    if endpoint.implementation == "hf"
                    else "layers."
                )
                for suffix, module in model.named_modules():
                    if (
                        not suffix
                        or suffix in {"model", "layers"}
                        or suffix.startswith(layer_prefix)
                    ):
                        continue
                    path = suffix

                    def capture_model_module(
                        _module,
                        _inputs,
                        output,
                        *,
                        label=label,
                        path=path,
                    ) -> None:
                        for branch, value in _tensor_leaves(output):
                            output_path = path + branch
                            trace.activations[label][output_path] = value.detach().cpu()
                            trace.dtype_flows[label][output_path] = _dtype_flow(
                                _module, _inputs, value
                            )
                            if value.requires_grad:
                                value.register_hook(
                                    lambda gradient,
                                    label=label,
                                    path=output_path: trace.gradients[label].__setitem__(
                                        path, gradient.detach().cpu()
                                    )
                                )

                    handles.append(module.register_forward_hook(capture_model_module))
        try:
            yield trace
        finally:
            for handle in handles:
                handle.remove()


def _run_causal_lm_endpoint(
    model: torch.nn.Module,
    endpoint: ModelEndpoint,
    batch: ParityBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one endpoint with a common explicit causal mask."""
    assert batch.tokens is not None
    labels = batch.tokens
    if endpoint.implementation == "hf":
        output = model(
            input_ids=batch.tokens,
            position_ids=batch.positions,
            attention_mask=batch.causal_mask,
            labels=labels,
            use_cache=False,
        )
        return output.logits, output.loss
    logits = model(
        batch.tokens,
        positions=batch.positions,
        attention_masks=batch.causal_mask,
    )
    loss = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
    )
    return logits, loss


class ComponentParityMixin:
    pair: ParityModelPair
    batch: ParityBatch
    precision: PrecisionPolicy

    def _recorder(self) -> ParityRecorder:
        return ParityRecorder(
            self.precision,
            precision_label=(
                f"titan:{self.precision.name} vs hf:{self.precision.name}"
            ),
        )

    def _record_component_trace(
        self,
        component: str,
        layer_indices: list[int],
        recorder: ParityRecorder,
    ) -> None:
        """Trace the selected component subtree across the requested layers."""
        if not layer_indices:
            return
        actual = ModelEndpoint("titan", self.precision)
        expected = ModelEndpoint("hf", self.precision)
        endpoints = {
            actual: self.pair.titan,
            expected: self.pair.hf,
        }
        spec = ComparisonSpec(
            actual=actual,
            expected=expected,
            scope="component_trace",
            component=component,
            rtol=self.precision.rtol,
            atol=self.precision.atol,
        )
        with RecursiveModuleTrace.install(endpoints, layer_indices) as trace:
            with torch.no_grad():
                for layer_index in layer_indices:
                    hf_layer = self.pair.hf_layer(layer_index)
                    titan_layer = self.pair.titan_layer(layer_index)
                    hf_layer(
                        self.batch.hidden_states,
                        attention_mask=self.batch.causal_mask,
                        position_ids=self.batch.positions,
                        position_embeddings=self.batch.hf_position_embeddings,
                        use_cache=False,
                    )
                    titan_layer(
                        self.batch.hidden_states,
                        self.batch.causal_mask,
                        self.batch.positions,
                    )
        self._record_recursive_trace(
            spec,
            trace,
            recorder,
            layer_indices=layer_indices,
            component_filter=self._normalize_component_name(
                layer_indices[0], component
            ),
            composition_checkpoints=False,
            skip_layer_roots=False,
            skip_filter_roots=True,
        )

    def _normalized(self, layer_index: int) -> torch.Tensor:
        return self.pair.hf_layer(layer_index).input_layernorm(
            self.batch.hidden_states
        )

    def _normalized_hidden_states(self, layer_index: int) -> torch.Tensor:
        """Compatibility alias for the original component-test helper."""
        return self._normalized(layer_index)

    @staticmethod
    def _module_path(layer_index: int, component: str) -> str:
        """Render both canonical module paths for one logical component."""
        paths = {
            "block": (f"model.layers.{layer_index}", f"layers.{layer_index}"),
            "attention": (
                f"model.layers.{layer_index}.self_attn",
                f"layers.{layer_index}.attention",
            ),
            "indexer": (
                f"model.layers.{layer_index}.self_attn.indexer",
                f"layers.{layer_index}.attention.indexer",
            ),
            "router": (
                f"model.layers.{layer_index}.mlp.gate",
                f"layers.{layer_index}.moe.router",
            ),
            "input_norm": (
                f"model.layers.{layer_index}.input_layernorm",
                f"layers.{layer_index}.attention_norm",
            ),
            "ffn_norm": (
                f"model.layers.{layer_index}.post_attention_layernorm",
                f"layers.{layer_index}.ffn_norm",
            ),
        }
        hf_path, titan_path = paths.get(
            component,
            (
                f"model.layers.{layer_index}.{component}",
                f"layers.{layer_index}.{component}",
            ),
        )
        return f"hf:{hf_path} <-> titan:{titan_path}"

    @staticmethod
    def _normalize_component_name(layer_index: int, component: str) -> str:
        """Normalize a user component selector to a layer-relative path."""
        name = component.strip().strip(".")
        for prefix in (
            f"model.layers.{layer_index}.",
            f"layers.{layer_index}.",
        ):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        name = re.sub(r"^(?:model\.)?layers\.\d+\.", "", name)
        if name in {"decoder_block", "layer", "block", ""}:
            return ""
        aliases = {
            "dense_ffn": "feed_forward",
            "ffn": "feed_forward",
            "self_attn": "attention",
            "indexer": "attention.indexer",
            "router": "moe.router",
            "gate": "moe.router",
        }
        return aliases.get(name, name)

    def _resolve_component_module(
        self,
        endpoint: ModelEndpoint,
        layer_index: int,
        component: str,
    ) -> torch.nn.Module:
        """Resolve any named module through the canonical report hierarchy."""
        target = self._normalize_component_name(layer_index, component)
        layer = (
            self._model(endpoint).model.layers[layer_index]
            if endpoint.implementation == "hf"
            else self._model(endpoint).layers[str(layer_index)]
        )
        if target == "mlp":
            is_moe = (
                hasattr(layer, "moe")
                if endpoint.implementation == "titan"
                else hasattr(layer.mlp, "gate")
            )
            target = "moe" if is_moe else "feed_forward"
        if not target:
            return layer
        endpoint_root = (
            f"model.layers.{layer_index}"
            if endpoint.implementation == "hf"
            else f"layers.{layer_index}"
        )
        for suffix, module in layer.named_modules():
            endpoint_path = endpoint_root if not suffix else f"{endpoint_root}.{suffix}"
            logical_path = self._logical_activation_path(endpoint, endpoint_path)
            canonical = logical_path.removeprefix(f"layers.{layer_index}.")
            if canonical == target:
                return module
        available = sorted(
            self._logical_activation_path(
                endpoint,
                endpoint_root if not suffix else f"{endpoint_root}.{suffix}",
            ).removeprefix(f"layers.{layer_index}.")
            for suffix, _module in layer.named_modules()
        )
        raise ValueError(
            f"component {component!r} is not present in layer {layer_index}; "
            f"available paths include {available[:20]}"
        )

    def get_component(
        self, layer_index: int, component: str
    ) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Return the HF and TorchTitan modules for any canonical component."""
        return (
            self._resolve_component_module(
                ModelEndpoint("hf", self.precision), layer_index, component
            ),
            self._resolve_component_module(
                ModelEndpoint("titan", self.precision), layer_index, component
            ),
        )

    def get_model_component(
        self, component: str
    ) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Return a top-level component such as embeddings, norm, or head."""
        aliases = {
            "embedding": "tok_embeddings",
            "embeddings": "tok_embeddings",
            "embed_tokens": "tok_embeddings",
            "model.embed_tokens": "tok_embeddings",
            "model.norm": "norm",
        }
        target = aliases.get(component.strip().strip("."), component.strip().strip("."))

        def resolve(endpoint: ModelEndpoint) -> torch.nn.Module:
            model = self._model(endpoint)
            for suffix, module in model.named_modules():
                if not suffix or suffix.startswith(
                    "model.layers." if endpoint.implementation == "hf" else "layers."
                ):
                    continue
                logical = self._logical_activation_path(endpoint, suffix)
                if logical == target:
                    return module
            raise ValueError(f"top-level component {component!r} was not found")

        return (
            resolve(ModelEndpoint("hf", self.precision)),
            resolve(ModelEndpoint("titan", self.precision)),
        )

    def component_modules(
        self, layer_index: int, component: str
    ) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Return corresponding HF/TorchTitan modules by logical component name."""
        hf_layer = self.pair.hf_layer(layer_index)
        titan_layer = self.pair.titan_layer(layer_index)
        paths = {
            "block": (hf_layer, titan_layer),
            "attention": (hf_layer.self_attn, titan_layer.attention),
            "indexer": (hf_layer.self_attn.indexer, titan_layer.attention.indexer),
            "input_norm": (hf_layer.input_layernorm, titan_layer.attention_norm),
            "ffn_norm": (hf_layer.post_attention_layernorm, titan_layer.ffn_norm),
            "q_norm": (hf_layer.self_attn.q_a_layernorm, titan_layer.attention.q_norm),
            "kv_norm": (hf_layer.self_attn.kv_a_layernorm, titan_layer.attention.kv_norm),
        }
        if component == "router":
            if not hasattr(titan_layer, "moe"):
                raise ValueError(f"layer {layer_index} does not contain an MoE router")
            return hf_layer.mlp.gate, titan_layer.moe.router
        if component in {"dense_ffn", "feed_forward"}:
            if not hasattr(titan_layer, "feed_forward"):
                raise ValueError(
                    f"layer {layer_index} does not contain a dense feed-forward module"
                )
            return hf_layer.mlp, titan_layer.feed_forward
        if component == "moe":
            if not hasattr(titan_layer, "moe"):
                raise ValueError(f"layer {layer_index} does not contain an MoE module")
            return hf_layer.mlp, titan_layer.moe
        if component in paths:
            return paths[component]
        return self.get_component(layer_index, component)

    def compare_model_component_trace(
        self,
        component: str,
        *,
        spec: ComparisonSpec | None = None,
    ) -> ParityRecorder:
        """Compare a top-level module activation such as ``lm_head``."""
        if spec is None:
            actual = ModelEndpoint("titan", self.precision)
            expected = ModelEndpoint("hf", self.precision)
            spec = ComparisonSpec(
                actual=actual,
                expected=expected,
                scope="model_component_trace",
                component=component,
                rtol=self.precision.rtol,
                atol=self.precision.atol,
            )
        endpoints = {
            spec.actual: self._model(spec.actual),
            spec.expected: self._model(spec.expected),
        }
        recorder = ParityRecorder(
            spec.actual.precision, precision_label=spec.label
        )
        with RecursiveModuleTrace.install(
            endpoints, [], include_model_modules=True
        ) as trace:
            for endpoint in (spec.actual, spec.expected):
                with torch.no_grad():
                    _run_causal_lm_endpoint(
                        endpoints[endpoint], endpoint, self._batch_for(endpoint)
                    )
        normalized = self._normalize_model_component(component)
        self._record_recursive_trace(
            spec,
            trace,
            recorder,
            layer_indices=[],
            component_filter=normalized,
        )
        if not recorder.results:
            raise ValueError(
                f"component {component!r} produced no trace rows for the selected model"
            )
        return recorder

    @staticmethod
    def _normalize_model_component(component: str) -> str:
        aliases = {
            "embedding": "tok_embeddings",
            "embeddings": "tok_embeddings",
            "embed_tokens": "tok_embeddings",
            "model.embed_tokens": "tok_embeddings",
            "model.norm": "norm",
        }
        return aliases.get(component.strip().strip("."), component.strip().strip("."))

    def compare_component_trace(
        self,
        component: str,
        layer_indices: list[int],
        *,
        spec: ComparisonSpec | None = None,
        sequential: bool = False,
    ) -> ParityRecorder:
        """Compare an arbitrary module and its descendants across layers.

        The selected module is a checkpoint; descendant outputs remain trace
        rows unless they are composition nodes.  ``sequential=True`` feeds
        each selected layer into the next one; the default probes each layer
        from the same input.  This gives callers a generic path for new
        modules without adding a bespoke runner.
        """
        if self._normalize_model_component(component) in TOP_LEVEL_COMPONENTS:
            return self.compare_model_component_trace(component, spec=spec)
        if not layer_indices:
            raise ValueError("at least one layer is required for a component trace")
        if spec is None:
            actual = ModelEndpoint("titan", self.precision)
            expected = ModelEndpoint("hf", self.precision)
            spec = ComparisonSpec(
                actual=actual,
                expected=expected,
                scope="component_trace",
                component=component,
                rtol=self.precision.rtol,
                atol=self.precision.atol,
            )
        endpoints = {
            spec.actual: self._model(spec.actual),
            spec.expected: self._model(spec.expected),
        }
        recorder = ParityRecorder(
            spec.actual.precision, precision_label=spec.label
        )
        with RecursiveModuleTrace.install(endpoints, layer_indices) as trace:
            for endpoint in (spec.actual, spec.expected):
                batch = self._batch_for(endpoint)
                model = endpoints[endpoint]
                layers = (
                    model.model.layers
                    if endpoint.implementation == "hf"
                    else model.layers
                )
                current = batch.hidden_states
                for layer_index in layer_indices:
                    layer = (
                        layers[layer_index]
                        if endpoint.implementation == "hf"
                        else layers[str(layer_index)]
                    )
                    if endpoint.implementation == "hf":
                        output = layer(
                            current,
                            attention_mask=batch.causal_mask,
                            position_ids=batch.positions,
                            position_embeddings=batch.hf_position_embeddings,
                            use_cache=False,
                        )
                    else:
                        output = layer(
                            current,
                            batch.causal_mask,
                            batch.positions,
                        )
                    if sequential:
                        value = (
                            output[0]
                            if endpoint.implementation == "hf"
                            and isinstance(output, (tuple, list))
                            else output
                        )
                        current = _first_tensor(value)
                    elif endpoint.implementation == "hf":
                        current = batch.hidden_states
        normalized = self._normalize_component_name(layer_indices[0], component)
        self._record_recursive_trace(
            spec,
            trace,
            recorder,
            layer_indices=layer_indices,
            component_filter=normalized,
        )
        if not recorder.results:
            raise ValueError(
                f"component {component!r} produced no trace rows for the selected layers"
            )
        return recorder

    def compare_component(
        self,
        component: str,
        layer_indices: list[int] | None = None,
        *,
        spec: ComparisonSpec | None = None,
        sequential: bool = False,
    ) -> ParityRecorder:
        """Public component entry point for unit and multi-layer checks."""
        selected = self._selected_layers() if layer_indices is None else layer_indices
        return self.compare_components(
            component, selected, spec=spec, sequential=sequential
        )

    def compare_components(
        self,
        component: str,
        layer_indices: list[int],
        *,
        spec: ComparisonSpec | None = None,
        sequential: bool = False,
    ) -> ParityRecorder:
        """Dispatch a reusable comparison by component and arbitrary layers."""
        runners: dict[str, Callable[[list[int]], ParityRecorder]] = {
            "indexer": self.compare_indexer,
            "router": self.compare_router,
            "attention": self.compare_attention,
            "block": self.compare_blocks,
        }
        if component in runners and spec is None and not sequential:
            recorder = runners[component](layer_indices)
            self._record_component_trace(
                component, layer_indices, recorder
            )
            return recorder
        return self.compare_component_trace(
            component, layer_indices, spec=spec, sequential=sequential
        )

    def compare_indexer(self, layer_indices: list[int]) -> ParityRecorder:
        recorder = self._recorder()
        for layer_index in layer_indices:
            hf_attention = self.pair.hf_layer(layer_index).self_attn
            titan_attention = self.pair.titan_layer(layer_index).attention
            hidden_states = self._normalized(layer_index)
            hf_q_resid = hf_attention.q_a_layernorm(hf_attention.q_a_proj(hidden_states))
            titan_q_resid = titan_attention.q_norm(titan_attention.wq_a(hidden_states))
            recorder.tensor(
                scope="component", component="q_residual", layer=layer_index,
                actual=titan_q_resid, expected=hf_q_resid,
                rtol=1e-6 if self.precision is FP32 else self.precision.rtol,
                atol=1e-7 if self.precision is FP32 else self.precision.atol,
                module_path=(
                    f"hf:model.layers.{layer_index}.self_attn.q_a_layernorm"
                    f" <-> titan:layers.{layer_index}.attention.q_norm"
                ),
                parent_path=f"layers.{layer_index}.attention",
                level=4,
                node_kind="activation_checkpoint",
            )
            common_q_resid = hf_q_resid
            hf_topk = hf_attention.indexer(
                hidden_states, common_q_resid, self.batch.hf_position_embeddings,
                self.batch.causal_mask[:, 0], self.batch.positions,
            )
            titan_topk = titan_attention.indexer(
                hidden_states, common_q_resid, self.batch.positions,
                self.batch.causal_mask[:, 0],
            )
            recorder.discrete(
                scope="component", component="indexer", layer=layer_index,
                actual=titan_topk, expected=hf_topk, positions=self.batch.positions,
                module_path=self._module_path(layer_index, "indexer"),
                parent_path=f"layers.{layer_index}.attention",
                level=4,
                node_kind="discrete_checkpoint",
            )
        return recorder

    def compare_router(self, layer_indices: list[int]) -> ParityRecorder:
        recorder = self._recorder()
        for layer_index in layer_indices:
            hf_layer = self.pair.hf_layer(layer_index)
            titan_moe = self.pair.titan_layer(layer_index).moe
            normalized = self._normalized(layer_index)
            # 比较topk索引和对应权重
            _, hf_weights, hf_indices = hf_layer.mlp.gate(normalized)
            titan_weights, titan_indices, _ = titan_moe.router(
                normalized, titan_moe.expert_bias_E
            )
            recorder.discrete(
                scope="component", component="router_indices", layer=layer_index,
                actual=titan_indices.view_as(hf_indices), expected=hf_indices,
                module_path=self._module_path(layer_index, "router"),
                parent_path=f"layers.{layer_index}.moe",
                level=4,
                node_kind="discrete_checkpoint",
            )
            recorder.tensor(
                scope="component", component="router_weights", layer=layer_index,
                actual=titan_weights,
                expected=hf_weights.view_as(titan_weights),
                rtol=1e-6,
                atol=1e-7,
                module_path=self._module_path(layer_index, "router"),
                parent_path=f"layers.{layer_index}.moe",
                level=4,
                node_kind="activation_checkpoint",
            )
        return recorder

    def compare_attention(self, layer_indices: list[int]) -> ParityRecorder:
        recorder = self._recorder()
        for layer_index in layer_indices:
            normalized = self._normalized(layer_index)
            hf_output = self.pair.hf_layer(layer_index).self_attn(
                hidden_states=normalized,
                position_embeddings=self.batch.hf_position_embeddings,
                attention_mask=self.batch.causal_mask,
                position_ids=self.batch.positions,
            )[0]
            titan_output = self.pair.titan_layer(layer_index).attention(
                normalized, self.batch.causal_mask, self.batch.positions
            )
            recorder.tensor(
                scope="component", component="attention", layer=layer_index,
                actual=titan_output, expected=hf_output,
                module_path=self._module_path(layer_index, "attention"),
                parent_path=f"layers.{layer_index}",
                level=3,
                node_kind="attention_checkpoint",
            )
        return recorder

    def compare_blocks(self, layer_indices: list[int]) -> ParityRecorder:
        recorder = self._recorder()
        for layer_index in layer_indices:
            hf_output = self.pair.hf_layer(layer_index)(
                self.batch.hidden_states,
                attention_mask=self.batch.causal_mask,
                position_ids=self.batch.positions,
                position_embeddings=self.batch.hf_position_embeddings,
                use_cache=False,
            )[0]
            titan_output = self.pair.titan_layer(layer_index)(
                self.batch.hidden_states,
                self.batch.causal_mask,
                self.batch.positions,
            )
            recorder.tensor(
                scope="composition", component="decoder_block", layer=layer_index,
                actual=titan_output, expected=hf_output,
                module_path=self._module_path(layer_index, "block"),
                parent_path="layers",
                level=2,
                node_kind="layer_checkpoint",
            )
        return recorder


class _ParityDiagnostics:
    def _check_format_reports_layer_position_and_discrete_mismatches(self) -> None:
        zero = torch.zeros(1, 3, 2)
        divergent = zero.clone()
        divergent[0, 1, 0] = 0.5
        indices = torch.tensor([[[0, 1], [0, 1], [0, 1]]])
        changed = torch.tensor([[[0, 1], [0, 1], [0, 2]]])
        records = {
            "hf_blocks": {0: zero, 1: zero},
            "titan_blocks": {0: zero, 1: divergent},
            "hf_indexer": {0: indices, 1: indices},
            "titan_indexer": {0: indices, 1: changed},
            "hf_router": {},
            "titan_router": {},
        }
        diagnostics = _format_parity_diagnostics(records, torch.tensor([[0, 1, 2]]))
        self.assertIn("layer 1: block_max_abs=0.5", diagnostics)
        self.assertIn("block_position_max_abs=[0:0, 1:0.5, 2:0]", diagnostics)
        self.assertIn("indexer_mismatched_queries=1 positions=[2]", diagnostics)

    def _check_format_reports_missing_expected_records(self) -> None:
        records = {
            "hf_blocks": {0: torch.zeros(1, 2, 2)},
            "titan_blocks": {0: None},
            "hf_indexer": {0: None},
            "titan_indexer": {0: None},
            "hf_router": {1: torch.zeros(1, 2, 2)},
            "titan_router": {1: None},
        }
        diagnostics = _format_parity_diagnostics(records, torch.tensor([[0, 1]]))
        self.assertIn("block_records=missing(hf=True, titan=False)", diagnostics)
        self.assertIn("indexer_records=missing(hf=False, titan=False)", diagnostics)
        self.assertIn("router_records=missing(hf=True, titan=False)", diagnostics)

    def _check_format_reports_incompatible_block_shape(self) -> None:
        malformed = torch.zeros(2, 2)
        records = {
            "hf_blocks": {0: malformed},
            "titan_blocks": {0: malformed},
            "hf_indexer": {},
            "titan_indexer": {},
            "hf_router": {},
            "titan_router": {},
        }
        diagnostics = _format_parity_diagnostics(records, torch.tensor([[0, 1]]))
        self.assertIn(
            "block_shape_error=expected [B, L, D] compatible with positions",
            diagnostics,
        )

    def _check_recorder_reports_tensor_and_discrete_rows(self) -> None:
        recorder = ParityRecorder(BF16)
        recorder.tensor(
            scope="component", component="block", layer=1,
            actual=torch.tensor([1.0, 1.5]),
            expected=torch.tensor([1.0, 1.0]),
            module_path="layers.1.moe.routed_experts.7.w1",
        )
        recorder.discrete(
            scope="component", component="indexer", layer=1,
            actual=torch.tensor([[[0, 2]]]), expected=torch.tensor([[[0, 1]]]),
        )
        report = recorder.table()
        self.assertIn("block", report)
        self.assertIn("layers.1.moe.routed_experts.7.w1", report)
        self.assertIn("indexer", report)
        self.assertIn("failed=2", report)

    def _check_recorder_writes_plain_text_report(self) -> None:
        recorder = ParityRecorder(FP32)
        recorder.tensor(
            scope="unit", component="identity", layer="-",
            actual=torch.ones(2), expected=torch.ones(2),
        )
        self.assertIn("pass_rate=100.0%", recorder.write())

    def _check_recorder_reports_extended_metrics_and_explosions(self) -> None:
        recorder = ParityRecorder(FP32)
        first = recorder.tensor(
            scope="trace",
            component="attention_norm",
            layer=0,
            actual=torch.tensor([1.0, 1.000001]),
            expected=torch.ones(2),
            module_path="layers.0.attention_norm",
            checkpoint=False,
        )
        second = recorder.tensor(
            scope="trace",
            component="attention",
            layer=0,
            actual=torch.tensor([1.0, 1.001]),
            expected=torch.ones(2),
            module_path="layers.0.attention",
        )
        recorder.ordered_results()
        self.assertIsNotNone(first.mean_abs)
        self.assertIsNotNone(second.mean_rel)
        self.assertIsNotNone(second.relative_l2)
        self.assertIsNotNone(second.cosine_similarity)
        self.assertTrue(second.explosion)
        self.assertIn("mean_abs", recorder.table())
        self.assertIn("explosion-row", recorder.html_table())


class _ParityRouterPrecision:
    def _check_router_gate_is_evaluated_in_float32(self) -> None:
        model_size = ParityModelSize.router_unit_profile()
        titan_model = _titan_config(model_size).build()
        titan_model.init_states()
        titan_model.bfloat16()
        router = titan_model.layers["1"].moe.router
        hidden_states = torch.randn(
            1, 16, model_size.dim, dtype=torch.bfloat16
        )
        _, _, scores = router(hidden_states, titan_model.layers["1"].moe.expert_bias_E)
        expected_scores = torch.sigmoid(
            F.linear(hidden_states.float(), router.gate.weight.float())
        )
        self.assertEqual(scores.dtype, torch.float32)
        torch.testing.assert_close(scores, expected_scores, rtol=0, atol=0)

    def _check_router_uses_gate_forward_for_fp32_biased_computation(self) -> None:
        router = TokenChoiceTopKRouter.Config(
            num_experts=4,
            gate=Linear.Config(in_features=3, out_features=4, bias=True),
            top_k=2,
            score_func="sigmoid",
        ).build()
        router.bfloat16()
        hidden_states = torch.randn(1, 2, 3, dtype=torch.bfloat16)
        calls = 0

        def count_gate_calls(*_args) -> None:
            nonlocal calls
            calls += 1

        hook = router.gate.register_forward_hook(count_gate_calls)
        try:
            _, _, scores = router(hidden_states)
        finally:
            hook.remove()
        expected_scores = torch.sigmoid(
            F.linear(hidden_states.float(), router.gate.weight.float(), router.gate.bias.float())
        )
        self.assertEqual(calls, 1)
        torch.testing.assert_close(scores, expected_scores, rtol=0, atol=0)
        scores.sum().backward()
        self.assertIsNotNone(router.gate.weight.grad)
        self.assertIsNotNone(router.gate.bias.grad)


class _ParityComponentTests(ComponentParityMixin):
    _active_suite_report: ParitySuiteReport | None = None

    def _publish_recorder(
        self,
        recorder: ParityRecorder,
        section_id: str,
        title: str,
    ) -> str:
        """Publish one test now or register it with the active suite report."""
        recorder.precision_label = self._configured_report_label()
        recorder.title = (
            f"{title} [{recorder.precision_label}] "
            f"(layers={self.LAYER_INDICES}, seed={self._active_data_seed})"
        )
        if self._active_suite_report is not None:
            self._active_suite_report.add(section_id, recorder)
            return recorder.table(color=False)
        report = recorder.write(self._report_path(section_id))
        if recorder.failed:
            recorder.assert_all_passed()
        return report

    def _uses_exact_transformers_component_path(self) -> bool:
        spec = self._configured_spec()
        return (
            spec.actual.implementation == "titan"
            and spec.expected.implementation == "hf"
            and spec.actual.precision is spec.expected.precision
        )

    def _configured_component_recorder(self, component: str) -> ParityRecorder:
        """Select the exact or generic compare implementation from config."""
        layers = self._selected_layers()
        exact_components = {"indexer", "router", "attention", "block"}
        if (
            self._uses_exact_transformers_component_path()
            and component in exact_components
        ):
            if component == "router":
                layers = [
                    layer
                    for layer in layers
                    if self.pair.titan_layer(layer).moe_enabled
                ]
            return self.compare_components(component, layers)
        return self.compare_component_trace(
            component,
            layers,
            spec=self._configured_spec(),
            sequential=self.COMPONENT_EXECUTION.lower() == "sequential",
        )

    def _check_indexer_topk_matches_transformers(self) -> ParityRecorder:
        return self._configured_component_recorder("indexer")

    def _check_router_selection_and_weights(self) -> ParityRecorder:
        return self._configured_component_recorder("router")

    def _check_attention_output(self) -> ParityRecorder:
        return self._configured_component_recorder("attention")

    def _check_dense_block_output(self) -> ParityRecorder:
        return self._configured_component_recorder("block")




# 主测试类
class TestGlm5Parity(
    unittest.TestCase,
    _ParityDiagnostics,
    _ParityRouterPrecision,
    _ParityComponentTests,
):
    """Single configurable GLM-5 parity suite.

    The suite owns one runtime endpoint selection.  By default it compares
    TorchTitan FP32 with Transformers FP32.  Set ``GLM5_PARITY_PRECISION`` to
    ``bf16`` for the same implementation comparison at BF16, or set the two
    endpoint variables for a cross-precision run:

    ``GLM5_PARITY_ACTUAL=titan:bf16``
    ``GLM5_PARITY_EXPECTED=hf:fp32``
    ``GLM5_PARITY_LAYERS=0,1`` (or ``all``)
    ``GLM5_PARITY_COMPONENTS=indexer,router,gradient``
    ``GLM5_PARITY_DATA_CASE=random|zeros|ones|extreme|alternating``
    ``GLM5_PARITY_COMPONENT_EXECUTION=independent|sequential``
    ``GLM5_PARITY_HF_ROUTED_EXPERT_COMPUTE=model|bf16|grouped_mm``
    ``GLM5_PARITY_TITAN_ROUTED_EXPERT_COMPUTE=model|fp32``
    ``GLM5_PARITY_REPORT_DIR=.``
    ``GLM5_PARITY_MODEL_DIM=2048``
    ``GLM5_PARITY_MODEL_LAYERS=11``

    Decoupled execution additionally supports:

    ``GLM5_PARITY_MODE=prepare|capture|compare|paired``
    ``GLM5_PARITY_FIXTURE=parity_fixtures/glm5.2-fp32``
    ``GLM5_PARITY_ENDPOINT=titan:fp32``
    ``GLM5_PARITY_ARTIFACT=parity_artifacts/titan-gpu-fp32``
    ``GLM5_PARITY_ACTUAL_ARTIFACT=parity_artifacts/titan-npu-fp32``
    ``GLM5_PARITY_EXPECTED_ARTIFACT=parity_artifacts/titan-gpu-fp32``

    The test methods do not construct a precision-specific class.  They use
    the models and batches prepared here.  Fixed runners preserve the original
    checks, while ``compare_component`` can trace any named module path.
    ``test_configured_precision_suite`` is the complete entry point.  It calls
    the independent component and end-to-end tests, then combines their tables
    into one automatically named HTML file.  ``GLM5_PARITY_REPORT`` remains an
    optional explicit path override.
    """

    # Runtime test configuration.
    RUN_MODE = os.environ.get("GLM5_PARITY_MODE", "paired").lower()
    ACTUAL_ENDPOINT = os.environ.get("GLM5_PARITY_ACTUAL", "titan:fp32")
    EXPECTED_ENDPOINT = os.environ.get("GLM5_PARITY_EXPECTED", "hf:fp32")
    CAPTURE_ENDPOINT = os.environ.get(
        "GLM5_PARITY_ENDPOINT", ACTUAL_ENDPOINT
    )
    ARTIFACT_PATH = os.environ.get("GLM5_PARITY_ARTIFACT")
    FIXTURE_PATH = os.environ.get("GLM5_PARITY_FIXTURE")
    COMPARE_ACTUAL_ARTIFACT = os.environ.get(
        "GLM5_PARITY_ACTUAL_ARTIFACT"
    )
    COMPARE_EXPECTED_ARTIFACT = os.environ.get(
        "GLM5_PARITY_EXPECTED_ARTIFACT"
    )
    PRECISION_OVERRIDE = os.environ.get("GLM5_PARITY_PRECISION")
    RUN_COMPONENTS = os.environ.get("GLM5_PARITY_COMPONENTS", "all")
    LAYER_INDICES = os.environ.get("GLM5_PARITY_LAYERS", "all")
    DATA_CASE = os.environ.get("GLM5_PARITY_DATA_CASE", "random")
    DATA_SEED = int(os.environ.get("GLM5_PARITY_DATA_SEED", "61"))
    MODEL_SEED = int(os.environ.get("GLM5_PARITY_MODEL_SEED", "61"))
    BATCH_SIZE = int(os.environ.get("GLM5_PARITY_BATCH_SIZE", "2"))
    SEQUENCE_LENGTH = int(os.environ.get("GLM5_PARITY_SEQUENCE_LENGTH", "16"))
    COMPONENT_EXECUTION = os.environ.get(
        "GLM5_PARITY_COMPONENT_EXECUTION", "independent"
    )
    HF_ROUTED_EXPERT_COMPUTE = os.environ.get(
        "GLM5_PARITY_HF_ROUTED_EXPERT_COMPUTE", "model"
    ).lower()
    TITAN_ROUTED_EXPERT_COMPUTE = os.environ.get(
        "GLM5_PARITY_TITAN_ROUTED_EXPERT_COMPUTE", "model"
    ).lower()

    @staticmethod
    def _endpoint(value: str) -> ModelEndpoint:
        implementation, _, precision_name = value.lower().partition(":")
        if not precision_name:
            precision_name = precision_name or "fp32"
        if implementation not in {"hf", "titan"}:
            raise ValueError(f"unsupported GLM-5 implementation: {implementation}")
        policies = {FP32.name: FP32, BF16.name: BF16, "bfloat16": BF16}
        try:
            precision = policies[precision_name]
        except KeyError as error:
            raise ValueError(f"unsupported GLM-5 precision: {precision_name}") from error
        return ModelEndpoint(implementation, precision)

    @classmethod
    def _configured_endpoints(cls) -> tuple[ModelEndpoint, ModelEndpoint]:
        if cls.PRECISION_OVERRIDE:
            precision = cls.PRECISION_OVERRIDE.lower()
            return (
                cls._endpoint(f"titan:{precision}"),
                cls._endpoint(f"hf:{precision}"),
            )
        return cls._endpoint(cls.ACTUAL_ENDPOINT), cls._endpoint(cls.EXPECTED_ENDPOINT)

    @classmethod
    def _configured_cases(cls) -> list[ParityCase]:
        """Materialize suite order so two servers execute the same cases."""
        selected = {
            item.strip().lower()
            for item in cls.RUN_COMPONENTS.split(",")
            if item.strip()
        }

        def enabled(name: str) -> bool:
            return not selected or "all" in selected or name in selected

        definitions: list[tuple[str, str, str]] = []
        for component in ("indexer", "router", "attention", "block"):
            if enabled(component):
                definitions.append(
                    (component, f"component-{component}", component)
                )
        fixed = {
            "all",
            "indexer",
            "router",
            "attention",
            "block",
            "gradient",
            "parameters",
            "logits",
            "loss",
            "model",
            "e2e",
        }
        for component in sorted(selected - fixed):
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", component)
            definitions.append(
                (f"component:{component}", f"component-{safe}", component)
            )
        definitions.append(("end-to-end", "end-to-end", "model"))
        return [
            ParityCase(
                ordinal=index,
                case_id=test_name,
                section_id=section_id,
                component=component,
                seed=cls._test_seed(cls.DATA_SEED, test_name),
            )
            for index, (test_name, section_id, component) in enumerate(definitions)
        ]

    @classmethod
    def _capture_test_plan(cls) -> dict[str, Any]:
        return {
            "suite": "glm5.2",
            "suite_version": GLM5_PARITY_SUITE_VERSION,
            "cases": [asdict(case) for case in cls.capture_cases],
            "layers": cls.LAYER_INDICES,
            "components": cls.RUN_COMPONENTS,
            "data_case": cls.DATA_CASE,
            "base_seed": cls.DATA_SEED,
            "model_seed": cls.MODEL_SEED,
            "batch_size": cls.BATCH_SIZE,
            "sequence_length": cls.SEQUENCE_LENGTH,
            "component_execution": cls.COMPONENT_EXECUTION,
            "model_size": asdict(cls.model_size),
        }

    @classmethod
    def _fixture_configuration(cls) -> dict[str, Any]:
        return {
            "layers": cls.LAYER_INDICES,
            "components": cls.RUN_COMPONENTS,
            "data_case": cls.DATA_CASE,
            "data_seed": cls.DATA_SEED,
            "model_seed": cls.MODEL_SEED,
            "batch_size": cls.BATCH_SIZE,
            "sequence_length": cls.SEQUENCE_LENGTH,
            "component_execution": cls.COMPONENT_EXECUTION,
            "model_size": asdict(cls.model_size),
            "estimated_fp32_model_gb": cls.model_size.estimated_fp32_size_gb,
            "scenario_id": os.environ.get("GLM5_PARITY_SCENARIO_ID", ""),
            "scenario_config_digest": os.environ.get(
                "GLM5_PARITY_SCENARIO_CONFIG_DIGEST", ""
            ),
            "scenario_configuration": os.environ.get(
                "GLM5_PARITY_SCENARIO_CONFIG_JSON", ""
            ),
        }

    @classmethod
    def _capture_configuration(cls) -> dict[str, Any]:
        configuration = cls._fixture_configuration()
        configuration["routed_expert_compute"] = {
            "hf": cls.HF_ROUTED_EXPERT_COMPUTE,
            "titan": cls.TITAN_ROUTED_EXPERT_COMPUTE,
        }
        return configuration

    @classmethod
    def _build_capture_model(
        cls,
        endpoint: ModelEndpoint,
        canonical_state: dict[str, torch.Tensor],
    ) -> torch.nn.Module:
        titan_config = _titan_config(cls.model_size)
        cls.adapter = Glm5StateDictAdapter(titan_config, hf_assets_path=None)
        if endpoint.implementation == "hf":
            if _TRANSFORMERS_IMPORT_ERROR is not None:
                raise RuntimeError(
                    "capturing an HF endpoint requires Transformers: "
                    f"{_TRANSFORMERS_IMPORT_ERROR!r}"
                )
            model = GlmMoeDsaForCausalLM(_hf_config(cls.model_size)).float()
            incompatible = model.load_state_dict(
                cls.adapter.to_hf(canonical_state), strict=False
            )
            unsupported_missing = set(incompatible.missing_keys).difference(
                cls.adapter._IGNORED_HF_KEYS
            )
            if unsupported_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "fixture state is incompatible with the HF endpoint: "
                    f"missing={sorted(unsupported_missing)}, "
                    f"unexpected={sorted(incompatible.unexpected_keys)}"
                )
        else:
            model = titan_config.build()
            model.init_states()
            model.load_state_dict(canonical_state, strict=True)

        model.to(dtype=endpoint.precision.dtype)
        for name, parameter in model.named_parameters():
            if name.endswith("indexer.weights_proj.weight"):
                parameter.data = parameter.data.float()
        model.to(device=cls.device).eval()
        return model

    @classmethod
    def _make_fixture_batches(cls) -> dict[str, ParityBatch]:
        return {
            case.case_id: ParityDataFactory.make(
                device=torch.device("cpu"),
                dtype=torch.float32,
                batch_size=cls.BATCH_SIZE,
                sequence_length=cls.SEQUENCE_LENGTH,
                hidden_size=cls.model_size.dim,
                vocab_size=cls.model_size.vocab_size,
                seed=case.seed,
                data_case=cls.DATA_CASE,
                make_tokens=True,
            )
            for case in cls.capture_cases
        }

    @classmethod
    def _set_up_prepare_class(cls) -> None:
        if not cls.FIXTURE_PATH:
            raise ValueError(
                "GLM5_PARITY_MODE=prepare requires GLM5_PARITY_FIXTURE"
            )
        if Path(cls.FIXTURE_PATH).exists():
            raise FileExistsError(
                f"GLM5_PARITY_FIXTURE already exists: {cls.FIXTURE_PATH}"
            )
        cls.gpu_ready = True
        cls.gpu_skip_reason = ""
        cls.device_type = "cpu"
        cls.model_size = ParityModelSize.from_env()
        cls.model_size.validate(sequence_length=cls.SEQUENCE_LENGTH)
        cls.capture_cases = cls._configured_cases()
        cls.capture_plan = cls._capture_test_plan()

    @classmethod
    def _load_fixture(
        cls,
    ) -> tuple[dict[str, torch.Tensor], dict[str, ParityBatch], str]:
        if not cls.FIXTURE_PATH:
            raise ValueError(
                "GLM5_PARITY_MODE=capture requires GLM5_PARITY_FIXTURE"
            )
        with ParityArtifactReader(cls.FIXTURE_PATH) as reader:
            if reader.manifest.get("status") != "success":
                raise ParityArtifactError(
                    "GLM5_PARITY_FIXTURE must be a successful fixture"
                )
            if reader.manifest.get("suite") != "glm5.2-fixture":
                raise ParityArtifactError(
                    "GLM5_PARITY_FIXTURE is not a GLM-5.2 fixture"
                )
            if reader.manifest.get("suite_version") != GLM5_PARITY_SUITE_VERSION:
                raise ParityArtifactError(
                    "GLM5_PARITY_FIXTURE suite version does not match this code"
                )
            if reader.manifest.get("test_plan_digest") != json_digest(
                cls.capture_plan
            ):
                raise ParityArtifactError(
                    "GLM5_PARITY_FIXTURE test plan does not match this capture"
                )
            if reader.manifest.get("configuration_digest") != json_digest(
                cls._fixture_configuration()
            ):
                raise ParityArtifactError(
                    "GLM5_PARITY_FIXTURE configuration does not match this capture"
                )
            canonical_state: dict[str, torch.Tensor] = {}
            fingerprints: dict[str, str] = {}
            for key in sorted(reader.observation_keys):
                metadata = reader.metadata(key)
                state_key = metadata.tags.get("state_key")
                if metadata.scope != "fixture_model" or not state_key:
                    continue
                value = reader.tensor(key).clone()
                canonical_state[state_key] = value
                fingerprints[f"model/{state_key}"] = tensor_digest(value)
            if not canonical_state:
                raise ParityArtifactError(
                    "GLM5_PARITY_FIXTURE contains no model state"
                )
            batches: dict[str, ParityBatch] = {}
            for case in cls.capture_cases:
                prefix = f"fixture/data/{case.case_id}"
                values = {
                    field_name: reader.tensor(f"{prefix}/{field_name}").clone()
                    for field_name in (
                        "hidden_states",
                        "positions",
                        "causal_mask",
                        "tokens",
                    )
                }
                batches[case.case_id] = ParityBatch(
                    hidden_states=values["hidden_states"],
                    positions=values["positions"],
                    causal_mask=values["causal_mask"],
                    hf_position_embeddings=(torch.empty(0), torch.empty(0)),
                    tokens=values["tokens"],
                )
                for field_name, value in values.items():
                    fingerprints[f"data/{case.case_id}/{field_name}"] = (
                        tensor_digest(value)
                    )
            fixture_digest = json_digest(fingerprints)
            if fixture_digest != reader.manifest.get("fixture_digest"):
                raise ParityArtifactError(
                    "GLM5_PARITY_FIXTURE tensors do not reproduce its digest"
                )
            return canonical_state, batches, fixture_digest

    @classmethod
    def _set_up_capture_class(cls) -> None:
        if not cls.FIXTURE_PATH:
            raise ValueError(
                "GLM5_PARITY_MODE=capture requires GLM5_PARITY_FIXTURE"
            )
        if not Path(cls.FIXTURE_PATH).is_dir():
            raise FileNotFoundError(
                f"GLM5_PARITY_FIXTURE does not exist: {cls.FIXTURE_PATH}"
            )
        if not cls.ARTIFACT_PATH:
            raise ValueError(
                "GLM5_PARITY_MODE=capture requires GLM5_PARITY_ARTIFACT"
            )
        if Path(cls.ARTIFACT_PATH).exists():
            raise FileExistsError(
                f"GLM5_PARITY_ARTIFACT already exists: {cls.ARTIFACT_PATH}"
            )
        cls.device, cls.device_module, cls.device_type = _parity_device()
        cls.gpu_ready = True
        cls.gpu_skip_reason = ""
        cls.capture_endpoint = cls._endpoint(cls.CAPTURE_ENDPOINT)
        cls.actual_endpoint = cls.capture_endpoint
        cls.expected_endpoint = cls.capture_endpoint
        cls.precision = cls.capture_endpoint.precision
        cls.model_size = ParityModelSize.from_env()
        cls.model_size.validate(sequence_length=cls.SEQUENCE_LENGTH)
        cls.capture_cases = cls._configured_cases()
        cls.capture_plan = cls._capture_test_plan()
        canonical_state, cls.capture_batches, cls.fixture_digest = (
            cls._load_fixture()
        )
        cls.capture_model = cls._build_capture_model(
            cls.capture_endpoint, canonical_state
        )
        del canonical_state
        if (
            cls.capture_endpoint.implementation == "hf"
            and cls.HF_ROUTED_EXPERT_COMPUTE != "model"
        ):
            _set_hf_routed_expert_compute_dtype(
                cls.capture_model,
                torch.bfloat16,
                use_grouped_mm=cls.HF_ROUTED_EXPERT_COMPUTE == "grouped_mm",
            )
        if (
            cls.capture_endpoint.implementation == "titan"
            and cls.TITAN_ROUTED_EXPERT_COMPUTE == "fp32"
        ):
            if cls.capture_endpoint.precision is not FP32:
                raise ValueError(
                    "the TorchTitan FP32 routed-expert experiment requires fp32"
                )
            _set_titan_routed_expert_compute_dtype(
                cls.capture_model, torch.float32
            )
        _annotate_endpoint_compute_dtypes(
            cls.capture_endpoint, cls.capture_model
        )
        cls.num_model_parameters = sum(
            parameter.numel() for parameter in cls.capture_model.parameters()
        )

    @classmethod
    def setUpClass(cls) -> None:
        if cls.RUN_MODE not in {"paired", "prepare", "capture", "compare"}:
            raise ValueError(
                "GLM5_PARITY_MODE must be paired, prepare, capture, or compare"
            )
        if cls.RUN_MODE == "compare":
            cls.gpu_ready = True
            cls.gpu_skip_reason = ""
            return
        if cls.RUN_MODE == "capture":
            cls._set_up_capture_class()
            return
        if cls.RUN_MODE == "prepare":
            cls._set_up_prepare_class()
            return
        if _TRANSFORMERS_IMPORT_ERROR is not None:
            cls.gpu_ready = False
            cls.gpu_skip_reason = f"Transformers unavailable: {_TRANSFORMERS_IMPORT_ERROR!r}"
            return
        try:
            cls.device, cls.device_module, cls.device_type = _parity_device()
        except RuntimeError as error:
            cls.gpu_ready = False
            cls.gpu_skip_reason = str(error)
            return

        cls.gpu_ready = True
        cls.actual_endpoint, cls.expected_endpoint = cls._configured_endpoints()
        cls.precision = cls.actual_endpoint.precision
        cls.model_size = ParityModelSize.from_env()
        cls.model_size.validate(sequence_length=cls.SEQUENCE_LENGTH)
        if cls.HF_ROUTED_EXPERT_COMPUTE not in {
            "model",
            "bf16",
            "grouped_mm",
        }:
            raise ValueError(
                "GLM5_PARITY_HF_ROUTED_EXPERT_COMPUTE must be "
                "model, bf16, or grouped_mm"
            )
        if cls.TITAN_ROUTED_EXPERT_COMPUTE not in {"model", "fp32"}:
            raise ValueError(
                "GLM5_PARITY_TITAN_ROUTED_EXPERT_COMPUTE must be model or fp32"
            )
        if cls.TITAN_ROUTED_EXPERT_COMPUTE == "fp32":
            titan_endpoints = (
                endpoint
                for endpoint in (cls.actual_endpoint, cls.expected_endpoint)
                if endpoint.implementation == "titan"
            )
            if any(endpoint.precision is not FP32 for endpoint in titan_endpoints):
                raise ValueError(
                    "the TorchTitan FP32 routed-expert experiment requires "
                    "every configured Titan endpoint to use fp32"
                )

        # Build the endpoint table once.  Test methods only select from this
        # table and never perform ad hoc dtype conversion.
        precisions = {
            cls.actual_endpoint.precision.name,
            cls.expected_endpoint.precision.name,
        }
        cls.models: dict[tuple[str, str], torch.nn.Module] = {}
        cls.pairs: dict[str, ParityModelPair] = {}
        for precision_name in sorted(precisions):
            precision = BF16 if precision_name == BF16.name else FP32
            pair = _build_pair(
                cls.device,
                model_size=cls.model_size,
                precision=precision,
                seed=cls.MODEL_SEED,
            )
            if cls.HF_ROUTED_EXPERT_COMPUTE != "model":
                _set_hf_routed_expert_compute_dtype(
                    pair.hf,
                    torch.bfloat16,
                    use_grouped_mm=(
                        cls.HF_ROUTED_EXPERT_COMPUTE == "grouped_mm"
                    ),
                )
            if (
                cls.TITAN_ROUTED_EXPERT_COMPUTE == "fp32"
                and precision is FP32
            ):
                _set_titan_routed_expert_compute_dtype(
                    pair.titan, torch.float32
                )
            _annotate_known_compute_dtypes(pair)
            cls.pairs[precision.name] = pair
            cls.models[("hf", precision.name)] = pair.hf
            cls.models[("titan", precision.name)] = pair.titan

        # Component methods operate on a same-precision HF/TorchTitan pair.
        cls.pair = cls.pairs[cls.actual_endpoint.precision.name]
        cls.hf_model = cls.pair.hf
        cls.titan_model = cls.pair.titan
        cls.num_model_parameters = sum(
            parameter.numel() for parameter in cls.hf_model.parameters()
        )
        cls.batch = ParityDataFactory.make(
            device=cls.device,
            dtype=cls.actual_endpoint.precision.dtype,
            batch_size=cls.BATCH_SIZE,
            sequence_length=cls.SEQUENCE_LENGTH,
            hidden_size=cls.pair.hf.config.hidden_size,
            vocab_size=cls.pair.hf.config.vocab_size,
            seed=cls.DATA_SEED,
            data_case=cls.DATA_CASE,
            make_tokens=True,
        )
        cls.batch = ParityDataFactory.attach_hf_position_embeddings(
            cls.batch, cls.pair.hf
        )
        cls.hidden_states = cls.batch.hidden_states
        cls.positions = cls.batch.positions
        cls.causal_mask = cls.batch.causal_mask
        cls.hf_position_embeddings = cls.batch.hf_position_embeddings
        cls.base_batch = ParityDataFactory.make(
            device=cls.device,
            dtype=torch.float32,
            batch_size=cls.BATCH_SIZE,
            sequence_length=cls.SEQUENCE_LENGTH,
            hidden_size=cls.pair.hf.config.hidden_size,
            vocab_size=cls.pair.hf.config.vocab_size,
            seed=cls.DATA_SEED,
            data_case=cls.DATA_CASE,
            make_tokens=True,
        )
        cls.adapter = Glm5StateDictAdapter(
            _titan_config(cls.model_size), hf_assets_path=None
        )
        cls._active_data_seed = cls.DATA_SEED

    def setUp(self) -> None:
        if (
            self.RUN_MODE in {"prepare", "capture", "compare"}
            and self._testMethodName != "test_configured_precision_suite"
        ):
            self.skipTest(
                f"GLM5_PARITY_MODE={self.RUN_MODE} uses the configured suite entry point"
            )

    @staticmethod
    def _test_seed(base_seed: int, test_name: str) -> int:
        offset = sum(
            (index + 1) * ord(character)
            for index, character in enumerate(test_name)
        )
        return base_seed + offset

    def _activate_test_data(self, test_name: str) -> None:
        """Create one deterministic batch owned by a single test method."""
        seed = self._test_seed(self.DATA_SEED, test_name)
        self._active_data_seed = seed
        self.base_batch = ParityDataFactory.make(
            device=self.device,
            dtype=torch.float32,
            batch_size=self.BATCH_SIZE,
            sequence_length=self.SEQUENCE_LENGTH,
            hidden_size=self.model_size.dim,
            vocab_size=self.model_size.vocab_size,
            seed=seed,
            data_case=self.DATA_CASE,
            make_tokens=True,
        )
        self.batch = ParityDataFactory.cast(
            self.base_batch,
            model=self.hf_model,
            dtype=self.actual_endpoint.precision.dtype,
        )
        self.hidden_states = self.batch.hidden_states
        self.positions = self.batch.positions
        self.causal_mask = self.batch.causal_mask
        self.hf_position_embeddings = self.batch.hf_position_embeddings

    def _model(self, endpoint: ModelEndpoint) -> torch.nn.Module:
        if self.RUN_MODE == "capture":
            if endpoint != self.capture_endpoint:
                raise KeyError(
                    f"capture run owns only endpoint {self.capture_endpoint.label}"
                )
            return self.capture_model
        return self.models[(endpoint.implementation, endpoint.precision.name)]

    def _batch_for(self, endpoint: ModelEndpoint) -> ParityBatch:
        if self.RUN_MODE == "capture":
            return self._capture_batch(self._active_capture_case)
        return ParityDataFactory.cast(
            self.base_batch,
            model=self._model(endpoint),
            dtype=endpoint.precision.dtype,
        )

    def _capture_batch(self, case: ParityCase) -> ParityBatch:
        source = self.capture_batches[case.case_id]
        dtype = self.capture_endpoint.precision.dtype
        device_batch = ParityBatch(
            hidden_states=source.hidden_states.to(dtype=dtype).to(self.device),
            positions=source.positions.to(self.device),
            causal_mask=source.causal_mask.to(dtype=dtype).to(self.device),
            hf_position_embeddings=(
                torch.empty(0, device=self.device),
                torch.empty(0, device=self.device),
            ),
            tokens=(
                source.tokens.to(self.device)
                if source.tokens is not None
                else None
            ),
        )
        if self.capture_endpoint.implementation == "hf":
            return ParityDataFactory.attach_hf_position_embeddings(
                device_batch, self.capture_model
            )
        return device_batch

    def _new_capture_writer(self) -> ParityArtifactWriter:
        run_id = os.environ.get(
            "GLM5_PARITY_RUN_ID",
            datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
        )
        try:
            device_name = self.device_module.get_device_name(self.device)
        except (AttributeError, RuntimeError):
            device_name = str(self.device)
        environment: dict[str, Any] = runtime_metadata()
        environment.update(_git_metadata())
        environment["argv"] = repr(sys.argv)
        environment["device_type"] = self.device_type
        environment["device_name"] = device_name
        environment["fixture_path"] = str(self.FIXTURE_PATH)
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "ASCEND_RT_VISIBLE_DEVICES",
            "ASCEND_HOME_PATH",
            "ASCEND_OPP_PATH",
            "PYTORCH_ALLOC_CONF",
            "PYTORCH_NPU_ALLOC_CONF",
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
        ):
            if name in os.environ:
                environment[f"env_{name}"] = os.environ[name]
        if self.device_type == "npu":
            try:
                import torch_npu

                environment["torch_npu"] = getattr(
                    torch_npu, "__version__", "unknown"
                )
            except ImportError:
                environment["torch_npu"] = "unavailable"
        identity = EndpointIdentity(
            implementation=self.capture_endpoint.implementation,
            precision=self.capture_endpoint.precision.name,
            device_type=self.device_type,
            device_name=device_name,
            run_id=run_id,
            rank=int(os.environ.get("RANK", "0")),
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
        )
        return ParityArtifactWriter(
            self.ARTIFACT_PATH,
            suite="glm5.2",
            suite_version=GLM5_PARITY_SUITE_VERSION,
            endpoint=identity,
            test_plan=self.capture_plan,
            configuration=self._capture_configuration(),
            fixture_digest=self.fixture_digest,
            environment=environment,
        )

    def _capture_add(
        self,
        *,
        key: str,
        section_id: str,
        tensor: torch.Tensor,
        scope: str,
        component: str,
        layer: int | str,
        value_kind: str = "tensor",
        module_path: str = "",
        parent_path: str = "",
        level: int = 0,
        node_kind: str = "checkpoint",
        checkpoint: bool = True,
        dtype_flow: str = "",
        positions_key: str = "",
        compare: bool = True,
        tags: dict[str, str] | None = None,
    ) -> None:
        effective_tags = dict(tags or {})
        active_case = getattr(self, "_active_capture_case", None)
        if active_case is not None:
            effective_tags.setdefault("case_id", active_case.case_id)
            effective_tags.setdefault("case_ordinal", str(active_case.ordinal))
            effective_tags.setdefault("case_seed", str(active_case.seed))
        self.capture_writer.add(
            ObservationMetadata(
                key=key,
                section_id=section_id,
                scope=scope,
                component=component,
                layer=layer,
                value_kind=value_kind,
                module_path=module_path,
                parent_path=parent_path,
                level=level,
                node_kind=node_kind,
                checkpoint=checkpoint,
                dtype_flow=dtype_flow,
                positions_key=positions_key,
                compare=compare,
                tags=effective_tags,
            ),
            tensor,
        )

    def _capture_fixture_batch(self, case: ParityCase) -> None:
        batch = self.capture_batches[case.case_id]
        prefix = f"fixture/data/{case.case_id}"
        for field_name in ("hidden_states", "positions", "causal_mask", "tokens"):
            value = getattr(batch, field_name)
            assert value is not None
            self._capture_add(
                key=f"{prefix}/{field_name}",
                section_id="fixture",
                tensor=value,
                scope="fixture",
                component=field_name,
                layer="global",
                value_kind="fixture",
                compare=False,
                tags={"case_id": case.case_id},
            )

    def _capture_recursive_observations(
        self,
        *,
        section_id: str,
        trace: RecursiveModuleTrace,
        include_gradients: bool,
        component_filter: str | None = None,
        skip_filter_root: bool = False,
    ) -> None:
        endpoint = self.capture_endpoint
        label = endpoint.label
        normalized_filter = (
            component_filter.strip().strip(".")
            if component_filter is not None
            else None
        )
        logical_values: dict[str, tuple[str, torch.Tensor, str]] = {}
        for endpoint_path, value in trace.activations[label].items():
            logical_path = self._logical_activation_path(endpoint, endpoint_path)
            base_path = logical_path.split("[", 1)[0]
            target_path = (
                base_path
                if re.fullmatch(r"layers\.\d+", base_path)
                or base_path.endswith((".attention", ".moe", ".feed_forward"))
                else logical_path
            )
            logical_values.setdefault(
                target_path,
                (
                    endpoint_path,
                    value,
                    trace.dtype_flows[label].get(endpoint_path, ""),
                ),
            )
        for logical_path, (endpoint_path, value, dtype_flow) in sorted(
            logical_values.items()
        ):
            base_path = logical_path.split("[", 1)[0]
            filter_root = False
            if normalized_filter:
                layer_match = re.match(r"layers\.(\d+)", base_path)
                targets = (
                    {
                        f"layers.{layer}.{normalized_filter}"
                        for layer in self._selected_layers()
                    }
                    if layer_match
                    else {normalized_filter}
                )
                if not any(
                    base_path == target or base_path.startswith(f"{target}.")
                    for target in targets
                ) and not base_path.endswith(f".{normalized_filter}"):
                    continue
                filter_root = base_path in targets or base_path.endswith(
                    f".{normalized_filter}"
                )
            layer_match = re.match(r"layers\.(\d+)", base_path)
            layer: int | str = int(layer_match.group(1)) if layer_match else "global"
            parent_path = base_path.rpartition(".")[0]
            component = logical_path.rsplit(".", 1)[-1]
            checkpoint = base_path.endswith(
                (".attention", ".moe", ".feed_forward")
            ) or bool(normalized_filter and base_path.endswith(f".{normalized_filter}"))
            is_discrete = base_path.endswith(".attention.indexer") or value.dtype in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
                torch.bool,
            }
            positions_key = (
                f"fixture/data/{self._active_capture_case.case_id}/positions"
                if is_discrete and value.ndim == 3
                else ""
            )
            self._capture_add(
                key=f"{section_id}/activation/{logical_path}",
                section_id=section_id,
                tensor=value,
                scope="trace",
                component=component,
                layer=layer,
                value_kind="discrete" if is_discrete else "tensor",
                module_path=endpoint_path,
                parent_path=parent_path,
                level=len(logical_path.split(".")) - 1,
                node_kind=(
                    "discrete_checkpoint"
                    if is_discrete
                    else "composition_checkpoint" if checkpoint else "activation"
                ),
                checkpoint=checkpoint,
                dtype_flow=dtype_flow,
                positions_key=positions_key,
                compare=not (
                    re.fullmatch(r"layers\.\d+", base_path)
                    or base_path.endswith(
                        (".attention.indexer", ".moe.router")
                    )
                    or (skip_filter_root and filter_root)
                ),
            )
        if not include_gradients:
            return
        logical_gradients: dict[str, tuple[str, torch.Tensor]] = {}
        for endpoint_path, value in trace.gradients[label].items():
            logical_path = self._logical_activation_path(endpoint, endpoint_path)
            base_path = logical_path.split("[", 1)[0]
            target_path = (
                base_path
                if base_path.endswith((".attention", ".moe", ".feed_forward"))
                else logical_path
            )
            logical_gradients.setdefault(target_path, (endpoint_path, value))
        for logical_path, (endpoint_path, value) in sorted(
            logical_gradients.items()
        ):
            base_path = logical_path.split("[", 1)[0]
            layer_match = re.match(r"layers\.(\d+)", base_path)
            layer = int(layer_match.group(1)) if layer_match else "global"
            parent_path = base_path.rpartition(".")[0]
            component = logical_path.rsplit(".", 1)[-1]
            checkpoint = base_path.endswith(
                (".attention", ".moe", ".feed_forward")
            )
            self._capture_add(
                key=f"{section_id}/activation_gradient/{logical_path}",
                section_id=section_id,
                tensor=value,
                scope="activation_gradient",
                component=component,
                layer=layer,
                module_path=f"{endpoint_path}.grad",
                parent_path=parent_path,
                level=len(logical_path.split(".")) - 1,
                node_kind=(
                    "gradient_checkpoint" if checkpoint else "gradient_activation"
                ),
                checkpoint=checkpoint,
                compare=not base_path.endswith(".moe.router"),
            )

    def _capture_component_case(self, case: ParityCase) -> None:
        endpoint = self.capture_endpoint
        model = self.capture_model
        batch = self._capture_batch(case)
        layers = model.model.layers if endpoint.implementation == "hf" else model.layers
        layer_indices = self._selected_layers()
        component = case.component
        if component in {"indexer", "router", "attention", "block"}:
            self._capture_exact_component_checkpoints(
                case=case,
                batch=batch,
                layer_indices=layer_indices,
            )
        normalized = self._normalize_model_component(component)
        if normalized in TOP_LEVEL_COMPONENTS:
            with RecursiveModuleTrace.install(
                {endpoint: model}, [], include_model_modules=True
            ) as trace:
                with torch.no_grad():
                    _run_causal_lm_endpoint(model, endpoint, batch)
            self._capture_recursive_observations(
                section_id=case.section_id,
                trace=trace,
                include_gradients=False,
                component_filter=normalized,
            )
            return

        component_filter = self._normalize_component_name(
            layer_indices[0], component
        )
        with RecursiveModuleTrace.install(
            {endpoint: model}, layer_indices
        ) as trace:
            with torch.no_grad():
                current = batch.hidden_states
                for layer_index in layer_indices:
                    layer = (
                        layers[layer_index]
                        if endpoint.implementation == "hf"
                        else layers[str(layer_index)]
                    )
                    if endpoint.implementation == "hf":
                        output = layer(
                            current,
                            attention_mask=batch.causal_mask,
                            position_ids=batch.positions,
                            position_embeddings=batch.hf_position_embeddings,
                            use_cache=False,
                        )
                    else:
                        output = layer(
                            current,
                            batch.causal_mask,
                            batch.positions,
                        )
                    if self.COMPONENT_EXECUTION.lower() == "sequential":
                        value = (
                            output[0]
                            if endpoint.implementation == "hf"
                            and isinstance(output, (tuple, list))
                            else output
                        )
                        current = _first_tensor(value)
                    else:
                        current = batch.hidden_states
        self._capture_recursive_observations(
            section_id=case.section_id,
            trace=trace,
            include_gradients=False,
            component_filter=component_filter or None,
            skip_filter_root=True,
        )

    def _capture_exact_component_checkpoints(
        self,
        *,
        case: ParityCase,
        batch: ParityBatch,
        layer_indices: list[int],
    ) -> None:
        endpoint = self.capture_endpoint
        component = case.component
        for layer_index in layer_indices:
            layer = (
                self.capture_model.model.layers[layer_index]
                if endpoint.implementation == "hf"
                else self.capture_model.layers[str(layer_index)]
            )
            with torch.no_grad():
                normalized = (
                    layer.input_layernorm(batch.hidden_states)
                    if endpoint.implementation == "hf"
                    else layer.attention_norm(batch.hidden_states)
                )
                parent_path = f"layers.{layer_index}"
                self._capture_add(
                    key=(
                        f"{case.section_id}/exact/"
                        f"layers.{layer_index}.attention.normalized_input"
                    ),
                    section_id=case.section_id,
                    tensor=normalized,
                    scope="component",
                    component="normalized_input",
                    layer=layer_index,
                    module_path=f"{parent_path}.attention.normalized_input",
                    parent_path=f"{parent_path}.attention",
                    level=4,
                    node_kind="activation_checkpoint",
                )
                if component == "indexer":
                    attention = (
                        layer.self_attn
                        if endpoint.implementation == "hf"
                        else layer.attention
                    )
                    q_residual = (
                        attention.q_a_layernorm(attention.q_a_proj(normalized))
                        if endpoint.implementation == "hf"
                        else attention.q_norm(attention.wq_a(normalized))
                    )
                    self._capture_add(
                        key=(
                            f"{case.section_id}/exact/"
                            f"layers.{layer_index}.attention.q_residual"
                        ),
                        section_id=case.section_id,
                        tensor=q_residual,
                        scope="component",
                        component="q_residual",
                        layer=layer_index,
                        module_path=(
                            f"model.layers.{layer_index}.self_attn.q_a_layernorm"
                            if endpoint.implementation == "hf"
                            else f"{parent_path}.attention.q_norm"
                        ),
                        parent_path=f"{parent_path}.attention",
                        level=4,
                        node_kind="activation_checkpoint",
                    )
                    topk = (
                        attention.indexer(
                            normalized,
                            q_residual,
                            batch.hf_position_embeddings,
                            batch.causal_mask[:, 0],
                            batch.positions,
                        )
                        if endpoint.implementation == "hf"
                        else attention.indexer(
                            normalized,
                            q_residual,
                            batch.positions,
                            batch.causal_mask[:, 0],
                        )
                    )
                    self._capture_add(
                        key=(
                            f"{case.section_id}/exact/"
                            f"layers.{layer_index}.attention.indexer"
                        ),
                        section_id=case.section_id,
                        tensor=topk,
                        scope="component",
                        component="indexer",
                        layer=layer_index,
                        value_kind="discrete",
                        module_path=(
                            f"model.layers.{layer_index}.self_attn.indexer"
                            if endpoint.implementation == "hf"
                            else f"{parent_path}.attention.indexer"
                        ),
                        parent_path=f"{parent_path}.attention",
                        level=4,
                        node_kind="discrete_checkpoint",
                        positions_key=f"fixture/data/{case.case_id}/positions",
                    )
                elif component == "router":
                    if endpoint.implementation == "hf":
                        if not hasattr(layer.mlp, "gate"):
                            continue
                        _, weights, indices = layer.mlp.gate(normalized)
                    else:
                        if not getattr(layer, "moe_enabled", False):
                            continue
                        weights, indices, _ = layer.moe.router(
                            normalized, layer.moe.expert_bias_E
                        )
                    indices = indices.view(
                        normalized.shape[0], normalized.shape[1], -1
                    )
                    weights = weights.view_as(indices)
                    for name, value, kind in (
                        ("indices", indices, "discrete"),
                        ("weights", weights, "tensor"),
                    ):
                        self._capture_add(
                            key=(
                                f"{case.section_id}/exact/"
                                f"layers.{layer_index}.moe.router.{name}"
                            ),
                            section_id=case.section_id,
                            tensor=value,
                            scope="component",
                            component=f"router_{name}",
                            layer=layer_index,
                            value_kind=kind,
                            module_path=(
                                f"model.layers.{layer_index}.mlp.gate"
                                if endpoint.implementation == "hf"
                                else f"{parent_path}.moe.router"
                            ),
                            parent_path=f"{parent_path}.moe",
                            level=4,
                            node_kind=(
                                "discrete_checkpoint"
                                if kind == "discrete"
                                else "activation_checkpoint"
                            ),
                            tags=(
                                {"rtol": "1e-6", "atol": "1e-7"}
                                if name == "weights"
                                else None
                            ),
                        )
                elif component == "attention":
                    output = (
                        layer.self_attn(
                            hidden_states=normalized,
                            position_embeddings=batch.hf_position_embeddings,
                            attention_mask=batch.causal_mask,
                            position_ids=batch.positions,
                        )[0]
                        if endpoint.implementation == "hf"
                        else layer.attention(
                            normalized, batch.causal_mask, batch.positions
                        )
                    )
                    self._capture_add(
                        key=f"{case.section_id}/exact/{parent_path}.attention",
                        section_id=case.section_id,
                        tensor=output,
                        scope="component",
                        component="attention",
                        layer=layer_index,
                        module_path=f"{parent_path}.attention",
                        parent_path=parent_path,
                        level=3,
                        node_kind="attention_checkpoint",
                    )
                else:
                    output = (
                        layer(
                            batch.hidden_states,
                            attention_mask=batch.causal_mask,
                            position_ids=batch.positions,
                            position_embeddings=batch.hf_position_embeddings,
                            use_cache=False,
                        )[0]
                        if endpoint.implementation == "hf"
                        else layer(
                            batch.hidden_states,
                            batch.causal_mask,
                            batch.positions,
                        )
                    )
                    self._capture_add(
                        key=f"{case.section_id}/exact/{parent_path}",
                        section_id=case.section_id,
                        tensor=output,
                        scope="composition",
                        component="decoder_block",
                        layer=layer_index,
                        module_path=parent_path,
                        parent_path="layers",
                        level=2,
                        node_kind="layer_checkpoint",
                    )

    def _capture_endpoint_trace(
        self,
        case: ParityCase,
        trace: EndpointTrace,
    ) -> None:
        endpoint = self.capture_endpoint
        label = endpoint.label
        positions_key = f"fixture/data/{case.case_id}/positions"
        groups: tuple[tuple[str, dict[int, torch.Tensor], str], ...] = (
            ("decoder_block", trace.blocks[label], "tensor"),
            ("indexer", trace.indexer[label], "discrete"),
            ("indexer_scores", trace.indexer_scores[label], "tensor"),
            ("router", trace.router[label], "discrete"),
            ("router_weights", trace.router_weights[label], "tensor"),
            ("router_scores", trace.router_scores[label], "tensor"),
            ("expert_load", trace.expert_load[label], "discrete"),
            ("moe_input", trace.moe_inputs[label], "tensor"),
        )
        for component, values, value_kind in groups:
            for layer, value in sorted(values.items()):
                parent_path = (
                    f"layers.{layer}.attention"
                    if component in {"indexer", "indexer_scores"}
                    else f"layers.{layer}.moe"
                    if component in {
                        "router",
                        "router_weights",
                        "router_scores",
                        "expert_load",
                        "moe_input",
                    }
                    else "layers"
                )
                logical_path = (
                    f"layers.{layer}"
                    if component == "decoder_block"
                    else f"{parent_path}.{component}"
                )
                self._capture_add(
                    key=f"{case.section_id}/checkpoint/{logical_path}",
                    section_id=case.section_id,
                    tensor=value,
                    scope="configured",
                    component=component,
                    layer=layer,
                    value_kind=value_kind,
                    module_path=logical_path,
                    parent_path=parent_path,
                    level=2 if component == "decoder_block" else 4,
                    node_kind=(
                        "discrete_checkpoint"
                        if value_kind == "discrete"
                        else "diagnostic_score"
                        if component in {"indexer_scores", "router_scores"}
                        else "activation_checkpoint"
                    ),
                    checkpoint=component not in {
                        "indexer_scores",
                        "router_scores",
                    },
                    compare=component not in {
                        "indexer_scores",
                        "router_scores",
                    },
                    positions_key=(
                        positions_key
                        if value_kind == "discrete" and value.ndim == 3
                        else ""
                    ),
                )

    def _capture_parameters_and_gradients(self, section_id: str) -> None:
        endpoint = self.capture_endpoint
        state = self._canonical_state(endpoint)
        for state_key, value in sorted(state.items()):
            layer_match = re.match(r"layers\.(\d+)", state_key)
            layer: int | str = (
                int(layer_match.group(1)) if layer_match else "global"
            )
            self._capture_add(
                key=f"{section_id}/parameter/{state_key}",
                section_id=section_id,
                tensor=value,
                scope="parameters",
                component="state_dict",
                layer=layer,
                module_path=state_key,
                parent_path=state_key.rpartition(".")[0],
                level=len(state_key.split(".")) - 1,
                node_kind="parameter",
                tags={"state_key": state_key},
            )
        gradients, missing = self._canonical_gradient(endpoint)
        for state_key, parameter_value in sorted(state.items()):
            layer_match = re.match(r"layers\.(\d+)", state_key)
            layer = int(layer_match.group(1)) if layer_match else "global"
            is_missing = state_key in missing or state_key not in gradients
            value = (
                torch.zeros_like(parameter_value)
                if is_missing
                else gradients[state_key]
            )
            self._capture_add(
                key=f"{section_id}/gradient/{state_key}",
                section_id=section_id,
                tensor=value,
                scope="gradient",
                component="gradient",
                layer=layer,
                module_path=f"{state_key}.grad",
                parent_path=state_key.rpartition(".")[0],
                level=len(state_key.split(".")) - 1,
                node_kind="gradient",
                tags={
                    "state_key": state_key,
                    "gradient_missing": "true" if is_missing else "false",
                },
            )
        self.capture_missing_gradients = sorted(missing)

    def _capture_local_moe_replay(
        self,
        case: ParityCase,
        endpoint_trace: EndpointTrace,
    ) -> None:
        endpoint = self.capture_endpoint
        label = endpoint.label
        for layer_index in self._selected_layers():
            layer = (
                self.capture_model.model.layers[layer_index]
                if endpoint.implementation == "hf"
                else self.capture_model.layers[str(layer_index)]
            )
            is_moe = (
                hasattr(layer.mlp, "gate")
                if endpoint.implementation == "hf"
                else getattr(layer, "moe_enabled", False)
            )
            if not is_moe:
                continue
            generated = endpoint_trace.moe_inputs[label].get(layer_index)
            if generated is None:
                raise RuntimeError(
                    f"missing captured MoE input for layer {layer_index}"
                )
            local_input = generated.float()
            with torch.no_grad():
                routed, weights, indices, expert_load = (
                    self._run_routed_experts_on_input(
                        endpoint, layer_index, local_input
                    )
                )
            parent_path = f"layers.{layer_index}.moe.routed_experts"
            for name, value, value_kind in (
                ("output", routed, "tensor"),
                ("weights", weights, "tensor"),
                ("indices", indices, "discrete"),
                ("expert_load", expert_load, "discrete"),
            ):
                self._capture_add(
                    key=(
                        f"{case.section_id}/local_replay/"
                        f"{parent_path}.{name}"
                    ),
                    section_id=case.section_id,
                    tensor=value,
                    scope="local_replay",
                    component=name,
                    layer=layer_index,
                    value_kind=value_kind,
                    module_path=f"{parent_path}.{name}",
                    parent_path=parent_path,
                    level=4,
                    node_kind=(
                        "discrete_checkpoint"
                        if value_kind == "discrete"
                        else "activation_checkpoint"
                    ),
                )

    def _capture_end_to_end_case(self, case: ParityCase) -> None:
        endpoint = self.capture_endpoint
        model = self.capture_model
        batch = self._capture_batch(case)
        layer_indices = self._selected_layers()
        model.zero_grad(set_to_none=True)
        with (
            EndpointTrace.install({endpoint: model}, layer_indices) as endpoint_trace,
            RecursiveModuleTrace.install(
                {endpoint: model}, layer_indices, include_model_modules=True
            ) as module_trace,
        ):
            logits, loss = _run_causal_lm_endpoint(model, endpoint, batch)
            loss.backward()
        self._capture_recursive_observations(
            section_id=case.section_id,
            trace=module_trace,
            include_gradients=True,
        )
        self._capture_endpoint_trace(case, endpoint_trace)
        self._capture_local_moe_replay(case, endpoint_trace)
        self._capture_add(
            key=f"{case.section_id}/output/logits",
            section_id=case.section_id,
            tensor=logits,
            scope="configured",
            component="logits",
            layer="all",
            module_path="lm_head",
            parent_path="model",
            level=1,
            node_kind="logits_checkpoint",
        )
        self._capture_add(
            key=f"{case.section_id}/output/loss",
            section_id=case.section_id,
            tensor=loss,
            scope="configured",
            component="loss",
            layer="all",
            module_path="loss",
            parent_path="model",
            level=1,
            node_kind="loss_checkpoint",
        )
        self._capture_parameters_and_gradients(case.section_id)

    def _run_prepare_fixture(self) -> None:
        torch.manual_seed(self.MODEL_SEED)
        titan_config = _titan_config(self.model_size)
        model = titan_config.build()
        model.init_states()
        model.float().eval()
        canonical_state = {
            name: value.detach()
            for name, value in model.state_dict().items()
        }
        batches = self._make_fixture_batches()
        fingerprints = {
            f"model/{name}": tensor_digest(value)
            for name, value in sorted(canonical_state.items())
        }
        for case in self.capture_cases:
            batch = batches[case.case_id]
            for field_name in (
                "hidden_states",
                "positions",
                "causal_mask",
                "tokens",
            ):
                value = getattr(batch, field_name)
                assert value is not None
                fingerprints[f"data/{case.case_id}/{field_name}"] = (
                    tensor_digest(value)
                )
        fixture_digest = json_digest(fingerprints)
        run_id = os.environ.get(
            "GLM5_PARITY_RUN_ID", Path(self.FIXTURE_PATH).name
        )
        environment: dict[str, Any] = runtime_metadata()
        environment.update(_git_metadata())
        environment["argv"] = repr(sys.argv)
        environment["fixture_role"] = "canonical_model_state_and_test_data"
        writer = ParityArtifactWriter(
            self.FIXTURE_PATH,
            suite="glm5.2-fixture",
            suite_version=GLM5_PARITY_SUITE_VERSION,
            endpoint=EndpointIdentity(
                implementation="titan-fixture",
                precision="fp32",
                device_type="cpu",
                device_name="cpu",
                run_id=run_id,
            ),
            test_plan=self.capture_plan,
            configuration=self._fixture_configuration(),
            fixture_digest=fixture_digest,
            environment=environment,
        )
        for state_key, value in sorted(canonical_state.items()):
            layer_match = re.match(r"layers\.(\d+)", state_key)
            layer: int | str = (
                int(layer_match.group(1)) if layer_match else "global"
            )
            writer.add(
                ObservationMetadata(
                    key=f"fixture/model/{state_key}",
                    section_id="fixture",
                    scope="fixture_model",
                    component="model_state",
                    layer=layer,
                    value_kind="fixture",
                    module_path=state_key,
                    parent_path=state_key.rpartition(".")[0],
                    level=len(state_key.split(".")) - 1,
                    node_kind="fixture_state",
                    checkpoint=True,
                    compare=False,
                    tags={"state_key": state_key},
                ),
                value,
            )
        for case in self.capture_cases:
            batch = batches[case.case_id]
            prefix = f"fixture/data/{case.case_id}"
            for field_name in (
                "hidden_states",
                "positions",
                "causal_mask",
                "tokens",
            ):
                value = getattr(batch, field_name)
                assert value is not None
                writer.add(
                    ObservationMetadata(
                        key=f"{prefix}/{field_name}",
                        section_id="fixture",
                        scope="fixture_data",
                        component=field_name,
                        layer="global",
                        value_kind="fixture",
                        compare=False,
                        tags={
                            "case_id": case.case_id,
                            "case_ordinal": str(case.ordinal),
                            "case_seed": str(case.seed),
                        },
                    ),
                    value,
                )
        del canonical_state, batches, model
        output = writer.write()
        print(f"GLM-5.2 parity fixture: {output}")
        print(f"fixture digest: {fixture_digest}")

    def _run_capture_suite(self) -> None:
        self.capture_writer = self._new_capture_writer()
        artifact_parent = Path(self.ARTIFACT_PATH).parent
        artifact_parent.mkdir(parents=True, exist_ok=True)
        log_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="glm5-parity-",
            suffix=".log",
            dir=artifact_parent,
            delete=False,
        )
        log_path = Path(log_file.name)
        failure: Exception | None = None
        try:
            with (
                log_file,
                redirect_stdout(_TeeStream(sys.stdout, log_file)),
                redirect_stderr(_TeeStream(sys.stderr, log_file)),
            ):
                print(
                    f"Starting GLM-5 parity capture: "
                    f"{self.capture_endpoint.label}@{self.device_type}"
                )
                try:
                    for case in self.capture_cases:
                        self._active_capture_case = case
                        self._active_data_seed = case.seed
                        print(
                            f"capture case {case.ordinal}: {case.case_id} "
                            f"seed={case.seed}"
                        )
                        self._capture_fixture_batch(case)
                        if case.section_id == "end-to-end":
                            self._capture_end_to_end_case(case)
                        else:
                            self._capture_component_case(case)
                except Exception as error:
                    failure = error
                    diagnostic = traceback.format_exc()
                    self.capture_writer.mark_failed(diagnostic)
                    print(diagnostic, file=sys.stderr)
            self.capture_writer.add_attachment(
                log_path, name="runtime.log"
            )
            try:
                properties = self.device_module.get_device_properties(
                    self.device
                )
                self.capture_writer.environment["peak_memory_bytes"] = str(
                    self.device_module.max_memory_allocated(self.device)
                )
                self.capture_writer.environment["total_memory_bytes"] = str(
                    properties.total_memory
                )
            except (AttributeError, RuntimeError) as error:
                self.capture_writer.environment["memory_metadata_error"] = repr(
                    error
                )
            output = self.capture_writer.write()
        except Exception:
            print(f"GLM-5 parity runtime log retained at: {log_path}")
            raise
        else:
            log_path.unlink()
        print(
            f"GLM-5 parity artifact: {output} "
            f"({self.capture_endpoint.label}@{self.device_type})"
        )
        if failure is not None:
            raise failure

    @staticmethod
    def _artifact_policy(
        actual: ParityArtifactReader,
        expected: ParityArtifactReader,
    ) -> PrecisionPolicy:
        if (
            actual.endpoint.precision == "fp32"
            and expected.endpoint.precision == "fp32"
        ):
            return FP32
        return BF16

    @staticmethod
    def _artifact_observation_is_comparable(
        key: str,
        metadata: ObservationMetadata,
    ) -> bool:
        if not metadata.compare:
            return False
        if metadata.scope not in {"trace", "activation_gradient"}:
            return True
        match = re.search(
            r"/(?:activation|activation_gradient)/"
            r"layers\.\d+\.moe\.router(?:\[\d+\])?$",
            key,
        )
        return match is None

    @staticmethod
    def _replay_artifact_indexer_scores(
        reader: ParityArtifactReader,
        metadata: ObservationMetadata,
    ) -> tuple[torch.Tensor | None, str]:
        if not isinstance(metadata.layer, int):
            return None, "indexer layer is unavailable"
        captured_score_key = (
            f"{metadata.section_id}/checkpoint/layers.{metadata.layer}."
            "attention.indexer_scores"
        )
        if captured_score_key in reader.observation_keys:
            return reader.tensor(captured_score_key), ""
        prefix = (
            f"{metadata.section_id}/activation/layers.{metadata.layer}."
            "attention.indexer"
        )
        keys = {
            "q_projection": f"{prefix}.wq_b",
            "q_rotated": f"{prefix}.rope[0]",
            "k_normalized": f"{prefix}.k_norm",
            "k_rotated": f"{prefix}.rope[1]",
            "projected_weights": f"{prefix}.weights_proj",
        }
        case_id = metadata.tags.get("case_id", metadata.section_id)
        mask_key = f"fixture/data/{case_id}/causal_mask"
        missing = [
            key for key in (*keys.values(), mask_key)
            if key not in reader.observation_keys
        ]
        if missing:
            return None, "missing replay tensors: " + ", ".join(missing)
        model_size = reader.manifest["test_plan"]["model_size"]
        try:
            scores = _replay_titan_indexer_scores(
                **{
                    name: reader.tensor(key)
                    for name, key in keys.items()
                },
                attention_mask=reader.tensor(mask_key),
                n_heads=int(model_size["index_heads"]),
                head_dim=int(model_size["index_head_dim"]),
                rope_head_dim=int(model_size["qk_rope_head_dim"]),
            )
        except (KeyError, RuntimeError, ValueError) as error:
            return None, f"indexer score replay failed: {error}"
        return scores, ""

    @classmethod
    def _offline_indexer_mismatch_detail(
        cls,
        *,
        actual_reader: ParityArtifactReader,
        expected_reader: ParityArtifactReader,
        actual_metadata: ObservationMetadata,
        expected_metadata: ObservationMetadata,
        actual_topk: torch.Tensor,
        expected_topk: torch.Tensor,
        positions: torch.Tensor | None,
    ) -> str:
        if positions is None or actual_topk.ndim != 3:
            return "score replay unavailable: indexer positions are missing"
        coordinates = _selection_mismatch_coordinates(
            actual_topk, expected_topk, positions
        )
        actual_scores, actual_error = cls._replay_artifact_indexer_scores(
            actual_reader, actual_metadata
        )
        expected_scores, expected_error = cls._replay_artifact_indexer_scores(
            expected_reader, expected_metadata
        )
        if actual_scores is None or expected_scores is None:
            errors = "; ".join(
                value for value in (actual_error, expected_error) if value
            )
            return f"score replay unavailable: {errors}"
        topk = actual_topk.shape[-1]
        entries = []
        for batch_index, position in coordinates:
            actual_ids = {
                int(value) for value in actual_topk[batch_index, position]
            }
            expected_ids = {
                int(value) for value in expected_topk[batch_index, position]
            }
            candidates = sorted(actual_ids.symmetric_difference(expected_ids))
            candidate_scores = ", ".join(
                f"id{candidate}:actual="
                f"{float(actual_scores[batch_index, position, candidate]):.8g},"
                f"expected="
                f"{float(expected_scores[batch_index, position, candidate]):.8g}"
                for candidate in candidates
            )
            actual_sorted = actual_scores[batch_index, position].sort(
                descending=True
            ).values
            expected_sorted = expected_scores[batch_index, position].sort(
                descending=True
            ).values
            actual_margin = (
                float(actual_sorted[topk - 1] - actual_sorted[topk])
                if actual_sorted.numel() > topk else float("nan")
            )
            expected_margin = (
                float(expected_sorted[topk - 1] - expected_sorted[topk])
                if expected_sorted.numel() > topk else float("nan")
            )
            actual_replay = actual_scores[batch_index, position].topk(
                topk
            ).indices
            expected_replay = expected_scores[batch_index, position].topk(
                topk
            ).indices
            entries.append(
                f"score_replay b{batch_index},p{position} "
                f"candidates=[{candidate_scores}] "
                f"cutoff_margin(actual={actual_margin:.8g},"
                f"expected={expected_margin:.8g}) "
                f"topk(actual={actual_replay.tolist()},"
                f"expected={expected_replay.tolist()})"
            )
        return "; ".join(entries)

    @staticmethod
    def _artifact_router_scores(
        reader: ParityArtifactReader,
        metadata: ObservationMetadata,
    ) -> tuple[torch.Tensor | None, str]:
        if not isinstance(metadata.layer, int):
            return None, "router layer is unavailable"
        score_key = (
            f"{metadata.section_id}/checkpoint/layers.{metadata.layer}."
            "moe.router_scores"
        )
        if score_key in reader.observation_keys:
            return reader.tensor(score_key), ""
        gate_key = (
            f"{metadata.section_id}/activation/layers.{metadata.layer}."
            "moe.router.gate"
        )
        if gate_key in reader.observation_keys:
            return torch.sigmoid(reader.tensor(gate_key).float()), ""
        return None, f"missing router score tensor: {score_key}"

    @classmethod
    def _offline_router_mismatch_detail(
        cls,
        *,
        actual_reader: ParityArtifactReader,
        expected_reader: ParityArtifactReader,
        actual_metadata: ObservationMetadata,
        expected_metadata: ObservationMetadata,
        actual_topk: torch.Tensor,
        expected_topk: torch.Tensor,
    ) -> str:
        coordinates = _ordered_mismatch_coordinates(
            actual_topk, expected_topk
        )
        actual_scores, actual_error = cls._artifact_router_scores(
            actual_reader, actual_metadata
        )
        expected_scores, expected_error = cls._artifact_router_scores(
            expected_reader, expected_metadata
        )
        if actual_scores is None or expected_scores is None:
            errors = "; ".join(
                value for value in (actual_error, expected_error) if value
            )
            return f"router score trace unavailable: {errors}"
        topk = actual_topk.shape[-1]
        entries = []
        for batch_index, position in coordinates:
            actual_ids = {
                int(value) for value in actual_topk[batch_index, position]
            }
            expected_ids = {
                int(value) for value in expected_topk[batch_index, position]
            }
            candidates = sorted(actual_ids.symmetric_difference(expected_ids))
            candidate_scores = ", ".join(
                f"id{candidate}:actual="
                f"{float(actual_scores[batch_index, position, candidate]):.8g},"
                f"expected="
                f"{float(expected_scores[batch_index, position, candidate]):.8g}"
                for candidate in candidates
            )
            actual_sorted = actual_scores[batch_index, position].sort(
                descending=True
            ).values
            expected_sorted = expected_scores[batch_index, position].sort(
                descending=True
            ).values
            actual_margin = (
                float(actual_sorted[topk - 1] - actual_sorted[topk])
                if actual_sorted.numel() > topk else float("nan")
            )
            expected_margin = (
                float(expected_sorted[topk - 1] - expected_sorted[topk])
                if expected_sorted.numel() > topk else float("nan")
            )
            entries.append(
                f"router_score b{batch_index},p{position} "
                f"candidates=[{candidate_scores}] "
                f"cutoff_margin(actual={actual_margin:.8g},"
                f"expected={expected_margin:.8g})"
            )
        return "; ".join(entries)

    def _compare_artifact_observation(
        self,
        *,
        key: str,
        actual: ParityArtifactReader,
        expected: ParityArtifactReader,
        recorder: ParityRecorder,
        policy: PrecisionPolicy,
    ) -> None:
        actual_present = key in actual.observation_keys
        expected_present = key in expected.observation_keys
        source = actual if actual_present else expected
        metadata = source.metadata(key)
        actual_path = (
            actual.metadata(key).module_path if actual_present else metadata.module_path
        )
        expected_path = (
            expected.metadata(key).module_path
            if expected_present
            else metadata.module_path
        )
        module_path = (
            f"{actual.endpoint.label}:{actual_path} <-> "
            f"{expected.endpoint.label}:{expected_path}"
        )
        if not actual_present or not expected_present:
            recorder.missing(
                scope=metadata.scope,
                component=metadata.component,
                layer=metadata.layer,
                module_path=module_path,
                parent_path=metadata.parent_path,
                level=metadata.level,
                node_kind=metadata.node_kind,
                checkpoint=metadata.checkpoint,
                detail=(
                    f"offline observation missing actual={actual_present}, "
                    f"expected={expected_present}; key={key}"
                ),
            )
            return
        actual_metadata = actual.metadata(key)
        expected_metadata = expected.metadata(key)
        if actual_metadata.value_kind != expected_metadata.value_kind:
            recorder.missing(
                scope=metadata.scope,
                component=metadata.component,
                layer=metadata.layer,
                module_path=module_path,
                parent_path=metadata.parent_path,
                level=metadata.level,
                node_kind=metadata.node_kind,
                checkpoint=True,
                detail=(
                    "offline observation kind mismatch actual=True, expected=True; "
                    f"{actual_metadata.value_kind} != {expected_metadata.value_kind}"
                ),
            )
            return
        actual_gradient_missing = actual_metadata.tags.get("gradient_missing")
        expected_gradient_missing = expected_metadata.tags.get("gradient_missing")
        if actual_gradient_missing != expected_gradient_missing:
            recorder.missing(
                scope=metadata.scope,
                component=metadata.component,
                layer=metadata.layer,
                module_path=module_path,
                parent_path=metadata.parent_path,
                level=metadata.level,
                node_kind=metadata.node_kind,
                checkpoint=True,
                detail=(
                    "gradient presence differs actual=True, expected=True; "
                    f"missing={actual_gradient_missing} != "
                    f"{expected_gradient_missing}"
                ),
            )
            return
        actual_tensor = actual.tensor(key)
        expected_tensor = expected.tensor(key)
        if (
            "moe.routed_experts" in key
            and actual_tensor.shape != expected_tensor.shape
            and actual_tensor.numel() == expected_tensor.numel()
        ):
            if actual_tensor.ndim > expected_tensor.ndim:
                expected_tensor = expected_tensor.reshape_as(actual_tensor)
            else:
                actual_tensor = actual_tensor.reshape_as(expected_tensor)
        actual_dtype = actual_metadata.dtype_flow or (
            f"out={str(actual_tensor.dtype).removeprefix('torch.')}"
        )
        expected_dtype = expected_metadata.dtype_flow or (
            f"out={str(expected_tensor.dtype).removeprefix('torch.')}"
        )
        if metadata.value_kind == "discrete":
            positions = None
            positions_key = (
                actual_metadata.positions_key or expected_metadata.positions_key
            )
            if positions_key:
                if positions_key in actual.observation_keys:
                    positions = actual.tensor(positions_key)
                elif positions_key in expected.observation_keys:
                    positions = expected.tensor(positions_key)
            result = recorder.discrete(
                scope=metadata.scope,
                component=metadata.component,
                layer=metadata.layer,
                actual=actual_tensor,
                expected=expected_tensor,
                positions=positions,
                module_path=module_path,
                parent_path=metadata.parent_path,
                level=metadata.level,
                node_kind=metadata.node_kind,
                checkpoint=metadata.checkpoint,
                actual_dtype=actual_dtype,
                expected_dtype=expected_dtype,
            )
            if not result.passed and metadata.component == "indexer":
                replay_detail = self._offline_indexer_mismatch_detail(
                    actual_reader=actual,
                    expected_reader=expected,
                    actual_metadata=actual_metadata,
                    expected_metadata=expected_metadata,
                    actual_topk=actual_tensor,
                    expected_topk=expected_tensor,
                    positions=positions,
                )
                if replay_detail:
                    result.detail = f"{result.detail}; {replay_detail}"
            elif not result.passed and metadata.component in {
                "router",
                "router_indices",
            }:
                score_detail = self._offline_router_mismatch_detail(
                    actual_reader=actual,
                    expected_reader=expected,
                    actual_metadata=actual_metadata,
                    expected_metadata=expected_metadata,
                    actual_topk=actual_tensor,
                    expected_topk=expected_tensor,
                )
                if score_detail:
                    result.detail = f"{result.detail}; {score_detail}"
            return
        state_key = actual_metadata.tags.get("state_key", "")
        routed_suffixes = (
            "moe.routed_experts.inner_experts.w1_EFD",
            "moe.routed_experts.inner_experts.w3_EFD",
            "moe.routed_experts.inner_experts.w2_EDF",
        )
        if (
            state_key.endswith(routed_suffixes)
            and actual_tensor.ndim >= 1
            and actual_tensor.shape == expected_tensor.shape
        ):
            layer_match = re.match(r"layers\.(\d+)", state_key)
            layer = int(layer_match.group(1)) if layer_match else metadata.layer
            for expert in range(actual_tensor.shape[0]):
                expert_path = f"layers.{layer}.moe.routed_experts.{expert}"
                recorder.tensor(
                    scope=metadata.scope,
                    component=(
                        "expert_parameter"
                        if metadata.scope == "parameters"
                        else "expert_gradient"
                    ),
                    layer=layer,
                    actual=actual_tensor[expert],
                    expected=expected_tensor[expert],
                    rtol=policy.rtol,
                    atol=policy.atol,
                    module_path=f"{module_path}[expert={expert}]",
                    parent_path=expert_path,
                    level=len(expert_path.split(".")) - 1,
                    node_kind=metadata.node_kind,
                    checkpoint=metadata.checkpoint,
                    actual_dtype=actual_dtype,
                    expected_dtype=expected_dtype,
                )
            return
        use_suite_tolerance = metadata.component == "q_residual"
        recorder.tensor(
            scope=metadata.scope,
            component=metadata.component,
            layer=metadata.layer,
            actual=actual_tensor,
            expected=expected_tensor,
            rtol=(
                policy.rtol
                if use_suite_tolerance
                else float(
                    actual_metadata.tags.get(
                        "rtol", expected_metadata.tags.get("rtol", policy.rtol)
                    )
                )
            ),
            atol=(
                policy.atol
                if use_suite_tolerance
                else float(
                    actual_metadata.tags.get(
                        "atol", expected_metadata.tags.get("atol", policy.atol)
                    )
                )
            ),
            module_path=module_path,
            parent_path=metadata.parent_path,
            level=metadata.level,
            node_kind=metadata.node_kind,
            checkpoint=metadata.checkpoint,
            actual_dtype=actual_dtype,
            expected_dtype=expected_dtype,
        )

    def _offline_report_path(
        self,
        actual: ParityArtifactReader,
        expected: ParityArtifactReader,
    ) -> str:
        explicit = os.environ.get("GLM5_PARITY_REPORT")
        if explicit:
            return explicit
        report_directory = Path(
            os.environ.get("GLM5_PARITY_REPORT_DIR", ".")
        )
        report_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        raw = (
            f"glm5_{actual.endpoint.implementation}-{actual.endpoint.precision}-"
            f"{actual.endpoint.device_type}_vs_"
            f"{expected.endpoint.implementation}-{expected.endpoint.precision}-"
            f"{expected.endpoint.device_type}_{timestamp}.html"
        )
        return str(report_directory / re.sub(r"[^A-Za-z0-9_.-]+", "-", raw))

    def _run_offline_compare(self) -> None:
        if not self.COMPARE_ACTUAL_ARTIFACT or not self.COMPARE_EXPECTED_ARTIFACT:
            raise ValueError(
                "GLM5_PARITY_MODE=compare requires "
                "GLM5_PARITY_ACTUAL_ARTIFACT and GLM5_PARITY_EXPECTED_ARTIFACT"
            )
        actual = ParityArtifactReader(self.COMPARE_ACTUAL_ARTIFACT)
        expected = ParityArtifactReader(self.COMPARE_EXPECTED_ARTIFACT)
        actual.validate_compatible(expected)
        fixture_keys = {
            key
            for reader in (actual, expected)
            for key in reader.observation_keys
            if reader.metadata(key).value_kind == "fixture"
        }
        fixture_mismatches = []
        for key in sorted(fixture_keys):
            if (
                key not in actual.observation_keys
                or key not in expected.observation_keys
            ):
                fixture_mismatches.append(f"{key}: missing on one side")
                continue
            if actual.entry(key)["sha256"] != expected.entry(key)["sha256"]:
                fixture_mismatches.append(f"{key}: tensor checksum differs")
        if fixture_mismatches:
            raise ParityArtifactError(
                "parity artifacts used different fixture tensors:\n"
                + "\n".join(fixture_mismatches[:20])
            )
        policy = self._artifact_policy(actual, expected)
        title = (
            f"GLM-5 offline parity: {actual.endpoint.label} "
            f"vs {expected.endpoint.label}"
        )
        report = ParitySuiteReport(title)
        comparable_keys = {
            key
            for reader in (actual, expected)
            for key in reader.observation_keys
            if self._artifact_observation_is_comparable(
                key, reader.metadata(key)
            )
        }
        sections = {
            reader.metadata(key).section_id
            for reader in (actual, expected)
            for key in reader.observation_keys
            if self._artifact_observation_is_comparable(
                key, reader.metadata(key)
            )
        }
        planned_sections = [
            case["section_id"] for case in actual.manifest["test_plan"]["cases"]
        ]
        section_order = list(dict.fromkeys(planned_sections))
        section_order.extend(sorted(sections - set(section_order)))
        for section_id in section_order:
            recorder = ParityRecorder(
                policy,
                precision_label=(
                    f"{actual.endpoint.label} vs {expected.endpoint.label}"
                ),
                title=section_id,
            )
            for key in sorted(comparable_keys):
                source = actual if key in actual.observation_keys else expected
                if source.metadata(key).section_id != section_id:
                    continue
                self._compare_artifact_observation(
                    key=key,
                    actual=actual,
                    expected=expected,
                    recorder=recorder,
                    policy=policy,
                )
            report.add(section_id, recorder)
        report.configuration = [
            ("artifact", "actual", str(actual.path)),
            ("artifact", "expected", str(expected.path)),
            ("identity", "actual", actual.endpoint.label),
            ("identity", "expected", expected.endpoint.label),
            (
                "integrity",
                "test_plan_digest",
                actual.manifest["test_plan_digest"],
            ),
            ("integrity", "fixture_digest", actual.manifest["fixture_digest"]),
            (
                "source",
                "actual_test_git_commit",
                str(actual.manifest["environment"].get("git_commit", "unknown")),
            ),
            (
                "source",
                "expected_test_git_commit",
                str(expected.manifest["environment"].get("git_commit", "unknown")),
            ),
            (
                "source",
                "actual_torchtitan_git_commit",
                str(
                    actual.manifest["environment"].get(
                        "torchtitan_git_commit", "unknown"
                    )
                ),
            ),
            (
                "source",
                "expected_torchtitan_git_commit",
                str(
                    expected.manifest["environment"].get(
                        "torchtitan_git_commit", "unknown"
                    )
                ),
            ),
        ]
        for side, reader in (("actual", actual), ("expected", expected)):
            report.configuration.extend(
                (f"{side}_environment", str(name), str(value))
                for name, value in sorted(reader.manifest["environment"].items())
            )
            report.configuration.extend(
                (f"{side}_configuration", str(name), str(value))
                for name, value in sorted(reader.manifest["configuration"].items())
            )
        output = self._offline_report_path(actual, expected)
        report.write(output)
        print(f"GLM-5 offline parity report: {output}")
        actual.close()
        expected.close()
        if report.failed:
            raise AssertionError(report.failure_message())

    def _report_path(self, _section_id: str = "suite") -> str:
        explicit = os.environ.get("GLM5_PARITY_REPORT")
        if explicit:
            return explicit
        timestamp = getattr(self, "_report_timestamp", None)
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            self._report_timestamp = timestamp
        raw_name = (
            f"glm5_{self.actual_endpoint.label}_vs_"
            f"{self.expected_endpoint.label}_{timestamp}.html"
        )
        filename = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw_name)
        report_directory = Path(
            os.environ.get("GLM5_PARITY_REPORT_DIR", ".")
        )
        report_directory.mkdir(parents=True, exist_ok=True)
        return str(report_directory / filename)

    def _effective_routed_expert_compute(
        self, endpoint: ModelEndpoint
    ) -> str:
        if endpoint.implementation == "titan":
            return (
                "fp32_eager"
                if self.TITAN_ROUTED_EXPERT_COMPUTE == "fp32"
                else "bf16_grouped_mm"
            )
        if self.HF_ROUTED_EXPERT_COMPUTE == "grouped_mm":
            return "bf16_grouped_mm"
        if self.HF_ROUTED_EXPERT_COMPUTE == "bf16":
            return "bf16_eager"
        return endpoint.precision.name

    def _report_configuration(
        self,
        spec: ComparisonSpec,
        *,
        report_path: str,
        peak_bytes: int,
        total_bytes: int,
    ) -> list[tuple[str, str, str]]:
        """Return every effective suite setting for the HTML report."""
        values: list[tuple[str, str, str]] = [
            (
                "scenario",
                "id",
                os.environ.get("GLM5_PARITY_SCENARIO_ID", "direct"),
            ),
            (
                "scenario",
                "configuration_digest",
                os.environ.get(
                    "GLM5_PARITY_SCENARIO_CONFIG_DIGEST", "unavailable"
                ),
            ),
            ("comparison", "model", "glm5"),
            ("comparison", "actual_endpoint", spec.actual.label),
            ("comparison", "expected_endpoint", spec.expected.label),
            ("comparison", "rtol", f"{spec.rtol:.8g}"),
            ("comparison", "atol", f"{spec.atol:.8g}"),
            ("selection", "layers", self.LAYER_INDICES),
            ("selection", "components", self.RUN_COMPONENTS),
            ("data", "case", self.DATA_CASE),
            ("data", "base_seed", str(self.DATA_SEED)),
            ("data", "per_test_batches", "independent_deterministic"),
            ("data", "batch_size", str(self.BATCH_SIZE)),
            ("data", "sequence_length", str(self.SEQUENCE_LENGTH)),
            ("execution", "component_execution", self.COMPONENT_EXECUTION),
            ("execution", "model_seed", str(self.MODEL_SEED)),
            (
                "execution",
                "actual_routed_expert_compute",
                self._effective_routed_expert_compute(spec.actual),
            ),
            (
                "execution",
                "expected_routed_expert_compute",
                self._effective_routed_expert_compute(spec.expected),
            ),
            (
                "execution",
                "hf_routed_expert_override",
                self.HF_ROUTED_EXPERT_COMPUTE,
            ),
            (
                "execution",
                "titan_routed_expert_override",
                self.TITAN_ROUTED_EXPERT_COMPUTE,
            ),
            (
                "report",
                "explosion_ratio",
                os.environ.get("GLM5_PARITY_EXPLOSION_RATIO", "8.0"),
            ),
            ("report", "path", report_path),
            ("hardware", "device", str(self.device)),
            (
                "hardware",
                "device_type",
                self.device_type,
            ),
            (
                "hardware",
                "device_name",
                self.device_module.get_device_name(self.device),
            ),
            ("hardware", "model_parameters", f"{self.num_model_parameters:,}"),
            ("hardware", "peak_memory_gib", f"{peak_bytes / 2**30:.4f}"),
            ("hardware", "total_memory_gib", f"{total_bytes / 2**30:.4f}"),
            (
                "hardware",
                "peak_memory_percent",
                f"{100.0 * peak_bytes / total_bytes:.2f}",
            ),
        ]
        values.extend(
            ("model_size", name, str(value))
            for name, value in asdict(self.model_size).items()
        )
        return values

    def _configured_spec(self) -> ComparisonSpec:
        actual, expected = self.actual_endpoint, self.expected_endpoint
        is_fp32 = actual.precision is FP32 and expected.precision is FP32
        tolerance = (1e-4, 1e-5) if is_fp32 else (5e-2, 5e-2)
        return ComparisonSpec(
            actual=actual,
            expected=expected,
            scope="configured",
            component="model",
            rtol=tolerance[0],
            atol=tolerance[1],
        )

    def _configured_report_label(self) -> str:
        def endpoint_label(endpoint: ModelEndpoint) -> str:
            expert_compute = self._effective_routed_expert_compute(endpoint)
            label = endpoint.label
            if expert_compute != endpoint.precision.name:
                label += f"(routed_experts={expert_compute})"
            return label

        return (
            f"{endpoint_label(self.actual_endpoint)} vs "
            f"{endpoint_label(self.expected_endpoint)}"
        )

    def _component_enabled(self, name: str) -> bool:
        selected = {
            item.strip().lower()
            for item in self.RUN_COMPONENTS.split(",")
            if item.strip()
        }
        return not selected or "all" in selected or name in selected

    def _selected_layers(self) -> list[int]:
        if self.LAYER_INDICES.strip().lower() == "all":
            if self.RUN_MODE == "capture":
                return list(range(self.model_size.num_layers))
            return list(range(len(self.pair.hf.model.layers)))
        return [
            int(value.strip())
            for value in self.LAYER_INDICES.split(",")
            if value.strip()
        ]

    @staticmethod
    def _endpoint_path(endpoint: ModelEndpoint, layer: int, component: str) -> str:
        base = "model.layers" if endpoint.implementation == "hf" else "layers"
        suffix = {
            "decoder_block": "",
            "indexer": ".self_attn.indexer"
            if endpoint.implementation == "hf"
            else ".attention.indexer",
            "router": ".mlp.gate"
            if endpoint.implementation == "hf"
            else ".moe.router",
        }[component]
        return f"{endpoint.label}:{base}.{layer}{suffix}"

    def _logical_activation_path(
        self, endpoint: ModelEndpoint, endpoint_path: str
    ) -> str:
        """Map HF/TorchTitan module names to one report hierarchy."""
        if endpoint.implementation == "titan":
            return endpoint_path
        path = (
            endpoint_path.replace("model.layers.", "layers.")
            .replace("model.embed_tokens", "tok_embeddings")
            .replace("model.norm", "norm")
        )
        layer_match = re.match(r"layers\.(\d+)", path)
        layer_is_moe = False
        if layer_match:
            layer = self._model(endpoint).model.layers[int(layer_match.group(1))]
            layer_is_moe = hasattr(layer.mlp, "gate")
        replacements = (
            (".self_attn.indexer", ".attention.indexer"),
            (".self_attn", ".attention"),
            (".input_layernorm", ".attention_norm"),
            (".post_attention_layernorm", ".ffn_norm"),
            (".q_a_proj", ".wq_a"),
            (".q_a_layernorm", ".q_norm"),
            (".q_b_proj", ".wq_b"),
            (".kv_a_proj_with_mqa", ".wkv_a"),
            (".kv_a_layernorm", ".kv_norm"),
            (".kv_b_proj", ".wkv_b"),
            (".o_proj", ".wo"),
            (".mlp.shared_experts.gate_proj", ".moe.shared_experts.w1"),
            (".mlp.shared_experts.up_proj", ".moe.shared_experts.w3"),
            (".mlp.shared_experts.down_proj", ".moe.shared_experts.w2"),
            (".mlp.experts.gate_up_proj", ".moe.routed_experts"),
            (".mlp.experts.down_proj", ".moe.routed_experts"),
            (".mlp.experts", ".moe.routed_experts"),
            (".mlp.gate", ".moe.router"),
            (".mlp.gate_proj", ".feed_forward.w1"),
            (".mlp.up_proj", ".feed_forward.w3"),
            (".mlp.down_proj", ".feed_forward.w2"),
        )
        for source, target in replacements:
            path = path.replace(source, target)
        if ".mlp" in path:
            path = path.replace(".mlp", ".moe" if layer_is_moe else ".feed_forward")
        return path

    def _record_recursive_trace(
        self,
        spec: ComparisonSpec,
        trace: RecursiveModuleTrace,
        recorder: ParityRecorder,
        *,
        layer_indices: list[int] | None = None,
        component_filter: str | None = None,
        composition_checkpoints: bool = True,
        skip_layer_roots: bool = True,
        skip_filter_roots: bool = False,
    ) -> None:
        """Add every captured module activation to the hierarchical report."""
        selected_layers = (
            self._selected_layers() if layer_indices is None else layer_indices
        )
        normalized_filter = (
            component_filter.strip().strip(".")
            if component_filter is not None
            else None
        )
        if component_filter is None:
            filter_paths = set()
        elif normalized_filter in TOP_LEVEL_COMPONENTS:
            filter_paths = {normalized_filter}
        else:
            filter_paths = {
                (
                    f"layers.{layer}.{normalized_filter}"
                    if normalized_filter
                    else f"layers.{layer}"
                )
                for layer in selected_layers
            }
        leaf_filter = bool(
            normalized_filter
            and normalized_filter not in TOP_LEVEL_COMPONENTS
            and "." not in normalized_filter
        )
        logical: dict[str, dict[str, tuple[str, torch.Tensor, str]]] = {}
        for endpoint in (spec.actual, spec.expected):
            label = endpoint.label
            for endpoint_path, value in trace.activations[label].items():
                logical_path = self._logical_activation_path(endpoint, endpoint_path)
                logical.setdefault(logical_path, {})[label] = (
                    endpoint_path,
                    value,
                    trace.dtype_flows[label].get(endpoint_path, ""),
                )

        # HF composition modules may return tuples while TorchTitan returns a
        # tensor.  Compare the first tensor leaf under the common module path.
        normalized_logical: dict[
            str, dict[str, tuple[str, torch.Tensor, str]]
        ] = {}
        for logical_path, values in logical.items():
            base_path = logical_path.split("[", 1)[0]
            target_path = (
                base_path
                if re.fullmatch(r"layers\.\d+", base_path)
                or base_path.endswith((".attention", ".moe", ".feed_forward"))
                else logical_path
            )
            target_values = normalized_logical.setdefault(target_path, {})
            for label, value in values.items():
                target_values.setdefault(label, value)
        logical = normalized_logical

        explicit_paths = (
            {f"layers.{layer}" for layer in selected_layers}
            if skip_layer_roots
            else set()
        )
        if normalized_filter is None:
            explicit_paths.update(
                path
                for layer in selected_layers
                for path in (
                    f"layers.{layer}.attention.indexer",
                    f"layers.{layer}.moe.router",
                )
            )
        for logical_path in sorted(logical):
            base_logical = logical_path.split("[", 1)[0]
            if base_logical in explicit_paths and base_logical not in filter_paths:
                continue
            if skip_filter_roots and base_logical in filter_paths:
                continue
            selected_match = bool(
                filter_paths
                and any(
                    base_logical == path or base_logical.startswith(f"{path}.")
                    for path in filter_paths
                )
            )
            if leaf_filter:
                selected_match = selected_match or base_logical.endswith(
                    f".{normalized_filter}"
                )
            if filter_paths and not selected_match:
                continue
            values = logical[logical_path]
            layer_match = re.match(r"layers\.(\d+)", base_logical)
            layer: int | str = (
                int(layer_match.group(1)) if layer_match else "global"
            )
            parent_path = base_logical.rpartition(".")[0]
            level = len(logical_path.split(".")) - 1
            component = logical_path.rsplit(".", 1)[-1]
            checkpoint = (
                composition_checkpoints
                and base_logical.endswith((".attention", ".moe", ".feed_forward"))
            ) or base_logical in filter_paths or (
                leaf_filter and base_logical.endswith(f".{normalized_filter}")
            )
            node_kind = "composition_checkpoint" if checkpoint else "activation"
            actual_value = values.get(spec.actual.label)
            expected_value = values.get(spec.expected.label)
            module_path = (
                f"{spec.actual.label}:{actual_value[0] if actual_value else logical_path}"
                f" <-> {spec.expected.label}:"
                f"{expected_value[0] if expected_value else logical_path}"
            )
            if (
                actual_value is not None
                and spec.actual.label == spec.expected.label
            ):
                expected_value = actual_value
            if actual_value is None or expected_value is None:
                recorder.missing(
                    scope="trace",
                    component=component,
                    layer=layer,
                    module_path=module_path,
                    parent_path=parent_path,
                    level=level,
                    node_kind=node_kind,
                    checkpoint=checkpoint,
                    detail=(
                        f"trace-only missing actual={actual_value is not None}, "
                        f"expected={expected_value is not None}"
                    ),
                )
                continue
            actual_tensor = actual_value[1]
            expected_tensor = expected_value[1]
            if (
                base_logical.endswith(".moe.routed_experts")
                and actual_tensor.shape != expected_tensor.shape
                and actual_tensor.numel() == expected_tensor.numel()
            ):
                # HF experts flatten B/L while TorchTitan routed experts keep
                # [B, L, D].  They represent the same token-major tensor.
                if actual_tensor.ndim > expected_tensor.ndim:
                    expected_tensor = expected_tensor.reshape_as(actual_tensor)
                else:
                    actual_tensor = actual_tensor.reshape_as(expected_tensor)
            if base_logical.endswith(".attention.indexer"):
                recorder.discrete(
                    scope="trace",
                    component=component,
                    layer=layer,
                    actual=actual_tensor,
                    expected=expected_tensor,
                    positions=self._batch_for(spec.actual).positions,
                    module_path=module_path,
                    parent_path=parent_path,
                    level=level,
                    node_kind="discrete_checkpoint",
                    checkpoint=checkpoint,
                    actual_dtype=actual_value[2],
                    expected_dtype=expected_value[2],
                )
                continue
            if actual_tensor.dtype in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }:
                recorder.discrete(
                    scope="trace",
                    component=component,
                    layer=layer,
                    actual=actual_tensor,
                    expected=expected_tensor,
                    positions=(
                        self._batch_for(spec.actual).positions
                        if actual_tensor.ndim == 3
                        else None
                    ),
                    module_path=module_path,
                    parent_path=parent_path,
                    level=level,
                    node_kind="discrete_checkpoint",
                    checkpoint=checkpoint,
                    actual_dtype=actual_value[2],
                    expected_dtype=expected_value[2],
                )
                continue
            recorder.tensor(
                scope="trace",
                component=component,
                layer=layer,
                actual=actual_tensor,
                expected=expected_tensor,
                rtol=spec.rtol,
                atol=spec.atol,
                module_path=module_path,
                parent_path=parent_path,
                level=level,
                node_kind=node_kind,
                checkpoint=checkpoint,
                actual_dtype=actual_value[2],
                expected_dtype=expected_value[2],
            )

    def _record_recursive_gradient_trace(
        self,
        spec: ComparisonSpec,
        trace: RecursiveModuleTrace,
        recorder: ParityRecorder,
        *,
        layer_indices: list[int],
        component_filter: str | None = None,
    ) -> None:
        """Add gradients of traced module outputs to the same hierarchy."""
        normalized_filter = (
            component_filter.strip().strip(".")
            if component_filter is not None
            else None
        )
        if component_filter is None:
            filter_paths = set()
        elif normalized_filter in TOP_LEVEL_COMPONENTS:
            filter_paths = {normalized_filter}
        else:
            filter_paths = {
                (
                    f"layers.{layer}.{normalized_filter}"
                    if normalized_filter
                    else f"layers.{layer}"
                )
                for layer in layer_indices
            }
        leaf_filter = bool(
            normalized_filter
            and normalized_filter not in TOP_LEVEL_COMPONENTS
            and "." not in normalized_filter
        )
        logical: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}
        for endpoint in (spec.actual, spec.expected):
            label = endpoint.label
            for endpoint_path, value in trace.gradients[label].items():
                logical_path = self._logical_activation_path(endpoint, endpoint_path)
                logical.setdefault(logical_path, {})[label] = (endpoint_path, value)
        normalized_logical: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}
        for logical_path, values in logical.items():
            base_path = logical_path.split("[", 1)[0]
            target_path = (
                base_path
                if base_path.endswith((".attention", ".moe", ".feed_forward"))
                else logical_path
            )
            target_values = normalized_logical.setdefault(target_path, {})
            for label, value in values.items():
                target_values.setdefault(label, value)
        logical = normalized_logical
        for logical_path in sorted(logical):
            base_logical = logical_path.split("[", 1)[0]
            selected_match = bool(
                filter_paths
                and any(
                    base_logical == path or base_logical.startswith(f"{path}.")
                    for path in filter_paths
                )
            )
            if leaf_filter:
                selected_match = selected_match or base_logical.endswith(
                    f".{normalized_filter}"
                )
            if filter_paths and not selected_match:
                continue
            values = logical[logical_path]
            layer_match = re.match(r"layers\.(\d+)", base_logical)
            layer: int | str = (
                int(layer_match.group(1)) if layer_match else "global"
            )
            parent_path = base_logical.rpartition(".")[0]
            level = len(logical_path.split(".")) - 1
            component = logical_path.rsplit(".", 1)[-1]
            checkpoint = base_logical.endswith(
                (".attention", ".moe", ".feed_forward")
            ) or base_logical in filter_paths or (
                leaf_filter and base_logical.endswith(f".{normalized_filter}")
            )
            node_kind = "gradient_checkpoint" if checkpoint else "gradient_activation"
            actual_value = values.get(spec.actual.label)
            expected_value = values.get(spec.expected.label)
            module_path = (
                f"{spec.actual.label}:{actual_value[0] if actual_value else logical_path}.grad"
                f" <-> {spec.expected.label}:"
                f"{expected_value[0] if expected_value else logical_path}.grad"
            )
            if (
                actual_value is not None
                and spec.actual.label == spec.expected.label
            ):
                expected_value = actual_value
            if actual_value is None or expected_value is None:
                recorder.missing(
                    scope="activation_gradient",
                    component=component,
                    layer=layer,
                    module_path=module_path,
                    parent_path=parent_path,
                    level=level,
                    node_kind=node_kind,
                    checkpoint=False,
                    detail=(
                        f"gradient trace missing actual={actual_value is not None}, "
                        f"expected={expected_value is not None}"
                    ),
                )
                continue
            recorder.tensor(
                scope="activation_gradient",
                component=component,
                layer=layer,
                actual=actual_value[1],
                expected=expected_value[1],
                rtol=spec.rtol,
                atol=spec.atol,
                module_path=module_path,
                parent_path=parent_path,
                level=level,
                node_kind=node_kind,
                checkpoint=checkpoint,
            )

    def _run_routed_experts_on_input(
        self,
        endpoint: ModelEndpoint,
        layer_index: int,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one routed-expert branch without the shared expert branch."""
        model = self._model(endpoint)
        hidden_states = hidden_states.to(
            device=self.device,
            dtype=endpoint.precision.dtype,
        )
        if endpoint.implementation == "hf":
            mlp = model.model.layers[layer_index].mlp
            _, weights, indices = mlp.gate(hidden_states)
            routed = mlp.experts(
                hidden_states.reshape(-1, hidden_states.shape[-1]),
                indices,
                weights,
            ).reshape_as(hidden_states)
            indices = indices.reshape(*hidden_states.shape[:2], -1)
            weights = weights.reshape(*hidden_states.shape[:2], -1)
            num_experts = mlp.experts.num_experts
        else:
            moe = model.layers[str(layer_index)].moe
            weights, indices, scores = moe.router(
                hidden_states, moe.expert_bias_E
            )
            routing_map = torch.zeros_like(scores, dtype=torch.bool).scatter_(
                -1, indices, True
            )
            expert_load = routing_map.sum(dim=(0, 1))
            routed = moe.routed_experts(
                hidden_states,
                weights,
                indices,
                expert_load,
                num_local_tokens_after_seq_dim_padding=(
                    hidden_states.shape[0] * hidden_states.shape[1]
                ),
            )
            return routed, weights, indices, expert_load

        expert_load = F.one_hot(
            indices,
            num_classes=num_experts,
        ).sum(dim=(0, 1, 2))
        return routed, weights, indices, expert_load

    def _record_common_input_moe_replay(
        self,
        spec: ComparisonSpec,
        trace: EndpointTrace,
        recorder: ParityRecorder,
        layer_indices: list[int],
    ) -> None:
        """Separate propagated MoE input error from routed-expert local error."""
        for layer in layer_indices:
            actual_input = trace.moe_inputs[spec.actual.label].get(layer)
            expected_input = trace.moe_inputs[spec.expected.label].get(layer)
            if actual_input is None and expected_input is None:
                continue
            if actual_input is None or expected_input is None:
                continue

            parent_path = f"layers.{layer}.moe"
            recorder.tensor(
                scope="causal_replay",
                component="input",
                layer=layer,
                actual=actual_input,
                expected=expected_input,
                rtol=spec.rtol,
                atol=spec.atol,
                module_path=(
                    f"{spec.actual.label}:{parent_path}.input <-> "
                    f"{spec.expected.label}:{parent_path}.input"
                ),
                parent_path=parent_path,
                level=3,
                node_kind="input_checkpoint",
            )

            # Use one FP32 source tensor, then cast it only at each configured
            # endpoint boundary.  This removes upstream activation divergence.
            common_input = expected_input.to(torch.float32)
            with torch.no_grad():
                actual = self._run_routed_experts_on_input(
                    spec.actual, layer, common_input
                )
                expected = self._run_routed_experts_on_input(
                    spec.expected, layer, common_input
                )

            common_path = f"{parent_path}.routed_experts.common_input"
            recorder.discrete(
                scope="causal_replay",
                component="indices",
                layer=layer,
                actual=actual[2],
                expected=expected[2],
                module_path=common_path,
                parent_path=f"{parent_path}.routed_experts",
                level=4,
                node_kind="discrete_checkpoint",
            )
            recorder.tensor(
                scope="causal_replay",
                component="weights",
                layer=layer,
                actual=actual[1],
                expected=expected[1],
                rtol=spec.rtol,
                atol=spec.atol,
                module_path=common_path,
                parent_path=f"{parent_path}.routed_experts",
                level=4,
                node_kind="activation_checkpoint",
            )
            recorder.discrete(
                scope="causal_replay",
                component="expert_load",
                layer=layer,
                actual=actual[3],
                expected=expected[3],
                module_path=common_path,
                parent_path=f"{parent_path}.routed_experts",
                level=4,
                node_kind="discrete_checkpoint",
            )
            recorder.tensor(
                scope="causal_replay",
                component="common_input",
                layer=layer,
                actual=actual[0],
                expected=expected[0],
                rtol=spec.rtol,
                atol=spec.atol,
                module_path=common_path,
                parent_path=f"{parent_path}.routed_experts",
                level=4,
                node_kind="activation_checkpoint",
            )

    def compare_end_to_end(self, spec: ComparisonSpec) -> ParityRecorder:
        """Compare every decoder layer plus final logits and loss."""
        actual_model = self._model(spec.actual)
        expected_model = self._model(spec.expected)
        actual_batch = self._batch_for(spec.actual)
        expected_batch = self._batch_for(spec.expected)
        endpoints = {
            spec.actual: actual_model,
            spec.expected: expected_model,
        }
        recorder = ParityRecorder(spec.actual.precision, precision_label=spec.label)
        actual_layers = (
            actual_model.model.layers
            if spec.actual.implementation == "hf"
            else actual_model.layers
        )
        layer_indices = self._selected_layers()
        if any(layer < 0 or layer >= len(actual_layers) for layer in layer_indices):
            raise ValueError(
                f"GLM5_PARITY_LAYERS is outside [0, {len(actual_layers) - 1}]: "
                f"{layer_indices}"
            )
        with (
            EndpointTrace.install(endpoints, layer_indices) as trace,
            RecursiveModuleTrace.install(
                endpoints,
                layer_indices,
                include_model_modules=True,
            ) as module_trace,
        ):
            actual_logits, actual_loss = _run_causal_lm_endpoint(
                actual_model, spec.actual, actual_batch
            )
            expected_logits, expected_loss = _run_causal_lm_endpoint(
                expected_model, spec.expected, expected_batch
            )
            actual_model.zero_grad(set_to_none=True)
            expected_model.zero_grad(set_to_none=True)
            actual_loss.backward()
            if expected_model is not actual_model:
                expected_loss.backward()

        self._record_common_input_moe_replay(
            spec, trace, recorder, layer_indices
        )

        self._record_recursive_trace(
            spec, module_trace, recorder, layer_indices=layer_indices
        )
        self._record_recursive_gradient_trace(
            spec, module_trace, recorder, layer_indices=layer_indices
        )

        for layer in layer_indices:
            actual_block = trace.blocks[spec.actual.label].get(layer)
            expected_block = trace.blocks[spec.expected.label].get(layer)
            block_path = (
                f"{self._endpoint_path(spec.actual, layer, 'decoder_block')} <-> "
                f"{self._endpoint_path(spec.expected, layer, 'decoder_block')}"
            )
            if actual_block is None or expected_block is None:
                recorder.missing(
                    scope=spec.scope,
                    component="decoder_block",
                    layer=layer,
                    module_path=block_path,
                    detail=(
                        f"missing trace actual={actual_block is not None}, "
                        f"expected={expected_block is not None}"
                    ),
                )
            else:
                recorder.tensor(
                    scope=spec.scope,
                    component="decoder_block",
                    layer=layer,
                    actual=actual_block,
                    expected=expected_block,
                    rtol=spec.rtol,
                    atol=spec.atol,
                    module_path=block_path,
                    parent_path="layers",
                    level=2,
                    node_kind="layer_checkpoint",
                )

            actual_indexer = trace.indexer[spec.actual.label].get(layer)
            expected_indexer = trace.indexer[spec.expected.label].get(layer)
            indexer_path = (
                f"{self._endpoint_path(spec.actual, layer, 'indexer')} <-> "
                f"{self._endpoint_path(spec.expected, layer, 'indexer')}"
            )
            if actual_indexer is None or expected_indexer is None:
                recorder.missing(
                    scope=spec.scope,
                    component="indexer",
                    layer=layer,
                    module_path=indexer_path,
                    parent_path=f"layers.{layer}.attention",
                    level=4,
                    node_kind="discrete_checkpoint",
                    detail=(
                        f"missing trace actual={actual_indexer is not None}, "
                        f"expected={expected_indexer is not None}"
                    ),
                )
            else:
                recorder.discrete(
                    scope=spec.scope,
                    component="indexer",
                    layer=layer,
                    actual=actual_indexer,
                    expected=expected_indexer,
                    positions=actual_batch.positions,
                    module_path=indexer_path,
                    parent_path=f"layers.{layer}.attention",
                    level=4,
                    node_kind="discrete_checkpoint",
                )

            actual_router = trace.router[spec.actual.label].get(layer)
            expected_router = trace.router[spec.expected.label].get(layer)
            if actual_router is None and expected_router is None:
                continue
            router_path = (
                f"{self._endpoint_path(spec.actual, layer, 'router')} <-> "
                f"{self._endpoint_path(spec.expected, layer, 'router')}"
            )
            if actual_router is None or expected_router is None:
                recorder.missing(
                    scope=spec.scope,
                    component="router",
                    layer=layer,
                    module_path=router_path,
                    parent_path=f"layers.{layer}.moe",
                    level=4,
                    node_kind="discrete_checkpoint",
                    detail=(
                        f"missing trace actual={actual_router is not None}, "
                        f"expected={expected_router is not None}"
                    ),
                )
            else:
                recorder.discrete(
                    scope=spec.scope,
                    component="router",
                    layer=layer,
                    actual=actual_router,
                    expected=expected_router,
                    module_path=router_path,
                    parent_path=f"layers.{layer}.moe",
                    level=4,
                    node_kind="discrete_checkpoint",
                )

                actual_weights = trace.router_weights[spec.actual.label].get(layer)
                expected_weights = trace.router_weights[spec.expected.label].get(layer)
                if actual_weights is not None and expected_weights is not None:
                    recorder.tensor(
                        scope=spec.scope,
                        component="router_weights",
                        layer=layer,
                        actual=actual_weights,
                        expected=expected_weights,
                        rtol=spec.rtol,
                        atol=spec.atol,
                        module_path=router_path,
                        parent_path=f"layers.{layer}.moe",
                        level=4,
                        node_kind="activation_checkpoint",
                    )

                actual_load = trace.expert_load[spec.actual.label].get(layer)
                expected_load = trace.expert_load[spec.expected.label].get(layer)
                if actual_load is not None and expected_load is not None:
                    recorder.discrete(
                        scope=spec.scope,
                        component="expert_load",
                        layer=layer,
                        actual=actual_load,
                        expected=expected_load,
                        module_path=router_path,
                        parent_path=f"layers.{layer}.moe",
                        level=4,
                        node_kind="discrete_checkpoint",
                    )

        recorder.tensor(
            scope=spec.scope,
            component="logits",
            layer="all",
            actual=actual_logits,
            expected=expected_logits,
            rtol=spec.rtol,
            atol=spec.atol,
            module_path=(
                f"{spec.actual.label}:lm_head <-> {spec.expected.label}:lm_head"
            ),
            parent_path="model",
            level=1,
            node_kind="logits_checkpoint",
        )
        recorder.tensor(
            scope=spec.scope,
            component="loss",
            layer="all",
            actual=actual_loss,
            expected=expected_loss,
            rtol=spec.rtol,
            atol=spec.atol,
            module_path=f"{spec.actual.label}:loss <-> {spec.expected.label}:loss",
            parent_path="model",
            level=1,
            node_kind="loss_checkpoint",
        )
        self._record_parameter_tree(spec, recorder)
        self._record_gradient_tree(spec, recorder)
        return recorder

    def _canonical_state(
        self, endpoint: ModelEndpoint
    ) -> dict[str, torch.Tensor]:
        # Parameters are the stable adapter surface; transient buffers such
        # as rotary caches are intentionally excluded from this tree.
        state = dict(self._model(endpoint).named_parameters())
        if endpoint.implementation == "hf":
            return self.adapter.from_hf(state)
        return state

    def _canonical_gradient(
        self, endpoint: ModelEndpoint
    ) -> tuple[dict[str, torch.Tensor], set[str]]:
        """Map gradients through the same HF/TorchTitan adapter surface."""
        model = self._model(endpoint)
        if endpoint.implementation == "titan":
            gradients = {
                name: parameter.grad
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
            missing = {
                name
                for name, parameter in model.named_parameters()
                if parameter.grad is None
            }
            return gradients, missing

        gradients: dict[str, torch.Tensor] = {}
        missing: set[str] = set()
        for name, parameter in model.named_parameters():
            is_missing = parameter.grad is None
            value = torch.zeros_like(parameter) if is_missing else parameter.grad
            mapped = self.adapter.from_hf({name: value})
            gradients.update(mapped)
            if is_missing:
                missing.update(mapped)
        return gradients, missing

    def _record_gradient_tree(
        self, spec: ComparisonSpec, recorder: ParityRecorder
    ) -> None:
        actual_state, actual_missing = self._canonical_gradient(spec.actual)
        expected_state, expected_missing = self._canonical_gradient(spec.expected)
        for key in sorted(
            set(actual_state)
            | set(expected_state)
            | actual_missing
            | expected_missing
        ):
            actual = actual_state.get(key)
            expected = expected_state.get(key)
            layer_match = self.adapter._TITAN_LAYER_KEY.fullmatch(key)
            layer = int(layer_match.group(1)) if layer_match else "global"
            actual_key = (
                self._hf_path_for_titan_key(key)
                if spec.actual.implementation == "hf"
                else key
            )
            expected_key = (
                self._hf_path_for_titan_key(key)
                if spec.expected.implementation == "hf"
                else key
            )
            module_path = (
                f"{spec.actual.label}:{actual_key}.grad <-> "
                f"{spec.expected.label}:{expected_key}.grad"
            )
            parent_path = key.rpartition(".")[0]
            level = len(key.split(".")) - 1
            if actual is None or expected is None:
                recorder.missing(
                    scope="gradient",
                    component="gradient",
                    layer=layer,
                    module_path=module_path,
                    parent_path=parent_path,
                    level=level,
                    node_kind="gradient",
                    checkpoint=not (
                        key in actual_missing and key in expected_missing
                    ),
                    detail=(
                        f"gradient missing actual={actual is not None}, "
                        f"expected={expected is not None}"
                    ),
                )
                continue
            routed_projection = {
                self.adapter._TITAN_GATE: "w1",
                self.adapter._TITAN_UP: "w3",
                self.adapter._TITAN_DOWN: "w2",
            }
            expert_name = next(
                (
                    projection
                    for suffix, projection in routed_projection.items()
                    if key.endswith(suffix)
                ),
                None,
            )
            if (
                expert_name is not None
                and actual.ndim >= 1
                and expected.ndim >= 1
                and actual.shape == expected.shape
            ):
                for expert in range(actual.shape[0]):
                    actual_expert_path = f"{actual_key}[expert={expert}]"
                    expected_expert_path = f"{expected_key}[expert={expert}]"
                    expert_parent = (
                        f"layers.{layer}.moe.routed_experts.{expert}"
                    )
                    recorder.tensor(
                        scope="gradient",
                        component=f"expert_{expert_name}_gradient",
                        layer=layer,
                        actual=actual[expert],
                        expected=expected[expert],
                        rtol=spec.rtol,
                        atol=spec.atol,
                        module_path=(
                            f"{spec.actual.label}:{actual_expert_path}.grad <-> "
                            f"{spec.expected.label}:{expected_expert_path}.grad"
                        ),
                        parent_path=expert_parent,
                        level=len(expert_parent.split(".")) - 1,
                        node_kind="gradient",
                        checkpoint=True,
                    )
                continue
            recorder.tensor(
                scope="gradient",
                component="gradient",
                layer=layer,
                actual=actual,
                expected=expected,
                rtol=spec.rtol,
                atol=spec.atol,
                module_path=module_path,
                parent_path=parent_path,
                level=level,
                node_kind="gradient",
                checkpoint=True,
            )

    def _hf_path_for_titan_key(self, titan_key: str) -> str:
        if titan_key in self.adapter._to_hf_top_level:
            return self.adapter._to_hf_top_level[titan_key]
        match = self.adapter._TITAN_LAYER_KEY.fullmatch(titan_key)
        if match is None:
            return titan_key
        layer, suffix = match.groups()
        if suffix in {
            self.adapter._TITAN_GATE,
            self.adapter._TITAN_UP,
        }:
            return f"model.layers.{layer}.mlp.experts.gate_up_proj"
        if suffix == self.adapter._TITAN_DOWN:
            return f"model.layers.{layer}.mlp.experts.down_proj"
        hf_suffix = self.adapter._to_hf_layer.get(suffix, suffix)
        return f"model.layers.{layer}.{hf_suffix}"

    def _record_parameter_tree(
        self, spec: ComparisonSpec, recorder: ParityRecorder
    ) -> None:
        """Compare every adapter-visible state key, including packed experts."""
        actual_state = self._canonical_state(spec.actual)
        expected_state = self._canonical_state(spec.expected)
        routed_paths = {
            self.adapter._TITAN_GATE: ("w1", "gate"),
            self.adapter._TITAN_UP: ("w3", "up"),
            self.adapter._TITAN_DOWN: ("w2", "down"),
        }
        for key in sorted(set(actual_state) | set(expected_state)):
            actual = actual_state.get(key)
            expected = expected_state.get(key)
            layer_match = self.adapter._TITAN_LAYER_KEY.fullmatch(key)
            layer = int(layer_match.group(1)) if layer_match else "global"
            actual_key = (
                self._hf_path_for_titan_key(key)
                if spec.actual.implementation == "hf"
                else key
            )
            expected_key = (
                self._hf_path_for_titan_key(key)
                if spec.expected.implementation == "hf"
                else key
            )
            path = (
                f"{spec.actual.label}:{actual_key} <-> "
                f"{spec.expected.label}:{expected_key}"
            )
            if actual is None or expected is None:
                recorder.missing(
                    scope="parameters",
                    component="state_dict",
                    layer=layer,
                    module_path=path,
                    parent_path=key.rpartition(".")[0],
                    level=len(key.split(".")) - 1,
                    node_kind="parameter",
                    checkpoint=False,
                    detail=(
                        f"missing key actual={actual is not None}, "
                        f"expected={expected is not None}"
                    ),
                )
                continue
            routed_suffix = None
            for suffix, value in routed_paths.items():
                if key.endswith(suffix):
                    routed_suffix = value
                    break
            if (
                routed_suffix is not None
                and actual.ndim >= 1
                and expected.ndim >= 1
                and actual.shape == expected.shape
            ):
                expert_name, hf_projection = routed_suffix
                for expert in range(actual.shape[0]):
                    def expert_path(endpoint: ModelEndpoint) -> str:
                        if endpoint.implementation == "titan":
                            return (
                                f"layers.{layer}.moe.routed_experts."
                                f"{expert}.{expert_name}"
                            )
                        if hf_projection == "gate":
                            projection = "gate_up_proj[:, :expert_width]"
                        elif hf_projection == "up":
                            projection = "gate_up_proj[:, expert_width:]"
                        else:
                            projection = "down_proj"
                        return (
                            f"model.layers.{layer}.mlp.experts.{projection}"
                            f"[expert={expert}]"
                        )

                    module_path = (
                        f"{spec.actual.label}:{expert_path(spec.actual)} <-> "
                        f"{spec.expected.label}:{expert_path(spec.expected)}"
                    )
                    recorder.tensor(
                        scope="parameters",
                        component="expert_parameter",
                        layer=layer,
                        actual=actual[expert],
                        expected=expected[expert],
                        rtol=spec.rtol,
                        atol=spec.atol,
                        module_path=module_path,
                        parent_path=(
                            f"layers.{layer}.moe.routed_experts.{expert}"
                        ),
                        level=len(
                            f"layers.{layer}.moe.routed_experts.{expert}".split(".")
                        ) - 1,
                        node_kind="parameter",
                        checkpoint=False,
                    )
                continue
            recorder.tensor(
                scope="parameters",
                component="state_dict",
                layer=layer,
                actual=actual,
                expected=expected,
                rtol=spec.rtol,
                atol=spec.atol,
                module_path=path,
                parent_path=key.rpartition(".")[0],
                level=len(key.split(".")) - 1,
                node_kind="parameter",
                checkpoint=False,
            )

    def test_format_reports_layer_position_and_discrete_mismatches(self) -> None:
        self._check_format_reports_layer_position_and_discrete_mismatches()

    def test_format_reports_missing_expected_records(self) -> None:
        self._check_format_reports_missing_expected_records()

    def test_format_reports_incompatible_block_shape(self) -> None:
        self._check_format_reports_incompatible_block_shape()

    def test_recorder_reports_tensor_and_discrete_rows(self) -> None:
        self._check_recorder_reports_tensor_and_discrete_rows()

    def test_recorder_writes_plain_text_report(self) -> None:
        self._check_recorder_writes_plain_text_report()

    def test_recorder_reports_extended_metrics_and_explosions(self) -> None:
        self._check_recorder_reports_extended_metrics_and_explosions()

    def test_router_gate_is_evaluated_in_float32(self) -> None:
        self._check_router_gate_is_evaluated_in_float32()

    def test_router_uses_gate_forward_for_fp32_biased_computation(self) -> None:
        self._check_router_uses_gate_forward_for_fp32_biased_computation()

    def test_indexer_topk_matches_transformers_exactly(self) -> None:
        if not self.gpu_ready:
            self.skipTest(self.gpu_skip_reason)
        self._activate_test_data("indexer")
        self._publish_recorder(
            self._check_indexer_topk_matches_transformers(),
            "component-indexer",
            "Indexer parity",
        )

    def test_router_selection_and_weights_match_transformers_exactly(self) -> None:
        if not self.gpu_ready:
            self.skipTest(self.gpu_skip_reason)
        self._activate_test_data("router")
        self._publish_recorder(
            self._check_router_selection_and_weights(),
            "component-router",
            "Router parity",
        )

    def test_attention_output_matches_transformers(self) -> None:
        if not self.gpu_ready:
            self.skipTest(self.gpu_skip_reason)
        self._activate_test_data("attention")
        self._publish_recorder(
            self._check_attention_output(),
            "component-attention",
            "Attention parity",
        )

    def test_dense_block_output_matches_transformers(self) -> None:
        if not self.gpu_ready:
            self.skipTest(self.gpu_skip_reason)
        self._activate_test_data("block")
        self._publish_recorder(
            self._check_dense_block_output(),
            "component-block",
            "Decoder block parity",
        )

    def test_selected_component_parity(self) -> None:
        """Run generic compare functions for configured custom components."""
        if not self.gpu_ready:
            self.skipTest(self.gpu_skip_reason)
        fixed_components = {
            "all",
            "indexer",
            "router",
            "attention",
            "block",
            "gradient",
            "parameters",
            "logits",
            "loss",
            "model",
            "e2e",
        }
        selected = {
            item.strip().lower()
            for item in self.RUN_COMPONENTS.split(",")
            if item.strip()
        }
        for component in sorted(selected - fixed_components):
            self._activate_test_data(f"component:{component}")
            safe_component = re.sub(r"[^A-Za-z0-9_.-]+", "-", component)
            self._publish_recorder(
                self._configured_component_recorder(component),
                f"component-{safe_component}",
                f"Component parity: {component}",
            )

    def test_end_to_end_output_loss_and_gradients(self) -> None:
        if not self.gpu_ready:
            self.skipTest(self.gpu_skip_reason)
        self._activate_test_data("end-to-end")
        self._publish_recorder(
            self.compare_end_to_end(self._configured_spec()),
            "end-to-end",
            "End-to-end parity",
        )

    # Complete entry point for the configured end-to-end and component tests.
    def test_configured_precision_suite(self) -> None:
        """Call independent tests and combine their results into one report."""
        if self.RUN_MODE == "prepare":
            self._run_prepare_fixture()
            return
        if self.RUN_MODE == "compare":
            self._run_offline_compare()
            return
        if not self.gpu_ready:
            self.skipTest(self.gpu_skip_reason)
        if self.RUN_MODE == "capture":
            self._run_capture_suite()
            return
        spec = self._configured_spec()
        suite_report = ParitySuiteReport(
            f"GLM-5 parity: {self._configured_report_label()}; "
            f"data={self.DATA_CASE}; "
            f"layers={self.LAYER_INDICES}; "
            f"model={self.model_size.label}; "
            f"params={self.num_model_parameters:,}"
        )
        self._active_suite_report = suite_report
        try:
            if self._component_enabled("indexer"):
                self.test_indexer_topk_matches_transformers_exactly()
            if self._component_enabled("router"):
                self.test_router_selection_and_weights_match_transformers_exactly()
            if self._component_enabled("attention"):
                self.test_attention_output_matches_transformers()
            if self._component_enabled("block"):
                self.test_dense_block_output_matches_transformers()
            self.test_selected_component_parity()
            self.test_end_to_end_output_loss_and_gradients()
        finally:
            self._active_suite_report = None

        peak_bytes = self.device_module.max_memory_allocated(self.device)
        total_bytes = self.device_module.get_device_properties(
            self.device
        ).total_memory
        suite_report.title += (
            f"; peak_{self.device_type}={peak_bytes / 2**30:.2f}GiB"
            f"/{total_bytes / 2**30:.2f}GiB"
            f" ({100.0 * peak_bytes / total_bytes:.1f}%)"
        )
        report_path = self._report_path("suite")
        suite_report.configuration = self._report_configuration(
            spec,
            report_path=report_path,
            peak_bytes=peak_bytes,
            total_bytes=total_bytes,
        )
        suite_report.write(report_path)
        print(f"GLM-5 parity report: {report_path}")
        if suite_report.failed:
            raise AssertionError(suite_report.failure_message())
