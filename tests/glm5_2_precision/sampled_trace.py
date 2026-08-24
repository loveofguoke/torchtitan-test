#!/usr/bin/env python3
"""Low-overhead sampled tensor traces for long-running precision diagnosis.

The hooks are installed only when ``GLM5_PRECISION_TRACE_STEPS`` is set. They
observe the existing Trainer/FSDP2 path without replacing collectives or the
optimizer implementation. Full tensors stay out of the artifact: each record
contains stable statistics, a digest, and 64 deterministic tensor samples.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


_INSTALLED = False
_ACTIVE_CAPTURE: "_SampledTrace | None" = None
_LAYER_RE = re.compile(r"(?:^|\.)layers\.\d+$")
_SAMPLE_COUNT = 64
_SUPPORTED_TRACE_DETAIL = frozenset({"indexer", "router"})


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    local = value.detach()
    to_local = getattr(local, "to_local", None)
    if callable(to_local):
        local = to_local()
    return local


def _canonical_cpu(value: torch.Tensor) -> torch.Tensor:
    local = _local_tensor(value)
    if local.is_floating_point() or local.is_complex():
        local = local.float()
    return local.to("cpu").contiguous()


def _clean_name(name: str) -> str:
    for fragment in ("._orig_mod", ".module"):
        name = name.replace(fragment, "")
    return name.strip(".") or "model"


def _detail_group(name: str) -> str | None:
    clean = _clean_name(name)
    if ".attention.indexer." in f".{clean}.":
        return "indexer"
    if ".moe.router." in f".{clean}.":
        return "router"
    return None


def _trace_module(name: str, detail: frozenset[str]) -> bool:
    clean = _clean_name(name)
    return (
        clean == "model"
        or clean.endswith(("tok_embeddings", "norm", "lm_head"))
        or bool(_LAYER_RE.search(clean))
        or clean.endswith((".attention", ".attention.indexer", ".moe.router"))
        or _detail_group(clean) in detail
    )


def _trace_parameter(name: str) -> bool:
    clean = _clean_name(name)
    return (
        "tok_embeddings" in clean
        or "lm_head" in clean
        or ".layers.0." in f".{clean}."
        or "router" in clean
        or clean.endswith("norm.weight")
    )


def _tensor_leaves(value: Any, prefix: str = "") -> Iterator[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield prefix or "tensor", value
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _tensor_leaves(value[key], child)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            child = f"{prefix}.{index}" if prefix else str(index)
            yield from _tensor_leaves(item, child)


def _summary(value: torch.Tensor) -> dict[str, Any]:
    cpu = _canonical_cpu(value)
    flat = cpu.reshape(-1)
    raw = flat.view(torch.uint8).numpy().tobytes()
    result: dict[str, Any] = {
        "shape": list(cpu.shape),
        "source_dtype": str(value.dtype),
        "stored_dtype": str(cpu.dtype),
        "numel": cpu.numel(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if not flat.numel():
        result["sample_indices"] = []
        result["sample"] = []
        return result
    count = min(_SAMPLE_COUNT, flat.numel())
    if count == flat.numel():
        indices = torch.arange(count, dtype=torch.int64)
    else:
        indices = torch.linspace(0, flat.numel() - 1, count).to(torch.int64)
    sample = flat[indices]
    result["sample_indices"] = [int(item) for item in indices]
    if (
        not cpu.is_floating_point()
        and not cpu.is_complex()
        and cpu.dtype != torch.bool
        and cpu.ndim >= 1
        and 0 < cpu.shape[-1] <= 256
    ):
        rows = cpu.reshape(-1, cpu.shape[-1])
        row_count = min(_SAMPLE_COUNT, rows.shape[0])
        if row_count == rows.shape[0]:
            row_indices = torch.arange(row_count, dtype=torch.int64)
        else:
            row_indices = torch.linspace(0, rows.shape[0] - 1, row_count).to(
                torch.int64
            )
        sorted_rows = torch.sort(rows, dim=-1).values.contiguous()
        result.update(
            {
                "unordered_last_dim_sha256": hashlib.sha256(
                    sorted_rows.view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
                "row_sample_indices": [int(item) for item in row_indices],
                "row_samples": [
                    [int(item) for item in row]
                    for row in rows[row_indices].tolist()
                ],
            }
        )
    if cpu.is_floating_point():
        numeric = flat.double()
        finite = torch.isfinite(numeric)
        result.update(
            {
                "finite": bool(finite.all()),
                "min": float(numeric.min()),
                "max": float(numeric.max()),
                "mean": float(numeric.mean()),
                "mean_abs": float(numeric.abs().mean()),
                "norm": float(torch.linalg.vector_norm(numeric)),
                "sample": [float(item) for item in sample],
            }
        )
    else:
        result["sample"] = [int(item) for item in sample]
    return result


class _SampledTrace:
    def __init__(
        self,
        trainer: Any,
        steps: frozenset[int],
        output: Path,
        detail: frozenset[str],
    ):
        self.trainer = trainer
        self.steps = steps
        self.output = output
        self.detail = detail
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.path = output / f"rank-{self.rank:03d}.jsonl"
        self.handles: list[Any] = []
        self.seen: set[tuple[int, str, str]] = set()
        self.parameter_names: dict[int, str] = {}
        self.parameter_before: dict[str, torch.Tensor] = {}
        output.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"sampled trace already exists: {self.path}")
        self._write(
            {
                "record_type": "manifest",
                "schema": "torchtitan.glm5_2.sampled_trace",
                "schema_version": 1,
                "rank": self.rank,
                "world_size": self.world_size,
                "backend": os.environ.get("TORCHTITAN_DEVICE", "unknown"),
                "steps": sorted(self.steps),
                "sample_count": _SAMPLE_COUNT,
                "detail": sorted(self.detail),
            }
        )
        self._index_parameters()
        self._register_forward_hooks()

    def _write(self, value: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")))
            stream.write("\n")

    def selected(self) -> bool:
        return int(self.trainer.step) in self.steps

    def _index_parameters(self) -> None:
        for part_index, model in enumerate(self.trainer.model_parts):
            for raw_name, parameter in model.named_parameters():
                name = f"part{part_index}.{_clean_name(raw_name)}"
                if _trace_parameter(name):
                    self.parameter_names[id(parameter)] = name

    def store(self, phase: str, key: str, value: torch.Tensor, *, kind: str) -> None:
        step = int(self.trainer.step)
        identity = (step, phase, key)
        if step not in self.steps or identity in self.seen:
            return
        self.seen.add(identity)
        self._write(
            {
                "record_type": "tensor",
                "rank": self.rank,
                "step": step,
                "phase": phase,
                "key": key.replace("/", "."),
                "kind": kind,
                **_summary(value),
            }
        )

    def store_tree(self, phase: str, prefix: str, value: Any, *, kind: str) -> None:
        for suffix, tensor in _tensor_leaves(value):
            self.store(phase, f"{prefix}.{suffix}", tensor, kind=kind)

    def _register_forward_hooks(self) -> None:
        for part_index, model in enumerate(self.trainer.model_parts):
            for raw_name, module in model.named_modules():
                clean = _clean_name(raw_name)
                if not _trace_module(clean, self.detail):
                    continue
                semantic = f"part{part_index}.{clean}"
                detail_group = _detail_group(clean)

                if detail_group in self.detail:

                    def capture_input(
                        _module: torch.nn.Module,
                        inputs: tuple[Any, ...],
                        *,
                        module_name: str = semantic,
                    ) -> None:
                        if self.selected():
                            self.store_tree(
                                "forward_input",
                                f"{module_name}.input",
                                inputs,
                                kind="activation_input",
                            )

                    self.handles.append(
                        module.register_forward_pre_hook(capture_input)
                    )

                def capture_output(
                    _module: torch.nn.Module,
                    _inputs: tuple[Any, ...],
                    output: Any,
                    *,
                    module_name: str = semantic,
                ) -> None:
                    if not self.selected():
                        return
                    for suffix, tensor in _tensor_leaves(output):
                        key = f"{module_name}.{suffix}"
                        self.store("forward", key, tensor, kind="activation")
                        if tensor.requires_grad:
                            tensor.register_hook(
                                lambda grad, trace=self, tensor_key=key: (
                                    trace.store(
                                        "backward_output_grad",
                                        tensor_key,
                                        grad,
                                        kind="backward_activation_gradient",
                                    )
                                    or grad
                                )
                            )

                    if module_name.endswith(".attention.indexer.topk"):
                        self.capture_indexer_topk_diagnostics(
                            module_name,
                            _module,
                            _inputs,
                            output,
                        )

                self.handles.append(module.register_forward_hook(capture_output))

    def capture_indexer_topk_diagnostics(
        self,
        module_name: str,
        module: torch.nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        """Record DSA scores, selected scores, and the top-k boundary margin."""
        if (
            not self.selected()
            or len(inputs) != 4
            or not isinstance(output, torch.Tensor)
        ):
            return
        q_BQNH, k_BKH, weights_BQN, attention_mask_BQK = inputs
        if not all(
            isinstance(value, torch.Tensor)
            for value in (q_BQNH, k_BKH, weights_BQN, attention_mask_BQK)
        ):
            return
        index_topk = getattr(module, "index_topk", None)
        softmax_scale = getattr(module, "softmax_scale", None)
        if not isinstance(index_topk, int) or not isinstance(
            softmax_scale, float
        ):
            return

        with torch.no_grad():
            scores_BNQK = (
                torch.matmul(
                    q_BQNH.float().transpose(1, 2),
                    k_BKH.float().transpose(1, 2).unsqueeze(1),
                )
                * softmax_scale
            )
            scores_BNQK = F.relu(scores_BNQK)
            index_scores_BQK = torch.matmul(
                weights_BQN.unsqueeze(-2), scores_BNQK.transpose(1, 2)
            ).squeeze(-2)
            index_scores_BQK = index_scores_BQK + attention_mask_BQK.float()
            selected_scores = torch.gather(
                index_scores_BQK,
                dim=-1,
                index=output.long(),
            )
            self.store(
                "forward_diagnostic",
                f"{module_name}.index_scores",
                index_scores_BQK,
                kind="index_score",
            )
            self.store(
                "forward_diagnostic",
                f"{module_name}.selected_scores",
                selected_scores,
                kind="selected_index_score",
            )
            if index_scores_BQK.shape[-1] > index_topk:
                boundary = torch.topk(
                    index_scores_BQK,
                    k=index_topk + 1,
                    dim=-1,
                    sorted=True,
                ).values
                margin = boundary[..., index_topk - 1] - boundary[..., index_topk]
                self.store(
                    "forward_diagnostic",
                    f"{module_name}.boundary_margin",
                    margin,
                    kind="topk_boundary_margin",
                )

    def capture_parameters_before(self) -> None:
        if self.rank != 0 or not self.selected():
            return
        self.parameter_before.clear()
        for model in self.trainer.model_parts:
            for parameter in model.parameters():
                name = self.parameter_names.get(id(parameter))
                if name is None:
                    continue
                before = _canonical_cpu(parameter).clone()
                self.parameter_before[name] = before
                self.store("parameter_before", name, parameter, kind="parameter")

    def capture_gradients(self, phase: str, parameters: Sequence[torch.Tensor]) -> None:
        if not self.selected():
            return
        for parameter in parameters:
            name = self.parameter_names.get(id(parameter))
            if name is not None and parameter.grad is not None:
                self.store(phase, name, parameter.grad, kind="gradient")

    def capture_optimizer_state(self, phase: str) -> None:
        if self.rank != 0 or not self.selected():
            return
        for optimizer_index, optimizer in enumerate(self.trainer.optimizers.optimizers):
            for parameter, state in optimizer.state.items():
                name = self.parameter_names.get(id(parameter))
                if name is None:
                    continue
                for state_name in ("step", "exp_avg", "exp_avg_sq"):
                    value = state.get(state_name)
                    if isinstance(value, torch.Tensor):
                        self.store(
                            phase,
                            f"optimizer{optimizer_index}.{name}.{state_name}",
                            value,
                            kind="optimizer_state",
                        )

    def capture_parameters_after(self) -> None:
        if self.rank != 0 or not self.selected():
            return
        for model in self.trainer.model_parts:
            for parameter in model.parameters():
                name = self.parameter_names.get(id(parameter))
                if name is None:
                    continue
                self.store("parameter_after", name, parameter, kind="parameter")
                before = self.parameter_before.get(name)
                if before is not None:
                    after = _canonical_cpu(parameter)
                    self.store(
                        "parameter_delta",
                        name,
                        after - before,
                        kind="parameter_delta",
                    )
        self.parameter_before.clear()

    def capture_input(self, input_dict: dict[str, Any], labels: torch.Tensor) -> None:
        self.store_tree("input", "input", input_dict, kind="input")
        self.store("input", "labels", labels, kind="input")

    def event(self, status: str, failure: str = "") -> None:
        self._write(
            {
                "record_type": "event",
                "rank": self.rank,
                "step": int(self.trainer.step),
                "status": status,
                "failure": failure,
            }
        )


class _CapturingIterator:
    def __init__(self, delegate: Iterator[Any], capture: _SampledTrace):
        self.delegate = delegate
        self.capture = capture
        self.captured = False

    def __iter__(self) -> "_CapturingIterator":
        return self

    def __next__(self) -> Any:
        item = next(self.delegate)
        if not self.captured:
            input_dict, labels = item
            self.capture.capture_input(input_dict, labels)
            self.captured = True
        return item


def _parse_steps(value: str) -> frozenset[int]:
    steps = frozenset(int(item.strip()) for item in value.split(",") if item.strip())
    if not steps or any(step < 1 for step in steps):
        raise ValueError("GLM5_PRECISION_TRACE_STEPS must contain positive steps")
    return steps


def _parse_detail(value: str) -> frozenset[str]:
    detail = frozenset(item.strip() for item in value.split(",") if item.strip())
    unknown = detail - _SUPPORTED_TRACE_DETAIL
    if unknown:
        raise ValueError(
            "GLM5_PRECISION_TRACE_DETAIL contains unsupported groups: "
            + ", ".join(sorted(unknown))
        )
    return detail


def install_sampled_trace() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    steps = _parse_steps(os.environ["GLM5_PRECISION_TRACE_STEPS"])
    output = Path(os.environ["GLM5_PRECISION_TRACE_DIR"]).resolve()
    detail = _parse_detail(os.environ.get("GLM5_PRECISION_TRACE_DETAIL", ""))

    from torchtitan.components.optimizer import OptimizersContainer
    from torchtitan.distributed import utils as dist_utils
    from torchtitan.trainer import Trainer

    original_init = Trainer.__init__
    original_train_step = Trainer.train_step
    original_forward_backward = Trainer.forward_backward_step
    original_clip_grad_norm = dist_utils.clip_grad_norm_
    original_optimizer_step = OptimizersContainer.step

    def init_with_trace(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self._glm5_sampled_trace = _SampledTrace(self, steps, output, detail)

    def forward_backward_with_trace(
        self: Any, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        loss = original_forward_backward(self, *args, **kwargs)
        trace: _SampledTrace = self._glm5_sampled_trace
        if trace.selected():
            trace.store("loss", "local", loss, kind="local_loss")
        return loss

    def clip_grad_norm_with_trace(
        parameters: Sequence[torch.Tensor], *args: Any, **kwargs: Any
    ) -> Any:
        parameter_list = list(parameters)
        trace = _ACTIVE_CAPTURE
        if trace is not None and trace.selected():
            trace.capture_gradients("gradient_post_reduce_pre_clip", parameter_list)
        result = original_clip_grad_norm(parameter_list, *args, **kwargs)
        if trace is not None and trace.selected():
            trace.capture_gradients("gradient_post_reduce_post_clip", parameter_list)
        return result

    def optimizer_step_with_trace(self: Any, *args: Any, **kwargs: Any) -> Any:
        trace = _ACTIVE_CAPTURE
        if trace is not None and trace.selected():
            trace.capture_optimizer_state("optimizer_state_before")
        result = original_optimizer_step(self, *args, **kwargs)
        if trace is not None and trace.selected():
            trace.capture_optimizer_state("optimizer_state_after")
            trace.capture_parameters_after()
        return result

    def train_step_with_trace(self: Any, data_iterator: Iterator[Any]) -> Any:
        global _ACTIVE_CAPTURE
        trace: _SampledTrace = self._glm5_sampled_trace
        if not trace.selected():
            return original_train_step(self, data_iterator)
        if _ACTIVE_CAPTURE is not None:
            raise RuntimeError("nested sampled trace capture is unsupported")
        trace.capture_parameters_before()
        _ACTIVE_CAPTURE = trace
        try:
            result = original_train_step(self, _CapturingIterator(data_iterator, trace))
            trace.event("step_complete")
            return result
        except BaseException as error:
            trace.event("failed", f"{type(error).__name__}: {error}")
            raise
        finally:
            _ACTIVE_CAPTURE = None

    Trainer.__init__ = init_with_trace
    Trainer.forward_backward_step = forward_backward_with_trace
    Trainer.train_step = train_step_with_trace
    dist_utils.clip_grad_norm_ = clip_grad_norm_with_trace
    OptimizersContainer.step = optimizer_step_with_trace
