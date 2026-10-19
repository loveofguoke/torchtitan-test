"""Executable inventory for TorchTitan feature acceptance.

The inventory is intentionally independent from TorchTitan imports. It can be
generated and reviewed while an upstream synchronization is in progress, and
it never treats a configuration surface as runtime acceptance evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import json
from pathlib import Path
from typing import Iterable


class Platform(StrEnum):
    GPU = "gpu"
    NPU = "npu"


class AcceptanceStatus(StrEnum):
    OFFICIAL = "official"
    PROTOTYPE = "prototype"
    SMOKE_PASSED = "smoke-passed"
    NUMERICS_PASSED = "numerics-passed"
    PERFORMANCE_PASSED = "performance-passed"
    BLOCKED_BY_CAPABILITY = "blocked-by-capability"
    BLOCKED_BY_VERSION = "blocked-by-version"
    NOT_STARTED = "not-started"


@dataclass(frozen=True, slots=True)
class PlatformAcceptance:
    status: AcceptanceStatus
    reason: str


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    feature_id: str
    category: str
    name: str
    upstream_surface: str
    prototype_owner: str
    required_topologies: tuple[str, ...]
    gates: tuple[str, ...]
    gpu: PlatformAcceptance
    npu: PlatformAcceptance

    def acceptance(self, platform: Platform) -> PlatformAcceptance:
        return self.gpu if platform is Platform.GPU else self.npu

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["gpu"]["status"] = self.gpu.status.value
        payload["npu"]["status"] = self.npu.status.value
        return payload


STANDARD_GATES = (
    "function",
    "numerics",
    "activation",
    "performance",
    "resources",
    "regression",
    "reproducibility",
)

# Feature identities advertised by the current TorchTitan README. Keep this
# list separate from extension prototypes so a matrix edit cannot silently
# drop an upstream capability while still passing uniqueness checks.
README_FEATURE_IDS = frozenset(
    {
        "ddp", "fsdp2", "hsdp", "tp", "async_tp", "pp", "cp", "ep",
        "deepep", "hybridep", "dist_moe", "full_ac", "selective_ac",
        "regional_ac", "regional_compile", "cuda_graphs", "mxfp8", "nvfp4",
        "flash_attention", "flex_attention", "varlen_attention",
        "bf16_optimizer", "fsdp_cpu_offload", "grain", "sft", "lora",
        "dist_muon", "ema", "mtp", "wsd_lr", "validation", "titan_rl",
        "dcp", "async_checkpoint", "hf_safetensors_io",
        "checkpoint_conversion", "observability", "memory_snapshot",
        "flight_recorder", "fake_backend", "deterministic_loss_compare",
        "sdc_replay", "structured_logging", "python_recipes",
        "component_overrides", "graph_trainer", "torchft",
        "transformers_backend",
    }
)


def _accept(status: AcceptanceStatus, reason: str) -> PlatformAcceptance:
    return PlatformAcceptance(status=status, reason=reason)


def _feature(
    feature_id: str,
    category: str,
    name: str,
    upstream_surface: str,
    prototype_owner: str,
    topologies: tuple[str, ...],
    gpu: tuple[AcceptanceStatus, str],
    npu: tuple[AcceptanceStatus, str],
) -> FeatureSpec:
    return FeatureSpec(
        feature_id=feature_id,
        category=category,
        name=name,
        upstream_surface=upstream_surface,
        prototype_owner=prototype_owner,
        required_topologies=topologies,
        gates=STANDARD_GATES,
        gpu=_accept(*gpu),
        npu=_accept(*npu),
    )


def build_feature_matrix() -> tuple[FeatureSpec, ...]:
    """Return the version-frozen official feature and extension inventory."""
    o = AcceptanceStatus.OFFICIAL
    p = AcceptanceStatus.PROTOTYPE
    s = AcceptanceStatus.SMOKE_PASSED
    perf = AcceptanceStatus.PERFORMANCE_PASSED
    cap = AcceptanceStatus.BLOCKED_BY_CAPABILITY
    new = AcceptanceStatus.NOT_STARTED
    return (
        _feature("ddp", "parallelism", "DDP", "ParallelDims", "torchtitan", ("ddp2", "ddp8"), (s, "GPU smoke evidence exists."), (s, "NPU smoke evidence exists.")),
        _feature("fsdp2", "parallelism", "FSDP2", "fully_shard", "torchtitan", ("fsdp2", "fsdp8"), (s, "GPU smoke evidence exists."), (s, "NPU smoke evidence exists.")),
        _feature("hsdp", "parallelism", "HSDP", "dp_replicate + dp_shard", "torchtitan-test", ("hsdp2x4", "hsdp4x2"), (s, "Single-node functional evidence exists."), (s, "Single-node functional evidence exists.")),
        _feature("tp", "parallelism", "Tensor parallel", "TP/SP plans", "torchtitan", ("tp2", "tp4", "tp8"), (s, "GPU smoke evidence exists."), (s, "NPU smoke evidence exists.")),
        _feature("async_tp", "parallelism", "Async tensor parallel", "GraphTrainer async TP", "torchtitanturbo", ("tp2",), (p, "Typed configuration prototype exists."), (p, "Requires NPU GraphTrainer numerics and timeline evidence.")),
        _feature("cp", "parallelism", "Context parallel", "all-gather KV / Ulysses", "torchtitan", ("cp2", "cp4", "cp8"), (s, "GPU smoke evidence exists."), (s, "NPU smoke evidence exists.")),
        _feature("pp", "parallelism", "Pipeline parallel schedules", "PipelineSchedule", "torchtitan", ("pp2", "pp4", "pp8"), (s, "GPU smoke evidence exists."), (s, "NPU smoke evidence exists.")),
        _feature("ep", "parallelism", "Expert parallel", "AllToAllTokenDispatcher", "torchtitan", ("ep2", "ep4", "ep8"), (s, "GPU smoke evidence exists."), (s, "NPU smoke evidence exists.")),
        _feature("deepep", "communication", "DeepEP dispatcher", "DeepEPTokenDispatcher", "torchtitanturbo", ("ep2", "ep8"), (o, "Official GPU seam; runtime validation remains."), (p, "DeepEP-Ascend provider and converter are default-off.")),
        _feature("hybridep", "communication", "HybridEP dispatcher", "HybridEP", "torchtitan", ("ep8",), (o, "NVIDIA-specific official path."), (cap, "No equivalent Ascend backend contract.")),
        _feature("dist_moe", "moe", "Dist-MoE", "models.common.dist_moe", "torchtitan", ("fsdp2-tp2-ep4",), (cap, "Requires SM100 or SM103."), (cap, "Official implementation targets Blackwell.")),
        _feature("local_compile", "compile", "Local torch.compile", "CompileConfig", "torchtitan-test", ("single", "ddp8", "tp8", "cp8", "pp8", "ep8"), (s, "CUDA Inductor smoke exists."), (s, "NPU Inductor with Triton and DVM smoke exists.")),
        _feature("regional_compile", "compile", "Regional torch.compile", "models.common.compile", "torchtitan-test", ("single", "fsdp2"), (o, "Official regional compile surface exists."), (p, "GLM NPU graph coverage and performance evidence are pending.")),
        _feature("cuda_graphs", "compile", "CUDA graphs", "training.enable_cuda_graphs", "torchtitan-test", ("single", "ddp2"), (o, "Official CUDA graph path exists."), (cap, "CUDA graphs are NVIDIA-specific; NPU graphs require a separate backend contract.")),
        _feature("graph_trainer", "compile", "GraphTrainer", "experiments.graph_trainer", "torchtitanturbo", ("single", "fsdp2"), (o, "Official experiment exists."), (p, "Whole-step NPU acceptance is pending.")),
        _feature("simple_fsdp", "compile", "GraphTrainer SimpleFSDP", "graph_trainer.simple_fsdp", "torchtitanturbo", ("fsdp2",), (o, "Official compiler-based FSDP exists."), (p, "NPU tracing, numerics, and collective scheduling are pending.")),
        _feature("graphpp", "compile", "GraphPP", "graph_trainer.graph_pp", "torchtitan-test", ("pp2",), (o, "Official stage-local graph runtime exists."), (p, "NPU stage tracing and schedule acceptance are pending.")),
        _feature("graph_precompile", "compile", "Compile-on-one-rank", "compile.precompile_artifact_dir", "torchtitanturbo", ("fsdp2",), (o, "Official rank-agnostic artifact path exists."), (p, "NPU artifact portability and fingerprint validation are pending; GraphPP is unsupported.")),
        _feature("graph_tensor_memory", "memory", "Tensor-granularity graph memory policy", "compile.memory_policy", "torchtitanturbo", ("single", "fsdp2"), (o, "Official save/recompute/offload policies exist."), (p, "NPU memory, transfer, and numerics acceptance are pending.")),
        _feature("autoparallel", "compile", "AutoParallel", "GraphTrainer AutoParallel", "torchtitanturbo", ("fsdp2",), (p, "Typed configuration prototype exists."), (p, "Generated placements need NPU validation.")),
        _feature("fsdp_overlap", "overlap", "FSDP AG/RS overlap", "GraphTrainer overlap flags", "torchtitanturbo", ("fsdp2",), (p, "Typed configuration prototype exists."), (p, "Profiler timeline acceptance is pending.")),
        _feature("fsdp_dense_overlap", "overlap", "FSDP dense-region overlap", "GraphTrainer overlap flags", "torchtitanturbo", ("fsdp2",), (p, "Typed configuration prototype exists."), (p, "Must be tested independently from AG/RS overlap.")),
        _feature("ep_overlap", "overlap", "EP overlap", "GraphTrainer EP overlap", "torchtitanturbo", ("ep2",), (p, "Typed chunk and module-scope configuration exists."), (p, "Numerics and communication coverage are pending.")),
        _feature("full_ac", "memory", "Full activation checkpointing", "ActivationCheckpointingConfig", "torchtitan", ("single", "fsdp2"), (o, "Official configuration exists."), (p, "GLM NPU A/B is pending.")),
        _feature("selective_ac", "memory", "Selective activation checkpointing", "ActivationCheckpointingConfig", "torchtitan", ("single", "fsdp2"), (o, "Official configuration exists."), (p, "GLM NPU A/B is pending.")),
        _feature("regional_ac", "memory", "Regional activation checkpointing", "REMAT", "torchtitan", ("single", "fsdp2"), (o, "Official interface exists."), (new, "GLM region configuration is not implemented.")),
        _feature("activation_cpu_offload", "memory", "Activation CPU offload", "GraphTrainer memory policy", "torchtitanturbo", ("single", "fsdp2"), (p, "Budget, prefetch, and defer prototype exists."), (p, "Host-memory and transfer evidence are pending.")),
        _feature("fsdp_cpu_offload", "memory", "FSDP CPU state offload", "training.enable_cpu_offload", "torchtitanturbo", ("fsdp2",), (o, "Official configuration exists."), (p, "Large-model NPU validation is pending.")),
        _feature("dcp", "checkpoint", "Distributed checkpoint", "CheckpointManager", "torchtitan-test", ("single", "fsdp2", "tp2"), (s, "Save/load experiments exist."), (s, "Save/load experiments exist.")),
        _feature("async_checkpoint", "checkpoint", "Async checkpoint", "checkpointer.async_mode", "torchtitanturbo", ("fsdp2",), (p, "Typed async mode exists."), (p, "Staging and resume validation are pending.")),
        _feature("async_pinned_checkpoint", "checkpoint", "Async pinned-memory checkpoint", "checkpointer.async_mode", "torchtitanturbo", ("fsdp2",), (p, "Typed pinned mode exists."), (p, "Pinned-memory peak and jitter validation are pending.")),
        _feature("hf_safetensors_io", "checkpoint", "Direct Hugging Face safetensors I/O", "components.checkpointer", "torchtitan-test", ("single", "fsdp2"), (o, "Official direct load/save surface exists."), (new, "GLM round-trip and distributed reshard evidence are pending.")),
        _feature("checkpoint_conversion", "checkpoint", "HF and DCP conversion", "scripts.checkpoint_conversion", "torchtitan-test", ("single",), (o, "Official conversion scripts exist."), (new, "GLM conversion parity is pending.")),
        _feature("grain", "data", "Grain data pipeline", "components.data", "torchtitan-test", ("single", "ddp2"), (o, "Official data interface exists."), (new, "GLM resume and throughput acceptance is pending.")),
        _feature("sft", "training", "Supervised fine-tuning", "training recipes", "torchtitan-test", ("single", "fsdp2"), (o, "Official chat-dataset SFT surface exists."), (new, "GLM NPU recipe and convergence smoke are pending.")),
        _feature("lora", "training", "LoRA", "training recipes", "torchtitan-test", ("single", "fsdp2"), (o, "Official LoRA path exists."), (new, "GLM adapter save/load and NPU numerics are pending.")),
        _feature("dist_muon", "optimizer", "Muon and DistMuon", "components.optim", "torchtitan-test", ("single", "fsdp2"), (o, "Official optimizer implementation exists."), (new, "GLM optimizer-state and NPU collective validation are pending.")),
        _feature("ema", "training", "Exponential moving average", "components", "torchtitan-test", ("single", "fsdp2"), (o, "Official EMA surface exists."), (new, "GLM checkpoint and distributed consistency evidence are pending.")),
        _feature("mtp", "training", "Multi-token prediction", "model and loss components", "torchtitan-test", ("single", "tp2"), (o, "Official MTP support is advertised."), (new, "GLM head, loss, state-dict, and NPU validation are pending.")),
        _feature("wsd_lr", "training", "Warmup-stable-decay LR", "components.lr_scheduler", "torchtitan-test", ("single",), (o, "Official scheduler surface exists."), (new, "GLM schedule trace and resume validation are pending.")),
        _feature("validation", "training", "Validation loop", "docs.evaluation", "torchtitan-test", ("single", "fsdp2"), (o, "Official validation workflow exists."), (new, "GLM dataset and distributed metric parity are pending.")),
        _feature("bf16_optimizer", "precision", "BF16 optimizer states", "components.optim", "torchtitan-test", ("single", "fsdp2"), (o, "Official configuration exists."), (new, "NPU numerics and checkpoint acceptance are pending.")),
        _feature("flash_attention", "attention", "FlashAttention 3/4", "attention backends", "torchtitan-test", ("single", "tp2"), (o, "Official GPU backend selection exists."), (cap, "NPU uses backend-native attention rather than NVIDIA FlashAttention 3/4.")),
        _feature("flex_attention", "attention", "FlexAttention", "torch.nn.attention.flex_attention", "torchtitan-test", ("single", "cp2"), (s, "CUDA FlexAttention smoke exists."), (s, "TorchNPU Triton and DVM FlexAttention smoke exists.")),
        _feature("varlen_attention", "attention", "Variable-length attention", "attention backends", "torchtitan-test", ("single", "cp2"), (o, "Official variable-length path exists."), (new, "GLM packed-sequence NPU numerics and kernel activation are pending.")),
        _feature("fp8", "precision", "FP8 training", "torchao float8", "torchtitan", ("single", "fsdp2", "tp2"), (o, "Requires supported GPU and matching torchao."), (cap, "Must use an Ascend-specific low-precision path.")),
        _feature("mxfp8", "precision", "MXFP8 training", "quantization.mxfp8", "torchtitan", ("single", "fsdp2"), (cap, "Requires SM100 or later."), (cap, "No matching NPU dtype contract.")),
        _feature("nvfp4", "precision", "NVFP4 training", "quantization.nvfp4", "torchtitan", ("single", "fsdp2"), (cap, "Requires SM100 or later."), (cap, "No matching NPU dtype contract.")),
        _feature("torchft", "reliability", "TorchFT", "experiments.torchft", "torchtitan-test", ("ddp2",), (o, "Official experiment exists."), (new, "Fault-injection validation is pending.")),
        _feature("transformers_backend", "modeling", "Transformers modeling backend", "experiments.transformers_modeling_backend", "torchtitan-test", ("single", "tp2"), (o, "Official experiment exists."), (new, "GLM state-dict and NPU validation are pending.")),
        _feature("titan_rl", "rl", "TitanRL", "torchtitan.rl", "torchtitan-test", ("single",), (o, "Official stack exists."), (new, "Kept outside pretraining performance claims.")),
        _feature("observability", "observability", "Metrics and profiler", "observability + profiler", "torchtitan-test", ("single", "fsdp8", "tp8", "cp8", "pp8", "ep8"), (s, "NVIDIA system workflow exists."), (perf, "Profiler, offline analysis, and Insight workflow are validated.")),
        _feature("memory_snapshot", "observability", "Memory snapshots", "docs.debugging", "torchtitan-test", ("single", "fsdp2"), (o, "Official memory snapshot workflow exists."), (p, "NPU memory report integration exists but acceptance is incomplete.")),
        _feature("flight_recorder", "observability", "Flight Recorder", "docs.debugging", "torchtitan-test", ("ddp2",), (o, "Official distributed debugging workflow exists."), (new, "NPU collective-failure replay is pending.")),
        _feature("fake_backend", "debugging", "Fake-backend large-layout dry run", "fake backend", "torchtitan-test", ("single",), (o, "Official dry-run capability exists."), (o, "Device-independent layout validation does not require NPU execution.")),
        _feature("deterministic_loss_compare", "debugging", "Deterministic mode and loss comparison", "scripts.loss_compare", "torchtitan-test", ("single", "ddp2"), (o, "Official cross-commit comparison exists."), (p, "NPU deterministic coverage and report integration are pending.")),
        _feature("sdc_replay", "reliability", "Silent-data-corruption replay", "observability.SILENT_DATA_CORRUPTION", "torchtitan-test", ("single",), (o, "Official SDC replay workflow exists."), (new, "NPU fault injection and replay evidence are pending.")),
        _feature("structured_logging", "observability", "Structured logging", "observability.structured_logger", "torchtitan-test", ("single",), (o, "Official structured logger exists."), (new, "Unified GPU/NPU artifact integration is pending.")),
        _feature("python_recipes", "extensibility", "Python recipes", "config module and config selectors", "torchtitan-test", ("single",), (o, "Official recipe mechanism exists."), (s, "GLM NPU experiments already select module and config recipes.")),
        _feature("component_overrides", "extensibility", "Component overrides", "config.OVERRIDE", "torchtitanturbo", ("single",), (o, "Official override mechanism exists."), (p, "Turbo providers use explicit seams; full acceptance remains per provider.")),
        _feature("deepselect", "extension", "DeepSelect DSA TopK", "DSAIndexerTopK seam", "torchtitanturbo", ("single",), (cap, "Ascend-specific library is not a GPU provider."), (p, "Default-off provider; FP32 GLM scores exceed the current BF16 capability.")),
        _feature("tilekernels_topk", "extension", "TileKernels MoE TopK", "Router TopK seam", "torchtitanturbo", ("single", "ep2"), (cap, "Ascend-specific library is not a GPU provider."), (p, "Capability and tie-semantics prototype exists.")),
        _feature("deepgemm_ascend", "extension", "DeepGEMM-Ascend grouped GEMM", "GroupedExperts converter", "torchtitanturbo", ("single", "ep2", "ep8"), (cap, "Use the upstream GPU DeepGEMM provider instead."), (p, "Forward and dX/dW adapter prototype exists.")),
        _feature("flashmla", "extension", "FlashMLA-Ascend sparse attention", "DSA sparse-attention seam", "torchtitanturbo", ("single",), (cap, "Ascend-specific library is not a GPU provider."), (cap, "The public Ascend API is forward-only for this training contract.")),
        _feature("deepep_ascend", "extension", "DeepEP-Ascend dispatcher", "BaseEPTokenDispatcher converter", "torchtitanturbo", ("ep2", "ep8"), (cap, "Use TorchTitan's official GPU DeepEP path."), (p, "Default-off expanded-layout and autograd lifecycle prototype exists.")),
    )


def validate_feature_matrix(features: Iterable[FeatureSpec]) -> tuple[FeatureSpec, ...]:
    """Validate stable identities and the mandatory acceptance contract."""
    materialized = tuple(features)
    feature_ids = [feature.feature_id for feature in materialized]
    duplicates = sorted({item for item in feature_ids if feature_ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"duplicate feature ids: {', '.join(duplicates)}")
    for feature in materialized:
        if not feature.required_topologies:
            raise ValueError(f"{feature.feature_id} has no acceptance topology")
        if feature.gates != STANDARD_GATES:
            raise ValueError(f"{feature.feature_id} does not use the standard gates")
        for platform in Platform:
            acceptance = feature.acceptance(platform)
            if not acceptance.reason.strip():
                raise ValueError(
                    f"{feature.feature_id} has no {platform.value} status reason"
                )
    return materialized


def render_markdown(features: Iterable[FeatureSpec]) -> str:
    """Render a compact review table without changing source evidence."""
    rows = [
        "# TorchTitan feature acceptance manifest",
        "",
        "| ID | Category | Feature | Owner | GPU | NPU | Required topologies |",
        "|---|---|---|---|---|---|---|",
    ]
    for feature in validate_feature_matrix(features):
        rows.append(
            "| "
            + " | ".join(
                (
                    f"`{feature.feature_id}`",
                    feature.category,
                    feature.name,
                    feature.prototype_owner,
                    f"`{feature.gpu.status.value}`",
                    f"`{feature.npu.status.value}`",
                    ", ".join(f"`{item}`" for item in feature.required_topologies),
                )
            )
            + " |"
        )
    return "\n".join(rows) + "\n"


def write_outputs(output_dir: Path) -> tuple[Path, Path]:
    """Write deterministic JSON and Markdown views of the same manifest."""
    features = validate_feature_matrix(build_feature_matrix())
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "feature_matrix.json"
    markdown_path = output_dir / "feature_matrix.md"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "features": [feature.to_dict() for feature in features],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(features), encoding="utf-8")
    return json_path, markdown_path

