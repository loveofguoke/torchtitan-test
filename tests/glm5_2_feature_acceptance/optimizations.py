"""Optimization registry shared by feature acceptance and A/B planning.

An entry is executable only when its activation changes the selected training
contract.  Unsupported provider, recipe, or hardware combinations remain
registered with an explicit block instead of receiving a baseline command.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import json
from pathlib import Path

from .matrix import Platform


class OptimizationScope(StrEnum):
    TRAINING = "training"
    INFERENCE = "inference"


class OptimizationReadiness(StrEnum):
    RUNNABLE = "runnable"
    REQUIRES_INTEGRATION = "requires-integration"
    BLOCKED_BY_CAPABILITY = "blocked-by-capability"


@dataclass(frozen=True, slots=True)
class OptimizationSpec:
    optimization_id: str
    feature_id: str
    name: str
    scope: OptimizationScope
    owner: str
    bottlenecks: tuple[str, ...]
    topologies: tuple[str, ...]
    metrics: tuple[str, ...]
    conflicts: tuple[str, ...] = ()
    gpu: OptimizationReadiness = OptimizationReadiness.REQUIRES_INTEGRATION
    npu: OptimizationReadiness = OptimizationReadiness.REQUIRES_INTEGRATION
    activation: tuple[str, ...] = ()
    reason: str = ""

    def readiness(self, platform: Platform) -> OptimizationReadiness:
        return self.gpu if platform is Platform.GPU else self.npu

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["scope"] = self.scope.value
        payload["gpu"] = self.gpu.value
        payload["npu"] = self.npu.value
        return payload


@dataclass(frozen=True, slots=True)
class OptimizationABPlan:
    """One bounded A/B contract for a platform and optimization identity."""

    optimization_id: str
    platform: Platform
    scope: OptimizationScope
    readiness: OptimizationReadiness
    topologies: tuple[str, ...]
    baseline_args: tuple[str, ...]
    candidate_args: tuple[str, ...]
    metrics: tuple[str, ...]
    conflicts: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["platform"] = self.platform.value
        payload["scope"] = self.scope.value
        payload["readiness"] = self.readiness.value
        return payload


COMMUNICATION_METRICS = (
    "median_step_seconds",
    "mean_throughput_tps",
    "median_exposed_communication_ms",
    "exposed_communication_percent",
    "communication_overlap_percent",
    "rank_ready_skew_ms",
    "collective_payload_bytes",
    "effective_bandwidth_gbps",
    "peak_active_memory_gib",
)


def build_optimization_registry() -> tuple[OptimizationSpec, ...]:
    """Return reviewed training and inference optimization identities."""
    run = OptimizationReadiness.RUNNABLE
    integrate = OptimizationReadiness.REQUIRES_INTEGRATION
    blocked = OptimizationReadiness.BLOCKED_BY_CAPABILITY
    training = OptimizationScope.TRAINING
    inference = OptimizationScope.INFERENCE
    return (
        OptimizationSpec(
            "mtp_training", "mtp", "GLM NextN auxiliary training", training,
            "torchtitan", ("decode_quality",), ("single", "tp2", "cp2"),
            ("loss", "grad_norm", "step_time", "tokens_per_second", "memory"),
            gpu=run, npu=run,
            activation=("--config", "glm5_2_debugmodel_mtp"),
            reason="Uses the explicit GLM MTP flavor and MTPLoss.",
        ),
        OptimizationSpec(
            "fsdp_bf16_reduce", "fsdp2", "BF16 FSDP gradient reduction", training,
            "torchtitan", ("fsdp_reduce_scatter_bandwidth",), ("fsdp2", "fsdp8"),
            COMMUNICATION_METRICS + ("loss_delta", "grad_norm_delta"),
            gpu=run, npu=run,
            activation=("--training.mixed_precision_reduce=bfloat16",),
            reason="Reduces collective bytes; numerical acceptance is mandatory.",
        ),
        OptimizationSpec(
            "fsdp_no_reshard", "fsdp2", "Keep FSDP parameters unsharded", training,
            "torchtitan", ("repeated_fsdp_all_gather",), ("fsdp2", "fsdp8"),
            COMMUNICATION_METRICS,
            conflicts=("memory_constrained",), gpu=run, npu=run,
            activation=("--parallelism.fsdp_reshard_after_forward=never",),
            reason="Trades HBM for fewer parameter all-gathers.",
        ),
        OptimizationSpec(
            "regional_compile", "regional_compile", "Regional compilation", training,
            "torchtitan", ("launch_overhead", "python_overhead"), ("single", "fsdp2"),
            ("compile_time", "graph_breaks", "step_time", "tokens_per_second"),
            gpu=run, npu=run,
            activation=("--graph", "inductor"),
            reason="Uses the existing local compile experiment path.",
        ),
        OptimizationSpec(
            "async_tp", "async_tp", "Async tensor parallel", training,
            "torchtitan", ("tp_collective_exposure",), ("tp2", "tp8"),
            COMMUNICATION_METRICS,
            gpu=integrate, npu=integrate,
            activation=("--compile.enable_async_tensor_parallel",),
            reason="GLM GraphTrainer model and recipe are not implemented yet.",
        ),
        OptimizationSpec(
            "fsdp_ag_rs_overlap", "fsdp_overlap", "FSDP AG/RS overlap", training,
            "torchtitan", ("fsdp_collective_exposure",), ("fsdp2", "fsdp8"),
            COMMUNICATION_METRICS,
            gpu=integrate, npu=integrate,
            activation=("--compile.enable_fsdp_ag_rs_overlap",),
            reason="Requires a GLM GraphTrainer model and recipe.",
        ),
        OptimizationSpec(
            "fsdp_dense_overlap", "fsdp_dense_overlap", "Dense-region FSDP overlap", training,
            "torchtitan", ("fsdp_collective_exposure",), ("fsdp2", "fsdp8"),
            COMMUNICATION_METRICS,
            conflicts=("ep_overlap",), gpu=integrate, npu=integrate,
            activation=("--compile.enable_fsdp_dense_region_overlap",),
            reason="Requires a GLM GraphTrainer model and recipe.",
        ),
        OptimizationSpec(
            "ep_overlap", "ep_overlap", "EP dispatch/combine overlap", training,
            "torchtitan", ("all_to_all_exposure", "moe_imbalance"), ("ep2", "ep8"),
            COMMUNICATION_METRICS + ("tokens_per_expert_cv",),
            conflicts=("fsdp_dense_overlap",), gpu=integrate, npu=integrate,
            activation=("--compile.ep_overlap.enabled",),
            reason="Requires GLM GraphTrainer integration and token-exchange markers.",
        ),
        OptimizationSpec(
            "deepep_gpu", "deepep", "DeepEP GPU dispatcher", training,
            "torchtitan", ("all_to_all_exposure",), ("ep2", "ep8"),
            COMMUNICATION_METRICS + ("dispatch_ms", "combine_ms"),
            gpu=integrate, npu=blocked,
            reason="GLM needs an explicit DeepEP recipe; NPU uses a different provider.",
        ),
        OptimizationSpec(
            "deepep_ascend", "deepep_ascend", "DeepEP-Ascend dispatcher", training,
            "torchtitanturbo", ("all_to_all_exposure",), ("ep2", "ep8"),
            COMMUNICATION_METRICS + ("dispatch_ms", "combine_ms"),
            gpu=blocked, npu=integrate,
            reason="Provider exists default-off; supported-hardware runtime acceptance is pending.",
        ),
        OptimizationSpec(
            "dtensor_collective_cleanup", "dtensor", "DTensor collective cleanup", training,
            "torchtitan", ("small_collectives", "partial_redistribution"),
            ("tp2", "tp8", "fsdp2-tp4"), COMMUNICATION_METRICS,
            gpu=integrate, npu=integrate,
            reason="Requires a reviewed placement change at the measured redistribution seam.",
        ),
        OptimizationSpec(
            "host_launch_skew", "host_runtime", "Reduce host launch/rank skew", training,
            "torchtitan-or-backend", ("rank_ready_skew", "host_overhead"),
            ("ddp8", "fsdp8", "tp8", "ep8"),
            COMMUNICATION_METRICS + ("host_self_time_ms", "synchronization_count"),
            gpu=integrate, npu=integrate,
            reason="The patch site must come from a distributed trace, not a global switch.",
        ),
        OptimizationSpec(
            "dsa_index_kernel", "dsa", "DSA index/scatter device kernel", training,
            "torchtitanturbo", ("aicpu_fallback", "index_memory_traffic"),
            ("single", "cp8"),
            ("aicpu_time_ms", "kernel_count", "step_time", "loss_delta", "index_mismatch"),
            gpu=blocked, npu=integrate,
            reason="Needs an Ascend-native implementation preserving global-token index semantics.",
        ),
        OptimizationSpec(
            "topk_mask_fusion", "dsa", "Fuse TopK mask pointwise chain", training,
            "compiler-or-torchtitanturbo", ("kernel_launches", "hbm_traffic"),
            ("single", "cp8"),
            ("kernel_count", "hbm_bytes", "step_time", "loss_delta", "topk_mismatch"),
            gpu=integrate, npu=integrate,
            reason="Select Inductor fusion, a native op, or a backend kernel only after hotspot proof.",
        ),
        OptimizationSpec(
            "deepselect", "dsa", "DeepSelect TopK provider", training,
            "torchtitanturbo", ("topk_latency",), ("single", "cp8"),
            ("topk_ms", "step_time", "tie_nan_mismatch", "loss_delta"),
            gpu=integrate, npu=blocked,
            reason="Default-off provider exists; package and full DSA semantic acceptance are pending.",
        ),
        OptimizationSpec(
            "tilekernels_topk", "moe", "TileKernels MoE TopK provider", training,
            "torchtitanturbo", ("router_topk_latency",), ("ep2", "ep8"),
            ("router_ms", "step_time", "route_mismatch", "loss_delta"),
            gpu=integrate, npu=integrate,
            reason="Provider exists; Ascend acceptance requires the documented 950/CANN 9.2 stack.",
        ),
        OptimizationSpec(
            "deepgemm_ascend", "moe", "DeepGEMM-Ascend grouped GEMM", training,
            "torchtitanturbo", ("expert_gemm",), ("ep2", "ep8"),
            ("grouped_gemm_ms", "step_time", "tokens_per_expert_cv", "loss_delta", "grad_delta"),
            gpu=blocked, npu=integrate,
            reason="Autograd provider exists; target-hardware kernel and end-to-end acceptance remain.",
        ),
        OptimizationSpec(
            "flashmla_dense", "attention", "FlashMLA dense attention", training,
            "torchtitanturbo", ("attention_latency", "attention_memory"),
            ("single", "tp8", "cp8"),
            ("attention_ms", "hbm_bytes", "step_time", "loss_delta", "grad_delta"),
            gpu=integrate, npu=integrate,
            reason="Needs exact GLM layout, mask, RoPE, dropout, dtype and backward compatibility.",
        ),
        OptimizationSpec(
            "activation_cpu_offload", "activation_offload", "Activation CPU offload", training,
            "torchtitan", ("activation_memory",), ("single", "fsdp2"),
            ("peak_active_memory_gib", "host_memory_gib", "transfer_ms", "step_time"),
            gpu=integrate, npu=integrate,
            reason="GraphTrainer policy exists but the GLM GraphTrainer recipe is missing.",
        ),
        OptimizationSpec(
            "state_cpu_offload", "state_offload", "FSDP state CPU offload", training,
            "torchtitan", ("parameter_optimizer_memory",), ("fsdp2", "fsdp8"),
            ("peak_active_memory_gib", "host_memory_gib", "transfer_ms", "step_time"),
            gpu=run, npu=run,
            activation=("--training.enable_cpu_offload",),
            reason="Uses the existing FSDP2 CPU-offload configuration and requires NUMA validation.",
        ),
        OptimizationSpec(
            "activation_recompute", "selective_ac", "Activation recomputation", training,
            "torchtitan", ("activation_memory",), ("single", "fsdp2"),
            ("peak_active_memory_gib", "step_time", "tokens_per_second", "loss_delta"),
            gpu=integrate, npu=integrate,
            reason=(
                "Current GLM recipes do not expose a reviewed eager activation-"
                "checkpoint policy; GraphTrainer memory policy needs GLM integration."
            ),
        ),
        OptimizationSpec(
            "async_checkpoint", "async_checkpoint", "Async distributed checkpoint", training,
            "torchtitan", ("checkpoint_pause",), ("fsdp2", "fsdp8"),
            ("checkpoint_pause_ms", "checkpoint_bandwidth_gbps", "step_jitter", "host_memory"),
            gpu=run, npu=run,
            activation=("--checkpointer.async_mode", "async"),
            reason="Measures checkpoint interference separately from steady-state compute.",
        ),
        OptimizationSpec(
            "deepep_deepgemm_overlap", "moe", "DeepEP/DeepGEMM overlap", training,
            "torchtitanturbo", ("all_to_all_exposure", "expert_gemm"), ("ep2", "ep8"),
            COMMUNICATION_METRICS + ("grouped_gemm_ms", "dispatch_ms", "combine_ms"),
            conflicts=("ep_overlap",), gpu=integrate, npu=integrate,
            reason="Requires a proved independent compute window and accepted providers first.",
        ),
        OptimizationSpec(
            "megamoe", "moe", "MegaMoE end-to-end fusion", training,
            "torchtitanturbo", ("all_to_all_exposure", "expert_gemm", "kernel_launches"),
            ("ep8",), COMMUNICATION_METRICS + ("grouped_gemm_ms", "loss_delta", "grad_delta"),
            gpu=integrate, npu=integrate,
            reason="Research adapter is not implemented; dispatcher, GEMM and combine semantics are coupled.",
        ),
        OptimizationSpec(
            "gtp_rematerialization", "memory", "Generalized TP weight rematerialization", training,
            "torchtitanturbo", ("model_state_memory",), ("tp8",),
            ("peak_active_memory_gib", "weight_all_gather_ms", "step_time", "loss_delta"),
            gpu=integrate, npu=integrate,
            reason="No PyTorch-native TorchTitan executor exists in the current version.",
        ),
        OptimizationSpec(
            "nvme_offload", "memory", "NVMe parameter/optimizer offload", training,
            "future-runtime", ("host_memory_capacity",), ("fsdp8",),
            ("peak_active_memory_gib", "host_memory_gib", "nvme_bandwidth_gbps", "step_time"),
            gpu=blocked, npu=blocked,
            reason="No TorchTitan executor exists; do not disguise CPU offload as NVMe offload.",
        ),
        OptimizationSpec(
            "flashcomm", "flashcomm", "FlashComm RS/AG fusion", inference,
            "vllm-ascend", ("tensor_parallel_collectives",), ("tp8",),
            ("ttft", "prefill_tokens_per_second", "exposed_communication_ms"),
            gpu=blocked, npu=integrate,
            reason="Official serving optimization; not a TorchTitan training implementation.",
        ),
        OptimizationSpec(
            "flashmla_sparse", "dsa", "FlashMLA sparse DSA forward", inference,
            "torchtitanturbo", ("sparse_attention_latency",), ("single", "tp8"),
            ("attention_ms", "prefill_tokens_per_second", "index_mismatch", "quality"),
            gpu=integrate, npu=integrate,
            reason="Forward-only provider exists; it must not replace training without backward.",
        ),
        OptimizationSpec(
            "fused_mc2", "fused_mc2", "Fused MC2/MegaMoE", inference,
            "vllm-ascend", ("all_to_all_exposure",), ("ep8",),
            ("ttft", "tpot", "dispatch_ms", "combine_ms"),
            conflicts=("mtp_overlap", "dynamic_eplb"), gpu=blocked, npu=integrate,
            reason="Serving-only candidate with hardware and quantization constraints.",
        ),
        OptimizationSpec(
            "mtp_overlap", "mtp", "Speculative MTP overlap", inference,
            "serving-runtime", ("decode_latency",), ("tp8",),
            ("acceptance_rate", "tpot", "decode_tokens_per_second", "quality"),
            gpu=integrate, npu=integrate,
            reason="Requires proposer/verifier scheduling; training MTP alone is insufficient.",
        ),
    )


def validate_optimization_registry(
    optimizations: tuple[OptimizationSpec, ...],
) -> tuple[OptimizationSpec, ...]:
    ids = [item.optimization_id for item in optimizations]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"duplicate optimization ids: {', '.join(duplicates)}")
    for item in optimizations:
        if not item.metrics or not item.topologies or not item.reason.strip():
            raise ValueError(f"incomplete optimization contract: {item.optimization_id}")
        if OptimizationReadiness.RUNNABLE in (item.gpu, item.npu) and not item.activation:
            raise ValueError(f"runnable optimization has no activation: {item.optimization_id}")
    return optimizations


def build_optimization_plans() -> tuple[OptimizationABPlan, ...]:
    """Expand the registry into honest platform-specific A/B contracts."""
    plans: list[OptimizationABPlan] = []
    for item in validate_optimization_registry(build_optimization_registry()):
        for platform in Platform:
            readiness = item.readiness(platform)
            plans.append(
                OptimizationABPlan(
                    optimization_id=item.optimization_id,
                    platform=platform,
                    scope=item.scope,
                    readiness=readiness,
                    topologies=item.topologies,
                    baseline_args=(),
                    candidate_args=(
                        item.activation
                        if readiness is OptimizationReadiness.RUNNABLE
                        else ()
                    ),
                    metrics=item.metrics,
                    conflicts=item.conflicts,
                    reason=item.reason,
                )
            )
    return tuple(plans)


def write_optimization_plans(output: Path) -> Path:
    plans = build_optimization_plans()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"schema_version": 1, "plans": [plan.to_dict() for plan in plans]},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def write_optimization_registry(output: Path) -> Path:
    items = validate_optimization_registry(build_optimization_registry())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"schema_version": 1, "optimizations": [item.to_dict() for item in items]},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output
