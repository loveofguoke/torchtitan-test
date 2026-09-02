"""Configuration primitives for GLM-5.2 performance experiments.

A preset is a collection policy, not a pass/fail standard. ``overview`` keeps
cross-topology cost bounded; deeper presets independently enable communication,
kernel, memory, stack, or host/system evidence. ``all`` expands to separate
captures because many Ascend collectors change profiler level or raw-event
retention and cannot be reconstructed from one lightweight trace.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from tests.glm5_2_common.topology import (
    ParallelTopology,
    standard_topologies as common_topologies,
)

from .collectors import PerformanceCollector


DeviceType = Literal["auto", "cuda", "npu"]
ParseMode = Literal["sync", "async", "offline"]


@dataclass(frozen=True)
class ProfilerPreset:
    """Top-down Ascend collection detail for a bounded step window.

    Rank selection is part of storage cost: an 8-rank distributed preset emits
    eight framework/CANN/device streams. Shape, stack, memory, L2, and system
    collectors further enlarge raw data, so suites should use deep presets on
    representative topologies rather than assuming they are free report views.
    """

    name: str
    level: Literal["level_none", "level0", "level1", "level2"]
    profile_ranks: str = "0"
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False
    with_modules: bool = False
    with_flops: bool = False
    export_stacks: bool = False
    export_memory_timeline: bool = False
    parse_mode: ParseMode = "sync"
    export_types: tuple[str, ...] = ("text", "db")
    aic_metrics: str = "none"
    l2_cache: bool = False
    op_attr: bool = False
    data_simplification: bool = True
    record_op_args: bool = False
    gc_detect_threshold: float | None = None
    msprof_tx: bool = False
    mstx: bool = False
    mstx_domain_include: tuple[str, ...] = ()
    mstx_domain_exclude: tuple[str, ...] = ()
    host_system: tuple[str, ...] = ()
    system_io: bool = False
    system_interconnection: bool = False

    def environment(self) -> dict[str, str]:
        prefix = "TORCHTITAN_NPU_PROFILER_"
        environment = {
            prefix + "LEVEL": self.level,
            prefix + "RANKS": self.profile_ranks,
            prefix + "RECORD_SHAPES": str(self.record_shapes).lower(),
            prefix + "PROFILE_MEMORY": str(self.profile_memory).lower(),
            prefix + "WITH_STACK": str(self.with_stack).lower(),
            prefix + "WITH_MODULES": str(self.with_modules).lower(),
            prefix + "PARSE_MODE": self.parse_mode,
            prefix + "AIC_METRICS": self.aic_metrics,
            prefix + "L2_CACHE": str(self.l2_cache).lower(),
            prefix + "OP_ATTR": str(self.op_attr).lower(),
            prefix + "DATA_SIMPLIFICATION": str(
                self.data_simplification
            ).lower(),
            prefix + "RECORD_OP_ARGS": str(self.record_op_args).lower(),
            prefix + "GC_DETECT_THRESHOLD": (
                str(self.gc_detect_threshold)
                if self.gc_detect_threshold is not None
                else "none"
            ),
            prefix + "HOST_SYSTEM": ",".join(self.host_system) or "none",
            prefix + "SYSTEM_IO": str(self.system_io).lower(),
            prefix + "SYSTEM_INTERCONNECTION": str(
                self.system_interconnection
            ).lower(),
        }
        if self.export_stacks:
            environment[prefix + "EXPORT_STACKS"] = "true"
        if self.export_memory_timeline:
            environment[prefix + "EXPORT_MEMORY_TIMELINE"] = "true"
        if self.with_flops:
            environment[prefix + "WITH_FLOPS"] = "true"
        if self.mstx:
            environment[prefix + "MSTX"] = "true"
        if self.msprof_tx:
            environment[prefix + "MSPROF_TX"] = "true"
        if self.mstx_domain_include:
            environment[prefix + "MSTX_DOMAIN_INCLUDE"] = ",".join(
                self.mstx_domain_include
            )
        if self.mstx_domain_exclude:
            environment[prefix + "MSTX_DOMAIN_EXCLUDE"] = ",".join(
                self.mstx_domain_exclude
            )
        if self.export_types != ("text", "db"):
            environment[prefix + "EXPORT_TYPES"] = ",".join(self.export_types)
        return environment


def profiler_presets() -> dict[str, ProfilerPreset]:
    """Return progressive presets; start shallow and only then drill down."""

    return {
        "overview": ProfilerPreset("overview", "level0"),
        "comparison": ProfilerPreset("comparison", "level0"),
        "standard": ProfilerPreset(
            "standard",
            "level1",
            profile_memory=True,
            aic_metrics="pipe_utilization",
        ),
        "distributed": ProfilerPreset(
            "distributed",
            "level1",
            profile_ranks="all",
            profile_memory=True,
            parse_mode="offline",
            aic_metrics="pipe_utilization",
            data_simplification=False,
            system_interconnection=True,
        ),
        "kernel": ProfilerPreset(
            "kernel",
            "level1",
            record_shapes=True,
            aic_metrics="arithmetic_utilization",
            l2_cache=True,
            data_simplification=False,
        ),
        "operator": ProfilerPreset(
            "operator",
            "level1",
            record_shapes=True,
            with_flops=True,
            aic_metrics="arithmetic_utilization",
            op_attr=True,
            data_simplification=False,
            record_op_args=True,
        ),
        "memory": ProfilerPreset(
            "memory",
            "level0",
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_modules=True,
            export_memory_timeline=True,
            data_simplification=False,
        ),
        "flamegraph": ProfilerPreset(
            "flamegraph",
            "level0",
            with_stack=True,
            with_modules=True,
            export_stacks=True,
            data_simplification=False,
        ),
        "runtime": ProfilerPreset(
            "runtime",
            "level2",
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_modules=True,
            export_stacks=True,
            aic_metrics="arithmetic_utilization",
            l2_cache=True,
            op_attr=True,
            data_simplification=False,
            gc_detect_threshold=1.0,
            host_system=("cpu", "mem"),
            export_memory_timeline=True,
        ),
        "system": ProfilerPreset(
            "system",
            "level2",
            profile_ranks="all",
            parse_mode="offline",
            aic_metrics="pipe_utilization",
            data_simplification=False,
            gc_detect_threshold=1.0,
            mstx=True,
            host_system=("cpu", "mem", "disk", "network", "osrt", "numa"),
            system_io=True,
            system_interconnection=True,
        ),
    }


def all_profiler_presets() -> tuple[str, ...]:
    """Return the non-redundant full profiler acquisition suite."""

    return (
        "overview",
        "distributed",
        "kernel",
        "operator",
        "memory",
        "flamegraph",
        "runtime",
        "system",
    )


def performance_topologies() -> dict[str, ParallelTopology]:
    """Extend canonical acceptance topologies with health-safe lab shapes.

    The two- and four-rank entries let profiler exploration avoid a known-bad
    physical card without changing the canonical cross-experiment topology
    matrix in ``glm5_2_common``.
    """

    topologies = dict(common_topologies())
    topologies.update(
        {
            "ddp4": ParallelTopology(
                "ddp4", 4, data_parallel_replicate_degree=4
            ),
            "fsdp2": ParallelTopology(
                "fsdp2", 2, data_parallel_shard_degree=2
            ),
            "fsdp4": ParallelTopology(
                "fsdp4", 4, data_parallel_shard_degree=4
            ),
            "tp2": ParallelTopology("tp2", 2, tensor_parallel_degree=2),
            "tp4": ParallelTopology("tp4", 4, tensor_parallel_degree=4),
            "cp2": ParallelTopology("cp2", 2, context_parallel_degree=2),
            "cp4": ParallelTopology("cp4", 4, context_parallel_degree=4),
            "pp2": ParallelTopology("pp2", 2, pipeline_parallel_degree=2),
            "pp4": ParallelTopology("pp4", 4, pipeline_parallel_degree=4),
            "ep2": ParallelTopology(
                "ep2",
                2,
                data_parallel_shard_degree=2,
                expert_parallel_degree=2,
            ),
            "ep4": ParallelTopology(
                "ep4",
                4,
                data_parallel_shard_degree=4,
                expert_parallel_degree=4,
            ),
            "fsdp2-tp2": ParallelTopology(
                "fsdp2-tp2",
                4,
                data_parallel_shard_degree=2,
                tensor_parallel_degree=2,
            ),
        }
    )
    return topologies


# Preserve the original public import used by existing performance tests and
# scripts while keeping these lab-only shapes out of glm5_2_common.
standard_topologies = performance_topologies


@dataclass(frozen=True)
class PerformanceConfig:
    """One reproducible profiler experiment."""

    name: str
    module: str = "glm5"
    model_config: str = "glm5_debugmodel"
    device: DeviceType = "auto"
    topology: str = "single"
    preset: str = "overview"
    steps: int = 12
    skip_steps: int = 3
    warmup_steps: int = 1
    active_steps: int = 3
    profiler_enabled: bool = True
    collector: str = PerformanceCollector.TORCH_NPU_PROFILER.value
    collector_args: tuple[str, ...] = ()
    local_batch_size: int = 2
    global_batch_size: int = 2
    sequence_length: int = 128
    seed: int = 61
    replicate: int = 0
    training_dtype: str = "float32"
    mixed_precision_param: str = "bfloat16"
    mixed_precision_reduce: str = "float32"
    deterministic: bool = False
    extra_args: tuple[str, ...] = ()
    run_root: str = "performance_runs"
    artifact_root: str = "performance_artifacts"
    report_root: str = "performance_reports"
    environment: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            PerformanceCollector(self.collector)
        except ValueError as error:
            raise ValueError(
                f"unknown performance collector: {self.collector!r}"
            ) from error
        if (
            self.collector_args
            and self.collector != PerformanceCollector.MSPROF.value
        ):
            raise ValueError("collector_args are supported only by msprof")
        reserved_msprof_prefixes = ("--output", "--application", "--dynamic")
        if any(
            argument == prefix or argument.startswith(prefix + "=")
            for argument in self.collector_args
            for prefix in reserved_msprof_prefixes
        ):
            raise ValueError(
                "collector_args must not override lifecycle-owned msprof "
                "output, application, or dynamic-launch options"
            )
        if min(
            self.steps,
            self.local_batch_size,
            self.global_batch_size,
            self.sequence_length,
            self.active_steps,
        ) < 1:
            raise ValueError("training and active profiler sizes must be positive")
        if self.skip_steps < 0 or self.warmup_steps < 0:
            raise ValueError("profiler skip and warmup steps must be non-negative")
        if self.replicate < 0:
            raise ValueError("replicate must be non-negative")
        if self.mixed_precision_reduce not in {
            "float32",
            "float16",
            "bfloat16",
        }:
            raise ValueError(
                "mixed_precision_reduce must be float32, float16, or bfloat16"
            )
        required_steps = self.skip_steps + self.warmup_steps + self.active_steps
        if self.steps < required_steps:
            raise ValueError(
                f"steps={self.steps} cannot reach profiler window ending at "
                f"step {required_steps}"
            )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
