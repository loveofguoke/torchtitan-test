"""Configuration primitives for GLM-5.2 performance experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


DeviceType = Literal["auto", "cuda", "npu"]


@dataclass(frozen=True)
class ParallelTopology:
    """TorchTitan parallel degrees for one profiling run."""

    name: str
    world_size: int
    dp_replicate: int = 1
    dp_shard: int = 1
    tp: int = 1
    pp: int = 1
    ep: int = 1
    pp_schedule: str = "1F1B"
    pp_microbatch_size: int = 1

    def __post_init__(self) -> None:
        dense_world_size = self.dp_replicate * self.dp_shard * self.tp * self.pp
        if self.world_size < 1 or dense_world_size != self.world_size:
            raise ValueError(
                f"topology {self.name!r} covers {dense_world_size} ranks, "
                f"not world_size={self.world_size}"
            )
        if (self.dp_shard * self.tp) % self.ep:
            raise ValueError("dp_shard * tp must be divisible by ep")

    def command_args(self) -> list[str]:
        args = [
            f"--parallelism.data_parallel_replicate_degree={self.dp_replicate}",
            f"--parallelism.data_parallel_shard_degree={self.dp_shard}",
            f"--parallelism.tensor_parallel_degree={self.tp}",
            f"--parallelism.pipeline_parallel_degree={self.pp}",
            f"--parallelism.expert_parallel_degree={self.ep}",
        ]
        if self.tp > 1:
            args.append("--parallelism.no-enable-sequence-parallel")
        if self.pp > 1:
            args.extend(
                [
                    f"--parallelism.pipeline_parallel_schedule={self.pp_schedule}",
                    "--parallelism.pipeline_parallel_microbatch_size="
                    f"{self.pp_microbatch_size}",
                ]
            )
        return args


def standard_topologies() -> dict[str, ParallelTopology]:
    """Return the topology names shared by probe and benchmark scripts."""

    return {
        "single": ParallelTopology("single", 1),
        "ddp2": ParallelTopology("ddp2", 2, dp_replicate=2),
        "ddp8": ParallelTopology("ddp8", 8, dp_replicate=8),
        "fsdp8": ParallelTopology("fsdp8", 8, dp_shard=8),
        "tp8": ParallelTopology("tp8", 8, tp=8),
        "pp8": ParallelTopology("pp8", 8, pp=8),
        "ep8": ParallelTopology("ep8", 8, dp_shard=8, ep=8),
        "fsdp2-tp4": ParallelTopology("fsdp2-tp4", 8, dp_shard=2, tp=4),
        "fsdp4-tp2": ParallelTopology("fsdp4-tp2", 8, dp_shard=4, tp=2),
        "fsdp2-pp4": ParallelTopology("fsdp2-pp4", 8, dp_shard=2, pp=4),
        "fsdp2-tp2-pp2": ParallelTopology(
            "fsdp2-tp2-pp2", 8, dp_shard=2, tp=2, pp=2
        ),
        "fsdp2-tp4-ep8": ParallelTopology(
            "fsdp2-tp4-ep8", 8, dp_shard=2, tp=4, ep=8
        ),
    }


@dataclass(frozen=True)
class ProfilerPreset:
    """Top-down Ascend collection detail for a bounded step window."""

    name: str
    level: Literal["level0", "level1", "level2"]
    profile_ranks: str = "0"
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False
    online_parse: bool = True
    aic_metrics: str = "none"
    l2_cache: bool = False
    op_attr: bool = False
    record_op_args: bool = False

    def environment(self) -> dict[str, str]:
        prefix = "TORCHTITAN_NPU_PROFILER_"
        return {
            prefix + "LEVEL": self.level,
            prefix + "RANKS": self.profile_ranks,
            prefix + "RECORD_SHAPES": str(self.record_shapes).lower(),
            prefix + "PROFILE_MEMORY": str(self.profile_memory).lower(),
            prefix + "WITH_STACK": str(self.with_stack).lower(),
            prefix + "ONLINE_PARSE": str(self.online_parse).lower(),
            prefix + "AIC_METRICS": self.aic_metrics,
            prefix + "L2_CACHE": str(self.l2_cache).lower(),
            prefix + "OP_ATTR": str(self.op_attr).lower(),
            prefix + "RECORD_OP_ARGS": str(self.record_op_args).lower(),
        }


def profiler_presets() -> dict[str, ProfilerPreset]:
    """Return progressive presets; start shallow and only then drill down."""

    return {
        "overview": ProfilerPreset("overview", "level0"),
        "distributed": ProfilerPreset(
            "distributed",
            "level1",
            profile_ranks="all",
            aic_metrics="pipe_utilization",
        ),
        "kernel": ProfilerPreset(
            "kernel",
            "level1",
            record_shapes=True,
            aic_metrics="arithmetic_utilization",
            l2_cache=True,
        ),
        "runtime": ProfilerPreset(
            "runtime",
            "level2",
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            aic_metrics="arithmetic_utilization",
            l2_cache=True,
            op_attr=True,
        ),
    }


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
    local_batch_size: int = 2
    global_batch_size: int = 2
    sequence_length: int = 128
    seed: int = 61
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
        required_steps = self.skip_steps + self.warmup_steps + self.active_steps
        if self.steps < required_steps:
            raise ValueError(
                f"steps={self.steps} cannot reach profiler window ending at "
                f"step {required_steps}"
            )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
