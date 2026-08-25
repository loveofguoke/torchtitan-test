"""Configuration for bounded GLM-5.2 CUDA profiler experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from tests.glm5_2_common.topology import ParallelTopology, standard_topologies


def gpu_topologies() -> dict[str, ParallelTopology]:
    """Return the shared topologies that fit on at most eight local GPUs."""

    return {
        name: topology
        for name, topology in standard_topologies().items()
        if topology.world_size <= 8
    }


@dataclass(frozen=True)
class GpuPerformanceConfig:
    """One reproducible Torch Profiler experiment on CUDA."""

    name: str
    module: str = "glm5"
    model_config: str = "glm5_debugmodel"
    topology: str = "single"
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
    profile_ranks: str = "0"
    record_shapes: bool = True
    profile_memory: bool = False
    with_stack: bool = False
    with_flops: bool = True
    run_root: str = "performance_gpu_runs"
    artifact_root: str = "performance_gpu_artifacts"
    report_root: str = "performance_gpu_reports"
    extra_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.topology not in gpu_topologies():
            raise ValueError(f"unknown GPU topology: {self.topology!r}")
        positive = (
            self.steps,
            self.active_steps,
            self.local_batch_size,
            self.global_batch_size,
            self.sequence_length,
        )
        if min(positive) < 1:
            raise ValueError("training and profiler sizes must be positive")
        if self.skip_steps < 0 or self.warmup_steps < 0:
            raise ValueError("skip and warmup steps must be non-negative")
        required = self.skip_steps + self.warmup_steps + self.active_steps
        if self.steps < required:
            raise ValueError(
                f"steps={self.steps} cannot reach profiler window ending at {required}"
            )
        if self.profile_ranks.strip().lower() not in {"all", "*", "-1"}:
            try:
                ranks = {int(value.strip()) for value in self.profile_ranks.split(",")}
            except ValueError as error:
                raise ValueError("profile_ranks must be 'all' or integers") from error
            if not ranks or min(ranks) < 0:
                raise ValueError("profile_ranks contains an invalid rank")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
