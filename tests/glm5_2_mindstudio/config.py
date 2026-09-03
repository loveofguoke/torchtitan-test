# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Configuration contracts for official MindStudio validation workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from tests.glm5_2_common.naming import config_name, slug
from tests.glm5_2_precision.workflow import (
    FormalExperimentConfig,
    FormalTrainingConfig,
    TrainingEndpoint,
)


OfficialWorkflow = Literal["config-check", "migration", "compile", "monitor"]
DumpTask = Literal[
    "statistics",
    "tensor",
    "structure",
    "overflow_check",
    "nan_check",
]
DumpLevel = Literal["L0", "L1", "L2", "mix"]


def _validate_repository_root(name: str, value: str) -> None:
    path = Path(value)
    if (
        not value.strip()
        or path.is_absolute()
        or path == Path(".")
        or ".." in path.parts
    ):
        raise ValueError(
            f"{name} must be a non-empty repository-relative directory"
        )


@dataclass(frozen=True)
class MsProbeDumpConfig:
    """One official ``PrecisionDebugger`` collection configuration."""

    task: DumpTask = "statistics"
    level: DumpLevel = "L0"
    steps: tuple[int, ...] = (0,)
    ranks: tuple[int, ...] = ()
    async_dump: bool = False
    extra_info: bool = True
    scope: tuple[str, ...] = ()
    module_or_api_list: tuple[str, ...] = ()
    tensor_list: tuple[str, ...] = ()
    data_mode: tuple[str, ...] = ("all",)
    summary_mode: Literal["statistics", "md5", "xor"] = "statistics"

    def __post_init__(self) -> None:
        if not self.steps or min(self.steps) < 0:
            raise ValueError("msProbe steps must contain non-negative values")
        if tuple(sorted(set(self.steps))) != self.steps:
            raise ValueError("msProbe steps must be sorted and unique")
        if any(rank < 0 for rank in self.ranks):
            raise ValueError("msProbe ranks must be non-negative")
        if tuple(sorted(set(self.ranks))) != self.ranks:
            raise ValueError("msProbe ranks must be sorted and unique")
        if not self.data_mode:
            raise ValueError("msProbe data_mode must not be empty")
        if self.async_dump and self.summary_mode == "md5":
            raise ValueError("msProbe async dump does not support md5 summaries")
        if self.async_dump and self.task == "tensor" and not self.module_or_api_list:
            raise ValueError("msProbe async tensor dump requires a non-empty list")
        if self.task == "nan_check" and self.level != "L1":
            raise ValueError("msProbe nan_check requires dump level L1")

    def official_config(self, dump_path: Path) -> dict[str, Any]:
        """Return the JSON object consumed by ``PrecisionDebugger``."""

        task_options: dict[str, Any] = {
            "scope": list(self.scope),
            "list": list(self.module_or_api_list),
            "data_mode": list(self.data_mode),
        }
        if self.task in {"statistics", "tensor"}:
            task_options["summary_mode"] = self.summary_mode
        if self.task == "statistics":
            task_options["tensor_list"] = list(self.tensor_list)
        return {
            "task": self.task,
            "dump_path": str(dump_path.resolve()),
            "rank": list(self.ranks),
            "step": list(self.steps),
            "level": self.level,
            "async_dump": self.async_dump,
            "extra_info": self.extra_info,
            self.task: task_options,
        }


@dataclass(frozen=True)
class MsProbeCompileConfig:
    """Options owned by msProbe's ``PrecisionChecker`` implementation."""

    backend: str = "inductor"
    dump_graphs: bool = True
    capture_input: bool = True
    policy: Literal["glm5-block", "children"] = "glm5-block"
    children_depth: int = 1

    def __post_init__(self) -> None:
        if not self.backend.strip():
            raise ValueError("compile backend must not be empty")
        if self.children_depth < 0:
            raise ValueError("children_depth must be non-negative")


@dataclass(frozen=True)
class MsProbeMonitorConfig:
    """Options consumed by the public ``TrainerMonitorV2`` API."""

    ranks: tuple[int, ...] = (0,)
    start_step: int = 0
    stop_step: int | None = None
    step_interval: int = 1
    step_count_per_record: int = 10
    collect_times: int = 100
    ops: tuple[str, ...] = ("min", "max", "mean", "norm", "nans")
    module: bool = False
    weight_grad: bool = True
    optimizer: bool = False
    param: bool = False
    cc: bool = False
    module_targets: tuple[str, ...] = ()
    monitor_mbs_grad: bool = False

    def __post_init__(self) -> None:
        if any(rank < 0 for rank in self.ranks):
            raise ValueError("monitor ranks must be non-negative")
        if tuple(sorted(set(self.ranks))) != self.ranks:
            raise ValueError("monitor ranks must be sorted and unique")
        if self.start_step < 0:
            raise ValueError("monitor start_step must be non-negative")
        if self.stop_step is not None and self.stop_step <= self.start_step:
            raise ValueError("monitor stop_step must be greater than start_step")
        if self.step_interval < 1:
            raise ValueError("monitor step_interval must be positive")
        if self.step_count_per_record < 1:
            raise ValueError("monitor step_count_per_record must be positive")
        if self.collect_times < 1:
            raise ValueError("monitor collect_times must be positive")
        supported_ops = {"min", "max", "mean", "norm", "nans"}
        if not self.ops or not set(self.ops).issubset(supported_ops):
            raise ValueError(
                "monitor ops must be a non-empty subset of "
                f"{sorted(supported_ops)}"
            )
        if not any(
            (self.module, self.weight_grad, self.optimizer, self.param, self.cc)
        ):
            raise ValueError("at least one monitor module must be enabled")

    def official_config(self, output_dir: Path) -> dict[str, Any]:
        """Return the documented Monitor V2 JSON configuration."""

        common = {"ops": list(self.ops), "eps": 1e-8}
        monitors: dict[str, dict[str, Any]] = {
            "module": {
                "enabled": self.module,
                "targets": list(self.module_targets),
                **common,
            },
            "weight_grad": {
                "enabled": self.weight_grad,
                "monitor_mbs_grad": self.monitor_mbs_grad,
                **common,
            },
            "optimizer": {
                "enabled": self.optimizer,
                "mv_distribution": True,
                **common,
            },
            "param": {
                "enabled": self.param,
                "param_distribution": True,
                **common,
            },
            "cc": {"enabled": self.cc, **common},
        }
        result: dict[str, Any] = {
            "framework": "pytorch",
            "output_dir": str(output_dir.resolve()),
            "rank": list(self.ranks),
            "start_step": self.start_step,
            "step_interval": self.step_interval,
            "step_count_per_record": self.step_count_per_record,
            "collect_times": self.collect_times,
            "format": "csv",
            # We call monitor.step() exactly once after Trainer.train_step().
            "patch_optimizer_step": False,
            "monitors": monitors,
        }
        if self.stop_step is not None:
            result["stop_step"] = self.stop_step
        return result


@dataclass(frozen=True)
class MindStudioExperimentConfig:
    """Official-tool experiment plus shared TorchTitan input contract."""

    name: str
    workflow: OfficialWorkflow
    reference: TrainingEndpoint
    candidate: TrainingEndpoint
    training: FormalTrainingConfig = field(
        default_factory=lambda: FormalTrainingConfig(
            steps=1,
            local_batch_size=8,
            global_batch_size=64,
            sequence_length=128,
        )
    )
    dump: MsProbeDumpConfig = field(default_factory=MsProbeDumpConfig)
    compile: MsProbeCompileConfig = field(default_factory=MsProbeCompileConfig)
    monitor: MsProbeMonitorConfig = field(default_factory=MsProbeMonitorConfig)
    fixture_root: str = "mindstudio_fixtures"
    run_root: str = "mindstudio_runs"
    artifact_root: str = "mindstudio_artifacts"
    report_root: str = "mindstudio_reports"

    def __post_init__(self) -> None:
        for name in ("fixture_root", "run_root", "artifact_root", "report_root"):
            _validate_repository_root(name, getattr(self, name))
        if self.workflow in {"config-check", "migration", "monitor"}:
            if self.reference.topology != self.candidate.topology:
                raise ValueError("migration requires equal reference/candidate topology")
        elif self.workflow == "compile" and (
            self.reference.device_type != self.candidate.device_type
        ):
            raise ValueError("compile accuracy requires one device type")
        if max(self.dump.steps) >= self.training.steps:
            raise ValueError(
                "training steps must be greater than every zero-based dump step"
            )
        if self.workflow == "monitor" and (
            self.monitor.start_step >= self.training.steps
        ):
            raise ValueError(
                "monitor start_step must be smaller than training steps"
            )

    @property
    def precision_name(self) -> str:
        name = self.training.precision_name
        return {
            "mixed-bfloat16": "bf16",
            "mixed-float32": "fp32",
        }.get(name, name)

    @property
    def storage_base_name(self) -> str:
        if self.workflow in {"config-check", "migration"}:
            prefix = (
                f"{self.workflow}-{self.reference.device_type}-"
                f"{self.candidate.device_type}"
            )
        elif self.workflow == "compile":
            prefix = f"compile-{self.candidate.device_type}-{self.compile.backend}"
        else:
            prefix = (
                f"monitor-{self.reference.device_type}-"
                f"{self.candidate.device_type}"
            )
        return slug(
            f"{prefix}-{self.precision_name}-random-s{self.training.steps}-"
            f"b{self.training.global_batch_size}-seq{self.training.sequence_length}-"
            f"seed{self.training.seed}"
        )

    @property
    def identity(self) -> dict[str, Any]:
        def endpoint_value(endpoint: TrainingEndpoint) -> dict[str, Any]:
            return {
                "device_type": endpoint.device_type,
                "num_nodes": endpoint.num_nodes,
                "environment": endpoint.environment,
                "extra_args": endpoint.extra_args,
            }

        identity = {
            "schema": "torchtitan.glm5_2.mindstudio_experiment",
            "schema_version": 1,
            "name": self.name,
            "workflow": self.workflow,
            "training": asdict(self.training),
            "dump": asdict(self.dump),
            "compile": asdict(self.compile),
            "reference": endpoint_value(self.reference),
            "candidate": endpoint_value(self.candidate),
        }
        # Monitor settings are irrelevant to the existing dump, compile, and
        # config-check workflows. Keeping them out of those identities avoids
        # invalidating previously captured experiments when Monitor V2 support
        # is added or its defaults evolve.
        if self.workflow == "monitor":
            identity["monitor"] = asdict(self.monitor)
        return identity

    @property
    def storage_name(self) -> str:
        return config_name(self.storage_base_name, self.identity)

    def formal_fixture_config(self) -> FormalExperimentConfig:
        """Build the existing fixed-input producer without reusing its verdict."""

        return FormalExperimentConfig(
            name=self.name,
            kind=(
                "migration"
                if self.workflow in {"config-check", "migration", "monitor"}
                else "self_consistency"
            ),
            reference=self.reference,
            candidate=self.candidate,
            training=self.training,
            fixture_root=self.fixture_root,
            artifact_root=self.artifact_root,
            report_root=self.report_root,
            run_root=self.run_root,
            fixture_name=self.storage_name,
            storage_name_override=self.storage_base_name,
            topology_subdirectory=True,
        )
