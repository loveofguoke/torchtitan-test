# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Independent feature configurations for combined training experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tests.glm5_2_common.execution import TrainingFeature
from tests.glm5_2_performance.config import ProfilerPreset


GraphMode = Literal["eager", "inductor", "npugraphs"]


@dataclass(frozen=True)
class GraphFeatureConfig:
    mode: GraphMode = "eager"
    components: tuple[str, ...] = ("model",)
    diagnostics: bool = False

    def feature(self, *, device_type: str) -> TrainingFeature:
        if self.mode == "eager":
            return TrainingFeature(name="graph:eager", metadata={"mode": "eager"})
        if self.mode == "npugraphs" and device_type != "npu":
            raise ValueError("npugraphs is available only for an NPU endpoint")
        if self.mode == "npugraphs" and self.components != ("model",):
            raise ValueError("npugraphs supports model compilation only")
        environment = (
            {"TORCH_LOGS": "graph_breaks,recompiles,dynamic"}
            if self.diagnostics
            else {}
        )
        return TrainingFeature(
            name=f"graph:{self.mode}",
            arguments=(
                "--compile.enable",
                f"--compile.components={','.join(self.components)}",
                f"--compile.backend={self.mode}",
            ),
            environment=environment,
            metadata={"mode": self.mode, "components": list(self.components)},
        )


@dataclass(frozen=True)
class ProfilerFeatureConfig:
    preset: ProfilerPreset
    skip_steps: int = 10
    warmup_steps: int = 1
    active_steps: int = 3

    def __post_init__(self) -> None:
        if self.skip_steps < 0 or self.warmup_steps < 0 or self.active_steps < 1:
            raise ValueError("invalid profiler schedule")

    def feature(self) -> TrainingFeature:
        frequency = self.warmup_steps + self.active_steps
        return TrainingFeature(
            name=f"profiler:{self.preset.name}",
            arguments=(
                "--profiler.enable_profiling",
                "--profiler.save_traces_folder=profiling/traces",
                f"--profiler.profile_freq={frequency}",
                "--profiler.profiler_repeat=1",
                f"--profiler.profiler_skip_first={self.skip_steps}",
                f"--profiler.profiler_warmup={self.warmup_steps}",
                f"--profiler.profiler_active={self.active_steps}",
            ),
            environment={
                **self.preset.environment(),
                "GLM5_PERFORMANCE_PROFILE_RANKS": self.preset.profile_ranks,
            },
            metadata={
                "preset": self.preset.name,
                "skip_steps": self.skip_steps,
                "warmup_steps": self.warmup_steps,
                "active_steps": self.active_steps,
            },
        )
