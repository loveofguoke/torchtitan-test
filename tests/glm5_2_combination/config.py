# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Independent feature configurations for combined training experiments.

These objects translate one feature into ``TrainingFeature`` without knowing
which other features are selected. ``ExecutionPlan`` is the only place allowed
to merge them and rejects duplicate TorchTitan options or conflicting runtime
environment, preserving standalone and combined behavior equivalence.
"""

from __future__ import annotations

from dataclasses import dataclass

from tests.glm5_2_common.execution import TrainingFeature
from tests.glm5_2_graph.config import GraphFeatureConfig, GraphMode
from tests.glm5_2_performance.config import ProfilerPreset


__all__ = ["GraphFeatureConfig", "GraphMode", "ProfilerFeatureConfig"]


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
