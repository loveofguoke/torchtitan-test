#!/usr/bin/env python3
"""Official MindStudio performance entry for GLM-5.2 training."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_performance.config import PerformanceConfig  # noqa: E402
from tests.glm5_2_performance.workflow import run_profiler_cli  # noqa: E402


CONFIG = PerformanceConfig(
    name="glm5-mindstudio-performance",
    device="npu",
    topology="single",
    preset="overview",
    collector="msprof",
    steps=30,
    skip_steps=10,
    warmup_steps=2,
    active_steps=5,
    local_batch_size=8,
    global_batch_size=64,
    sequence_length=128,
    run_root="mindstudio_runs",
    artifact_root="mindstudio_artifacts",
    report_root="mindstudio_reports",
)


if __name__ == "__main__":
    run_profiler_cli(CONFIG, __file__)
