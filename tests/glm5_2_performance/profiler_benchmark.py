#!/usr/bin/env python3
"""Configurable GLM-5.2 profiler benchmark for single or distributed runs."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_performance.config import PerformanceConfig  # noqa: E402
from tests.glm5_2_performance.workflow import run_profiler_cli  # noqa: E402


CONFIG = PerformanceConfig(
    name="glm5-profile",
    topology="single",
    preset="overview",
    steps=30,
    skip_steps=10,
    warmup_steps=2,
    active_steps=5,
    local_batch_size=8,
    global_batch_size=64,
    sequence_length=128,
)


if __name__ == "__main__":
    run_profiler_cli(CONFIG, __file__)
