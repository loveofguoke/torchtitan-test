#!/usr/bin/env python3
"""Small, fast GLM-5.2 profiler probe for GPU or Ascend NPU."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_performance.config import PerformanceConfig  # noqa: E402
from tests.glm5_2_performance.workflow import run_profiler_cli  # noqa: E402


CONFIG = PerformanceConfig(
    name="glm5-probe",
    topology="single",
    preset="overview",
    steps=10,
    skip_steps=2,
    warmup_steps=1,
    active_steps=3,
    local_batch_size=1,
    global_batch_size=1,
    sequence_length=32,
)


if __name__ == "__main__":
    run_profiler_cli(CONFIG, __file__)
