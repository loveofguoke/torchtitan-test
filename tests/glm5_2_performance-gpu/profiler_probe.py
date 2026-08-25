#!/usr/bin/env python3
"""Small CUDA profiler smoke run for GLM-5.2."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SUITE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SUITE))

from gpu_config import GpuPerformanceConfig  # noqa: E402
from gpu_workflow import run_gpu_profiler_cli  # noqa: E402


CONFIG = GpuPerformanceConfig(
    name="glm5-gpu-probe",
    steps=10,
    skip_steps=2,
    warmup_steps=1,
    active_steps=3,
    local_batch_size=1,
    global_batch_size=1,
    sequence_length=32,
)


if __name__ == "__main__":
    run_gpu_profiler_cli(CONFIG, __file__)
