#!/usr/bin/env python3
"""Configurable CUDA profiler benchmark for GLM-5.2."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SUITE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SUITE))

from gpu_config import GpuPerformanceConfig  # noqa: E402
from gpu_workflow import run_gpu_profiler_cli  # noqa: E402


CONFIG = GpuPerformanceConfig(
    name="glm5-gpu-profile",
    steps=30,
    skip_steps=10,
    warmup_steps=2,
    active_steps=5,
    local_batch_size=8,
    global_batch_size=64,
    sequence_length=128,
)


if __name__ == "__main__":
    run_gpu_profiler_cli(CONFIG, __file__)
