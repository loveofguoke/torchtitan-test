#!/usr/bin/env python3
"""Profile selected CUDA kernels with NVIDIA Nsight Compute."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_nvidia.ncu_workflow import run_cli  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run_cli())
