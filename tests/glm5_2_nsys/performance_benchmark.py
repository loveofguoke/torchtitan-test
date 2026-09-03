#!/usr/bin/env python3
"""Profile GLM CUDA training with NVIDIA Nsight Systems."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_nsys.workflow import run_cli  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run_cli())
