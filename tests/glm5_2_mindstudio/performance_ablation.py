#!/usr/bin/env python3
"""Validate and report a profiler-off performance ablation."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_performance.comparison import run_cli  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run_cli())
