#!/usr/bin/env python3
"""Run official msOpProf operator tuning without system-profiler coupling."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_mindstudio.specialized_tuning import run_operator_cli  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run_operator_cli())
