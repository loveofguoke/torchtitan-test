#!/usr/bin/env python3
"""Run official msMemScope memory tuning around TorchTitan training."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_mindstudio.specialized_tuning import run_memory_cli  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run_memory_cli())
