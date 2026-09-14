#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Manage a stateful MindStudio accuracy diagnosis case."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_mindstudio.accuracy_diagnostics import (  # noqa: E402
    run_diagnostic_cli,
)


if __name__ == "__main__":
    run_diagnostic_cli(Path(__file__).resolve().parents[2])
