#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Single-topology graph precision entry point."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_combination.combination_benchmark import (  # noqa: E402
    CONFIG,
    TOPOLOGIES,
)
from tests.glm5_2_combination.workflow import run_combination_cli  # noqa: E402


if __name__ == "__main__":
    run_combination_cli(
        CONFIG,
        __file__,
        topologies=TOPOLOGIES,
        topology_names=("single",),
    )
