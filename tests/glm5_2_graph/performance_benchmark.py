#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""NPU eager-versus-graph performance entry point."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_graph.precision_benchmark import (
    CONFIG,
    TOPOLOGIES,
    TOPOLOGY_NAMES,
)
from tests.glm5_2_combination.workflow import run_combination_cli


if __name__ == "__main__":
    run_combination_cli(
        CONFIG,
        __file__,
        topologies=TOPOLOGIES,
        topology_names=TOPOLOGY_NAMES,
        default_topology="single",
        default_objectives="performance",
    )
