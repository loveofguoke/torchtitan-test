#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Fast single-CUDA eager-versus-Inductor precision probe."""

from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_combination.workflow import run_combination_cli  # noqa: E402
from tests.glm5_2_graph.gpu_precision_benchmark import (  # noqa: E402
    CONFIG,
    TOPOLOGIES,
    TOPOLOGY_NAMES,
)


PROBE_CONFIG = replace(
    CONFIG,
    training=replace(CONFIG.training, steps=10),
)


if __name__ == "__main__":
    run_combination_cli(
        PROBE_CONFIG,
        __file__,
        topologies=TOPOLOGIES,
        topology_names=TOPOLOGY_NAMES,
        default_topology="single",
        default_objectives="precision",
    )
