#!/usr/bin/env python3
"""10-step BF16 eager versus SiLU-materialized Inductor experiment."""

from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_combination.workflow import run_combination_cli  # noqa: E402
from tests.glm5_2_graph.gpu_silu_materialization_benchmark import (  # noqa: E402
    MATERIALIZATION_CONFIG,
    TOPOLOGIES,
    TOPOLOGY_NAMES,
)


PROBE_CONFIG = replace(
    MATERIALIZATION_CONFIG,
    training=replace(MATERIALIZATION_CONFIG.training, steps=10),
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
