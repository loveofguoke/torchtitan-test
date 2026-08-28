#!/usr/bin/env python3
"""10-step BF16 eager/Inductor control with SiLU materialized on both sides."""

from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_combination.workflow import run_combination_cli  # noqa: E402
from tests.glm5_2_graph.gpu_silu_materialization_probe import (  # noqa: E402
    PROBE_CONFIG,
    TOPOLOGIES,
    TOPOLOGY_NAMES,
)


_MARKER = {"GLM5_GPU_MATERIALIZE_DENSE_SILU": "1"}
SYMMETRIC_CONFIG = replace(
    PROBE_CONFIG,
    name="cuda-bf16-silu-materialization-symmetric-control",
    reference=replace(
        PROBE_CONFIG.reference,
        environment={**PROBE_CONFIG.reference.environment, **_MARKER},
    ),
)


if __name__ == "__main__":
    run_combination_cli(
        SYMMETRIC_CONFIG,
        __file__,
        topologies=TOPOLOGIES,
        topology_names=TOPOLOGY_NAMES,
        default_topology="single",
        default_objectives="precision",
    )
