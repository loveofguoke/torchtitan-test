#!/usr/bin/env python3
"""1000-step BF16 eager versus SiLU-materialized Inductor experiment."""

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


_CAPTURE = "tests.glm5_2_graph.gpu_diagnostic_capture"
_MARKER = {"GLM5_GPU_MATERIALIZE_DENSE_SILU": "1"}
MATERIALIZATION_CONFIG = replace(
    CONFIG,
    name="cuda-bf16-silu-materialization",
    reference=replace(CONFIG.reference, entry_module=_CAPTURE),
    candidate=replace(
        CONFIG.candidate,
        entry_module=_CAPTURE,
        environment={**CONFIG.candidate.environment, **_MARKER},
    ),
)


if __name__ == "__main__":
    run_combination_cli(
        MATERIALIZATION_CONFIG,
        __file__,
        topologies=TOPOLOGIES,
        topology_names=TOPOLOGY_NAMES,
        default_topology="single",
        default_objectives="precision",
    )
