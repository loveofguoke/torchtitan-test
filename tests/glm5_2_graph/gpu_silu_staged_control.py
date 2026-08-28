#!/usr/bin/env python3
"""Parameterized BF16 symmetric materialization control."""

from dataclasses import replace
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_combination.workflow import run_combination_cli  # noqa: E402
from tests.glm5_2_graph.gpu_silu_symmetric_probe import (  # noqa: E402
    SYMMETRIC_CONFIG,
    TOPOLOGIES,
    TOPOLOGY_NAMES,
)


VARIANT_POINTS = {
    "silu-product": "silu,product",
    "silu-product-down": "silu,product,down",
    "all": "w1,w3,silu,product,down",
}
LENGTH_STEPS = {"step1": 1, "probe": 10}
variant = os.environ.get("GLM5_GPU_SILU_STAGE_VARIANT", "silu-product")
length = os.environ.get("GLM5_GPU_SILU_STAGE_LENGTH", "step1")
if variant not in VARIANT_POINTS:
    raise ValueError(f"unknown staged SiLU variant: {variant}")
if length not in LENGTH_STEPS:
    raise ValueError(f"unknown staged SiLU length: {length}")

_POINTS = {
    "GLM5_GPU_DENSE_SILU_MATERIALIZATION_POINTS": VARIANT_POINTS[variant]
}
STAGED_CONFIG = replace(
    SYMMETRIC_CONFIG,
    name=f"cuda-bf16-{variant}-{length}",
    reference=replace(
        SYMMETRIC_CONFIG.reference,
        environment={**SYMMETRIC_CONFIG.reference.environment, **_POINTS},
    ),
    candidate=replace(
        SYMMETRIC_CONFIG.candidate,
        environment={**SYMMETRIC_CONFIG.candidate.environment, **_POINTS},
    ),
    training=replace(
        SYMMETRIC_CONFIG.training,
        steps=LENGTH_STEPS[length],
    ),
)


if __name__ == "__main__":
    run_combination_cli(
        STAGED_CONFIG,
        __file__,
        topologies=TOPOLOGIES,
        topology_names=TOPOLOGY_NAMES,
        default_topology="single",
        default_objectives="precision",
    )
