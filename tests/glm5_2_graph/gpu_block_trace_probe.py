#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Short CUDA eager/Inductor run with Block 0 tensor tracing enabled."""

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


_MARKER = {"GLM5_GPU_DIAGNOSTIC": "block-trace-v1"}
TRACE_CONFIG = replace(
    CONFIG,
    name="cuda-graph-block-trace",
    reference=replace(
        CONFIG.reference,
        entry_module="tests.glm5_2_graph.gpu_diagnostic_capture",
        environment={**CONFIG.reference.environment, **_MARKER},
    ),
    candidate=replace(
        CONFIG.candidate,
        entry_module="tests.glm5_2_graph.gpu_diagnostic_capture",
        environment={**CONFIG.candidate.environment, **_MARKER},
    ),
    training=replace(
        CONFIG.training,
        steps=20,
        extra_args=(
            "--training.disable_cuda_graphs",
            "--lr_scheduler.decay_ratio=0.0",
        ),
    ),
)


if __name__ == "__main__":
    run_combination_cli(
        TRACE_CONFIG,
        __file__,
        topologies=TOPOLOGIES,
        topology_names=TOPOLOGY_NAMES,
        default_topology="single",
        default_objectives="precision",
    )
