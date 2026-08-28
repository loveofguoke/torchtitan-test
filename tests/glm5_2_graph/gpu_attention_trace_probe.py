#!/usr/bin/env python3
"""One-step FP32 eager/Inductor Block 0 attention localization."""

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
_MARKER = {"GLM5_GPU_DIAGNOSTIC": "attention-internal-trace-v1"}
ATTENTION_TRACE_CONFIG = replace(
    CONFIG,
    name="cuda-fp32-attention-internal-trace",
    reference=replace(
        CONFIG.reference,
        entry_module=_CAPTURE,
        environment={**CONFIG.reference.environment, **_MARKER},
    ),
    candidate=replace(
        CONFIG.candidate,
        entry_module=_CAPTURE,
        environment={**CONFIG.candidate.environment, **_MARKER},
    ),
    training=replace(
        CONFIG.training,
        steps=1,
        extra_args=(
            "--training.disable_cuda_graphs",
            "--lr_scheduler.decay_ratio=0.0",
        ),
    ),
)


if __name__ == "__main__":
    run_combination_cli(
        ATTENTION_TRACE_CONFIG,
        __file__,
        topologies=TOPOLOGIES,
        topology_names=TOPOLOGY_NAMES,
        default_topology="single",
        default_objectives="precision",
    )
