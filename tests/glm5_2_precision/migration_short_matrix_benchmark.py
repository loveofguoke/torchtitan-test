#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Short GPU-to-NPU precision matrix across eight-device topologies."""

from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_precision.migration_benchmark import (  # noqa: E402
    CONFIG as BASE_CONFIG,
    TOPOLOGIES,
)
from tests.glm5_2_precision.workflow import run_formal_cli  # noqa: E402


CONFIG = replace(
    BASE_CONFIG,
    name="glm5-2-gpu-npu-training-short-matrix",
    training=replace(BASE_CONFIG.training, steps=10),
    standard=replace(
        BASE_CONFIG.standard,
        migration=replace(
            BASE_CONFIG.standard.migration,
            warmup_steps=0,
            minimum_observations=10,
        ),
    ),
    artifact_root="precision_short_matrix_artifacts",
    report_root="precision_short_matrix_reports",
    run_root="precision_short_matrix_runs",
)


def _config_for_requested_topology():
    """Choose enough local microbatches for 1F1B pipeline schedules."""

    topology = None
    for index, argument in enumerate(sys.argv):
        if argument == "--topology" and index + 1 < len(sys.argv):
            topology = sys.argv[index + 1]
        elif argument.startswith("--topology="):
            topology = argument.split("=", 1)[1]
    local_batch_size = {
        "pp8": 8,
        "fsdp2-pp4": 4,
    }.get(topology, CONFIG.training.local_batch_size)
    return replace(
        CONFIG,
        training=replace(CONFIG.training, local_batch_size=local_batch_size),
    )


if __name__ == "__main__":
    run_formal_cli(_config_for_requested_topology(), __file__, topologies=TOPOLOGIES)
