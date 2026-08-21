#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Fast single-topology graph probe using the combination executor."""

from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_graph.precision_benchmark import CONFIG, TOPOLOGIES
from tests.glm5_2_combination.workflow import run_combination_cli


PROBE_CONFIG = replace(
    CONFIG,
    training=replace(
        CONFIG.training,
        steps=10,
        local_batch_size=1,
        global_batch_size=1,
        sequence_length=32,
    ),
)


if __name__ == "__main__":
    run_combination_cli(
        PROBE_CONFIG,
        __file__,
        topologies=TOPOLOGIES,
        topology_names=("single",),
    )
