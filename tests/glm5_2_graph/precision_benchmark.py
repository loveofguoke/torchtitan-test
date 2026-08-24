#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""NPU eager-versus-graph precision entry point."""

from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_combination.combination_benchmark import (  # noqa: E402
    CONFIG as COMBINATION_CONFIG,
    TOPOLOGIES,
    TOPOLOGY_NAMES,
)
from tests.glm5_2_combination.workflow import run_combination_cli  # noqa: E402


# Graph-mode validation compares two executions on the same NPU backend. The
# central combination entry point composes the same NPU graph contract with
# optional performance profiling and distributed candidate topologies.
CONFIG = replace(
    COMBINATION_CONFIG,
    name="npu-graph",
    kind="self_consistency",
    reference=replace(
        COMBINATION_CONFIG.reference,
        name="npu-eager-reference",
        device_type="npu",
    ),
    candidate=replace(
        COMBINATION_CONFIG.candidate,
        name="npu-graph-candidate",
        device_type="npu",
    ),
)


if __name__ == "__main__":
    run_combination_cli(
        CONFIG,
        __file__,
        topologies=TOPOLOGIES,
        topology_names=TOPOLOGY_NAMES,
        default_topology="single",
        default_objectives="precision",
    )
