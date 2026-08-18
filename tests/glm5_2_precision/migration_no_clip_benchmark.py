#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""GPU-to-NPU diagnostic benchmark with gradient clipping disabled."""

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
    name="glm5-2-gpu-npu-training-no-clip",
    training=replace(
        BASE_CONFIG.training,
        steps=400,
        extra_args=(*BASE_CONFIG.training.extra_args, "--training.max_norm=inf"),
    ),
    standard=replace(
        BASE_CONFIG.standard,
        migration=replace(
            BASE_CONFIG.standard.migration,
            minimum_observations=400,
        ),
    ),
    artifact_root="precision_no_clip_artifacts",
    report_root="precision_no_clip_reports",
    run_root="precision_no_clip_runs",
)


if __name__ == "__main__":
    run_formal_cli(CONFIG, __file__, topologies=TOPOLOGIES)
