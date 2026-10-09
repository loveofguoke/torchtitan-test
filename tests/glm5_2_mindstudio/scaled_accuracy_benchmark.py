#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Long-run accuracy workflow for the scaled GLM-5.2 validation model."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_common.topology import standard_topologies  # noqa: E402
from tests.glm5_2_mindstudio.config import (  # noqa: E402
    MindStudioExperimentConfig,
    MsProbeDumpConfig,
    MsProbeMonitorConfig,
)
from tests.glm5_2_mindstudio.workflow import run_mindstudio_cli  # noqa: E402
from tests.glm5_2_precision.workflow import (  # noqa: E402
    FormalTrainingConfig,
    TrainingEndpoint,
)


TOPOLOGIES = standard_topologies()
ALL_DEVICES = "0,1,2,3,4,5,6,7"
TRAINING = FormalTrainingConfig(
    config="glm5_2_scaled_debugmodel",
    steps=1000,
    local_batch_size=8,
    global_batch_size=64,
    sequence_length=512,
    seed=61,
    deterministic=True,
    training_dtype="float32",
    mixed_precision_param="bfloat16",
    checkpoint_kind="random_seed",
)

_UNSCOPED = MindStudioExperimentConfig(
    name="glm5-2-scaled-long-run-accuracy",
    workflow="migration",
    reference=TrainingEndpoint(
        name="gpu-reference",
        device_type="cuda",
        visible_devices=ALL_DEVICES,
        topology=TOPOLOGIES["single"],
        repeats=1,
    ),
    candidate=TrainingEndpoint(
        name="npu-candidate",
        device_type="npu",
        visible_devices=ALL_DEVICES,
        topology=TOPOLOGIES["single"],
        repeats=1,
    ),
    training=TRAINING,
    dump=MsProbeDumpConfig(
        task="statistics",
        level="L0",
        steps=(0, 1),
        summary_mode="statistics",
    ),
)
BASE_CONFIG = replace(
    _UNSCOPED,
    experiment_storage_name=_UNSCOPED.storage_name,
)
STAGE_CONFIGS = {
    "config-check": replace(
        BASE_CONFIG,
        workflow="config-check",
        training=replace(TRAINING, steps=1),
        fixture_training=TRAINING,
        output_subdirectory="diagnostics/configuration-check",
        owns_fixture=False,
    ),
    "observation": replace(BASE_CONFIG, workflow="observation"),
    "monitor": replace(
        BASE_CONFIG,
        workflow="monitor",
        training=replace(TRAINING, steps=100),
        fixture_training=TRAINING,
        monitor=MsProbeMonitorConfig(
            ranks=(0,),
            start_step=0,
            step_interval=1,
            step_count_per_record=10,
            collect_times=100,
            weight_grad=True,
        ),
    ),
    "dump": BASE_CONFIG,
}


def _select_stage(arguments: list[str]) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--stage",
        choices=tuple(STAGE_CONFIGS),
        default="observation",
    )
    parsed, remaining = parser.parse_known_args(arguments)
    return parsed.stage, remaining


if __name__ == "__main__":
    stage, arguments = _select_stage(sys.argv[1:])
    sys.argv = [sys.argv[0], *arguments]
    run_mindstudio_cli(STAGE_CONFIGS[stage], __file__)
