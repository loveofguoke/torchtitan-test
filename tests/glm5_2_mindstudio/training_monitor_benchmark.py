#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Official msProbe Monitor V2 training-status workflow."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_common.topology import standard_topologies  # noqa: E402
from tests.glm5_2_mindstudio.config import (  # noqa: E402
    MindStudioExperimentConfig,
    MsProbeMonitorConfig,
)
from tests.glm5_2_mindstudio.workflow import run_mindstudio_cli  # noqa: E402
from tests.glm5_2_precision.workflow import (  # noqa: E402
    FormalTrainingConfig,
    TrainingEndpoint,
)


TOPOLOGIES = standard_topologies()
ALL_DEVICES = "0,1,2,3,4,5,6,7"

CONFIG = MindStudioExperimentConfig(
    name="glm5-2-official-training-monitor",
    workflow="monitor",
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
    training=FormalTrainingConfig(
        steps=100,
        local_batch_size=8,
        global_batch_size=64,
        sequence_length=128,
        seed=61,
        deterministic=True,
        training_dtype="float32",
        mixed_precision_param="bfloat16",
        checkpoint_kind="random_seed",
    ),
    monitor=MsProbeMonitorConfig(
        ranks=(0,),
        start_step=0,
        step_interval=1,
        step_count_per_record=10,
        collect_times=100,
        weight_grad=True,
    ),
    formal_precision_reports=(
        "precision_reports/migration-cuda-npu-*-s5000-*",
    ),
)


if __name__ == "__main__":
    run_mindstudio_cli(CONFIG, __file__)
