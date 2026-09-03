#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Official msProbe same-GPU reference/candidate validation workflow."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_common.topology import standard_topologies  # noqa: E402
from tests.glm5_2_mindstudio.config import (  # noqa: E402
    MindStudioExperimentConfig,
    MsProbeDumpConfig,
)
from tests.glm5_2_mindstudio.workflow import run_mindstudio_cli  # noqa: E402
from tests.glm5_2_precision.workflow import (  # noqa: E402
    FormalTrainingConfig,
    TrainingEndpoint,
)


TOPOLOGIES = standard_topologies()
ALL_DEVICES = "0,1,2,3,4,5,6,7"

CONFIG = MindStudioExperimentConfig(
    name="glm5-2-official-accuracy-self",
    workflow="migration",
    reference=TrainingEndpoint(
        name="gpu-reference",
        device_type="cuda",
        visible_devices=ALL_DEVICES,
        topology=TOPOLOGIES["single"],
        repeats=1,
    ),
    candidate=TrainingEndpoint(
        name="gpu-candidate",
        device_type="cuda",
        visible_devices=ALL_DEVICES,
        topology=TOPOLOGIES["single"],
        repeats=1,
    ),
    training=FormalTrainingConfig(
        steps=2,
        local_batch_size=8,
        global_batch_size=64,
        sequence_length=128,
        seed=61,
        deterministic=True,
        training_dtype="float32",
        mixed_precision_param="bfloat16",
        checkpoint_kind="random_seed",
    ),
    dump=MsProbeDumpConfig(
        task="statistics",
        level="L0",
        steps=(0, 1),
        summary_mode="statistics",
    ),
)


if __name__ == "__main__":
    run_mindstudio_cli(CONFIG, __file__)
