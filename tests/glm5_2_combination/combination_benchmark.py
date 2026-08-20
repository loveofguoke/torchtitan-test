#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""GPU/NPU graph, distributed, precision, and performance experiment."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_combination.workflow import run_combination_cli  # noqa: E402
from tests.glm5_2_common.topology import (  # noqa: E402
    DISTRIBUTED_TOPOLOGY_NAMES,
    standard_topologies,
)
from tests.glm5_2_precision.workflow import (  # noqa: E402
    FormalExperimentConfig,
    FormalTrainingConfig,
    TrainingEndpoint,
)


TOPOLOGIES = standard_topologies()
TOPOLOGY_NAMES = ("single", *DISTRIBUTED_TOPOLOGY_NAMES)
ALL_DEVICES = "0,1,2,3,4,5,6,7"
CONFIG = FormalExperimentConfig(
    name="combined",
    kind="migration",
    reference=TrainingEndpoint(
        name="gpu-reference",
        device_type="cuda",
        visible_devices=ALL_DEVICES,
        topology=TOPOLOGIES["single"],
        repeats=2,
    ),
    candidate=TrainingEndpoint(
        name="npu-candidate",
        device_type="npu",
        visible_devices=ALL_DEVICES,
        topology=TOPOLOGIES["single"],
        repeats=2,
    ),
    training=FormalTrainingConfig(
        steps=5000,
        local_batch_size=8,
        global_batch_size=64,
        sequence_length=128,
        seed=61,
        deterministic=True,
        training_dtype="float32",
        mixed_precision_param="bfloat16",
        checkpoint_kind="random_seed",
    ),
)


if __name__ == "__main__":
    run_combination_cli(
        CONFIG,
        __file__,
        topologies=TOPOLOGIES,
        topology_names=TOPOLOGY_NAMES,
    )
