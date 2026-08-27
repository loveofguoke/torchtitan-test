#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Single-CUDA eager-versus-Inductor precision benchmark."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_combination.workflow import run_combination_cli  # noqa: E402
from tests.glm5_2_common.topology import standard_topologies  # noqa: E402
from tests.glm5_2_precision.workflow import (  # noqa: E402
    FormalExperimentConfig,
    FormalTrainingConfig,
    TrainingEndpoint,
)


TOPOLOGIES = standard_topologies()
TOPOLOGY_NAMES = ("single",)
CONFIG = FormalExperimentConfig(
    name="cuda-graph",
    kind="self_consistency",
    reference=TrainingEndpoint(
        name="cuda-eager-reference",
        device_type="cuda",
        visible_devices="0",
        topology=TOPOLOGIES["single"],
        repeats=2,
    ),
    candidate=TrainingEndpoint(
        name="cuda-inductor-candidate",
        device_type="cuda",
        visible_devices="0",
        topology=TOPOLOGIES["single"],
        repeats=2,
    ),
    training=FormalTrainingConfig(
        steps=1000,
        local_batch_size=1,
        global_batch_size=1,
        sequence_length=32,
        seed=61,
        deterministic=True,
        deterministic_warn_only=False,
        training_dtype="float32",
        mixed_precision_param="bfloat16",
        checkpoint_kind="random_seed",
        # Keep whole-step CUDA Graph out of this experiment. The only intended
        # execution difference is eager versus torch.compile/Inductor.
        extra_args=("--training.disable_cuda_graphs",),
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
