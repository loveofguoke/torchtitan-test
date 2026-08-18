#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""GPU to NPU formal precision benchmark with selectable parallel topology."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_precision.standards import (  # noqa: E402
    AnyOfErrorLimit,
    MigrationStandard,
    PrecisionStandard,
    QuantileLimit,
)
from tests.glm5_2_precision.workflow import (  # noqa: E402
    FormalExperimentConfig,
    FormalTrainingConfig,
    TrainingEndpoint,
    run_formal_cli,
    standard_topologies,
)


TOPOLOGIES = standard_topologies()
ALL_DEVICES = "0,1,2,3,4,5,6,7"

CONFIG = FormalExperimentConfig(
    name="glm5-2-gpu-npu-training",
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
        # Keep one topology-independent profile valid through eight-way DP and
        # 1F1B PP. This lets any built-in <=8-card migration topology use the
        # same global samples, local batch, and optimizer-step definition.
        local_batch_size=8,
        global_batch_size=64,
        sequence_length=128,
        seed=61,
        deterministic=True,
        training_dtype="float32",
        mixed_precision_param="bfloat16",
        checkpoint_kind="random_seed",
        # Add a few steps here when sampled parameter/activation diagnostics
        # are needed. Formal PASS/FAIL remains based on loss and grad norm.
        exploratory_steps=(),
    ),
    standard=PrecisionStandard(
        migration=MigrationStandard(
            first_loss=AnyOfErrorLimit(
                absolute=0.005,
                relative_absolute=0.005,
            ),
            all_loss=AnyOfErrorLimit(
                absolute=0.01,
                relative_absolute=0.01,
            ),
            loss_quantiles=(
                QuantileLimit(0.99, 0.01),
                QuantileLimit(0.999, 0.01),
                QuantileLimit(0.9999, 0.05),
            ),
            warmup_steps=100,
            outlier_fraction=0.0,
            minimum_observations=1000,
            required_reference_repeats=2,
            required_candidate_repeats=2,
        )
    ),
)


if __name__ == "__main__":
    run_formal_cli(CONFIG, __file__, topologies=TOPOLOGIES)
