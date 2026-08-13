#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Formal single/distributed self-consistency benchmark on one device backend."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_precision.standards import (  # noqa: E402
    MigrationStandard,
    PrecisionStandard,
    SelfConsistencyStandard,
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
    name="glm5-2-distributed-self-consistency",
    kind="self_consistency",
    reference=TrainingEndpoint(
        name="single-card-reference",
        device_type="cuda",
        visible_devices=ALL_DEVICES,
        topology=TOPOLOGIES["single"],
        repeats=2,
    ),
    candidate=TrainingEndpoint(
        name="distributed-candidate",
        device_type="cuda",
        visible_devices=ALL_DEVICES,
        topology=TOPOLOGIES["fsdp8"],
        repeats=2,
    ),
    training=FormalTrainingConfig(
        steps=1000,
        local_batch_size=2,
        global_batch_size=16,
        sequence_length=128,
        seed=61,
        deterministic=True,
        training_dtype="float32",
        mixed_precision_param="bfloat16",
        checkpoint_kind="random_seed",
        exploratory_steps=(),
    ),
    standard=PrecisionStandard(
        self_consistency=SelfConsistencyStandard(
            # Changing the parallel decomposition changes reduction order. The
            # guidance treats that as randomness-impacting self-consistency and
            # therefore applies migration tolerances instead of bitwise equality.
            randomness_impacted=True,
            migration_fallback=MigrationStandard(
                warmup_steps=100,
                minimum_observations=1000,
                required_reference_repeats=2,
                required_candidate_repeats=2,
            ),
            required_reference_repeats=2,
            required_candidate_repeats=2,
        )
    ),
)


if __name__ == "__main__":
    run_formal_cli(CONFIG, __file__, topologies=TOPOLOGIES)
