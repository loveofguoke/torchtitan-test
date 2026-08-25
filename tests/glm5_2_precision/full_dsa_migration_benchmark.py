#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""GPU-to-NPU formal precision benchmark for the Full DSA reference path."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_common.topology import DISTRIBUTED_TOPOLOGY_NAMES  # noqa: E402
from tests.glm5_2_precision.standards import (  # noqa: E402
    AnyOfErrorLimit,
    MigrationStandard,
    PrecisionStandard,
    QuantileLimit,
)
from tests.glm5_2_precision.topology_suite import (  # noqa: E402
    run_topology_suite_cli,
)
from tests.glm5_2_precision.workflow import (  # noqa: E402
    FormalExperimentConfig,
    FormalTrainingConfig,
    TrainingEndpoint,
    standard_topologies,
)


TOPOLOGIES = standard_topologies()
ALL_DEVICES = "0,1,2,3,4,5,6,7"
# Cross-stage top-k transport is intentionally not part of the readable model.
# Keep --topology all useful by restricting this benchmark to non-PP layouts.
FULL_DSA_TOPOLOGIES = (
    "single",
    *(
        name
        for name in DISTRIBUTED_TOPOLOGY_NAMES
        if TOPOLOGIES[name].pipeline_parallel_degree == 1
    ),
)

CONFIG = FormalExperimentConfig(
    name="glm5-2-full-dsa-gpu-npu",
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
    storage_tag="full-dsa",
    training=FormalTrainingConfig(
        module="glm5",
        config="glm5_full_dsa_debugmodel",
        steps=5000,
        local_batch_size=8,
        global_batch_size=64,
        sequence_length=128,
        seed=61,
        deterministic=True,
        training_dtype="float32",
        mixed_precision_param="bfloat16",
        checkpoint_kind="random_seed",
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
    run_topology_suite_cli(
        CONFIG,
        __file__,
        topologies=TOPOLOGIES,
        topology_names=FULL_DSA_TOPOLOGIES,
    )
