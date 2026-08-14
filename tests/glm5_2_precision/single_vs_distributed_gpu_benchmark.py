#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Formal GLM5 debugmodel single-GPU versus four-GPU precision benchmark."""

import os
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
    ParallelTopology,
    TrainingEndpoint,
    run_formal_cli,
)


def _steps() -> int:
    value = int(os.environ.get("GLM5_ALIGNMENT_STEPS", "1000"))
    if value < 1:
        raise ValueError("GLM5_ALIGNMENT_STEPS must be positive")
    return value


STEPS = _steps()
TOPOLOGIES = {
    "single": ParallelTopology("single", 1),
    "ddp4": ParallelTopology("ddp4", 4, data_parallel_replicate_degree=4),
    "fsdp4": ParallelTopology("fsdp4", 4, data_parallel_shard_degree=4),
    "hsdp2x2": ParallelTopology(
        "hsdp2x2",
        4,
        data_parallel_replicate_degree=2,
        data_parallel_shard_degree=2,
    ),
    "tp4": ParallelTopology("tp4", 4, tensor_parallel_degree=4),
    "pp4": ParallelTopology(
        "pp4",
        4,
        pipeline_parallel_degree=4,
        extra_args=(
            "--parallelism.pipeline_parallel_first_stage_less_layers=0",
            "--parallelism.pipeline_parallel_last_stage_less_layers=0",
        ),
    ),
    "fsdp2-tp2": ParallelTopology(
        "fsdp2-tp2",
        4,
        data_parallel_shard_degree=2,
        tensor_parallel_degree=2,
    ),
    "pp2-fsdp2": ParallelTopology(
        "pp2-fsdp2",
        4,
        data_parallel_shard_degree=2,
        pipeline_parallel_degree=2,
    ),
    "pp2-tp2": ParallelTopology(
        "pp2-tp2",
        4,
        tensor_parallel_degree=2,
        pipeline_parallel_degree=2,
    ),
    "fsdp4-ep2": ParallelTopology(
        "fsdp4-ep2",
        4,
        data_parallel_shard_degree=4,
        expert_parallel_degree=2,
    ),
    "fsdp4-ep4": ParallelTopology(
        "fsdp4-ep4",
        4,
        data_parallel_shard_degree=4,
        expert_parallel_degree=4,
    ),
    "fsdp2-tp2-ep2": ParallelTopology(
        "fsdp2-tp2-ep2",
        4,
        data_parallel_shard_degree=2,
        tensor_parallel_degree=2,
        expert_parallel_degree=2,
    ),
    "pp2-fsdp2-ep2": ParallelTopology(
        "pp2-fsdp2-ep2",
        4,
        data_parallel_shard_degree=2,
        pipeline_parallel_degree=2,
        expert_parallel_degree=2,
    ),
    "pp2-tp2-ep2": ParallelTopology(
        "pp2-tp2-ep2",
        4,
        tensor_parallel_degree=2,
        pipeline_parallel_degree=2,
        expert_parallel_degree=2,
    ),
}

CONFIG = FormalExperimentConfig(
    name="glm5-2-distributed-self-consistency",
    kind="self_consistency",
    reference=TrainingEndpoint(
        name="single-card-reference",
        device_type="cuda",
        visible_devices="0",
        topology=TOPOLOGIES["single"],
        repeats=2,
    ),
    candidate=TrainingEndpoint(
        name="distributed-candidate",
        device_type="cuda",
        visible_devices="0,1,2,3",
        topology=TOPOLOGIES["fsdp4"],
        repeats=2,
    ),
    training=FormalTrainingConfig(
        steps=STEPS,
        local_batch_size=2,
        global_batch_size=16,
        sequence_length=128,
        seed=61,
        deterministic=True,
        training_dtype="float32",
        mixed_precision_param="bfloat16",
        checkpoint_kind="random_seed",
        fixed_global_batches=True,
        exploratory_steps=(),
    ),
    standard=PrecisionStandard(
        self_consistency=SelfConsistencyStandard(
            randomness_impacted=True,
            migration_fallback=MigrationStandard(
                warmup_steps=min(100, STEPS // 10),
                minimum_observations=STEPS,
                required_reference_repeats=2,
                required_candidate_repeats=2,
            ),
            required_reference_repeats=2,
            required_candidate_repeats=2,
        )
    ),
    shared_reference_group=(
        os.environ.get("GLM5_ALIGNMENT_SHARED_REFERENCE_GROUP") or None
    ),
)


if __name__ == "__main__":
    run_formal_cli(CONFIG, __file__, topologies=TOPOLOGIES)
