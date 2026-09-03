#!/usr/bin/env python3
"""NPU eager single-card versus four-card topology precision benchmark."""

from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_precision.single_vs_distributed_gpu_benchmark import (  # noqa: E402
    TOPOLOGIES,
)
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
)


def _steps() -> int:
    value = int(os.environ.get("GLM5_EAGER_ALIGNMENT_STEPS", "10"))
    if value < 1:
        raise ValueError("GLM5_EAGER_ALIGNMENT_STEPS must be positive")
    return value


def _param_dtype() -> str:
    value = os.environ.get("GLM5_EAGER_PARAM_DTYPE", "bfloat16")
    if value not in {"float32", "bfloat16"}:
        raise ValueError("GLM5_EAGER_PARAM_DTYPE must be float32 or bfloat16")
    return value


STEPS = _steps()
PARAM_DTYPE = _param_dtype()
ALL_DEVICES = os.environ.get("GLM5_EAGER_NPU_DEVICES", "0,1,2,3")
REFERENCE_DEVICE = os.environ.get("GLM5_EAGER_REFERENCE_NPU", "0")

CONFIG = FormalExperimentConfig(
    name="glm5-2-npu-eager-topology-mindstudio",
    kind="self_consistency",
    reference=TrainingEndpoint(
        name="single-card-eager-reference",
        device_type="npu",
        visible_devices=REFERENCE_DEVICE,
        topology=TOPOLOGIES["single"],
        repeats=2,
    ),
    candidate=TrainingEndpoint(
        name="distributed-eager-candidate",
        device_type="npu",
        visible_devices=ALL_DEVICES,
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
        mixed_precision_param=PARAM_DTYPE,
        checkpoint_kind="random_seed",
        fixed_global_batches=True,
        exploratory_steps=(),
    ),
    standard=PrecisionStandard(
        self_consistency=SelfConsistencyStandard(
            randomness_impacted=True,
            migration_fallback=MigrationStandard(
                warmup_steps=0,
                minimum_observations=STEPS,
                required_reference_repeats=2,
                required_candidate_repeats=2,
            ),
            required_reference_repeats=2,
            required_candidate_repeats=2,
        )
    ),
    fixture_root="npu_eager_mindstudio_fixtures",
    artifact_root="npu_eager_mindstudio_artifacts",
    report_root="npu_eager_mindstudio_reports",
    run_root="npu_eager_mindstudio_runs",
    # All candidates intentionally share one fixture and one single-card capture.
    shared_reference_group="four-card-topology-suite",
)


if __name__ == "__main__":
    run_formal_cli(CONFIG, __file__, topologies=TOPOLOGIES)
