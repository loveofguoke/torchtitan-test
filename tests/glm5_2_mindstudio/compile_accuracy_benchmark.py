#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Official msProbe NPU eager-vs-compiled module accuracy workflow."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_common.topology import standard_topologies  # noqa: E402
from tests.glm5_2_mindstudio.config import (  # noqa: E402
    MindStudioExperimentConfig,
    MsProbeCompileConfig,
    MsProbeDumpConfig,
)
from tests.glm5_2_mindstudio.workflow import run_mindstudio_cli  # noqa: E402
from tests.glm5_2_precision.workflow import (  # noqa: E402
    FormalTrainingConfig,
    TrainingEndpoint,
)


TOPOLOGIES = standard_topologies()
ALL_DEVICES = "0,1,2,3,4,5,6,7"

NPU_ENDPOINT = TrainingEndpoint(
    name="npu",
    device_type="npu",
    visible_devices=ALL_DEVICES,
    topology=TOPOLOGIES["single"],
    repeats=1,
)

CONFIG = MindStudioExperimentConfig(
    name="glm5-2-official-compile-accuracy",
    workflow="compile",
    reference=NPU_ENDPOINT,
    candidate=NPU_ENDPOINT,
    training=FormalTrainingConfig(
        steps=1,
        local_batch_size=8,
        global_batch_size=64,
        sequence_length=128,
        seed=61,
        deterministic=True,
        training_dtype="float32",
        mixed_precision_param="bfloat16",
        checkpoint_kind="random_seed",
    ),
    dump=MsProbeDumpConfig(steps=(0,)),
    compile=MsProbeCompileConfig(
        backend="inductor",
        dump_graphs=True,
        capture_input=True,
        policy="glm5-block",
    ),
    formal_precision_reports=(
        "graph_reports/precision/*",
    ),
    performance_reports=(
        "combination_reports/performance/*",
    ),
)


if __name__ == "__main__":
    run_mindstudio_cli(CONFIG, __file__)
