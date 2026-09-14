#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Official msProbe Monitor V2 training-status workflow."""

from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_mindstudio.config import MsProbeMonitorConfig  # noqa: E402
from tests.glm5_2_mindstudio.migration_benchmark import (  # noqa: E402
    CONFIG as ACCURACY_CONFIG,
)
from tests.glm5_2_mindstudio.workflow import run_mindstudio_cli  # noqa: E402


CONFIG = replace(
    ACCURACY_CONFIG,
    workflow="monitor",
    reference=replace(ACCURACY_CONFIG.reference, repeats=1),
    candidate=replace(ACCURACY_CONFIG.candidate, repeats=1),
    training=replace(ACCURACY_CONFIG.training, steps=100),
    monitor=MsProbeMonitorConfig(
        ranks=(0,),
        start_step=0,
        step_interval=1,
        step_count_per_record=10,
        collect_times=100,
        weight_grad=True,
    ),
)


if __name__ == "__main__":
    run_mindstudio_cli(CONFIG, __file__)
