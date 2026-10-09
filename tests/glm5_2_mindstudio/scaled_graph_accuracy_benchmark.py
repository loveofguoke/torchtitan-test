#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Graph accuracy workflow for the scaled GLM-5.2 validation model."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_mindstudio.graph_accuracy_benchmark import (  # noqa: E402
    _select_options,
    _stage_configs,
)
from tests.glm5_2_mindstudio.scaled_accuracy_benchmark import (  # noqa: E402
    BASE_CONFIG as SCALED_ACCURACY_CONFIG,
    TRAINING,
)
from tests.glm5_2_mindstudio.workflow import run_mindstudio_cli  # noqa: E402


if __name__ == "__main__":
    stage, device, graph_backend, codegen_backend, arguments = (
        _select_options(sys.argv[1:])
    )
    _base_config, stage_configs = _stage_configs(
        device,
        graph_backend=graph_backend,
        codegen_backend=codegen_backend,
        training=TRAINING,
        experiment_storage_name=SCALED_ACCURACY_CONFIG.storage_name,
    )
    sys.argv = [sys.argv[0], *arguments]
    run_mindstudio_cli(stage_configs[stage], __file__)
