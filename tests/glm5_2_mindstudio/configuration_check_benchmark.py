#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Official msProbe configuration check for GPU-to-NPU validation."""

from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_mindstudio.migration_benchmark import (  # noqa: E402
    CONFIG as ACCURACY_CONFIG,
)
from tests.glm5_2_mindstudio.workflow import run_mindstudio_cli  # noqa: E402


CONFIG = replace(
    ACCURACY_CONFIG,
    workflow="config-check",
    output_subdirectory="diagnostics/configuration-check",
    owns_fixture=False,
)


if __name__ == "__main__":
    run_mindstudio_cli(CONFIG, __file__)
