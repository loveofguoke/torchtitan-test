#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Run every official MindStudio accuracy stage under one experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_mindstudio.configuration_check_benchmark import (  # noqa: E402
    CONFIG as CONFIG_CHECK_CONFIG,
)
from tests.glm5_2_mindstudio.training_baseline_benchmark import (  # noqa: E402
    CONFIG as BASELINE_CONFIG,
)
from tests.glm5_2_mindstudio.migration_benchmark import (  # noqa: E402
    CONFIG as DUMP_CONFIG,
)
from tests.glm5_2_mindstudio.training_monitor_benchmark import (  # noqa: E402
    CONFIG as MONITOR_CONFIG,
)
from tests.glm5_2_mindstudio.workflow import run_mindstudio_cli  # noqa: E402


STAGE_CONFIGS = {
    "config-check": CONFIG_CHECK_CONFIG,
    "observation": BASELINE_CONFIG,
    "monitor": MONITOR_CONFIG,
    "dump": DUMP_CONFIG,
}


def _select_stage(arguments: list[str]) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--stage", choices=tuple(STAGE_CONFIGS), default="baseline"
    )
    parsed, remaining = parser.parse_known_args(arguments)
    return parsed.stage, remaining


if __name__ == "__main__":
    stage, arguments = _select_stage(sys.argv[1:])
    sys.argv = [sys.argv[0], *arguments]
    run_mindstudio_cli(STAGE_CONFIGS[stage], __file__)
