#!/usr/bin/env python3
"""One-step msProbe tensor comparison at GLM5 block boundaries."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_precision.single_vs_distributed_npu_eager_benchmark import (  # noqa: E402
    CONFIG as BASE_CONFIG,
    TOPOLOGIES,
)
from tests.glm5_2_precision.workflow import run_formal_cli  # noqa: E402


CONFIG = replace(
    BASE_CONFIG,
    name="glm5-2-npu-eager-block-boundary",
)


if __name__ == "__main__":
    run_formal_cli(CONFIG, __file__, topologies=TOPOLOGIES)
