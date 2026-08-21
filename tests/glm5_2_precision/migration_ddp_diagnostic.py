#!/usr/bin/env python3
"""Targeted 1600-step DDP8 trace using the canonical 5000-step fixture."""

from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_precision.migration_benchmark import (  # noqa: E402
    CONFIG as BASE_CONFIG,
)
from tests.glm5_2_precision.workflow import (  # noqa: E402
    FormalExperimentConfig,
    run_formal_cli,
    standard_topologies,
)


TOPOLOGIES = standard_topologies()
DDP8 = TOPOLOGIES["ddp8"]
SAMPLE_STEPS = (1, 100, 500, 800, 1100, 1300, 1400, 1600)

CONFIG = FormalExperimentConfig(
    name="glm5-2-gpu-npu-ddp-diagnostic",
    kind="migration",
    reference=replace(BASE_CONFIG.reference, topology=DDP8, repeats=1),
    candidate=replace(BASE_CONFIG.candidate, topology=DDP8, repeats=1),
    training=replace(
        BASE_CONFIG.training,
        steps=1600,
        fixture_steps=5000,
        exploratory_steps=SAMPLE_STEPS,
        # Stop after the suspected split while preserving the exact LR values
        # from the original 5000-step trajectory.
        extra_args=(
            *BASE_CONFIG.training.extra_args,
            "--lr_scheduler.total_steps=5000",
        ),
    ),
    standard=BASE_CONFIG.standard,
    fixture_root=BASE_CONFIG.fixture_root,
    artifact_root="precision_diagnostic_artifacts",
    report_root="precision_diagnostic_reports",
    run_root="precision_diagnostic_runs",
    shared_fixture=True,
)


if __name__ == "__main__":
    run_formal_cli(CONFIG, __file__, topologies={"ddp8": DDP8})
