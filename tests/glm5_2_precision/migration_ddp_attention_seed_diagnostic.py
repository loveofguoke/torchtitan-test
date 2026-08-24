#!/usr/bin/env python3
"""One-step DDP8 BF16 capture around the first Attention divergence."""

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

CONFIG = FormalExperimentConfig(
    name="glm5-2-gpu-npu-ddp-attention-seed-diagnostic",
    kind="migration",
    reference=replace(BASE_CONFIG.reference, topology=DDP8, repeats=1),
    candidate=replace(BASE_CONFIG.candidate, topology=DDP8, repeats=1),
    training=replace(
        BASE_CONFIG.training,
        steps=1,
        fixture_steps=5000,
        exploratory_steps=(1,),
        exploratory_trace_detail=("attention_projection", "indexer"),
        extra_args=(
            *BASE_CONFIG.training.extra_args,
            "--lr_scheduler.total_steps=5000",
        ),
    ),
    standard=BASE_CONFIG.standard,
    fixture_root=BASE_CONFIG.fixture_root,
    artifact_root="precision_attention_seed_diagnostic_artifacts",
    report_root="precision_attention_seed_diagnostic_reports",
    run_root="precision_attention_seed_diagnostic_runs",
    shared_fixture=True,
    fixture_name=BASE_CONFIG.fixture_storage_name,
)


if __name__ == "__main__":
    run_formal_cli(CONFIG, __file__, topologies={"ddp8": DDP8})
