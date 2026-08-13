# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from tests.glm5_2_precision.artifacts import (
    PrecisionArtifactError,
    PrecisionArtifactReader,
    PrecisionArtifactWriter,
    TrainingMetric,
)
from tests.glm5_2_precision.report import compare_and_write_report
from tests.glm5_2_precision.migrate_legacy_outputs import compact_scenario_name
from tests.glm5_2_precision.standards import (
    AnyOfErrorLimit,
    MigrationStandard,
    PrecisionStandard,
    QuantileLimit,
    SelfConsistencyStandard,
    compute_error_metrics,
    evaluate_exact_pair,
    evaluate_migration_pair,
)
from tests.glm5_2_precision.workflow import (
    FormalExperimentConfig,
    FormalTrainingConfig,
    ParallelTopology,
    TrainingEndpoint,
    _endpoint_from_process_environment,
    _fixture_endpoint_from_environment,
)


def test_documented_error_formulas() -> None:
    metrics = compute_error_metrics([1.0, 2.0], [0.5, 3.0])

    assert metrics.normal == pytest.approx(-0.25)
    assert metrics.absolute == pytest.approx(0.75)
    assert metrics.relative_normal == pytest.approx(0.0)
    assert metrics.relative_absolute == pytest.approx(0.5)
    assert metrics.max_absolute == pytest.approx(1.0)


def test_first_loss_is_evaluated_before_warmup() -> None:
    evaluation = evaluate_migration_pair(
        reference_loss={1: 1.0, 2: 1.0, 3: 1.0},
        candidate_loss={1: 1.1, 2: 1.0, 3: 1.0},
        reference_grad_norm={1: 1.0, 2: 1.0, 3: 1.0},
        candidate_grad_norm={1: 1.0, 2: 1.0, 3: 1.0},
        standard=MigrationStandard(
            first_loss=AnyOfErrorLimit(absolute=0.05),
            all_loss=AnyOfErrorLimit(absolute=0.0),
            warmup_steps=1,
            minimum_observations=3,
        ),
    )

    first_loss = next(item for item in evaluation.criteria if item.name == "First loss")
    assert not first_loss.passed
    assert evaluation.loss.metrics.absolute == 0.0


def test_loss_quantiles_are_diagnostic_only() -> None:
    evaluation = evaluate_migration_pair(
        reference_loss={1: 1.0, 2: 1.0},
        candidate_loss={1: 1.001, 2: 1.001},
        reference_grad_norm={1: 1.0, 2: 1.0},
        candidate_grad_norm={1: 1.0, 2: 1.0},
        standard=MigrationStandard(
            loss_quantiles=(QuantileLimit(0.99, 0.0),),
            minimum_observations=2,
        ),
    )

    assert evaluation.passed
    assert all("P99" not in item.name for item in evaluation.criteria)


def test_exact_self_consistency_uses_float_bits() -> None:
    exact = evaluate_exact_pair(
        reference_loss={1: 1.0},
        candidate_loss={1: 1.0},
        reference_grad_norm={1: 2.0},
        candidate_grad_norm={1: 2.0},
    )
    changed = evaluate_exact_pair(
        reference_loss={1: 1.0},
        candidate_loss={1: 1.0 + 1e-12},
        reference_grad_norm={1: 2.0},
        candidate_grad_norm={1: 2.0},
    )

    assert exact.passed
    assert not changed.passed


def _metrics(loss_offset: float = 0.0) -> tuple[TrainingMetric, ...]:
    return tuple(
        TrainingMetric(
            step=step,
            loss=2.0 - step * 0.1 + loss_offset,
            global_max_loss=2.0 - step * 0.1 + loss_offset,
            grad_norm=1.0 + step * 0.01,
            extras={"lr": 0.001},
        )
        for step in range(1, 5)
    )


def test_artifact_checksums(tmp_path: Path) -> None:
    path = PrecisionArtifactWriter(tmp_path / "artifact").write(
        metadata={"role": "reference"},
        training_contract={"seed": 61},
        metrics=_metrics(),
    )
    reader = PrecisionArtifactReader(path)
    assert reader.loss_series()[1] == pytest.approx(1.9)

    with (path / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("corruption\n")
    with pytest.raises(PrecisionArtifactError, match="checksum mismatch"):
        PrecisionArtifactReader(path)


def test_topology_validates_rank_product() -> None:
    topology = ParallelTopology(
        "fsdp2-tp4", 8, data_parallel_shard_degree=2, tensor_parallel_degree=4
    )
    assert "--parallelism.data_parallel_shard_degree=2" in topology.command_args()
    assert "--parallelism.tensor_parallel_degree=4" in topology.command_args()
    with pytest.raises(ValueError, match="dense ranks"):
        ParallelTopology("invalid", 8, data_parallel_shard_degree=2)


def test_endpoint_uses_exported_visible_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    endpoint = TrainingEndpoint(
        "npu", "npu", "0,1,2,3,4,5,6,7", ParallelTopology("single", 1)
    )
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4")

    resolved = _endpoint_from_process_environment(endpoint)

    assert resolved.visible_devices == "4"


def test_fixture_backend_is_inferred_without_endpoint_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = ParallelTopology("single", 1)
    config = FormalExperimentConfig(
        name="fixture-backend",
        kind="migration",
        reference=TrainingEndpoint("gpu", "cuda", "0", topology, repeats=2),
        candidate=TrainingEndpoint("npu", "npu", "0", topology, repeats=2),
    )
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    endpoint = _fixture_endpoint_from_environment(config, None)

    assert endpoint.device_type == "npu"


def test_fixture_backend_requires_unambiguous_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = ParallelTopology("single", 1)
    config = FormalExperimentConfig(
        name="fixture-backend",
        kind="migration",
        reference=TrainingEndpoint("gpu", "cuda", "0", topology, repeats=2),
        candidate=TrainingEndpoint("npu", "npu", "0", topology, repeats=2),
    )
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    with pytest.raises(ValueError, match="fixture backend is ambiguous"):
        _fixture_endpoint_from_environment(config, None)


def test_compare_writes_html_and_json_report(tmp_path: Path) -> None:
    topology = ParallelTopology("single", 1)
    reference = TrainingEndpoint(
        "gpu", "cuda", "0", topology, repeats=2
    )
    candidate = TrainingEndpoint(
        "npu", "npu", "0", topology, repeats=2
    )
    standard = PrecisionStandard(
        migration=MigrationStandard(
            warmup_steps=0,
            minimum_observations=4,
            required_candidate_repeats=2,
        ),
        self_consistency=SelfConsistencyStandard(
            randomness_impacted=True,
            required_candidate_repeats=2,
        ),
    )
    config = FormalExperimentConfig(
        name="synthetic",
        kind="migration",
        reference=reference,
        candidate=candidate,
        training=FormalTrainingConfig(steps=4, global_batch_size=2),
        standard=standard,
        artifact_root="artifacts",
        report_root="reports",
    )
    from tests.glm5_2_precision.workflow import _artifact_directory

    contract = {
        "scenario_name": config.scenario_name,
        "training": {
            **asdict(config.training),
            "converged_checkpoint": None,
        },
        "checkpoint_sha256": "checkpoint",
        "data_sha256": {"data": "digest"},
        "topology": {
            "name": topology.name,
            "world_size": topology.world_size,
        },
    }
    for repeat in (1, 2):
        PrecisionArtifactWriter(
            _artifact_directory(tmp_path, config, "reference", reference, repeat)
        ).write(
            metadata={"role": "reference", "repeat": repeat},
            training_contract=contract,
            metrics=_metrics(),
        )
    for repeat in (1, 2):
        PrecisionArtifactWriter(
            _artifact_directory(tmp_path, config, "candidate", candidate, repeat)
        ).write(
            metadata={"role": "candidate", "repeat": repeat},
            training_contract=contract,
            metrics=_metrics(loss_offset=0.001 if repeat == 1 else 0.0011),
        )

    report = compare_and_write_report(tmp_path, config)
    assert report.is_file()
    report_text = report.read_text(encoding="utf-8")
    assert "Formal GLM 5.2 precision benchmark" in report_text
    assert "Repeat-run diagnostics" in report_text
    assert "Historical customer standards (reference only)" in report_text
    assert "step 1" in report_text
    summary = json.loads(
        (report.parent / "precision_summary.json").read_text(encoding="utf-8")
    )
    assert summary["passed"]
    assert not summary["candidate_reproducible_loss"]
    assert not summary["strict_reproducibility_required"]


def test_storage_name_and_capture_paths_are_compact(tmp_path: Path) -> None:
    topology = ParallelTopology("single", 1)
    config = FormalExperimentConfig(
        name="glm5-2-gpu-npu-training",
        kind="migration",
        reference=TrainingEndpoint("gpu-reference", "cuda", "0", topology),
        candidate=TrainingEndpoint("npu-candidate", "npu", "0", topology),
    )

    assert config.storage_name == (
        "migration-cuda-npu-single-bf16-random-s1000-b16-seq128-seed61"
    )
    from tests.glm5_2_precision.workflow import _artifact_directory

    assert _artifact_directory(
        tmp_path, config, "candidate", config.candidate, 1
    ).name == "candidate-r1"


def test_legacy_scenario_name_maps_to_current_storage_name() -> None:
    legacy = (
        "glm5-2-gpu-npu-training-migration-cuda-single-vs-npu-single-"
        "mixed-bfloat16-random-seed-steps1000-global-batch16-seq128-seed61"
    )

    assert compact_scenario_name(legacy) == (
        "migration-cuda-npu-single-bf16-random-s1000-b16-seq128-seed61"
    )
