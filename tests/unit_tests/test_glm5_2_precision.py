# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tests.glm5_2_precision.artifacts import (
    PrecisionArtifactError,
    PrecisionArtifactReader,
    PrecisionArtifactWriter,
    TrainingMetric,
)
from tests.glm5_2_precision.fixed_batches import FixedGlobalBatchDataLoader
from tests.glm5_2_precision.report import compare_and_write_report
from tests.glm5_2_precision.standards import (
    AnyOfErrorLimit,
    MigrationStandard,
    PrecisionStandard,
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


def test_fixed_global_batches_are_identical_across_dp_degrees(
    tmp_path: Path,
) -> None:
    steps = 2
    global_batch_size = 16
    sequence_length = 3
    rows = steps * global_batch_size
    base = torch.arange(rows * sequence_length, dtype=torch.int64).reshape(
        rows, sequence_length
    )
    path = tmp_path / "fixed.safetensors"
    save_file(
        {"input": base, "positions": base + 100, "labels": base + 200},
        path,
        metadata={
            "steps": str(steps),
            "global_batch_size": str(global_batch_size),
            "sequence_length": str(sequence_length),
        },
    )

    single = FixedGlobalBatchDataLoader(
        path,
        dp_world_size=1,
        dp_rank=0,
        local_batch_size=2,
        sequence_length=sequence_length,
    )
    single_iterator = iter(single)
    distributed = [
        iter(
            FixedGlobalBatchDataLoader(
                path,
                dp_world_size=4,
                dp_rank=rank,
                local_batch_size=2,
                sequence_length=sequence_length,
            )
        )
        for rank in range(4)
    ]

    for _step in range(steps):
        single_batches = [next(single_iterator) for _wave in range(8)]
        distributed_batches = [
            next(rank_iterator)
            for _wave in range(2)
            for rank_iterator in distributed
        ]
        for key in ("input", "positions"):
            single_step = torch.cat([batch[0][key] for batch in single_batches])
            distributed_step = torch.cat(
                [batch[0][key] for batch in distributed_batches]
            )
            assert torch.equal(single_step, distributed_step)
        assert torch.equal(
            torch.cat([batch[1] for batch in single_batches]),
            torch.cat([batch[1] for batch in distributed_batches]),
        )


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
    runtime_log = tmp_path / "runtime.log"
    runtime_log.write_text("training completed\n", encoding="utf-8")
    for repeat in (1, 2):
        PrecisionArtifactWriter(
            _artifact_directory(tmp_path, config, "reference", reference, repeat)
        ).write(
            metadata={"role": "reference", "repeat": repeat},
            training_contract=contract,
            metrics=_metrics(),
            attachments={"runtime.log": runtime_log},
        )
    for repeat in (1, 2):
        PrecisionArtifactWriter(
            _artifact_directory(tmp_path, config, "candidate", candidate, repeat)
        ).write(
            metadata={"role": "candidate", "repeat": repeat},
            training_contract=contract,
            metrics=_metrics(loss_offset=0.001),
            attachments={"runtime.log": runtime_log},
        )

    report = compare_and_write_report(tmp_path, config)
    assert report.is_file()
    assert "Formal GLM 5.2 precision benchmark" in report.read_text(encoding="utf-8")
    summary = json.loads(
        (report.parent / "precision_summary.json").read_text(encoding="utf-8")
    )
    assert summary["passed"]


def test_matrix_candidates_share_reference_but_keep_separate_reports(
    tmp_path: Path,
) -> None:
    from tests.glm5_2_precision.workflow import _artifact_directory

    single = ParallelTopology("single", 1)
    ddp4 = ParallelTopology("ddp4", 4, data_parallel_replicate_degree=4)
    fsdp4 = ParallelTopology("fsdp4", 4, data_parallel_shard_degree=4)
    reference = TrainingEndpoint("reference", "cuda", "0", single, repeats=2)
    common = {
        "name": "synthetic",
        "kind": "self_consistency",
        "reference": reference,
        "shared_reference_group": "four-gpu-matrix",
    }
    ddp_config = FormalExperimentConfig(
        **common,
        candidate=TrainingEndpoint("candidate", "cuda", "0,1,2,3", ddp4),
    )
    fsdp_config = FormalExperimentConfig(
        **common,
        candidate=TrainingEndpoint("candidate", "cuda", "0,1,2,3", fsdp4),
    )

    assert ddp_config.scenario_name == fsdp_config.scenario_name
    assert _artifact_directory(
        tmp_path, ddp_config, "reference", reference, 1
    ) == _artifact_directory(tmp_path, fsdp_config, "reference", reference, 1)
    assert _artifact_directory(
        tmp_path, ddp_config, "candidate", ddp_config.candidate, 1
    ) != _artifact_directory(
        tmp_path, fsdp_config, "candidate", fsdp_config.candidate, 1
    )
    assert ddp_config.report_subdirectory == "ddp4"
    assert fsdp_config.report_subdirectory == "fsdp4"


def _resumable_config(root: Path) -> FormalExperimentConfig:
    (root / "data").mkdir()
    (root / "data" / "sample.txt").write_text("data\n", encoding="utf-8")
    (root / "tokenizer").mkdir()
    (root / "tokenizer" / "tokenizer.txt").write_text(
        "tokenizer\n", encoding="utf-8"
    )
    topology = ParallelTopology("single", 1)
    endpoint = TrainingEndpoint("endpoint", "cuda", "0", topology, repeats=2)
    return FormalExperimentConfig(
        name="resume",
        kind="self_consistency",
        reference=endpoint,
        candidate=endpoint,
        training=FormalTrainingConfig(
            steps=4,
            global_batch_size=2,
            data_paths=("data", "tokenizer"),
        ),
    )


def _write_resumable_fixture(root: Path, config: FormalExperimentConfig) -> Path:
    from tests.glm5_2_precision.workflow import (
        _directory_digest,
        _fixture_directory,
        _normalized_training,
    )

    fixture = _fixture_directory(root, config)
    checkpoint = fixture / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.bin").write_bytes(b"checkpoint")
    manifest = {
        "schema": "torchtitan.glm5_2.precision_fixture",
        "schema_version": 1,
        "scenario_name": config.scenario_name,
        "checkpoint_kind": config.training.checkpoint_kind,
        "checkpoint_relative_path": "checkpoint",
        "checkpoint_sha256": _directory_digest(checkpoint),
        "data_sha256": {
            path: _directory_digest(root / path)
            for path in config.training.data_paths
        },
        "fixed_batches_relative_path": None,
        "fixed_batches_sha256": None,
        "training": _normalized_training(config.training),
    }
    (fixture / "fixture.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return fixture


def test_resume_skips_valid_fixture_and_capture(tmp_path: Path) -> None:
    from tests.glm5_2_precision.workflow import (
        _artifact_directory,
        _training_contract,
        capture_endpoint,
        prepare_fixture,
    )

    config = _resumable_config(tmp_path)
    fixture = _write_resumable_fixture(tmp_path, config)
    manifest = json.loads((fixture / "fixture.json").read_text(encoding="utf-8"))
    artifact_path = _artifact_directory(
        tmp_path, config, "reference", config.reference, 1
    )
    PrecisionArtifactWriter(artifact_path).write(
        metadata={
            "role": "reference",
            "repeat": 1,
            "endpoint_name": config.reference.name,
            "device_type": config.reference.device_type,
        },
        training_contract=_training_contract(config, config.reference, manifest),
        metrics=_metrics(),
    )

    assert prepare_fixture(tmp_path, config, force=False, resume=True) == fixture
    assert capture_endpoint(
        tmp_path,
        config,
        role="reference",
        repeat=1,
        force=False,
        resume=True,
    ) == artifact_path


def test_resume_replaces_incomplete_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.glm5_2_precision import workflow

    config = _resumable_config(tmp_path)
    _write_resumable_fixture(tmp_path, config)
    artifact_path = workflow._artifact_directory(
        tmp_path, config, "candidate", config.candidate, 1
    )
    artifact_path.mkdir(parents=True)
    (artifact_path / "partial.txt").write_text("partial", encoding="utf-8")
    run_path = workflow._run_directory(
        tmp_path, config, "candidate", config.candidate, 1
    )
    run_path.mkdir(parents=True)
    (run_path / "stale.txt").write_text("stale", encoding="utf-8")

    def fake_run_process(command, *, root, environment, log_path) -> None:
        del command, root
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("completed\n", encoding="utf-8")
        metrics_path = Path(environment["GLM5_PRECISION_METRICS_PATH"])
        with metrics_path.open("w", encoding="utf-8") as stream:
            for metric in _metrics():
                stream.write(
                    json.dumps(
                        {
                            "step": metric.step,
                            "metrics": {
                                "loss_metrics/global_avg_loss": metric.loss,
                                "loss_metrics/global_max_loss": metric.global_max_loss,
                                "grad_norm": metric.grad_norm,
                            },
                        }
                    )
                    + "\n"
                )

    monkeypatch.setattr(workflow, "_run_process", fake_run_process)
    result = workflow.capture_endpoint(
        tmp_path,
        config,
        role="candidate",
        repeat=1,
        force=False,
        resume=True,
    )

    assert result == artifact_path
    assert PrecisionArtifactReader(artifact_path).metrics[-1].step == 4
    assert not (run_path / "stale.txt").exists()
