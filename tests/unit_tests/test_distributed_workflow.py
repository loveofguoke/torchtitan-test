from __future__ import annotations

import json
from pathlib import Path

import pytest

from distributed.artifacts import (
    FIXTURE_SCHEMA_NAME,
    FIXTURE_SCHEMA_VERSION,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    ArtifactError,
    file_manifest,
    fixture_digest,
    json_digest,
    validate_fixture,
    write_json,
)
from distributed.suite import _scenario_paths
from distributed.workflow import _load_scenario, compare


def _scenario() -> dict:
    return {
        "name": "unit",
        "module": "glm5",
        "config": "debug",
        "world_size": 2,
        "steps": 2,
        "performance_warmup_steps": 1,
        "seed": 42,
        "deterministic": True,
        "dataset_source": "data",
        "hf_assets_source": "assets",
        "seed_args": [],
        "training_args": [],
        "tolerances": {
            "default": {"atol": 1e-4, "rtol": 1e-4},
            "grad_norm": {"atol": 1e-3, "rtol": 1e-3},
        },
    }


def _fixture(path: Path, scenario_digest: str) -> str:
    (path / "data").mkdir(parents=True)
    (path / "data" / "input.json").write_text('{"text":"fixed"}\n')
    manifest = {
        "schema": FIXTURE_SCHEMA_NAME,
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "scenario_digest": scenario_digest,
        "files": file_manifest(path),
    }
    manifest["fixture_digest"] = fixture_digest(manifest)
    write_json(path / "manifest.json", manifest)
    return manifest["fixture_digest"]


def _capture(
    path: Path,
    *,
    backend: str,
    scenario_digest: str,
    fixture_digest_value: str,
    delta: float = 0.0,
) -> None:
    metrics = {
        "loss_metrics/global_avg_loss": {"1": 2.0 + delta, "2": 1.0 + delta},
        "loss_metrics/global_max_loss": {"1": 2.1 + delta, "2": 1.1 + delta},
        "grad_norm": {"1": 0.5 + delta, "2": 0.4 + delta},
        "throughput(tps)": {"1": 10.0, "2": 100.0 if backend == "gpu" else 120.0},
        "time_metrics/end_to_end(s)": {
            "1": 5.0,
            "2": 1.0 if backend == "gpu" else 0.8,
        },
    }
    write_json(
        path,
        {
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "status": "success",
            "scenario_digest": scenario_digest,
            "fixture_digest": fixture_digest_value,
            "backend": backend,
            "world_size": 2,
            "metrics": metrics,
        },
    )


def test_fixture_detects_modified_input(tmp_path: Path) -> None:
    digest = json_digest(_scenario())
    _fixture(tmp_path, digest)
    validate_fixture(tmp_path, digest)
    (tmp_path / "data" / "input.json").write_text("changed")
    with pytest.raises(ArtifactError, match="checksum"):
        validate_fixture(tmp_path, digest)


def test_compare_precision_and_performance(tmp_path: Path) -> None:
    scenario = _scenario()
    scenario_digest = json_digest(scenario)
    fixture = tmp_path / "fixture"
    fixture_digest_value = _fixture(fixture, scenario_digest)
    gpu = tmp_path / "gpu.json"
    npu = tmp_path / "npu.json"
    report = tmp_path / "report.json"
    _capture(
        gpu,
        backend="gpu",
        scenario_digest=scenario_digest,
        fixture_digest_value=fixture_digest_value,
    )
    _capture(
        npu,
        backend="npu",
        scenario_digest=scenario_digest,
        fixture_digest_value=fixture_digest_value,
        delta=5e-5,
    )

    assert compare(scenario, scenario_digest, fixture, gpu, npu, report)
    result = json.loads(report.read_text())
    assert result["precision_passed"] is True
    assert result["performance"]["metrics"]["throughput(tps)"]["npu_over_gpu"] == 1.2


def test_compare_returns_failure_for_precision_drift(tmp_path: Path) -> None:
    scenario = _scenario()
    scenario_digest = json_digest(scenario)
    fixture = tmp_path / "fixture"
    fixture_digest_value = _fixture(fixture, scenario_digest)
    gpu = tmp_path / "gpu.json"
    npu = tmp_path / "npu.json"
    _capture(
        gpu,
        backend="gpu",
        scenario_digest=scenario_digest,
        fixture_digest_value=fixture_digest_value,
    )
    _capture(
        npu,
        backend="npu",
        scenario_digest=scenario_digest,
        fixture_digest_value=fixture_digest_value,
        delta=0.1,
    )

    assert not compare(
        scenario,
        scenario_digest,
        fixture,
        gpu,
        npu,
        tmp_path / "report.json",
    )


def test_scenario_inheritance_appends_training_args(tmp_path: Path) -> None:
    parent = _scenario()
    parent["training_args"] = ["--training.seq_len=128"]
    write_json(tmp_path / "parent.json", parent)
    write_json(
        tmp_path / "child.json",
        {
            "extends": "parent.json",
            "name": "child",
            "training_args_append": ["--parallelism.tensor_parallel_degree=4"],
        },
    )

    scenario, _digest = _load_scenario(tmp_path / "child.json")

    assert scenario["name"] == "child"
    assert scenario["training_args"] == [
        "--training.seq_len=128",
        "--parallelism.tensor_parallel_degree=4",
    ]


def test_checked_in_four_card_suite_resolves() -> None:
    root = Path(__file__).resolve().parents[2]
    suite = root / "distributed/scenarios/glm5_debugmodel_4card_suite.json"
    paths = _scenario_paths(suite)

    assert len(paths) == 15
    for path in paths:
        scenario, _digest = _load_scenario(path)
        assert scenario["world_size"] == 4
        assert scenario["training_args"][:3] == [
            "--training.local_batch_size=4",
            "--training.global_batch_size=16",
            "--training.seq_len=128",
        ]
