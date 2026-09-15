# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.glm5_2_precision.artifacts import (
    PrecisionArtifactWriter,
    TrainingMetric,
)
from tests.glm5_2_precision.ddp_long_v2 import (
    DdpLongV2Config,
    compare_distributed_long_convergence_v2,
)
from tests.glm5_2_precision.distributed_long_convergence_v2 import (
    SCENARIO_PREFIX,
    SCENARIO_SUFFIX,
    compare_topology_matrix,
)


def _contract(topology: str, *, world_size: int = 8) -> dict[str, object]:
    degrees = {
        "data_parallel_replicate_degree": 8 if topology == "ddp8" else 1,
        "data_parallel_shard_degree": 1,
        "tensor_parallel_degree": 8 if topology == "tp8" else 1,
        "pipeline_parallel_degree": 1,
        "expert_parallel_degree": 1,
    }
    return {
        "scenario_name": f"distributed-long-{topology}",
        "checkpoint_sha256": "checkpoint",
        "data_sha256": {"tokens": "data"},
        "training": {"seed": 61, "steps": 1000},
        "topology": {
            "name": topology,
            "world_size": world_size,
            **degrees,
        },
    }


def _artifact(
    root: Path,
    name: str,
    losses: list[float],
    contract: dict[str, object],
) -> Path:
    metrics = (
        TrainingMetric(
            step=index,
            loss=loss,
            global_max_loss=loss,
            grad_norm=1.0 + index / 1000.0,
            extras={},
        )
        for index, loss in enumerate(losses, start=1)
    )
    return PrecisionArtifactWriter(root / name).write(
        metadata={"role": "reference" if name.startswith("reference") else "candidate"},
        training_contract=contract,
        metrics=metrics,
    )


def _write_scenario(root: Path, topology: str) -> Path:
    scenario = f"{SCENARIO_PREFIX}{topology}{SCENARIO_SUFFIX}"
    scenario_root = root / scenario
    reference = [8.0 - index * 0.004 for index in range(1000)]
    candidate = [value * 1.015 for value in reference]
    contract = _contract(topology)
    _artifact(scenario_root, "reference-r1", reference, contract)
    _artifact(scenario_root, "reference-r2", reference, contract)
    _artifact(scenario_root, "candidate-r1", candidate, contract)
    _artifact(scenario_root, "candidate-r2", candidate, contract)
    return scenario_root


def _config() -> DdpLongV2Config:
    return DdpLongV2Config(
        minimum_steps=1000,
        sustained_window_size=200,
    )


def test_topology_neutral_evaluator_accepts_tp(tmp_path: Path) -> None:
    scenario_root = _write_scenario(tmp_path, "tp8")

    result = compare_distributed_long_convergence_v2(
        (scenario_root / "reference-r1", scenario_root / "reference-r2"),
        (scenario_root / "candidate-r1", scenario_root / "candidate-r2"),
        config=_config(),
    )

    assert result.status == "PASS"
    assert result.curve.area_relative_error == pytest.approx(0.015)


def test_topology_neutral_evaluator_rejects_single_card(tmp_path: Path) -> None:
    reference = [8.0 - index * 0.004 for index in range(1000)]
    contract = _contract("single", world_size=1)
    paths = [
        _artifact(tmp_path, f"reference-r{repeat}", reference, contract)
        for repeat in (1, 2)
    ]
    candidates = [
        _artifact(tmp_path, f"candidate-r{repeat}", reference, contract)
        for repeat in (1, 2)
    ]

    with pytest.raises(ValueError, match="world_size > 1"):
        compare_distributed_long_convergence_v2(
            paths,
            candidates,
            config=_config(),
        )


def test_matrix_reassesses_multiple_topologies(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    output_root = tmp_path / "reports"
    _write_scenario(artifact_root, "ddp8")
    _write_scenario(artifact_root, "tp8")

    status, results, summary_path, report_path = compare_topology_matrix(
        artifact_root,
        output_root,
        topologies=("ddp8", "tp8"),
        config=_config(),
    )

    assert status == "PASS"
    assert [result.status for result in results] == ["PASS", "PASS"]
    assert summary_path.is_file()
    assert report_path.is_file()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert [result["topology"] for result in payload["results"]] == [
        "ddp8",
        "tp8",
    ]
    assert (output_root / "ddp8" / "ddp_long_v2_report.md").is_file()
    assert (output_root / "tp8" / "ddp_long_v2_report.md").is_file()


def test_matrix_reports_missing_topology_as_inconclusive(tmp_path: Path) -> None:
    status, results, summary_path, report_path = compare_topology_matrix(
        tmp_path / "artifacts",
        tmp_path / "reports",
        topologies=("ep8",),
        config=_config(),
    )

    assert status == "INCONCLUSIVE"
    assert results[0].status == "INVALID"
    assert results[0].error
    assert summary_path.is_file()
    assert "INVALID" in report_path.read_text(encoding="utf-8")
