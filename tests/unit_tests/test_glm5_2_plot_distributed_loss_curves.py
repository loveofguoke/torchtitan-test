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
from tests.glm5_2_precision.distributed_long_convergence_v2 import (
    SCENARIO_PREFIX,
    SCENARIO_SUFFIX,
)
from tests.glm5_2_precision.plot_distributed_loss_curves import (
    plot_topology_matrix,
)


def _artifact(root: Path, name: str, losses: list[float]) -> None:
    role = "reference" if name.startswith("reference") else "candidate"
    PrecisionArtifactWriter(root / name).write(
        metadata={"role": role},
        training_contract={"topology": {"world_size": 8}},
        metrics=(
            TrainingMetric(
                step=index,
                loss=loss,
                global_max_loss=loss,
                grad_norm=1.0,
                extras={},
            )
            for index, loss in enumerate(losses, start=1)
        ),
    )


def _scenario(root: Path, topology: str, *, candidate_steps: int = 40) -> None:
    scenario = f"{SCENARIO_PREFIX}{topology}{SCENARIO_SUFFIX}"
    scenario_root = root / scenario
    reference = [6.0 - index * 0.05 for index in range(40)]
    candidate = [value * 1.01 for value in reference[:candidate_steps]]
    for repeat in (1, 2):
        _artifact(scenario_root, f"reference-r{repeat}", reference)
        _artifact(scenario_root, f"candidate-r{repeat}", candidate)


def test_plot_matrix_writes_detail_overview_and_manifest(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    output_root = tmp_path / "plots"
    _scenario(artifact_root, "ddp8")
    _scenario(artifact_root, "tp8")

    overview, details, manifest = plot_topology_matrix(
        artifact_root,
        output_root,
        topologies=("ddp8", "tp8"),
        smoothing_window=10,
    )

    assert overview.is_file()
    assert [path.name for path in details] == [
        "loss_step_ddp8.svg",
        "loss_step_tp8.svg",
    ]
    assert all(path.is_file() for path in details)
    assert manifest.is_file()
    overview_text = overview.read_text(encoding="utf-8")
    detail_text = details[0].read_text(encoding="utf-8")
    assert "GPU vs NPU loss-step overview" in overview_text
    assert "DDP8" in overview_text
    assert "TP8" in overview_text
    assert "Optimizer step" in detail_text
    assert "Training loss" in detail_text
    assert "GPU raw repeats" in detail_text
    assert "GPU 10-step MA" in detail_text
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["smoothing_window"] == 10
    assert [item["steps"] for item in payload["topologies"]] == [40, 40]


def test_plot_matrix_rejects_mismatched_steps(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _scenario(artifact_root, "ddp8", candidate_steps=39)

    with pytest.raises(ValueError, match="step sequences do not match"):
        plot_topology_matrix(
            artifact_root,
            tmp_path / "plots",
            topologies=("ddp8",),
        )


def test_plot_matrix_rejects_invalid_smoothing_window(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="smoothing window must be positive"):
        plot_topology_matrix(
            tmp_path / "artifacts",
            tmp_path / "plots",
            topologies=("ddp8",),
            smoothing_window=0,
        )
