# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree

import pytest

from tests.glm5_2_precision.artifacts import (
    PrecisionArtifactWriter,
    TrainingMetric,
)
from tests.glm5_2_precision.export_distributed_alignment_results import (
    LONG_SCENARIO_SUFFIX,
)
from tests.glm5_2_precision.plot_distributed_gpu_npu_alignment import (
    plot_distributed_gpu_npu_alignment,
)


def _contract(topology: str, steps: int) -> dict[str, object]:
    return {
        "training": {
            "global_batch_size": 64,
            "sequence_length": 128,
            "seed": 61,
            "mixed_precision_param": "bfloat16",
        },
        "token_plan": {"steps": steps},
        "topology": {
            "name": topology,
            "world_size": 8,
        },
    }


def _artifact(
    scenario_root: Path,
    name: str,
    *,
    role: str,
    device: str,
    repeat: int,
    losses: list[float],
    grad_norms: list[float],
    contract: dict[str, object],
) -> None:
    PrecisionArtifactWriter(scenario_root / name).write(
        metadata={
            "role": role,
            "device_type": device,
            "repeat": repeat,
        },
        training_contract=contract,
        metrics=(
            TrainingMetric(
                step=index,
                loss=loss,
                global_max_loss=loss,
                grad_norm=grad_norm,
                extras={},
            )
            for index, (loss, grad_norm) in enumerate(
                zip(losses, grad_norms), start=1
            )
        ),
    )


def _scenario(
    artifact_root: Path,
    topology: str,
    *,
    candidate_steps: int = 250,
) -> None:
    scenario_root = artifact_root / (
        f"migration-cuda-npu-{topology}{LONG_SCENARIO_SUFFIX}"
    )
    gpu_loss = [4.0 - index * 0.006 for index in range(250)]
    gpu_grad = [1.1 - index * 0.002 for index in range(250)]
    npu_loss = [value * 1.01 for value in gpu_loss[:candidate_steps]]
    npu_grad = [value * 0.98 for value in gpu_grad[:candidate_steps]]
    contract = _contract(topology, 250)
    for repeat in (1, 2):
        _artifact(
            scenario_root,
            f"reference-r{repeat}",
            role="reference",
            device="cuda",
            repeat=repeat,
            losses=gpu_loss,
            grad_norms=gpu_grad,
            contract=contract,
        )
        _artifact(
            scenario_root,
            f"candidate-r{repeat}",
            role="candidate",
            device="npu",
            repeat=repeat,
            losses=npu_loss,
            grad_norms=npu_grad,
            contract=contract,
        )


def test_plotter_writes_four_panel_svg_and_manifest(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    output_root = tmp_path / "plots"
    _scenario(artifact_root, "ddp8")

    paths, manifest = plot_distributed_gpu_npu_alignment(
        artifact_root,
        output_root,
        topologies=("ddp8",),
    )

    assert [path.name for path in paths] == ["gpu-vs-npu-ddp8-250steps.svg"]
    ElementTree.parse(paths[0])
    text = paths[0].read_text(encoding="utf-8")
    assert "GPU vs NPU distributed precision alignment" in text
    assert "Loss trajectory (31-step moving average)" in text
    assert "Global grad norm trajectory" in text
    assert "Running loss mean absolute relative error" in text
    assert "Running grad norm signed mean relative error" in text
    assert "diagnostic and do not independently decide PASS/FAIL" in text
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["plots"][0]["topology"] == "ddp8"
    assert payload["plots"][0]["final_running_loss_mare_percent"] == pytest.approx(
        1.0
    )
    assert payload["plots"][0][
        "final_running_grad_signed_mean_percent"
    ] == pytest.approx(2.0)


def test_plotter_rejects_mismatched_gpu_npu_steps(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    _scenario(artifact_root, "tp8", candidate_steps=249)

    with pytest.raises(ValueError, match="steps do not match"):
        plot_distributed_gpu_npu_alignment(
            artifact_root,
            tmp_path / "plots",
            topologies=("tp8",),
        )
