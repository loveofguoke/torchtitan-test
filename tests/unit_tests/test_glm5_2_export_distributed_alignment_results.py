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
from tests.glm5_2_precision.ddp_long_v2 import DdpLongV2Config
from tests.glm5_2_precision.export_distributed_alignment_results import (
    LONG_SCENARIO_SUFFIX,
    SHORT_SCENARIO_SUFFIX,
    collect_short_results,
    export_results,
)


def _contract(topology: str, steps: int) -> dict[str, object]:
    return {
        "scenario_name": f"migration-cuda-npu-{topology}",
        "checkpoint_sha256": "checkpoint",
        "data_sha256": {"tokens": "data"},
        "training": {"seed": 61, "steps": steps},
        "topology": {"name": topology, "world_size": 8},
    }


def _write_artifact(
    path: Path,
    *,
    role: str,
    device: str,
    repeat: int,
    topology: str,
    losses: list[float],
) -> None:
    metrics = (
        TrainingMetric(
            step=index,
            loss=loss,
            global_max_loss=loss,
            grad_norm=1.0 + index / 100.0,
            extras={},
        )
        for index, loss in enumerate(losses, start=1)
    )
    PrecisionArtifactWriter(path).write(
        metadata={
            "role": role,
            "device_type": device,
            "repeat": repeat,
        },
        training_contract=_contract(topology, len(losses)),
        metrics=metrics,
    )


def _write_scenario(
    root: Path,
    *,
    topology: str,
    suffix: str,
    gpu_losses: list[float],
    npu_losses: list[float],
) -> None:
    scenario = root / f"migration-cuda-npu-{topology}{suffix}"
    for repeat in (1, 2):
        _write_artifact(
            scenario / f"reference-r{repeat}",
            role="reference",
            device="cuda",
            repeat=repeat,
            topology=topology,
            losses=gpu_losses,
        )
        _write_artifact(
            scenario / f"candidate-r{repeat}",
            role="candidate",
            device="npu",
            repeat=repeat,
            topology=topology,
            losses=npu_losses,
        )


def test_export_results_writes_requested_short_and_long_metrics(
    tmp_path: Path,
) -> None:
    short_root = tmp_path / "short"
    long_root = tmp_path / "long"
    output = tmp_path / "reports"
    _write_scenario(
        short_root,
        topology="ddp8",
        suffix=SHORT_SCENARIO_SUFFIX,
        gpu_losses=[8.0] * 10,
        npu_losses=[8.004] * 10,
    )
    gpu_long = [8.0 - index * 0.02 for index in range(30)]
    npu_long = [loss * 1.01 for loss in gpu_long]
    _write_scenario(
        long_root,
        topology="ddp8",
        suffix=LONG_SCENARIO_SUFFIX,
        gpu_losses=gpu_long,
        npu_losses=npu_long,
    )

    overall, json_path, markdown_path = export_results(
        short_root,
        long_root,
        output,
        short_topologies=("ddp8",),
        long_topologies=("ddp8",),
        long_config=DdpLongV2Config(
            minimum_steps=20,
            warmup_steps=2,
            window_size=3,
            sustained_window_size=5,
        ),
    )

    assert overall == "PASS"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["short_results"][0]["gpu_first_loss"] == 8.0
    assert payload["short_results"][0]["npu_first_loss"] == 8.004
    assert payload["short_results"][0]["relative_error"] == pytest.approx(0.0005)
    long_result = payload["long_results"][0]
    assert long_result["area_relative_error"] == pytest.approx(0.01)
    assert long_result["final_mean_relative_error"] == pytest.approx(0.01)
    assert long_result["smoothed_correlation"] == pytest.approx(1.0)
    assert long_result["maximum_sustained_window_relative_error"] == pytest.approx(
        0.01
    )
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "| ddp8 | PASS | 8 | 8.004 | 0.050000% |" in markdown
    assert "| ddp8 | PASS | 30 | 1.000000%" in markdown


def test_short_results_reject_mislabeled_gpu_artifact(tmp_path: Path) -> None:
    _write_scenario(
        tmp_path,
        topology="ddp8",
        suffix=SHORT_SCENARIO_SUFFIX,
        gpu_losses=[8.0] * 10,
        npu_losses=[8.0] * 10,
    )
    manifest_path = (
        tmp_path
        / f"migration-cuda-npu-ddp8{SHORT_SCENARIO_SUFFIX}"
        / "reference-r1"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["device_type"] = "npu"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = collect_short_results(tmp_path, topologies=("ddp8",))[0]

    assert result.status == "INVALID"
    assert "expected role/device reference/cuda" in (result.error or "")
