# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from tests.glm5_2_precision.artifacts import (
    PrecisionArtifactWriter,
    TrainingMetric,
)
from tests.glm5_2_precision.ddp_long_v2 import (
    DdpLongV2Config,
    compare_ddp_long_v2,
    main,
)


def _contract(*, seed: int = 61) -> dict[str, object]:
    return {
        "scenario_name": "migration-cuda-npu-ddp8-long-v2",
        "checkpoint_sha256": "checkpoint",
        "data_sha256": {"tokens": "data"},
        "training": {"seed": seed, "steps": 1000},
        "topology": {
            "name": "ddp8",
            "world_size": 8,
            "data_parallel_replicate_degree": 8,
            "data_parallel_shard_degree": 1,
            "tensor_parallel_degree": 1,
            "pipeline_parallel_degree": 1,
            "expert_parallel_degree": 1,
        },
    }


def _artifact(
    root: Path,
    name: str,
    losses: list[float],
    *,
    grad_scale: float = 1.0,
    contract: dict[str, object] | None = None,
) -> Path:
    metrics = (
        TrainingMetric(
            step=index,
            loss=loss,
            global_max_loss=loss,
            grad_norm=(1.0 + index / 1000.0) * grad_scale,
            extras={},
        )
        for index, loss in enumerate(losses, start=1)
    )
    return PrecisionArtifactWriter(root / name).write(
        metadata={"role": "reference" if name.startswith("gpu") else "candidate"},
        training_contract=contract or _contract(),
        metrics=metrics,
    )


def _curves(steps: int = 1000) -> tuple[list[float], list[float]]:
    reference = [8.0 - index * 0.004 for index in range(steps)]
    candidate = [value * 1.005 for value in reference]
    return reference, candidate


def test_ddp_long_v2_passes_overall_curve_with_pointwise_offset(
    tmp_path: Path,
) -> None:
    reference, candidate = _curves()
    # A uniform 1.5% offset fails the old one-percent pointwise MARE rule but
    # remains within the convergence-curve limits.
    candidate = [value * 1.015 for value in reference]
    gpu = [
        _artifact(tmp_path, "gpu-r1", reference),
        _artifact(tmp_path, "gpu-r2", reference),
    ]
    npu = [
        _artifact(tmp_path, "npu-r1", candidate),
        _artifact(tmp_path, "npu-r2", candidate, grad_scale=1.1),
    ]

    result = compare_ddp_long_v2(
        gpu,
        npu,
        config=DdpLongV2Config(
            minimum_steps=1000,
            sustained_window_size=200,
        ),
    )

    assert result.status == "PASS"
    assert result.loss.mean_absolute_relative_error == pytest.approx(0.015)
    assert result.curve.area_relative_error == pytest.approx(0.015)
    assert result.curve.final_mean_relative_error == pytest.approx(0.015)
    assert result.curve.smoothed_correlation == pytest.approx(1.0)
    assert result.npu_repeat_grad_norm_mare > 0.09
    assert all(item.passed for item in result.criteria)


def test_ddp_long_v2_detects_large_sustained_curve_gap(tmp_path: Path) -> None:
    reference, candidate = _curves()
    # A 200-step middle window differs by 4%. Its whole-run area and final mean
    # remain acceptable, so the sustained-window convergence gate catches it.
    candidate = list(reference)
    for index in range(300, 500):
        candidate[index] *= 1.04
    gpu = [
        _artifact(tmp_path, "gpu-r1", reference),
        _artifact(tmp_path, "gpu-r2", reference),
    ]
    npu = [
        _artifact(tmp_path, "npu-r1", candidate),
        _artifact(tmp_path, "npu-r2", candidate),
    ]

    result = compare_ddp_long_v2(
        gpu,
        npu,
        config=DdpLongV2Config(
            minimum_steps=1000,
            sustained_window_size=200,
        ),
    )

    assert result.status == "FAIL"
    sustained = next(
        item
        for item in result.criteria
        if item.name == "Maximum sustained-window mean loss difference"
    )
    assert not sustained.passed
    assert result.curve.maximum_sustained_window_relative_error > 0.03


def test_ddp_long_v2_rejects_opposite_smoothed_trend(tmp_path: Path) -> None:
    reference, _ = _curves()
    candidate = list(reversed(reference))
    gpu = [
        _artifact(tmp_path, "gpu-r1", reference),
        _artifact(tmp_path, "gpu-r2", reference),
    ]
    npu = [
        _artifact(tmp_path, "npu-r1", candidate),
        _artifact(tmp_path, "npu-r2", candidate),
    ]

    result = compare_ddp_long_v2(
        gpu,
        npu,
        config=DdpLongV2Config(
            minimum_steps=1000,
            sustained_window_size=200,
        ),
    )

    assert result.status == "FAIL"
    correlation = next(
        item
        for item in result.criteria
        if item.name == "Smoothed loss-curve correlation"
    )
    assert not correlation.passed
    assert result.curve.smoothed_correlation == pytest.approx(-1.0)


def test_ddp_long_v2_marks_unstable_endpoint_inconclusive(tmp_path: Path) -> None:
    reference, candidate = _curves()
    unstable_candidate = [value * 1.03 for value in candidate]
    gpu = [
        _artifact(tmp_path, "gpu-r1", reference),
        _artifact(tmp_path, "gpu-r2", reference),
    ]
    npu = [
        _artifact(tmp_path, "npu-r1", candidate),
        _artifact(tmp_path, "npu-r2", unstable_candidate),
    ]

    result = compare_ddp_long_v2(
        gpu,
        npu,
        config=DdpLongV2Config(
            minimum_steps=1000,
            sustained_window_size=200,
        ),
    )

    assert result.status == "INCONCLUSIVE"
    assert result.inconclusive_reasons


def test_ddp_long_v2_rejects_contract_mismatch(tmp_path: Path) -> None:
    reference, candidate = _curves()
    gpu = [
        _artifact(tmp_path, "gpu-r1", reference),
        _artifact(tmp_path, "gpu-r2", reference),
    ]
    npu = [
        _artifact(tmp_path, "npu-r1", candidate),
        _artifact(tmp_path, "npu-r2", candidate, contract=_contract(seed=62)),
    ]

    with pytest.raises(ValueError, match="different training contracts"):
        compare_ddp_long_v2(
            gpu,
            npu,
            config=DdpLongV2Config(minimum_steps=1000),
        )


def test_ddp_long_v2_cli_writes_reports(tmp_path: Path) -> None:
    reference, candidate = _curves()
    gpu = [
        _artifact(tmp_path, "gpu-r1", reference),
        _artifact(tmp_path, "gpu-r2", reference),
    ]
    npu = [
        _artifact(tmp_path, "npu-r1", candidate),
        _artifact(tmp_path, "npu-r2", candidate),
    ]
    output = tmp_path / "report"

    arguments: list[str] = []
    for path in gpu:
        arguments.extend(("--gpu-artifact", str(path)))
    for path in npu:
        arguments.extend(("--npu-artifact", str(path)))
    arguments.extend(("--output-dir", str(output), "--minimum-steps", "1000"))
    arguments.extend(("--sustained-window-size", "200"))

    assert main(arguments) == 0
    assert (output / "ddp_long_v2_summary.json").is_file()
    assert (output / "ddp_long_v2_report.md").is_file()
    assert "Whole-run loss AUC" in (
        output / "ddp_long_v2_report.md"
    ).read_text(encoding="utf-8")
