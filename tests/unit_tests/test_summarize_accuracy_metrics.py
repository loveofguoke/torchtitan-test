# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.summarize_accuracy_metrics import (
    _discover_pairs,
    _read_metrics,
    _series,
    _summarize,
)


def _write(path: Path, rows: list[tuple[int, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                {
                    "step": step,
                    "metrics": {
                        "loss_metrics/global_avg_loss": loss,
                        "grad_norm": grad_norm,
                    },
                }
            )
            + "\n"
            for step, loss, grad_norm in rows
        ),
        encoding="utf-8",
    )


def test_summarize_accuracy_metrics(tmp_path: Path) -> None:
    gpu_path = tmp_path / "gpu.jsonl"
    npu_path = tmp_path / "npu.jsonl"
    _write(gpu_path, [(1, 4.0, 2.0), (2, 2.0, 1.0)])
    _write(npu_path, [(1, 5.0, 2.5), (2, 1.5, 1.1)])

    gpu = _read_metrics(gpu_path)
    npu = _read_metrics(npu_path)
    summary = _summarize(_series(gpu, "loss"), _series(npu, "loss"))

    assert summary.reference_final == 2.0
    assert summary.candidate_final == 1.5
    assert summary.final_signed_difference == -0.5
    assert summary.mean_signed_difference == 0.25
    assert summary.mean_absolute_difference == 0.75
    assert summary.max_absolute_difference == 1.0
    assert summary.max_absolute_difference_step == 1
    assert summary.mean_relative_error == 0.25


def test_rejects_duplicate_steps(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    _write(path, [(1, 4.0, 2.0), (1, 3.0, 1.0)])

    with pytest.raises(ValueError, match="duplicate step 1"):
        _read_metrics(path)


def test_discovers_all_topology_pairs_for_profile(tmp_path: Path) -> None:
    for topology in ("single", "tp8"):
        base = tmp_path / topology / "s500" / "observations" / "training"
        _write(base / "reference-r1" / "training_metrics.jsonl", [(1, 4.0, 2.0)])
        _write(base / "candidate-r1" / "training_metrics.jsonl", [(1, 4.1, 2.1)])
    ignored = tmp_path / "cp8" / "s150" / "observations" / "training"
    _write(ignored / "reference-r1" / "training_metrics.jsonl", [(1, 4.0, 2.0)])
    _write(ignored / "candidate-r1" / "training_metrics.jsonl", [(1, 4.1, 2.1)])

    pairs = _discover_pairs(tmp_path, "s500")

    assert set(pairs) == {"single", "tp8"}
