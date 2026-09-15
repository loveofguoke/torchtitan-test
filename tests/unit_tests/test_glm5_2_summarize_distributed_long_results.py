# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

from tests.glm5_2_precision.summarize_distributed_long_results import (
    summarize_distributed_long_results,
)


def _write_result(
    report_root: Path,
    topology: str,
    *,
    status: str = "PASS",
) -> None:
    directory = report_root / topology
    directory.mkdir(parents=True)
    (directory / "ddp_long_v2_summary.json").write_text(
        json.dumps(
            {
                "status": status,
                "step_range": [1, 5000],
                "loss": {"count": 5000},
                "curve": {
                    "area_relative_error": 0.012,
                    "final_mean_relative_error": 0.015,
                    "smoothed_correlation": 0.998,
                    "maximum_sustained_window_relative_error": 0.021,
                },
            }
        ),
        encoding="utf-8",
    )


def test_summarizer_writes_compact_outputs(tmp_path: Path) -> None:
    report_root = tmp_path / "reports"
    _write_result(report_root, "ddp8")
    _write_result(report_root, "tp8")

    status, results, json_path, markdown_path, table = (
        summarize_distributed_long_results(
            report_root,
            topologies=("ddp8", "tp8"),
        )
    )

    assert status == "PASS"
    assert len(results) == 2
    assert json_path.is_file()
    assert markdown_path.is_file()
    assert "Overall: **PASS** (2/2 PASS)" in markdown_path.read_text(
        encoding="utf-8"
    )
    assert "5000 (1..5000)" in table
    assert "1.200%" in table
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["passed"] == 2
    assert payload["results"][0]["topology"] == "ddp8"


def test_summarizer_marks_missing_topology_incomplete(tmp_path: Path) -> None:
    report_root = tmp_path / "reports"
    _write_result(report_root, "ddp8")

    status, results, _, markdown_path, _ = summarize_distributed_long_results(
        report_root,
        topologies=("ddp8", "ep8"),
    )

    assert status == "INCOMPLETE"
    assert results[1].status == "MISSING"
    assert "Missing or invalid reports" in markdown_path.read_text(encoding="utf-8")
