# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.glm5_2_mindstudio.monitor_analysis import (
    align_monitor_rows,
    read_monitor_rows,
    write_monitor_analysis,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class MonitorAnalysisTest(unittest.TestCase):
    def test_aligns_official_rows_by_rank_step_parameter_and_scope(self) -> None:
        reference = [{
            "rank": 0, "monitor": "weight_grad", "vpp_stage": "0",
            "step": "22", "module_name": "embedding.weight",
            "scope": "unreduced", "micro_step": "1", "norm": "2.0",
        }]
        candidate = [{
            "rank": 0, "monitor": "weight_grad", "vpp_stage": "0",
            "step": "22", "module_name": "embedding.weight",
            "scope": "unreduced", "micro_step": "1", "norm": "5.0",
        }]

        rows = align_monitor_rows(reference, candidate)

        self.assertEqual("matched", rows[0]["match_status"])
        self.assertEqual(3.0, rows[0]["signed_norm_difference"])
        self.assertEqual(1.5, rows[0]["relative_norm_error"])

    def test_report_uses_official_csv_and_writes_machine_readable_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference"
            candidate = root / "candidate"
            output = root / "analysis"
            fields = {
                "vpp_stage": "0", "step": "22",
                "module_name": "embedding.weight", "scope": "unreduced",
                "micro_step": "1", "min": "-1", "max": "1",
                "mean": "0", "norm": "2", "nans": "0",
            }
            _write_csv(
                reference / "rank_0" / "weight_grad_step22-22.csv",
                [fields],
            )
            _write_csv(
                candidate / "rank_0" / "weight_grad_step22-22.csv",
                [{**fields, "norm": "6"}],
            )

            with (
                patch(
                    "tests.glm5_2_mindstudio.monitor_analysis.section_heading",
                    return_value=object(),
                ),
                patch(
                    "tests.glm5_2_mindstudio.monitor_analysis.summary_table",
                    return_value=object(),
                ),
                patch(
                    "tests.glm5_2_mindstudio.monitor_analysis._layer_charts",
                    return_value=[],
                ),
                patch(
                    "tests.glm5_2_mindstudio.monitor_analysis._rank_step_heatmap",
                    return_value=None,
                ),
                patch(
                    "tests.glm5_2_mindstudio.monitor_analysis.interactive_table",
                    return_value=object(),
                ),
                patch(
                    "tests.glm5_2_mindstudio.monitor_analysis.save_panel_report",
                    side_effect=lambda **kwargs: kwargs["path"],
                ),
            ):
                summary = write_monitor_analysis(
                    reference_official=reference,
                    candidate_official=candidate,
                    output_directory=output,
                )

            self.assertEqual(1, summary["matched_rows"])
            self.assertEqual("22", summary["focus_step"])
            self.assertTrue((output / "aligned_metrics.csv").is_file())
            self.assertTrue((output / "anomaly_summary.csv").is_file())
            self.assertTrue((output / "rank_step_summary.csv").is_file())
            self.assertTrue((output / "analysis.json").is_file())

    def test_reads_rank_and_monitor_kind_from_official_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_csv(
                root / "rank_7" / "timestamp" / "weight_grad_step18-18.csv",
                [{
                    "vpp_stage": "0", "step": "18", "module_name": "p",
                    "scope": "reduced", "norm": "1",
                }],
            )

            rows = read_monitor_rows(root, "candidate")

            self.assertEqual(7, rows[0]["rank"])
            self.assertEqual("weight_grad", rows[0]["monitor"])
            self.assertEqual("candidate", rows[0]["role"])


if __name__ == "__main__":
    unittest.main()
