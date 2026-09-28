# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.glm5_2_mindstudio.monitor_analysis import (
    _rank_scope_charts,
    _rank_scope_rows,
    _rank_step_summary,
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
    def test_rank_scope_rows_keep_rank_aggregates_separate(self) -> None:
        rows = []
        for rank, reference_norm, candidate_norm in (
            (0, "3", "4"),
            (1, "5", "12"),
        ):
            rows.extend(
                align_monitor_rows(
                    [{
                        "rank": rank, "monitor": "weight_grad",
                        "vpp_stage": "0", "step": "22",
                        "module_name": "p", "scope": "unreduced",
                        "micro_step": "1", "norm": reference_norm,
                    }],
                    [{
                        "rank": rank, "monitor": "weight_grad",
                        "vpp_stage": "0", "step": "22",
                        "module_name": "p", "scope": "unreduced",
                        "micro_step": "1", "norm": candidate_norm,
                    }],
                )
            )

        summary = _rank_scope_rows(rows)

        self.assertEqual([0, 1], [row["rank"] for row in summary])
        self.assertEqual([4.0, 12.0], [row["npu_aggregate_norm"] for row in summary])

    def test_rank_scope_charts_put_each_rank_in_a_separate_tab(self) -> None:
        rows = []
        for rank in (0, 1):
            rows.extend(
                align_monitor_rows(
                    [{
                        "rank": rank, "monitor": "weight_grad",
                        "vpp_stage": "0", "step": "22",
                        "module_name": "p", "scope": "unreduced",
                        "micro_step": "1", "norm": "3",
                    }],
                    [{
                        "rank": rank, "monitor": "weight_grad",
                        "vpp_stage": "0", "step": "22",
                        "module_name": "p", "scope": "unreduced",
                        "micro_step": "1", "norm": "4",
                    }],
                )
            )

        with (
            patch(
                "tests.glm5_2_mindstudio.monitor_analysis.echarts_line",
                side_effect=lambda **kwargs: kwargs,
            ),
            patch(
                "tests.glm5_2_mindstudio.monitor_analysis.tabbed_views",
                side_effect=lambda **kwargs: kwargs,
            ),
        ):
            charts = _rank_scope_charts(rows)

        self.assertEqual(1, len(charts))
        self.assertEqual(
            ["Rank 0", "Rank 1"],
            [name for name, _ in charts[0]["views"]],
        )
        self.assertEqual(
            ["GPU Rank 1", "NPU Rank 1"],
            [series[0] for series in charts[0]["views"][1][1]["series"]],
        )

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
        self.assertEqual(0.6, rows[0]["scaled_norm_error"])

    def test_reduce_localization_pairs_the_same_parameter(self) -> None:
        rows = align_monitor_rows(
            [
                {
                    "rank": 0, "monitor": "weight_grad", "vpp_stage": "0",
                    "step": "22", "module_name": "p", "scope": scope,
                    "micro_step": "1", "norm": reference,
                }
                for scope, reference in (("unreduced", "10"), ("reduced", "10"))
            ],
            [
                {
                    "rank": 0, "monitor": "weight_grad", "vpp_stage": "0",
                    "step": "22", "module_name": "p", "scope": scope,
                    "micro_step": "1", "norm": candidate,
                }
                for scope, candidate in (("unreduced", "10.1"), ("reduced", "20"))
            ],
        )

        summary = _rank_step_summary(rows)

        self.assertEqual("reduced", summary[0]["worst_scope"])
        self.assertIn("reduce 后误差明显放大", summary[0]["localization_hint"])

    def test_near_zero_reference_keeps_absolute_and_bounded_scaled_error(self) -> None:
        rows = align_monitor_rows(
            [{
                "rank": 0, "monitor": "weight_grad", "vpp_stage": "0",
                "step": "22", "module_name": "p", "scope": "reduced",
                "micro_step": "1", "norm": "1e-14",
            }],
            [{
                "rank": 0, "monitor": "weight_grad", "vpp_stage": "0",
                "step": "22", "module_name": "p", "scope": "reduced",
                "micro_step": "1", "norm": "1e-6",
            }],
        )

        self.assertTrue(rows[0]["reference_norm_near_zero"])
        self.assertAlmostEqual(1e-6, rows[0]["absolute_norm_difference"])
        self.assertAlmostEqual(1.0, rows[0]["scaled_norm_error"])

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
                    "tests.glm5_2_mindstudio.monitor_analysis._overall_scope_charts",
                    return_value=[],
                ),
                patch(
                    "tests.glm5_2_mindstudio.monitor_analysis._rank_scope_charts",
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
            self.assertEqual("23", summary["focus_step"])
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
