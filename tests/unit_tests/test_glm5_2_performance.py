import json
import tempfile
import unittest
from pathlib import Path

from tests.glm5_2_performance.analysis import (
    build_analysis,
    render_html_report,
)
from tests.glm5_2_performance.config import (
    PerformanceConfig,
    profiler_presets,
    standard_topologies,
)


class TestPerformanceConfig(unittest.TestCase):
    def test_standard_topologies_cover_declared_world_size(self):
        topologies = standard_topologies()
        self.assertEqual(topologies["single"].world_size, 1)
        self.assertEqual(topologies["fsdp2-tp4"].world_size, 8)
        self.assertIn(
            "--parallelism.tensor_parallel_degree=4",
            topologies["fsdp2-tp4"].command_args(),
        )

    def test_profile_window_must_fit_training_steps(self):
        with self.assertRaisesRegex(ValueError, "cannot reach"):
            PerformanceConfig(
                name="invalid",
                steps=4,
                skip_steps=2,
                warmup_steps=1,
                active_steps=2,
            )

    def test_presets_progress_from_overview_to_runtime(self):
        presets = profiler_presets()
        self.assertEqual(presets["overview"].level, "level0")
        self.assertEqual(presets["distributed"].profile_ranks, "all")
        self.assertTrue(presets["kernel"].record_shapes)
        self.assertTrue(presets["runtime"].with_stack)


class TestPerformanceAnalysis(unittest.TestCase):
    def test_metrics_and_csv_are_rendered_into_report(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run = root / "run"
            profiler = run / "trainer_output" / "profiling" / "traces"
            profiler.mkdir(parents=True)
            records = [
                {
                    "step": step,
                    "rank": 0,
                    "metrics": {
                        "time_metrics/end_to_end(s)": 0.5 + step * 0.01,
                        "throughput(tps)": 1000 + step,
                    },
                }
                for step in range(1, 5)
            ]
            (run / "metrics.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            (profiler / "kernel_details.csv").write_text(
                "Name,Duration(us)\nmatmul,20\nmatmul,30\nall_reduce,10\n",
                encoding="utf-8",
            )

            analysis = build_analysis(run)
            self.assertEqual(analysis["profiler_inventory"]["file_count"], 1)
            self.assertEqual(
                analysis["top_csv_tables"][0]["rows"][0]["name"], "matmul"
            )
            self.assertEqual(
                analysis["top_csv_tables"][0]["category_totals_us"][
                    "communication"
                ],
                10.0,
            )

            config = PerformanceConfig(name="test", steps=8)
            manifest = {
                "run_name": "test-npu-single-overview",
                "device": "npu",
                "topology": "single",
                "preset": "overview",
                "config": config.as_dict(),
            }
            report = root / "report.html"
            render_html_report(
                manifest=manifest,
                analysis=analysis,
                output_path=report,
            )
            content = report.read_text(encoding="utf-8")
            self.assertIn("GLM-5.2 performance probe", content)
            self.assertIn("matmul", content)
            self.assertIn("blue region", content)


if __name__ == "__main__":
    unittest.main()
