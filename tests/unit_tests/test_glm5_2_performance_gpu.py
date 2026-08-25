from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SUITE = Path(__file__).resolve().parents[1] / "glm5_2_performance-gpu"
sys.path.insert(0, str(SUITE))

from gpu_analysis import build_analysis, parse_trace, render_html_report  # noqa: E402
from gpu_config import GpuPerformanceConfig  # noqa: E402
from gpu_workflow import _training_command  # noqa: E402


class TestGpuPerformanceConfig(unittest.TestCase):
    def test_profile_window_must_fit_training(self):
        with self.assertRaisesRegex(ValueError, "cannot reach"):
            GpuPerformanceConfig(
                name="bad", steps=4, skip_steps=2, warmup_steps=1, active_steps=2
            )

    def test_rank_selector_is_validated(self):
        with self.assertRaisesRegex(ValueError, "profile_ranks"):
            GpuPerformanceConfig(name="bad", profile_ranks="rank-zero")

    def test_training_command_enables_bounded_torch_profiler(self):
        config = GpuPerformanceConfig(
            name="probe", steps=10, skip_steps=2, warmup_steps=1, active_steps=3
        )
        with tempfile.TemporaryDirectory() as temporary_directory, patch.dict(
            "os.environ", {"CUDA_VISIBLE_DEVICES": "0"}, clear=False
        ):
            command = _training_command(
                root=Path(temporary_directory),
                config=config,
                run_directory=Path(temporary_directory) / "run",
            )

        self.assertIn("--profiler.enable_profiling", command)
        self.assertIn("--profiler.profiler_skip_first=2", command)
        self.assertIn("--profiler.profiler_warmup=1", command)
        self.assertIn("--profiler.profiler_active=3", command)
        self.assertTrue(
            any(value.endswith("glm5_2_performance-gpu/capture_metrics.py") for value in command)
        )


class TestGpuTraceAnalysis(unittest.TestCase):
    def test_trace_categories_and_top_events(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            trace = Path(temporary_directory) / "rank3_trace.json"
            trace.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            {"ph": "X", "cat": "kernel", "name": "gemm", "dur": 120},
                            {"ph": "X", "cat": "kernel", "name": "ncclKernel_AllReduce", "dur": 80},
                            {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "dur": 10},
                            {"ph": "X", "cat": "cpu_op", "name": "aten::linear", "dur": 30},
                            {"ph": "M", "name": "process_name", "args": {}},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            result = parse_trace(trace)

        self.assertEqual(result["rank"], 3)
        self.assertEqual(result["categories"]["cuda_kernel"]["total_us"], 120.0)
        self.assertEqual(result["categories"]["communication"]["total_us"], 80.0)
        self.assertEqual(result["categories"]["cuda_runtime"]["count"], 1)
        self.assertEqual(result["categories"]["cpu_operator"]["count"], 1)

    def test_build_analysis_and_html_report(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = Path(temporary_directory) / "run"
            traces = run / "trainer_output" / "profiling" / "traces" / "iteration_4"
            traces.mkdir(parents=True)
            (traces / "rank0_trace.json").write_text(
                json.dumps({"traceEvents": [{"ph": "X", "cat": "kernel", "name": "gemm", "dur": 50}]}),
                encoding="utf-8",
            )
            (run / "metrics.jsonl").write_text(
                json.dumps({"step": 4, "rank": 0, "metrics": {"throughput(tps)": 1234.0}}) + "\n",
                encoding="utf-8",
            )
            analysis = build_analysis(run)
            report = run / "report.html"
            render_html_report(
                manifest={"run_name": "gpu-test", "topology": "single", "visible_devices": [0]},
                analysis=analysis,
                output_path=report,
            )
            content = report.read_text(encoding="utf-8")

        self.assertEqual(analysis["profiler"]["parsed_count"], 1)
        self.assertIn("throughput(tps)", content)
        self.assertIn("gemm", content)


if __name__ == "__main__":
    unittest.main()
