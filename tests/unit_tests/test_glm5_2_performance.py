import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tests.glm5_2_performance.analysis import (
    build_analysis,
    inspect_profiler_deliverables,
    render_html_report,
    summarize_profile_phases,
)
from tests.glm5_2_performance.config import (
    PerformanceConfig,
    profiler_presets,
    standard_topologies,
)
from tests.glm5_2_performance.dynamic_profile import (
    build_dynamic_profile_config,
    write_dynamic_profile_config,
)
from tests.glm5_2_performance.workflow import _run_name


class TestPerformanceConfig(unittest.TestCase):
    def test_run_name_uses_readable_config_and_digest(self):
        config = PerformanceConfig(name="glm5-probe")
        run_name = _run_name(config, "npu")

        self.assertTrue(run_name.startswith("npu-single-bf16-s12-"))
        self.assertNotIn("glm5-probe", run_name)
        self.assertEqual(len(run_name.rsplit("-", 1)[1]), 8)

    def test_run_name_distinguishes_effective_parse_mode(self):
        config = PerformanceConfig(name="glm5-probe")
        sync = profiler_presets()["overview"]
        offline = replace(sync, parse_mode="offline")

        sync_name = _run_name(config, "npu", sync)
        offline_name = _run_name(config, "npu", offline)

        self.assertIn("-overview-", sync_name)
        self.assertNotIn("-overview-sync-", sync_name)
        self.assertIn("-overview-offline-", offline_name)
        self.assertNotEqual(sync_name, offline_name)

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
        self.assertEqual(presets["standard"].aic_metrics, "pipe_utilization")
        self.assertEqual(presets["distributed"].profile_ranks, "all")
        self.assertEqual(presets["distributed"].parse_mode, "offline")
        self.assertTrue(presets["distributed"].system_interconnection)
        self.assertTrue(presets["kernel"].record_shapes)
        self.assertTrue(presets["runtime"].with_stack)
        self.assertTrue(presets["runtime"].with_modules)
        self.assertEqual(presets["runtime"].gc_detect_threshold, 1.0)

    def test_preset_exports_official_profiler_controls(self):
        environment = profiler_presets()["runtime"].environment()
        prefix = "TORCHTITAN_NPU_PROFILER_"
        self.assertEqual(environment[prefix + "PARSE_MODE"], "sync")
        self.assertEqual(environment[prefix + "HOST_SYSTEM"], "cpu,mem")
        self.assertEqual(environment[prefix + "WITH_MODULES"], "true")

    def test_dynamic_config_uses_official_schema_and_atomic_writer(self):
        preset = replace(profiler_presets()["overview"], parse_mode="async")
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = build_dynamic_profile_config(
                preset=preset,
                profile_directory=root / "profiles",
                start_step=10,
                active_steps=3,
                warmup_steps=1,
                enabled=True,
            )
            path = write_dynamic_profile_config(root / "control", config)
            loaded = json.loads(path.read_text(encoding="utf-8"))

        self.assertTrue(loaded["is_valid"])
        self.assertEqual(loaded["rank_list"], [0])
        self.assertEqual(
            loaded["experimental_config"]["profiler_level"], "Level0"
        )
        self.assertEqual(
            loaded["experimental_config"]["export_type"], ["text", "db"]
        )

    def test_rank_filtered_dynamic_config_rejects_synchronous_parse(self):
        with self.assertRaisesRegex(ValueError, "rank-filtered"):
            build_dynamic_profile_config(
                preset=profiler_presets()["overview"],
                profile_directory=Path("profiles"),
                start_step=10,
                active_steps=3,
                warmup_steps=1,
                enabled=True,
            )


class TestPerformanceAnalysis(unittest.TestCase):
    def test_profile_phases_separate_collection_and_sync_parse_tax(self):
        records = [
            {
                "step": step,
                "metrics": {
                    "time_metrics/end_to_end(s)": value,
                    "throughput(tps)": 1000.0 / value,
                },
            }
            for step, value in enumerate(
                (9.0, 1.0, 1.1, 1.25, 1.35, 11.0, 1.0, 1.0),
                start=1,
            )
        ]
        config = PerformanceConfig(
            name="phases",
            steps=8,
            skip_steps=2,
            warmup_steps=1,
            active_steps=2,
        ).as_dict()
        phases = summarize_profile_phases(
            records,
            config=config,
            profiler_environment={
                "TORCHTITAN_NPU_PROFILER_PARSE_MODE": "sync"
            },
        )

        self.assertEqual(phases["phases"]["baseline"]["steps"], [2])
        self.assertEqual(phases["phases"]["profile_active"]["steps"], [4, 5])
        self.assertEqual(phases["phases"]["sync_parse_boundary"]["steps"], [6])
        self.assertAlmostEqual(
            phases["comparison"]["active_step_overhead_percent"], 30.0
        )
        self.assertAlmostEqual(
            phases["comparison"]["parse_stall_excess_seconds"], 10.0
        )

    def test_official_ascend_deliverables_are_classified(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            profiler = Path(temporary_directory)
            profile = profiler / "rank_0_ascend_pt"
            output = profile / "ASCEND_PROFILER_OUTPUT"
            output.mkdir(parents=True)
            (profile / "profiler_info.json").write_text("{}", encoding="utf-8")
            (output / "trace_view.json").write_text("{}", encoding="utf-8")
            (output / "analysis.db").write_bytes(b"sqlite")

            deliverables = inspect_profiler_deliverables(profiler)

            self.assertTrue(deliverables["insight_ready"])
            self.assertTrue(deliverables["ascend_profiles"][0]["raw_ready"])
            self.assertTrue(
                deliverables["ascend_profiles"][0]["parsed_database"]
            )

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
                "Name,Duration(us),Input Shapes,Input Data Types\n"
                "matmul,20,2x4,FP16\nmatmul,30,2x4,FP16\n"
                "all_reduce,10,2x4,FP16\nsort_aicpu,5,8,INT64\n",
                encoding="utf-8",
            )
            (profiler / "l2_cache.csv").write_text(
                "Op Name,Hit Rate,Victim Rate\n"
                "matmul,0.8,0.1\nmatmul,0.6,0.3\n",
                encoding="utf-8",
            )
            (run / "runtime.log").write_text(
                "Compiling each TransformerBlock with torch.compile\n"
                "Graph break from user code\n"
                "CAUTION: an op will fall back to run on the CPU\n"
                "kernel is running on AiCpu\n"
                "CANN profiling data parsed in a total time of 0:01:21.25\n"
                "All profiling data parsed in a total time of 0:01:47.5\n",
                encoding="utf-8",
            )

            analysis = build_analysis(run)
            self.assertEqual(analysis["profiler_inventory"]["file_count"], 2)
            self.assertEqual(
                analysis["top_csv_tables"][0]["rows"][0]["name"], "matmul"
            )
            self.assertEqual(
                analysis["top_csv_tables"][0]["category_totals_us"][
                    "communication"
                ],
                10.0,
            )
            self.assertEqual(
                analysis["top_csv_tables"][0]["category_totals_us"]["aicpu"],
                5.0,
            )
            self.assertEqual(
                analysis["compiler_diagnostics"]["npu_cpu_fallbacks"], 1
            )
            self.assertEqual(
                analysis["compiler_diagnostics"]["aicpu_fallbacks"], 1
            )
            self.assertEqual(
                analysis["compiler_diagnostics"]["all_parse_seconds"], 107.5
            )
            self.assertEqual(
                analysis["kernel_shape_tables"][0]["rows"][0]["input_shapes"],
                "2x4",
            )
            self.assertEqual(
                analysis["kernel_shape_tables"][0]["rows"][0]["calls"], 2
            )
            self.assertEqual(
                analysis["l2_cache_tables"][0]["rows"][0]["mean_hit_rate"],
                0.7,
            )

            config = PerformanceConfig(
                name="test",
                steps=8,
                extra_args=("--compile.enable",),
            )
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
            self.assertIn("Official Ascend deliverables", content)
            self.assertIn("Open in MindStudio Insight", content)
            self.assertIn("torch.compile health", content)
            self.assertIn("Graph-break messages", content)
            self.assertIn("Top kernel shape signatures", content)
            self.assertIn("L2 cache samples", content)


if __name__ == "__main__":
    unittest.main()
