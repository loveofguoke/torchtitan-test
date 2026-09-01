import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from tests.glm5_2_performance.analysis import (
    build_analysis,
    inspect_profiler_deliverables,
    render_html_report,
    summarize_profile_phases,
)
from tests.glm5_2_performance.config import (
    PerformanceConfig,
    all_profiler_presets,
    performance_topologies,
    profiler_presets,
    standard_topologies,
)
from tests.glm5_2_performance.dynamic_profile import (
    build_dynamic_profile_config,
    write_dynamic_profile_config,
)
from tests.glm5_2_performance.workflow import (
    _config_is_compatible,
    _msprof_analyze_executable,
    _msprof_analyze_workers,
    _profiled_rank_count,
    _run_name,
    _safe_distributed_parse_preset,
    _write_suite_report,
)
from tests.glm5_2_performance.visualization import (
    inspect_analysis_outputs,
    inspect_memory_visualizations,
    inspect_stack_visualizations,
    inspect_tensorboard,
    render_mindstudio_flamegraphs,
)


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

    def test_distributed_npu_sync_parse_moves_offline(self):
        sync = profiler_presets()["overview"]

        distributed = _safe_distributed_parse_preset(
            sync,
            device="npu",
            world_size=8,
        )

        self.assertEqual(distributed.parse_mode, "offline")
        self.assertIs(
            _safe_distributed_parse_preset(
                sync,
                device="npu",
                world_size=1,
            ),
            sync,
        )
        self.assertIs(
            _safe_distributed_parse_preset(
                sync,
                device="cuda",
                world_size=8,
            ),
            sync,
        )

    def test_run_name_exposes_nondefault_reduction_precision(self):
        config = PerformanceConfig(
            name="glm5-probe", mixed_precision_reduce="bfloat16"
        )

        run_name = _run_name(config, "npu")

        self.assertIn("-reduce-bf16-", run_name)

    def test_run_name_exposes_profiler_off_baseline(self):
        config = PerformanceConfig(name="glm5-probe", profiler_enabled=False)

        self.assertIn("-profiler-off-", _run_name(config, "npu"))

    def test_run_name_exposes_repeat_index(self):
        config = PerformanceConfig(name="glm5-probe", replicate=2)

        self.assertIn("-r2-", _run_name(config, "npu"))

    def test_isolated_msprof_analyze_executable_can_be_selected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            executable = Path(temporary_directory) / "msprof-analyze"
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            executable.chmod(0o755)
            with mock.patch.dict(
                os.environ,
                {"TORCHTITAN_MSPROF_ANALYZE": str(executable)},
            ):
                selected = _msprof_analyze_executable()

        self.assertEqual(selected, str(executable.resolve()))

    def test_msprof_analyze_workers_are_bounded(self):
        with mock.patch.dict(
            os.environ,
            {"TORCHTITAN_MSPROF_ANALYZE_WORKERS": "2"},
        ):
            self.assertEqual(_msprof_analyze_workers(), 2)
        for value in ("0", "9", "invalid"):
            with self.subTest(value=value), mock.patch.dict(
                os.environ,
                {"TORCHTITAN_MSPROF_ANALYZE_WORKERS": value},
            ):
                with self.assertRaisesRegex(
                    ValueError, "TORCHTITAN_MSPROF_ANALYZE_WORKERS"
                ):
                    _msprof_analyze_workers()

    def test_additive_default_fields_match_old_capture_manifests(self):
        config = PerformanceConfig(name="legacy")
        saved = json.loads(json.dumps(config.as_dict()))
        saved.pop("profiler_enabled")
        saved.pop("replicate")

        self.assertTrue(_config_is_compatible(saved, config))
        self.assertFalse(
            _config_is_compatible(
                saved, replace(config, profiler_enabled=False)
            )
        )

    def test_standard_topologies_cover_declared_world_size(self):
        topologies = standard_topologies()
        self.assertEqual(topologies["single"].world_size, 1)
        self.assertEqual(topologies["fsdp2-tp4"].world_size, 8)
        self.assertIn(
            "--parallelism.tensor_parallel_degree=4",
            topologies["fsdp2-tp4"].command_args(),
        )

    def test_performance_topologies_add_health_safe_lab_shapes(self):
        topologies = performance_topologies()

        self.assertEqual(topologies["ddp4"].world_size, 4)
        self.assertEqual(topologies["fsdp4"].dp_shard, 4)
        self.assertEqual(topologies["tp4"].tp, 4)
        self.assertEqual(topologies["cp4"].cp, 4)
        self.assertEqual(topologies["pp4"].pp, 4)
        self.assertEqual(topologies["ep4"].ep, 4)
        self.assertEqual(topologies["fsdp2-tp2"].world_size, 4)

    def test_profile_window_must_fit_training_steps(self):
        with self.assertRaisesRegex(ValueError, "cannot reach"):
            PerformanceConfig(
                name="invalid",
                steps=4,
                skip_steps=2,
                warmup_steps=1,
                active_steps=2,
            )

    def test_reduction_precision_is_bounded_for_performance_override(self):
        with self.assertRaisesRegex(ValueError, "mixed_precision_reduce"):
            PerformanceConfig(name="invalid", mixed_precision_reduce="int8")

    def test_presets_progress_from_overview_to_runtime(self):
        presets = profiler_presets()
        self.assertEqual(presets["overview"].level, "level0")
        self.assertEqual(presets["standard"].aic_metrics, "pipe_utilization")
        self.assertEqual(presets["distributed"].profile_ranks, "all")
        self.assertEqual(presets["distributed"].parse_mode, "offline")
        self.assertTrue(presets["distributed"].system_interconnection)
        self.assertTrue(presets["kernel"].record_shapes)
        self.assertTrue(presets["flamegraph"].with_stack)
        self.assertTrue(presets["flamegraph"].export_stacks)
        self.assertTrue(presets["runtime"].with_stack)
        self.assertTrue(presets["runtime"].with_modules)
        self.assertTrue(presets["runtime"].export_stacks)
        self.assertTrue(presets["runtime"].export_memory_timeline)
        self.assertEqual(presets["runtime"].gc_detect_threshold, 1.0)
        self.assertTrue(presets["operator"].with_flops)
        self.assertTrue(presets["operator"].record_op_args)
        self.assertTrue(presets["memory"].export_memory_timeline)
        self.assertTrue(presets["system"].system_io)
        self.assertTrue(presets["system"].system_interconnection)
        self.assertNotIn("comparison", all_profiler_presets())
        self.assertNotIn("standard", all_profiler_presets())

    def test_cluster_requires_a_preset_that_captures_multiple_ranks(self):
        presets = profiler_presets()

        self.assertEqual(_profiled_rank_count(presets["overview"], 8), 1)
        self.assertEqual(_profiled_rank_count(presets["distributed"], 8), 8)
        self.assertEqual(
            _profiled_rank_count(
                replace(presets["overview"], profile_ranks="0,2,9"), 8
            ),
            2,
        )

    def test_preset_exports_official_profiler_controls(self):
        environment = profiler_presets()["runtime"].environment()
        overview_environment = profiler_presets()["overview"].environment()
        prefix = "TORCHTITAN_NPU_PROFILER_"
        self.assertEqual(environment[prefix + "PARSE_MODE"], "sync")
        self.assertEqual(environment[prefix + "HOST_SYSTEM"], "cpu,mem")
        self.assertEqual(environment[prefix + "WITH_MODULES"], "true")
        self.assertEqual(environment[prefix + "EXPORT_STACKS"], "true")
        self.assertEqual(
            environment[prefix + "EXPORT_MEMORY_TIMELINE"], "true"
        )
        self.assertNotIn(prefix + "EXPORT_STACKS", overview_environment)
        self.assertNotIn(
            prefix + "EXPORT_MEMORY_TIMELINE", overview_environment
        )
        self.assertNotIn(prefix + "MSPROF_TX", overview_environment)

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
    def test_official_mindstudio_flamegraph_renderer(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            profiler = root / "profiling"
            database = (
                profiler
                / "rank_0_123_ascend_pt"
                / "ASCEND_PROFILER_OUTPUT"
                / "ascend_pytorch_profiler_0.db"
            )
            database.parent.mkdir(parents=True)
            database.write_bytes(b"database")
            renderer = root / "flamegraph.py"
            renderer.write_text(
                "import argparse\n"
                "from pathlib import Path\n"
                "parser = argparse.ArgumentParser()\n"
                "parser.add_argument('db_path')\n"
                "parser.add_argument('-o', '--output', required=True)\n"
                "args = parser.parse_args()\n"
                "output = Path(args.output)\n"
                "output.mkdir(parents=True, exist_ok=True)\n"
                "(output / 'flamegraph.html').write_text('<html></html>')\n",
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {"TORCHTITAN_MSINSIGHT_FLAMEGRAPH": str(renderer)},
            ):
                generated = render_mindstudio_flamegraphs(profiler)
                inventory = inspect_stack_visualizations(profiler)

        self.assertEqual(len(generated), 1)
        self.assertEqual(len(inventory["mindstudio_flamegraphs"]), 1)
        self.assertEqual(len(inventory["mindstudio_flamegraph_logs"]), 1)

    def test_visualization_artifacts_are_discovered(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = Path(temporary_directory) / "run"
            profiler = run / "trainer_output" / "profiling" / "traces"
            stacks = profiler / "stacks"
            stacks.mkdir(parents=True)
            stack = stacks / "rank_0_step_12_npu_stacks.log"
            stack.write_text("train;forward 42\n", encoding="utf-8")
            flamegraph = stacks / "rank_0_step_12_npu_stacks_flamegraph.svg"
            flamegraph.write_text("<svg></svg>\n", encoding="utf-8")
            official_flamegraph = (
                profiler
                / "mindstudio_flamegraphs"
                / "rank_0"
                / "flamegraph.html"
            )
            official_flamegraph.parent.mkdir(parents=True)
            official_flamegraph.write_text(
                "<html></html>\n", encoding="utf-8"
            )
            tensorboard = run / "trainer_output" / "tensorboard"
            tensorboard.mkdir(parents=True)
            event = tensorboard / "events.out.tfevents.1.host.1.0"
            event.write_bytes(b"event")
            ascend_output = (
                profiler
                / "rank_0_123_ascend_pt"
                / "ASCEND_PROFILER_OUTPUT"
            )
            ascend_output.mkdir(parents=True)
            (ascend_output / "trace_view.json").write_text(
                "{}\n", encoding="utf-8"
            )
            graph_root = run / "graph_visualization" / "rank_0"
            (graph_root / "torch_trace").mkdir(parents=True)
            (graph_root / "torch_trace" / "dedicated_log_torch_trace.log").write_text(
                "trace\n", encoding="utf-8"
            )
            memory_root = profiler / "memory_timeline"
            memory_root.mkdir(parents=True)
            (memory_root / "rank_0_memory_timeline.html").write_text(
                "<html></html>\n", encoding="utf-8"
            )
            (memory_root / "rank_0_memory_timeline.json.gz").write_bytes(
                b"series"
            )
            (memory_root / "rank_0_memory_timeline_raw.json.gz").write_bytes(
                b"raw"
            )
            advisor_output = run / "advisor" / "advice.html"
            advisor_output.parent.mkdir(parents=True)
            advisor_output.write_text("advice", encoding="utf-8")
            (run / "advisor.json").write_text(
                json.dumps(
                    {
                        "return_code": 0,
                        "command": [
                            "msprof-analyze",
                            "advisor",
                            "-o",
                            str(advisor_output.parent),
                        ],
                    }
                ),
                encoding="utf-8",
            )

            stack_inventory = inspect_stack_visualizations(profiler)
            memory_inventory = inspect_memory_visualizations(profiler)
            advisor_inventory = inspect_analysis_outputs(run)
            tensorboard_inventory = inspect_tensorboard(run)
            analysis = build_analysis(run)

        self.assertEqual(
            stack_inventory["stack_files"],
            ["stacks/rank_0_step_12_npu_stacks.log"],
        )
        self.assertEqual(
            stack_inventory["flamegraphs"],
            ["stacks/rank_0_step_12_npu_stacks_flamegraph.svg"],
        )
        self.assertEqual(
            stack_inventory["mindstudio_flamegraphs"],
            ["mindstudio_flamegraphs/rank_0/flamegraph.html"],
        )
        self.assertTrue(tensorboard_inventory["ready"])
        self.assertTrue(memory_inventory["ready"])
        self.assertTrue(advisor_inventory["ready"])
        self.assertEqual(advisor_inventory["entries"][0]["file_count"], 1)
        self.assertEqual(
            memory_inventory["raw"],
            ["memory_timeline/rank_0_memory_timeline_raw.json.gz"],
        )
        self.assertEqual(tensorboard_inventory["event_count"], 1)
        self.assertIn("tensorboard", tensorboard_inventory["command"])
        self.assertTrue(
            analysis["deliverables"]["ascend_profiles"][0][
                "trace_view_files"
            ]
        )
        self.assertTrue(analysis["graph_visualizations"]["structured_traces"])
        self.assertTrue(analysis["memory_visualizations"]["html"])
        self.assertTrue(analysis["analysis_outputs"]["ready"])

    def test_suite_report_indexes_topology_and_preset(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            report = root / "suites" / "all.html"
            child = root / "8-card" / "tp8" / "runtime.html"
            child.parent.mkdir(parents=True)
            child.write_text("report", encoding="utf-8")

            _write_suite_report(report, [("tp8", "runtime", child)])
            content = report.read_text(encoding="utf-8")

        self.assertIn("tp8", content)
        self.assertIn("runtime", content)
        self.assertIn("性能采集套件", content)

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
            topology="ddp2",
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
        self.assertEqual(phases["comparison"]["world_size"], 2)
        self.assertAlmostEqual(
            phases["comparison"]["baseline_job_throughput_tps"], 2000.0
        )

    def test_distributed_trace_and_collectives_are_summarized_once(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            profiler = root / "trainer_output" / "profiling" / "traces"
            output = (
                profiler
                / "rank_0_123_ascend_pt"
                / "ASCEND_PROFILER_OUTPUT"
            )
            output.mkdir(parents=True)
            (output / "step_trace_time.csv").write_text(
                "Device_id,Step,Computing,Communication(Not Overlapped),"
                "Overlapped,Communication,Free,Stage\n"
                "1,12,200,100,50,150,650,1000\n"
                "1,13,220,80,40,120,680,1000\n",
                encoding="utf-8",
            )
            communication = {
                f"step{step}": {
                    "collective": {
                        f"hcom_allReduce__{step}": {
                            "Communication Time Info": {
                                "Elapse Time(ms)": 10,
                                "Transit Time(ms)": 2,
                                "Wait Time(ms)": 8,
                                "Synchronization Time(ms)": 8,
                                "Idle Time(ms)": 0,
                            },
                            "Communication Bandwidth Info": {
                                "HCCS": {
                                    "Transit Size(MB)": 40,
                                    "Transit Time(ms)": 2,
                                },
                                "SDMA": {
                                    "Transit Size(MB)": 40,
                                    "Transit Time(ms)": 2,
                                },
                            },
                        }
                    },
                    "p2p": {},
                }
                for step in (12, 13)
            }
            encoded = json.dumps(communication)
            (output / "communication.json").write_text(encoded, encoding="utf-8")
            duplicate = output.parent / "PROF_1" / "analyze"
            duplicate.mkdir(parents=True)
            (duplicate / "communication.json").write_text(
                encoded, encoding="utf-8"
            )

            analysis = build_analysis(root)

            rank = analysis["distributed_step_trace"]["ranks"][0]
            self.assertEqual(rank["rank"], 0)
            self.assertEqual(rank["median_stage_ms"], 1.0)
            self.assertEqual(rank["median_compute_ms"], 0.21)
            collective = analysis["communication_summary"]["rows"][0]
            self.assertEqual(collective["calls"], 2)
            self.assertEqual(collective["calls_per_step"], 1.0)
            self.assertEqual(collective["payload_mb_per_step"], 40.0)
            self.assertEqual(collective["effective_bandwidth_gbps"], 20.0)

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

    def test_msprof_deliverables_are_classified_as_insight_inputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            profiler = Path(temporary_directory)
            profile = profiler / "PROF_000001_20260901000000"
            output = profile / "mindstudio_profiler_output"
            output.mkdir(parents=True)
            (profile / "msprof_20260901000000.db").write_bytes(b"sqlite")
            (output / "op_summary_20260901000000.csv").write_text(
                "name,duration\n", encoding="utf-8"
            )

            deliverables = inspect_profiler_deliverables(profiler)

        self.assertTrue(deliverables["insight_ready"])
        self.assertEqual(
            deliverables["ascend_profiles"][0]["layout"], "msprof"
        )
        self.assertTrue(deliverables["ascend_profiles"][0]["raw_ready"])
        self.assertTrue(
            deliverables["ascend_profiles"][0]["parsed_database"]
        )

    def test_profiler_off_uses_post_warmup_steps_as_authoritative_steady_state(self):
        records = [
            {
                "step": step,
                "metrics": {
                    "time_metrics/end_to_end(s)": float(value),
                    "throughput(tps)": 1000.0 / value,
                },
            }
            for step, value in enumerate((9, 8, 1, 2), start=1)
        ]
        config = PerformanceConfig(
            name="off",
            steps=4,
            skip_steps=2,
            active_steps=1,
            profiler_enabled=False,
            topology="ddp2",
        ).as_dict()

        phases = summarize_profile_phases(
            records, config=config, profiler_environment={}
        )

        self.assertEqual(phases["phase_order"], ["startup", "steady"])
        self.assertEqual(phases["phases"]["steady"]["steps"], [3, 4])
        self.assertEqual(phases["comparison"]["parse_mode"], "disabled")
        self.assertEqual(
            phases["comparison"]["baseline_job_throughput_tps"], 1500.0
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
            self.assertIn("中文阅读路线", content)
            self.assertIn("Ascend memory timeline", content)
            self.assertIn("msprof-analyze output index", content)
            self.assertIn("Official MindStudio interactive HTML", content)


if __name__ == "__main__":
    unittest.main()
