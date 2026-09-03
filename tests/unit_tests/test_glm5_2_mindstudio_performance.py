# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from __future__ import annotations

import json
import runpy
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from tests.glm5_2_common.topology import ParallelTopology
from tests.glm5_2_mindstudio.toolchain import canonical_doctor_scope
from tests.glm5_2_performance.advanced_analysis import (
    parse_recipe_policy,
    resolve_recipe_plan,
)
from tests.glm5_2_performance.collectors import (
    PerformanceCollector,
    collector_inventory,
    require_training_collector,
)
from tests.glm5_2_performance.config import PerformanceConfig, profiler_presets
from tests.glm5_2_performance.visualization import mindstudio_insight_handoff
from tests.glm5_2_performance.workflow import (
    _adopt_legacy_cluster_outputs,
    _analysis_request,
    _prepare_analysis_stage,
    _analysis_toolchain_metadata,
    _collector_toolchain_metadata,
    _config_is_compatible,
    _path_tree_identity,
    _parse_compare_ranks,
    _rank_comparison_input,
    _run_name,
    _scoped_parent,
    _training_command,
    _validate_analysis_capabilities,
    analyze,
    run_advisor,
    run_cluster_analysis,
    run_performance_compare,
    run_profiler_cli,
)


class MindStudioPerformanceTest(unittest.TestCase):
    def test_legacy_cluster_tree_is_adopted_into_profiler_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            profile = run / "trainer_output" / "profiling" / "traces"
            profile.mkdir(parents=True)
            legacy = run / "cluster" / "cluster_analysis_output"
            legacy.mkdir(parents=True)
            (legacy / "cluster_analysis.db").write_bytes(b"db")
            (run / "cluster" / "cluster_text.json").write_text(
                "{}", encoding="utf-8"
            )
            recipe = run / "free_analysis" / "cluster_analysis_output"
            recipe.mkdir(parents=True)
            (recipe / "cluster_analysis.db").write_bytes(b"legacy-recipe-db")
            (recipe / "free_analysis.csv").write_text("x\n", encoding="utf-8")

            delivery = _adopt_legacy_cluster_outputs(run)

            self.assertTrue((delivery / "cluster_analysis.db").is_file())
            self.assertEqual(
                (delivery / "cluster_analysis.db").read_bytes(),
                b"db",
            )
            self.assertTrue((delivery / "free_analysis.csv").is_file())
            self.assertFalse((run / "cluster").exists())
            self.assertFalse((run / "free_analysis").exists())
            self.assertTrue((run / "cluster_text.json").is_file())

    def test_run_name_binds_default_profiler_capture_contract(self) -> None:
        config = PerformanceConfig(
            name="identity",
            topology="single",
            preset="distributed",
        )
        preset = profiler_presets()["distributed"]

        with_shapes = _run_name(config, "npu", preset)
        without_shapes = _run_name(
            config,
            "npu",
            replace(preset, record_shapes=False),
        )

        self.assertNotEqual(with_shapes, without_shapes)
        self.assertIn("-distributed-shapes-", with_shapes)
        self.assertNotIn("-shapes-", without_shapes)

    def test_necessary_advanced_recipes_cover_slow_rank_and_host_dispatch(
        self,
    ) -> None:
        topology = ParallelTopology(
            "ep2",
            2,
            data_parallel_shard_degree=2,
            expert_parallel_degree=2,
        )
        plan = resolve_recipe_plan(
            "necessary",
            topology=topology,
            record_shapes=True,
            mstx_enabled=False,
            with_flops=False,
            cluster_summary_baseline=True,
        )
        self.assertTrue(
            {
                "cluster_time_summary",
                "cluster_time_compare_summary",
                "slow_rank",
                "slow_link",
                "communication_bottleneck",
                "free_analysis",
                "cann_api_sum",
                "ep_load_balance",
            }.issubset(plan.recipes)
        )

    def test_all_advanced_recipes_skip_missing_capture_contracts(self) -> None:
        plan = resolve_recipe_plan(
            "all",
            topology=ParallelTopology("single", 1),
            record_shapes=False,
            mstx_enabled=False,
            with_flops=False,
            cluster_summary_baseline=False,
        )
        self.assertNotIn("module_statistic", plan.recipes)
        self.assertNotIn("operator_mfu", plan.recipes)
        self.assertNotIn("ep_load_balance", plan.recipes)
        self.assertEqual(
            plan.skipped["ep_load_balance"],
            "requires record_shapes capture data",
        )
        self.assertEqual(
            plan.skipped["cluster_time_compare_summary"],
            "requires --cluster-summary-baseline",
        )

    def test_moe_load_balance_applies_without_expert_parallelism(self) -> None:
        plan = resolve_recipe_plan(
            "necessary",
            topology=ParallelTopology("fsdp2", 2, data_parallel_shard_degree=2),
            record_shapes=True,
            mstx_enabled=False,
            with_flops=False,
            cluster_summary_baseline=False,
        )
        self.assertIn("ep_load_balance", plan.recipes)

    def test_explicit_mutating_recipe_is_accepted_but_never_automatic(
        self,
    ) -> None:
        self.assertEqual(parse_recipe_policy("pp_chart"), ("pp_chart",))
        plan = resolve_recipe_plan(
            "all",
            topology=ParallelTopology("pp2", 2, pipeline_parallel_degree=2),
            record_shapes=True,
            mstx_enabled=True,
            with_flops=True,
            cluster_summary_baseline=False,
        )
        self.assertNotIn("pp_chart", plan.recipes)

    def test_analysis_stages_are_additive_and_keep_independent_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            artifact = root / "artifact"
            profile = run / "trainer_output" / "profiling" / "traces"
            profile.mkdir(parents=True)
            artifact.mkdir()
            request = {
                "capture_identity": {"attempt_id": "capture-1"},
                "operations": {
                    "offline_parse": False,
                    "parse_workers": None,
                    "advisor": True,
                    "cluster": True,
                    "cluster_mode": "all",
                    "cluster_agent_output": False,
                    "cluster_bypass_input_safety_checks": False,
                    "compare_baseline": None,
                },
                "analysis_toolchain": {"version": "26.1"},
            }

            advisor_attempt, should_advise = _prepare_analysis_stage(
                run_directory=run,
                artifact_directory=artifact,
                request=request,
                stage="advisor",
                force=False,
            )
            self.assertTrue(should_advise)
            assert advisor_attempt is not None
            (run / "advisor").mkdir()
            (run / "advisor.json").write_text("{}", encoding="utf-8")
            advisor_attempt.update("completed")

            cluster_attempt, should_cluster = _prepare_analysis_stage(
                run_directory=run,
                artifact_directory=artifact,
                request=request,
                stage="cluster",
                force=False,
            )
            self.assertTrue(should_cluster)
            self.assertTrue((run / "advisor").is_dir())
            assert cluster_attempt is not None
            (profile / "cluster_analysis_output").mkdir()
            (run / "cluster.json").write_text("{}", encoding="utf-8")
            cluster_attempt.update("completed")

            repeated_attempt, should_repeat = _prepare_analysis_stage(
                run_directory=run,
                artifact_directory=artifact,
                request=request,
                stage="advisor",
                force=False,
            )
            self.assertFalse(should_repeat)
            self.assertIsNone(repeated_attempt)
            self.assertTrue((profile / "cluster_analysis_output").is_dir())

    def test_stale_derived_stage_is_rebuilt_without_touching_capture_or_peers(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            artifact = root / "artifact"
            run.mkdir()
            artifact.mkdir()
            capture = run / "trainer_output"
            capture.mkdir()
            cluster = run / "cluster"
            cluster.mkdir()
            (run / "cluster.json").write_text("{}", encoding="utf-8")
            request = {
                "capture_identity": {"attempt_id": "capture-1"},
                "operations": {
                    "offline_parse": False,
                    "parse_workers": None,
                    "advisor": True,
                    "cluster": True,
                    "cluster_mode": "all",
                    "cluster_agent_output": False,
                    "cluster_bypass_input_safety_checks": False,
                    "compare_baseline": None,
                },
                "analysis_toolchain": {"version": "26.1"},
            }

            old_attempt, should_run = _prepare_analysis_stage(
                run_directory=run,
                artifact_directory=artifact,
                request=request,
                stage="advisor",
                force=False,
            )
            self.assertTrue(should_run)
            assert old_attempt is not None
            (run / "advisor").mkdir()
            (run / "advisor.json").write_text("{}", encoding="utf-8")
            old_attempt.update("completed")

            changed_request = {
                **request,
                "analysis_toolchain": {"version": "26.1.1"},
            }
            new_attempt, should_rebuild = _prepare_analysis_stage(
                run_directory=run,
                artifact_directory=artifact,
                request=changed_request,
                stage="advisor",
                force=False,
            )

            self.assertTrue(should_rebuild)
            self.assertIsNotNone(new_attempt)
            self.assertFalse((run / "advisor").exists())
            self.assertFalse((run / "advisor.json").exists())
            self.assertTrue(capture.is_dir())
            self.assertTrue(cluster.is_dir())
            self.assertTrue((run / "cluster.json").is_file())

    def test_official_entry_defaults_to_ascend_pytorch_profiler(self) -> None:
        entry = runpy.run_path(
            str(
                Path(__file__).parents[1]
                / "glm5_2_mindstudio"
                / "performance_benchmark.py"
            ),
            run_name="mindstudio_performance_defaults",
        )
        self.assertEqual(entry["CONFIG"].collector, "torch_npu_profiler")

    def test_collector_registry_keeps_unverified_tools_disabled(self) -> None:
        self.assertEqual(
            [entry["collector"] for entry in collector_inventory()],
            list(PerformanceCollector),
        )
        self.assertEqual(
            require_training_collector("msprof"),
            PerformanceCollector.MSPROF,
        )
        with self.assertRaisesRegex(NotImplementedError, "isolated operator"):
            require_training_collector("msopprof")
        with self.assertRaisesRegex(NotImplementedError, "Server validation"):
            require_training_collector("msmemscope")
        with self.assertRaisesRegex(NotImplementedError, "online request"):
            require_training_collector("service_profiler")

    def test_msprof_command_wraps_application_before_training(self) -> None:
        config = PerformanceConfig(
            name="official-performance",
            collector="msprof",
            collector_args=("--task-time=on",),
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "tests.glm5_2_performance.workflow.resolve_msprof_executable",
            return_value="/opt/ascend/bin/msprof",
        ), patch.dict(
            "os.environ",
            {
                "ASCEND_RT_VISIBLE_DEVICES": "0",
            },
            clear=False,
        ):
            command = _training_command(
                root=Path(directory),
                config=config,
                device="npu",
                preset=profiler_presets()["overview"],
                run_directory=Path(directory) / "run",
            )
        self.assertEqual(command[0], "/opt/ascend/bin/msprof")
        self.assertIn("--type=text", command)
        self.assertIn("--task-time=on", command)
        self.assertIn("--nproc_per_node=1", command)
        self.assertNotIn("--profiler.enable_profiling", command)
        self.assertLess(
            command.index("--task-time=on"),
            command.index("--nproc_per_node=1"),
        )

    def test_msprof_type_override_replaces_default(self) -> None:
        config = PerformanceConfig(
            name="official-performance",
            collector="msprof",
            collector_args=("--type=db",),
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "tests.glm5_2_performance.workflow.resolve_msprof_executable",
            return_value="msprof",
        ):
            command = _training_command(
                root=Path(directory),
                config=config,
                device="npu",
                preset=profiler_presets()["overview"],
                run_directory=Path(directory) / "run",
            )
        self.assertIn("--type=db", command)
        self.assertNotIn("--type=text", command)

    def test_collector_args_cannot_override_lifecycle(self) -> None:
        with self.assertRaisesRegex(ValueError, "lifecycle-owned"):
            PerformanceConfig(
                name="invalid",
                collector="msprof",
                collector_args=("--output=/tmp/wrong",),
            )
        with self.assertRaisesRegex(ValueError, "only by msprof"):
            PerformanceConfig(
                name="invalid",
                collector="torch_npu_profiler",
                collector_args=("--type=db",),
            )

    def test_old_manifest_defaults_remain_compatible(self) -> None:
        config = PerformanceConfig(name="legacy")
        saved = json.loads(json.dumps(config.as_dict()))
        saved.pop("collector")
        saved.pop("collector_args")
        self.assertTrue(_config_is_compatible(saved, config))

    def test_msprof_capture_requires_audited_toolchain_identity(self) -> None:
        config = PerformanceConfig(name="official", collector="msprof")
        metadata = {
            "doctor": {
                "ok": True,
                "checks": {
                    "framework": {"ok": True},
                    "cann": {"ok": True},
                    "msprof": {"ok": True},
                    "msprof-analyze": {"ok": True},
                },
            },
            "versions": {
                "torch": "2.14",
                "torch_npu": "2.14",
                "msprof": "26.1",
                "msprof-analyze": "26.1",
            },
            "sources": {
                "msprof": {"commit": "capture"},
                "msprof-analyze": {"commit": "analysis"},
            },
        }
        with patch(
            "tests.glm5_2_performance.workflow._performance_stage_metadata",
            return_value=metadata,
        ) as collect:
            identity = _collector_toolchain_metadata(config)
        collect.assert_called_once_with("performance-capture")
        self.assertEqual(identity["scope"], "performance-capture")
        self.assertNotIn("msprof-analyze", json.dumps(identity))
        self.assertEqual(identity["versions"]["msprof"], "26.1")
        changed_analyzer = json.loads(json.dumps(metadata))
        changed_analyzer["versions"]["msprof-analyze"] = "99.0"
        changed_analyzer["sources"]["msprof-analyze"]["commit"] = "changed"
        with patch(
            "tests.glm5_2_performance.workflow._performance_stage_metadata",
            return_value=changed_analyzer,
        ):
            self.assertEqual(
                _collector_toolchain_metadata(config),
                identity,
            )
        with patch(
            "tests.glm5_2_performance.workflow._performance_stage_metadata",
            return_value={"doctor": {"ok": False, "required_failures": ["msprof"]}},
        ), self.assertRaisesRegex(RuntimeError, "msprof"):
            _collector_toolchain_metadata(config)

    def test_profiler_off_baseline_does_not_require_msprof(self) -> None:
        config = PerformanceConfig(
            name="baseline",
            collector="msprof",
            profiler_enabled=False,
        )
        self.assertIsNone(_collector_toolchain_metadata(config))

    def test_torch_npu_capture_binds_framework_and_adapter_identity(self) -> None:
        config = PerformanceConfig(
            name="deep",
            collector="torch_npu_profiler",
        )
        metadata = {
            "doctor": {
                "ok": True,
                "scope": "performance-torch-npu-capture",
                "checks": {
                    "framework": {"ok": True},
                    "cann": {"ok": True},
                    "msprof": {"ok": False},
                },
            },
            "versions": {
                "torch": "2.14",
                "torch_npu": "2.14",
                "msprof": None,
            },
            "sources": {},
            "compatibility": {"cann": {"version": "9.1"}},
        }
        adapters = {"torchtitanturbo": {"file": {"sha256": "adapter"}}}
        with patch(
            "tests.glm5_2_performance.workflow._performance_stage_metadata",
            return_value=metadata,
        ) as collect, patch(
            "tests.glm5_2_performance.workflow."
            "_torch_npu_profiler_adapter_identity",
            return_value=adapters,
        ):
            identity = _collector_toolchain_metadata(config)

        collect.assert_called_once_with("performance-torch-npu-capture")
        self.assertEqual(identity["scope"], "performance-torch-npu-capture")
        self.assertEqual(identity["adapters"], adapters)
        self.assertEqual(identity["versions"]["torch_npu"], "2.14")
        self.assertNotIn("msprof", identity["versions"])

    def test_compare_baseline_identity_tracks_nested_file_rewrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline"
            nested = baseline / "rank0" / "profile.db"
            nested.parent.mkdir(parents=True)
            nested.write_bytes(b"first")
            first = _path_tree_identity(baseline)
            nested.write_bytes(b"second-version")
            second = _path_tree_identity(baseline)

        self.assertEqual(first["file_count"], 1)
        self.assertNotEqual(
            first["tree_stat_sha256"], second["tree_stat_sha256"]
        )

    def test_analysis_request_binds_optional_visualizer_identity(self) -> None:
        manifest = {"format_version": 1, "run_name": "run"}
        visualizers = {
            "mindstudio_insight_flamegraph": {"sha256": "v1"},
            "brendangregg_flamegraph": None,
        }
        with patch(
            "tests.glm5_2_performance.workflow."
            "_visualization_toolchain_identity",
            return_value=visualizers,
        ):
            first = _analysis_request(
                manifest=manifest,
                parse_offline=False,
                parse_workers=None,
                advisor=False,
                cluster=False,
                compare_baseline=None,
                analysis_toolchain=None,
            )
        changed = json.loads(json.dumps(visualizers))
        changed["mindstudio_insight_flamegraph"]["sha256"] = "v2"
        with patch(
            "tests.glm5_2_performance.workflow."
            "_visualization_toolchain_identity",
            return_value=changed,
        ):
            second = _analysis_request(
                manifest=manifest,
                parse_offline=False,
                parse_workers=None,
                advisor=False,
                cluster=False,
                compare_baseline=None,
                analysis_toolchain=None,
            )

        self.assertEqual(first["visualization_toolchain"], visualizers)
        self.assertNotEqual(first["identity"], second["identity"])

    def test_offline_analysis_uses_analyzer_only_toolchain_scope(self) -> None:
        metadata = {
            "doctor": {
                "ok": True,
                "scope": "performance-analysis",
                "checks": {
                    "framework": {"ok": False},
                    "msprof-analyze": {"ok": True},
                },
            },
            "versions": {
                "torch": "different-host",
                "msprof": "different-host",
                "msprof-analyze": "26.1",
            },
            "sources": {
                "msprof": {"commit": "capture"},
                "msprof-analyze": {"commit": "analysis"},
            },
        }
        with patch(
            "tests.glm5_2_performance.workflow._performance_stage_metadata",
            return_value=metadata,
        ) as collect:
            identity = _analysis_toolchain_metadata()
        collect.assert_called_once_with("performance-analysis")
        self.assertEqual(identity["scope"], "performance-analysis")
        self.assertEqual(
            set(identity["versions"]),
            {"msprof-analyze"},
        )
        self.assertNotIn('"msprof"', json.dumps(identity))

    def test_analyzer_status_records_toolchain_provenance(self) -> None:
        toolchain = {
            "scope": "performance-analysis",
            "versions": {"msprof-analyze": "26.1"},
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "tests.glm5_2_performance.workflow._msprof_analyze_executable",
            return_value="msprof-analyze",
        ), patch(
            "tests.glm5_2_performance.workflow.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, "ok", ""),
        ):
            run = Path(directory)
            (run / "trainer_output" / "profiling" / "traces").mkdir(
                parents=True
            )
            run_advisor(run, analysis_toolchain=toolchain)
            status = json.loads(
                (run / "advisor.json").read_text(encoding="utf-8")
            )
        self.assertEqual(status["analysis_toolchain"], toolchain)

    def test_standard_cluster_avoids_legacy_recipe_fanout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "trainer_output" / "profiling" / "msprof").mkdir(
                parents=True
            )
            with patch(
                "tests.glm5_2_performance.workflow._msprof_analyze_executable",
                return_value="msprof-analyze",
            ), patch(
                "tests.glm5_2_performance.workflow._run_msprof_analyze",
                return_value={"return_code": 0},
            ) as execute:
                result = run_cluster_analysis(run, extended=False)
        self.assertEqual(set(result), {"cluster", "cluster_text"})
        self.assertEqual(result["cluster_text"]["status"], "missing_text_input")
        self.assertEqual(execute.call_count, 1)
        command = execute.call_args.kwargs["command"]
        self.assertEqual(command[1:4], ["cluster", "-m", "all"])

    def test_standard_cluster_exposes_official_modes_and_flags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "trainer_output" / "profiling" / "msprof").mkdir(
                parents=True
            )
            with patch(
                "tests.glm5_2_performance.workflow._msprof_analyze_executable",
                return_value="msprof-analyze",
            ), patch(
                "tests.glm5_2_performance.workflow._run_msprof_analyze",
                return_value={"return_code": 0},
            ) as execute:
                run_cluster_analysis(
                    run,
                    extended=False,
                    mode="communication_matrix",
                    agent_output=True,
                    bypass_input_safety_checks=True,
                )
        command = execute.call_args.kwargs["command"]
        self.assertEqual(
            command[1:4], ["cluster", "-m", "communication_matrix"]
        )
        self.assertIn("--agent", command)
        self.assertIn("--force", command)

    def test_standard_cluster_runs_topology_aware_advanced_recipes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            profile = run / "trainer_output" / "profiling" / "traces"
            for rank in (0, 1):
                rank_root = profile / f"rank_{rank}_capture_ascend_pt"
                rank_root.mkdir(parents=True)
                (rank_root / f"profiler_info_{rank}.json").write_text(
                    "{}", encoding="utf-8"
                )
            with patch(
                "tests.glm5_2_performance.workflow._msprof_analyze_executable",
                return_value="msprof-analyze",
            ), patch(
                "tests.glm5_2_performance.workflow._run_msprof_analyze",
                return_value={"return_code": 0},
            ) as execute:
                result = run_cluster_analysis(
                    run,
                    extended=False,
                    advanced_recipe_policy="necessary",
                    topology=ParallelTopology(
                        "ddp2", 2, data_parallel_replicate_degree=2
                    ),
                )
        commands = [call.kwargs["command"] for call in execute.call_args_list]
        self.assertIn("advanced", result)
        self.assertTrue(
            any("cluster_time_summary" in command for command in commands)
        )
        recipe_commands = [
            command
            for command in commands
            if len(command) > 2 and command[1] == "-m"
        ]
        self.assertTrue(recipe_commands)
        self.assertTrue(
            all(
                Path(command[command.index("-o") + 1]) == profile
                for command in recipe_commands
            )
        )
        bottleneck_commands = [
            command
            for command in commands
            if "communication_bottleneck" in command
        ]
        self.assertEqual(len(bottleneck_commands), 4)
        self.assertEqual(
            {
                command[command.index("--export_type") + 1]
                for command in bottleneck_commands
            },
            {"db", "text"},
        )
        self.assertEqual(
            {
                command[command.index("--rank_id") + 1]
                for command in bottleneck_commands
            },
            {"0", "1"},
        )

    def test_cluster_merges_official_db_and_text_deliveries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            profile = run / "trainer_output" / "profiling" / "traces"
            for rank in (0, 1):
                rank_root = profile / f"rank_{rank}_capture_ascend_pt"
                output = rank_root / "ASCEND_PROFILER_OUTPUT"
                output.mkdir(parents=True)
                (rank_root / f"profiler_info_{rank}.json").write_text(
                    "{}", encoding="utf-8"
                )
                (output / "step_trace_time.csv").write_text(
                    "Step,Computing\n1,1\n", encoding="utf-8"
                )
                (output / "communication.json").write_text(
                    "{}", encoding="utf-8"
                )
                (output / "communication_matrix.json").write_text(
                    "{}", encoding="utf-8"
                )

            def execute(
                run_directory: Path,
                *,
                name: str,
                command: list[str],
                analysis_toolchain: dict[str, object] | None = None,
                status_path: Path | None = None,
            ) -> dict[str, object]:
                output = Path(command[command.index("-o") + 1])
                delivery = output / "cluster_analysis_output"
                delivery.mkdir(parents=True)
                if name == "cluster":
                    (delivery / "cluster_analysis.db").write_bytes(b"db")
                else:
                    for file_name in (
                        "cluster_step_trace_time.csv",
                        "cluster_communication.json",
                        "cluster_communication_matrix.json",
                        "communication_group.json",
                    ):
                        (delivery / file_name).write_text("{}", encoding="utf-8")
                return {"return_code": 0}

            with patch(
                "tests.glm5_2_performance.workflow._msprof_analyze_executable",
                return_value="msprof-analyze",
            ), patch(
                "tests.glm5_2_performance.workflow._run_msprof_analyze",
                side_effect=execute,
            ):
                run_cluster_analysis(run, extended=False)

            delivery = profile / "cluster_analysis_output"
            self.assertEqual(
                {path.name for path in delivery.iterdir()},
                {
                    "cluster_analysis.db",
                    "cluster_step_trace_time.csv",
                    "cluster_communication.json",
                    "cluster_communication_matrix.json",
                    "communication_group.json",
                },
            )
            self.assertFalse((run / "cluster_time_summary").exists())
            self.assertFalse((run / "free_analysis").exists())
            self.assertFalse((run / "cluster").exists())

    def test_standard_compare_uses_documented_output_path_option(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            baseline = Path(directory) / "baseline"
            (run / "trainer_output" / "profiling" / "msprof").mkdir(
                parents=True
            )
            baseline.mkdir()
            with patch(
                "tests.glm5_2_performance.workflow._msprof_analyze_executable",
                return_value="msprof-analyze",
            ), patch(
                "tests.glm5_2_performance.workflow._run_msprof_analyze",
                return_value={"return_code": 0},
            ) as execute:
                run_performance_compare(run, baseline, standard_cli=True)
        command = execute.call_args.kwargs["command"]
        self.assertIn("--output_path", command)
        self.assertNotIn("-o", command)
        output = Path(command[command.index("--output_path") + 1])
        self.assertEqual(
            output,
            run
            / "trainer_output"
            / "profiling"
            / "msprof"
            / "compare_analysis_output"
            / "profile_compare",
        )

    def test_rank_compare_uses_two_rank_profile_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            profile = run / "trainer_output" / "profiling" / "traces"
            rank_roots = []
            for rank in (0, 1):
                rank_root = profile / f"rank_{rank}_capture_ascend_pt"
                rank_root.mkdir(parents=True)
                (rank_root / f"profiler_info_{rank}.json").write_text(
                    "{}", encoding="utf-8"
                )
                rank_roots.append(rank_root)
            self.assertEqual(_parse_compare_ranks("0,1"), (0, 1))
            self.assertEqual(_rank_comparison_input(run, 1), rank_roots[1])
            with patch(
                "tests.glm5_2_performance.workflow._msprof_analyze_executable",
                return_value="msprof-analyze",
            ), patch(
                "tests.glm5_2_performance.workflow._run_msprof_analyze",
                return_value={"return_code": 0},
            ) as execute:
                run_performance_compare(
                    run,
                    rank_roots[0],
                    candidate=rank_roots[1],
                    comparison_name="rank_0_vs_rank_1",
                    standard_cli=True,
                )
        command = execute.call_args.kwargs["command"]
        self.assertEqual(Path(command[command.index("-d") + 1]), rank_roots[1])
        self.assertEqual(Path(command[command.index("-bp") + 1]), rank_roots[0])
        output = Path(command[command.index("--output_path") + 1])
        self.assertEqual(output.name, "rank_0_vs_rank_1")
        self.assertEqual(output.parent.name, "compare_analysis_output")

    def test_msprof_capability_gate_rejects_pytorch_only_analysis(self) -> None:
        config = PerformanceConfig(name="official", collector="msprof")
        with self.assertRaisesRegex(ValueError, r"\*_ascend_pt"):
            _validate_analysis_capabilities(
                config,
                advisor=True,
                cluster=False,
                compare_baseline=None,
            )
        with self.assertRaisesRegex(ValueError, r"\*_ascend_pt"):
            _validate_analysis_capabilities(
                config,
                advisor=False,
                cluster=False,
                compare_baseline=Path("baseline"),
            )

    def test_msprof_cluster_requires_database_capture(self) -> None:
        config = PerformanceConfig(name="official", collector="msprof")
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            profile = run / "trainer_output" / "profiling" / "msprof"
            profile.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, r"no \*\.db file"):
                _validate_analysis_capabilities(
                    config,
                    advisor=False,
                    cluster=True,
                    compare_baseline=None,
                    run_directory=run,
                )
            (profile / "ascend.db").write_bytes(b"db")
            _validate_analysis_capabilities(
                config,
                advisor=False,
                cluster=True,
                compare_baseline=None,
                run_directory=run,
            )

    def test_force_analyze_replaces_only_derived_outputs(self) -> None:
        config = PerformanceConfig(
            name="official",
            collector="torch_npu_profiler",
        )
        preset = profiler_presets()["overview"]
        toolchain_v1 = {
            "scope": "performance-analysis",
            "versions": {"msprof-analyze": "26.1"},
        }
        toolchain_v2 = {
            "scope": "performance-analysis",
            "versions": {"msprof-analyze": "26.2"},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_name = _run_name(config, "npu", preset)
            run = _scoped_parent(
                root, config.run_root, config.topology
            ) / run_name
            artifact = _scoped_parent(
                root, config.artifact_root, config.topology
            ) / run_name
            report = _scoped_parent(
                root, config.report_root, config.topology
            ) / f"{run_name}.html"
            profile = run / "trainer_output" / "profiling" / "traces"
            profile.mkdir(parents=True)
            raw = profile / "raw_capture.bin"
            raw.write_bytes(b"raw")
            old_advisor = run / "advisor" / "old.txt"
            old_advisor.parent.mkdir()
            old_advisor.write_text("old", encoding="utf-8")
            artifact.mkdir(parents=True)
            manifest = {
                "format_version": 1,
                "run_name": run_name,
                "attempt_id": "capture-1",
                "device": "npu",
                "topology": config.topology,
                "preset": config.preset,
                "collector": config.collector,
                "config": config.as_dict(),
                "profiler_environment": preset.environment(),
                "source": {},
                "collector_toolchain": None,
            }
            (artifact / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (artifact / "analysis.json").write_text("{}", encoding="utf-8")
            report.parent.mkdir(parents=True)
            report.write_text("old", encoding="utf-8")

            def advisor_side_effect(
                run_directory: Path,
                *,
                analysis_toolchain: dict[str, object] | None = None,
            ) -> dict[str, object]:
                self.assertFalse(old_advisor.exists())
                self.assertEqual(analysis_toolchain, toolchain_v1)
                output = run_directory / "advisor"
                output.mkdir()
                (output / "new.txt").write_text("new", encoding="utf-8")
                return {"return_code": 0}

            def report_side_effect(*, output_path: Path, **_: object) -> None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text("new", encoding="utf-8")

            handoff = {
                "import_targets": [],
                "portable_import_targets": [],
            }
            with patch(
                "tests.glm5_2_performance.workflow._analysis_toolchain_metadata",
                return_value=toolchain_v1,
            ), patch(
                "tests.glm5_2_performance.workflow.run_advisor",
                side_effect=advisor_side_effect,
            ), patch(
                "tests.glm5_2_performance.workflow.render_mindstudio_flamegraphs"
            ), patch(
                "tests.glm5_2_performance.workflow.render_flamegraphs"
            ), patch(
                "tests.glm5_2_performance.workflow.build_analysis",
                return_value={},
            ), patch(
                "tests.glm5_2_performance.workflow.mindstudio_insight_handoff",
                return_value=handoff,
            ), patch(
                "tests.glm5_2_performance.workflow.render_html_report",
                side_effect=report_side_effect,
            ), patch(
                "tests.glm5_2_performance.workflow._sync_exploration_bundle"
            ):
                result = analyze(
                    root,
                    config,
                    device="npu",
                    preset=preset,
                    parse_offline=False,
                    parse_workers=None,
                    advisor=True,
                    cluster=False,
                    compare_baseline=None,
                    force=True,
                )
            self.assertEqual(result, report)
            self.assertEqual(raw.read_bytes(), b"raw")
            self.assertTrue((run / "advisor" / "new.txt").is_file())
            analysis = json.loads(
                (artifact / "analysis.json").read_text(encoding="utf-8")
            )
            self.assertEqual(analysis["analysis_toolchain"], toolchain_v1)
            state = json.loads(
                (artifact / "analysis_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["status"], "completed")
            self.assertEqual(
                state["context"]["analysis_toolchain"],
                toolchain_v1,
            )

            def changed_advisor_side_effect(
                run_directory: Path,
                *,
                analysis_toolchain: dict[str, object] | None = None,
            ) -> dict[str, object]:
                self.assertFalse((run_directory / "advisor").exists())
                self.assertEqual(analysis_toolchain, toolchain_v2)
                output = run_directory / "advisor"
                output.mkdir()
                (output / "updated.txt").write_text("updated", encoding="utf-8")
                return {"return_code": 0}

            with patch(
                "tests.glm5_2_performance.workflow._analysis_toolchain_metadata",
                return_value=toolchain_v2,
            ), patch(
                "tests.glm5_2_performance.workflow.run_advisor",
                side_effect=changed_advisor_side_effect,
            ), patch(
                "tests.glm5_2_performance.workflow.render_mindstudio_flamegraphs"
            ), patch(
                "tests.glm5_2_performance.workflow.render_flamegraphs"
            ), patch(
                "tests.glm5_2_performance.workflow.build_analysis",
                return_value={},
            ), patch(
                "tests.glm5_2_performance.workflow.mindstudio_insight_handoff",
                return_value=handoff,
            ), patch(
                "tests.glm5_2_performance.workflow.render_html_report",
                side_effect=report_side_effect,
            ), patch(
                "tests.glm5_2_performance.workflow._sync_exploration_bundle"
            ):
                repeated_result = analyze(
                    root,
                    config,
                    device="npu",
                    preset=preset,
                    parse_offline=False,
                    parse_workers=None,
                    advisor=True,
                    cluster=False,
                    compare_baseline=None,
                )
            self.assertEqual(repeated_result, report)
            self.assertEqual(raw.read_bytes(), b"raw")
            self.assertTrue((run / "advisor" / "updated.txt").is_file())

    def test_msprof_all_expands_to_cluster_only(self) -> None:
        config = PerformanceConfig(
            name="official",
            collector="msprof",
            collector_args=("--type=db",),
            topology="ddp2",
        )
        with tempfile.TemporaryDirectory() as directory, patch.object(
            sys,
            "argv",
            [
                "performance.py",
                "--analyze",
                "--device",
                "npu",
                "--topology",
                "ddp2",
                "--analysis-tools",
                "all",
                "--cluster-mode",
                "communication_time",
                "--cluster-agent-output",
                "--cluster-bypass-input-safety-checks",
            ],
        ), patch(
            "tests.glm5_2_performance.workflow._repository_root",
            return_value=Path(directory),
        ), patch(
            "tests.glm5_2_performance.workflow._resolve_device",
            return_value="npu",
        ), patch(
            "tests.glm5_2_performance.workflow.analyze",
            return_value=Path(directory) / "report.html",
        ) as analyze_mock, patch(
            "tests.glm5_2_performance.workflow._record_exploration_command"
        ):
            run_profiler_cli(config, "performance.py")
        kwargs = analyze_mock.call_args.kwargs
        self.assertFalse(kwargs["advisor"])
        self.assertTrue(kwargs["cluster"])
        self.assertFalse(kwargs["parse_offline"])
        self.assertEqual(kwargs["cluster_mode"], "communication_time")
        self.assertTrue(kwargs["cluster_agent_output"])
        self.assertTrue(kwargs["cluster_bypass_input_safety_checks"])

    def test_invalid_msprof_analysis_fails_before_force_reset(self) -> None:
        config = PerformanceConfig(
            name="official",
            collector="msprof",
            topology="ddp2",
        )
        with patch.object(
            sys,
            "argv",
            [
                "performance.py",
                "--analyze",
                "--force",
                "--device",
                "npu",
                "--topology",
                "ddp2",
                "--advisor",
            ],
        ), patch(
            "tests.glm5_2_performance.workflow._resolve_device",
            return_value="npu",
        ), patch(
            "tests.glm5_2_performance.workflow.reset_output_generation"
        ) as reset, self.assertRaises(SystemExit):
            run_profiler_cli(config, "performance.py")
        reset.assert_not_called()

    def test_compare_baseline_rejects_multiple_topologies_before_reset(self) -> None:
        config = PerformanceConfig(
            name="official",
            collector="torch_npu_profiler",
        )
        with patch.object(
            sys,
            "argv",
            [
                "performance.py",
                "--analyze",
                "--force",
                "--device",
                "npu",
                "--topologies",
                "ddp2,fsdp8",
                "--compare-baseline",
                "baseline",
            ],
        ), patch(
            "tests.glm5_2_performance.workflow._resolve_device",
            return_value="npu",
        ), patch(
            "tests.glm5_2_performance.workflow.reset_output_generation"
        ) as reset, self.assertRaises(SystemExit):
            run_profiler_cli(config, "performance.py")
        reset.assert_not_called()

    def test_insight_handoff_indexes_full_roots_and_views(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profile"
            profile.mkdir()
            (profile / "rank0.db").write_bytes(b"db")
            (profile / "trace_view.json").write_text("{}", encoding="utf-8")
            cluster = root / "analysis" / "cluster_analysis_output"
            cluster.mkdir(parents=True)
            handoff = mindstudio_insight_handoff(
                profile,
                analysis_root=root / "analysis",
                portable_base=root,
            )
        self.assertEqual(handoff["profile_inventory"]["databases"], ["rank0.db"])
        self.assertIn(str(cluster.resolve()), handoff["import_targets"])
        self.assertEqual(
            handoff["portable_import_targets"][0]["relative_path"],
            "profile",
        )
        self.assertEqual(
            handoff["portable_import_targets"][1]["relative_path"],
            "analysis/cluster_analysis_output",
        )
        self.assertEqual(
            {view["name"] for view in handoff["views"]},
            {"Summary", "Communication", "Timeline", "Operator", "Memory", "RL"},
        )
        json.dumps(handoff)

    def test_insight_handoff_prefers_one_colocated_import_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profile"
            (profile / "rank_0_capture_ascend_pt").mkdir(parents=True)
            (profile / "cluster_analysis_output").mkdir()
            separate = root / "analysis" / "cluster_analysis_output"
            separate.mkdir(parents=True)

            handoff = mindstudio_insight_handoff(
                profile,
                analysis_root=root / "analysis",
                portable_base=root,
            )

        self.assertEqual(handoff["import_targets"], [str(profile.resolve())])
        self.assertEqual(
            handoff["portable_import_targets"],
            [
                {
                    "kind": "profile",
                    "server_path": str(profile.resolve()),
                    "relative_path": "profile",
                }
            ],
        )

    def test_performance_doctor_scope_is_public(self) -> None:
        self.assertEqual(canonical_doctor_scope("profiler"), "performance")
        self.assertEqual(
            canonical_doctor_scope("torch_npu_profiler_capture"),
            "performance-torch-npu-capture",
        )


if __name__ == "__main__":
    unittest.main()
