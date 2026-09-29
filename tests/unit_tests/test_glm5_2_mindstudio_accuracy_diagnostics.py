# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""CPU-only tests for stateful MindStudio accuracy diagnosis cases."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from tests.glm5_2_common.reporting import echarts_line
from tests.glm5_2_mindstudio.accuracy_benchmark import _select_stage
from tests.glm5_2_mindstudio.accuracy_diagnostics import (
    _load_case,
    add_hypothesis,
    analyze_training_observation,
    build_plan,
    close_case,
    compare_repeats,
    create_case,
    record_hypothesis,
    record_stage,
)
from tests.glm5_2_mindstudio.capture_training import (
    _install_training_metrics_capture,
)
from tests.glm5_2_mindstudio.configuration_check_benchmark import (
    CONFIG as CONFIG_CHECK_CONFIG,
)
from tests.glm5_2_mindstudio.graph_accuracy_benchmark import (
    BASE_CONFIG as GRAPH_BASE_CONFIG,
    STAGE_CONFIGS as GRAPH_STAGE_CONFIGS,
    _stage_configs as graph_stage_configs,
    _select_stage as select_graph_stage,
)
from tests.glm5_2_mindstudio.migration_benchmark import CONFIG as MIGRATION_CONFIG
from tests.glm5_2_mindstudio.training_observation_benchmark import (
    CONFIG as OBSERVATION_CONFIG,
)
from tests.glm5_2_mindstudio.training_monitor_benchmark import (
    CONFIG as MONITOR_CONFIG,
)
from tests.glm5_2_mindstudio.training_observation import compare_training_metrics
from tests.glm5_2_mindstudio.workflow import (
    _compatible_fixture_directory,
    _compatible_fixture_manifest,
    _finalizable_monitor_capture,
    _fixture_directory,
    _paths,
    _stage_scoped_config,
    _training_profile_from_fixture,
    reset_selected_outputs,
)
from tests.glm5_2_precision.workflow import (
    _fixture_directory as _formal_fixture_directory,
)


class MindStudioDiagnosticsTest(unittest.TestCase):
    def test_echarts_category_steps_use_names_not_numeric_indexes(self) -> None:
        chart = echarts_line(
            title="Early steps",
            subtitle="Step coordinates",
            x_values=(1, 2, 3),
            series=(("Loss", (3.0, 2.0, 1.0), "#2563eb"),),
            y_name="Loss",
            mark_areas=(("Early window", 1, 3, "#2563eb"),),
            mark_points=(("First", 1, 3.0, "#dc2626"),),
        )
        option = chart[1].object
        self.assertEqual(["1", "2", "3"], option["xAxis"][0]["data"])
        self.assertEqual(["1", 3.0], option["series"][0]["data"][0])
        self.assertEqual(
            "1",
            option["series"][0]["markArea"]["data"][0][0]["xAxis"],
        )
        self.assertEqual(
            ["1", 3.0],
            option["series"][0]["markPoint"]["data"][0]["coord"],
        )

    def test_training_metrics_capture_does_not_require_tensorboard(self) -> None:
        class BaseLogger:
            def log(self, metrics, step) -> None:
                pass

        class LoggerContainer(BaseLogger):
            def log(self, metrics, step) -> None:
                pass

        metrics_module = types.ModuleType("torchtitan.components.metrics")
        metrics_module.BaseLogger = BaseLogger
        metrics_module.LoggerContainer = LoggerContainer
        components_module = types.ModuleType("torchtitan.components")
        components_module.metrics = metrics_module
        torchtitan_module = types.ModuleType("torchtitan")
        torchtitan_module.components = components_module
        modules = {
            "torchtitan": torchtitan_module,
            "torchtitan.components": components_module,
            "torchtitan.components.metrics": metrics_module,
        }
        values = {
            "loss_metrics/global_avg_loss": 1.25,
            "loss_metrics/global_max_loss": 1.5,
            "grad_norm": 0.75,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "training_metrics.jsonl"
            environment = {
                "GLM5_MINDSTUDIO_METRICS_PATH": str(path),
                "RANK": "0",
                "LOG_RANK": "0",
            }
            with patch.dict(sys.modules, modules), patch.dict(
                os.environ, environment, clear=False
            ):
                _install_training_metrics_capture()
                BaseLogger().log(values, 1)
                LoggerContainer().log(values, 2)

            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([1, 2], [record["step"] for record in records])
            self.assertEqual(values, records[0]["metrics"])

    def test_accuracy_stages_share_one_experiment_root(self) -> None:
        self.assertEqual(
            MIGRATION_CONFIG.storage_name,
            CONFIG_CHECK_CONFIG.storage_name,
        )
        self.assertEqual(
            MIGRATION_CONFIG.storage_name,
            MONITOR_CONFIG.storage_name,
        )
        self.assertEqual(
            MIGRATION_CONFIG.storage_name,
            OBSERVATION_CONFIG.storage_name,
        )
        self.assertEqual(
            Path(MIGRATION_CONFIG.storage_name),
            CONFIG_CHECK_CONFIG.output_relative_root,
        )
        self.assertEqual(
            Path("diagnostics/configuration-check"),
            CONFIG_CHECK_CONFIG.operation_relative_root,
        )
        observation = _stage_scoped_config(OBSERVATION_CONFIG, MIGRATION_CONFIG)
        self.assertEqual(
            Path(MIGRATION_CONFIG.storage_name),
            observation.output_relative_root,
        )
        self.assertTrue(
            observation.operation_relative_root.as_posix().startswith(
                "observations/training/s100-"
            )
        )
        monitor = _stage_scoped_config(MONITOR_CONFIG, MIGRATION_CONFIG)
        self.assertEqual(
            Path(MIGRATION_CONFIG.storage_name),
            monitor.output_relative_root,
        )

    def test_unified_accuracy_entry_removes_only_its_stage_option(self) -> None:
        stage, remaining = _select_stage(
            ["--capture", "candidate", "--stage", "monitor", "--topology", "fsdp8"]
        )
        self.assertEqual("monitor", stage)
        self.assertEqual(
            ["--capture", "candidate", "--topology", "fsdp8"], remaining
        )

    def test_unified_accuracy_entry_defaults_to_observation(self) -> None:
        stage, remaining = _select_stage(
            ["--capture", "candidate", "--topology", "single"]
        )
        self.assertEqual("observation", stage)
        self.assertEqual(
            ["--capture", "candidate", "--topology", "single"], remaining
        )

    def test_graph_accuracy_stages_share_one_experiment_root(self) -> None:
        for config in GRAPH_STAGE_CONFIGS.values():
            self.assertEqual(GRAPH_BASE_CONFIG.storage_name, config.storage_name)
            self.assertEqual(
                "graph/npu-inductor-ascend-triton",
                config.execution_branch,
            )
            self.assertFalse(config.owns_fixture)
        self.assertEqual(
            "candidate",
            GRAPH_STAGE_CONFIGS[
                "observation"
            ].reuse_eager_role_as_reference,
        )
        self.assertIsNone(
            GRAPH_STAGE_CONFIGS[
                "compile-checker"
            ].reuse_eager_role_as_reference
        )
        self.assertEqual(
            MIGRATION_CONFIG.storage_name,
            GRAPH_BASE_CONFIG.storage_name,
        )
        self.assertEqual(500, GRAPH_BASE_CONFIG.training.steps)
        self.assertEqual(1, GRAPH_STAGE_CONFIGS["config-check"].training.steps)
        self.assertEqual(
            500,
            GRAPH_STAGE_CONFIGS["config-check"].fixture_training.steps,
        )
        self.assertEqual(500, GRAPH_STAGE_CONFIGS["observation"].training.steps)
        self.assertEqual(
            (),
            GRAPH_STAGE_CONFIGS["compile-checker"].candidate.extra_args,
        )
        self.assertIn(
            "--compile.enable",
            GRAPH_STAGE_CONFIGS["observation"].candidate.extra_args,
        )
        self.assertEqual(
            "aten.sum,_c10d_functional.all_reduce",
            GRAPH_STAGE_CONFIGS["observation"].candidate.environment[
                "NPU_INDUCTOR_FALLBACK_LIST"
            ],
        )
        graph_checklist = _stage_scoped_config(
            GRAPH_STAGE_CONFIGS["config-check"],
            GRAPH_BASE_CONFIG,
        )
        self.assertTrue(
            graph_checklist.output_subdirectory.startswith(
                "s500-"
                ""
            )
        )
        self.assertEqual(
            graph_checklist.output_subdirectory.split("/", 1)[0]
            + "/inputs",
            graph_checklist.fixture_subdirectory,
        )
        checker = _stage_scoped_config(
            GRAPH_STAGE_CONFIGS["compile-checker"],
            GRAPH_BASE_CONFIG,
            training_profile="s500-contract",
        )
        self.assertTrue(
            checker.output_subdirectory.startswith(
                "s500-contract/graph/npu-inductor-ascend-triton/"
                "compile-checker/"
            )
        )
        self.assertEqual(
            "s500-contract/inputs",
            checker.fixture_subdirectory,
        )
        graph_observation = _stage_scoped_config(
            GRAPH_STAGE_CONFIGS["observation"],
            GRAPH_BASE_CONFIG,
            training_profile="s500-contract",
        )
        eager_observation = _stage_scoped_config(
            replace(
                OBSERVATION_CONFIG,
                training=graph_observation.training,
            ),
            MIGRATION_CONFIG,
            training_profile="s500-contract",
        )
        root = Path("repo")
        graph_reference = _paths(
            root,
            graph_observation,
            graph_observation.reference.topology,
            "reference",
            1,
        )[1]
        eager_candidate = _paths(
            root,
            eager_observation,
            eager_observation.candidate.topology,
            "candidate",
            1,
        )[1]
        self.assertEqual(eager_candidate, graph_reference)
        formal = GRAPH_BASE_CONFIG.formal_fixture_config(
            GRAPH_BASE_CONFIG.candidate.topology
        )
        self.assertEqual("self_consistency", formal.kind)

    def test_graph_accuracy_entry_selects_compile_checker(self) -> None:
        stage, remaining = select_graph_stage(
            [
                "--stage",
                "compile-checker",
                "--device",
                "npu",
                "--graph-backend",
                "inductor",
                "--codegen-backend",
                "ascend-triton",
                "--capture",
                "candidate",
            ]
        )
        self.assertEqual("compile-checker", stage)
        self.assertEqual(["--capture", "candidate"], remaining)

    def test_graph_accuracy_supports_gpu_self_consistency(self) -> None:
        base, stages = graph_stage_configs(
            "gpu",
            graph_backend="inductor",
            codegen_backend="triton",
        )
        self.assertEqual("cuda", base.reference.device_type)
        self.assertEqual("cuda", base.candidate.device_type)
        self.assertEqual({}, base.reference.environment)
        self.assertEqual(
            "reference",
            stages["observation"].reuse_eager_role_as_reference,
        )
        self.assertIn("--compile.enable", stages["dump"].candidate.extra_args)
        formal = base.formal_fixture_config(base.candidate.topology)
        self.assertEqual("self_consistency", formal.kind)

    def test_graph_accuracy_backend_selection_changes_identity(self) -> None:
        triton, _ = graph_stage_configs(
            "npu",
            graph_backend="inductor",
            codegen_backend="ascend-triton",
        )
        dvm, _ = graph_stage_configs(
            "npu",
            graph_backend="inductor",
            codegen_backend="dvm",
        )
        self.assertEqual(triton.storage_name, dvm.storage_name)
        self.assertNotEqual(triton.execution_branch, dvm.execution_branch)
        self.assertEqual(
            "default",
            triton.candidate.environment["TORCHINDUCTOR_NPU_BACKEND"],
        )
        self.assertEqual(
            "dvm",
            dvm.candidate.environment["TORCHINDUCTOR_NPU_BACKEND"],
        )
        with self.assertRaisesRegex(ValueError, "invalid for gpu"):
            graph_stage_configs(
                "gpu",
                graph_backend="inductor",
                codegen_backend="dvm",
            )

    def test_named_experiment_contains_variable_operation_scopes(self) -> None:
        experiment = "fsdp8-accuracy-001"
        short = _stage_scoped_config(
            MIGRATION_CONFIG,
            MIGRATION_CONFIG,
            experiment,
        )
        long = _stage_scoped_config(
            replace(
                MONITOR_CONFIG,
                training=replace(MONITOR_CONFIG.training, steps=5000),
            ),
            MIGRATION_CONFIG,
            experiment,
        )
        self.assertEqual(MIGRATION_CONFIG.storage_name, short.storage_name)
        self.assertEqual(MIGRATION_CONFIG.storage_name, long.storage_name)
        self.assertIn("/dump/", short.output_subdirectory)
        self.assertIn("/observations/monitor/", long.output_subdirectory)
        self.assertNotEqual(short.fixture_subdirectory, long.fixture_subdirectory)

    def test_short_monitor_reuses_longer_compatible_fixture(self) -> None:
        scoped = _stage_scoped_config(
            replace(
                MONITOR_CONFIG,
                training=replace(MONITOR_CONFIG.training, steps=27),
            ),
            MIGRATION_CONFIG,
            "accuracy-experiment",
        )
        topology = scoped.candidate.topology
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            exact = _fixture_directory(root, scoped, topology)
            longer = exact.parent.parent / "s500-compatible" / "inputs"
            longer.mkdir(parents=True)
            stored_training = asdict(replace(scoped.training, steps=500))
            stored_training["converged_checkpoint"] = None
            (longer / "fixture.json").write_text(
                json.dumps(
                    {
                        "training": stored_training,
                        "token_plan": {"steps": 500},
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                longer,
                _compatible_fixture_directory(root, scoped, topology),
            )
            manifest = _compatible_fixture_manifest(root, scoped, topology)
            self.assertEqual(manifest["token_plan"]["steps"], 500)

    def test_step_21_dump_reuses_parent_training_window_layout(self) -> None:
        requested = _stage_scoped_config(
            replace(
                MIGRATION_CONFIG,
                training=replace(MIGRATION_CONFIG.training, steps=22),
                dump=replace(
                    MIGRATION_CONFIG.dump,
                    steps=(21,),
                    ranks=tuple(range(8)),
                    data_mode=("backward",),
                ),
            ),
            MIGRATION_CONFIG,
            "accuracy-experiment",
        )
        topology = requested.candidate.topology
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            exact = _fixture_directory(root, requested, topology)
            parent = exact.parent.parent / "s500-parent" / "inputs"
            parent.mkdir(parents=True)
            stored_training = asdict(replace(requested.training, steps=500))
            stored_training["converged_checkpoint"] = None
            (parent / "fixture.json").write_text(
                json.dumps(
                    {
                        "training": stored_training,
                        "token_plan": {"steps": 500},
                    }
                ),
                encoding="utf-8",
            )

            selected = _compatible_fixture_directory(root, requested, topology)
            self.assertEqual(parent, selected)
            profile = _training_profile_from_fixture(selected)
            scoped = _stage_scoped_config(
                replace(
                    MIGRATION_CONFIG,
                    training=replace(MIGRATION_CONFIG.training, steps=22),
                    dump=requested.dump,
                ),
                MIGRATION_CONFIG,
                "accuracy-experiment",
                training_profile=profile,
            )
            self.assertEqual("s500-parent/inputs", scoped.fixture_subdirectory)
            self.assertTrue(
                scoped.output_subdirectory.startswith("s500-parent/dump/")
            )

    def test_finished_monitor_run_can_be_finalized_without_training(self) -> None:
        topology = MONITOR_CONFIG.candidate.topology
        endpoint = replace(MONITOR_CONFIG.candidate, topology=topology)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            runtime_log = root / "runtime.log"
            runtime_log.write_text("training completed\n", encoding="utf-8")
            expected = {"valid": True, "steps": MONITOR_CONFIG.training.steps}
            with (
                patch(
                    "tests.glm5_2_mindstudio.workflow.load_token_plan",
                    return_value=object(),
                ),
                patch(
                    "tests.glm5_2_mindstudio.workflow.validate_runtime_input_contract",
                    return_value=expected,
                ) as validate_contract,
                patch(
                    "tests.glm5_2_mindstudio.workflow._validate_monitor_outputs"
                ) as validate_monitor,
            ):
                result = _finalizable_monitor_capture(
                    config=MONITOR_CONFIG,
                    topology=topology,
                    endpoint=endpoint,
                    official_output=root / "official",
                    input_contract=root / "input_contract",
                    runtime_log=runtime_log,
                    token_plan_path=root / "token_plan",
                )

            self.assertEqual(expected, result)
            validate_contract.assert_called_once()
            validate_monitor.assert_called_once()

    def test_named_experiment_fixture_path_is_shared_with_formal_producer(self) -> None:
        experiment = "glm5-debug-bf16-b64-seq128-seed61"
        scoped = _stage_scoped_config(
            MIGRATION_CONFIG,
            MIGRATION_CONFIG,
            experiment,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            topology = scoped.candidate.topology
            self.assertEqual(
                _fixture_directory(root, scoped, topology),
                _formal_fixture_directory(
                    root,
                    scoped.formal_fixture_config(topology),
                ),
            )

    def test_diagnostic_case_lives_below_canonical_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = create_case(
                Path(temporary_directory),
                case_id="display-alias",
                title="Display name",
                symptom="unknown",
                topologies=("fsdp8",),
                repeat=1,
                notes="",
            )
            self.assertEqual("display-alias", path.parent.name)
            self.assertEqual("diagnoses", path.parent.parent.name)
            self.assertEqual(
                MIGRATION_CONFIG.storage_name,
                path.parent.parent.parent.name,
            )

    def test_forcing_scoped_stage_preserves_default_dump(self) -> None:
        topology = MIGRATION_CONFIG.candidate.topology
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            experiment = "fsdp8-accuracy-001"
            dump_scope = _stage_scoped_config(
                MIGRATION_CONFIG,
                MIGRATION_CONFIG,
                experiment,
            )
            default_run, default_artifact, default_report = _paths(
                root,
                dump_scope,
                topology,
                "candidate",
                1,
            )
            scoped = _stage_scoped_config(
                CONFIG_CHECK_CONFIG,
                MIGRATION_CONFIG,
                experiment,
            )
            scoped_run, scoped_artifact, scoped_report = _paths(
                root,
                scoped,
                topology,
                "candidate",
                1,
            )
            for path in (
                default_run,
                default_artifact,
                default_report,
                scoped_run,
                scoped_artifact,
                scoped_report,
            ):
                path.mkdir(parents=True)

            reset_selected_outputs(
                root,
                scoped,
                topologies=(topology,),
                role="candidate",
                include_fixture=False,
            )

            self.assertTrue(default_run.is_dir())
            self.assertTrue(default_artifact.is_dir())
            self.assertTrue(default_report.is_dir())
            self.assertFalse(scoped_run.exists())
            self.assertFalse(scoped_artifact.exists())
            self.assertFalse(scoped_report.exists())

    def test_topology_precedes_every_accuracy_operation_scope(self) -> None:
        topology = MIGRATION_CONFIG.candidate.topology
        scoped = _stage_scoped_config(
            CONFIG_CHECK_CONFIG,
            MIGRATION_CONFIG,
            None,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run, artifact, report = _paths(
                root,
                scoped,
                topology,
                "candidate",
                1,
            )
            relative = (
                Path(MIGRATION_CONFIG.storage_name)
                / topology.slug
                / "checklist/configuration-check"
            )
            self.assertEqual(
                root / scoped.run_root / relative / "candidate-r1",
                run,
            )
            self.assertEqual(
                root / scoped.artifact_root / relative / "candidate-r1",
                artifact,
            )
            self.assertEqual(root / scoped.report_root / relative, report)

    def test_training_metrics_generate_csv_summary_and_charts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference.jsonl"
            candidate = root / "candidate.jsonl"

            def write_metrics(
                path: Path,
                losses: tuple[object, ...],
                grad_norms: tuple[float, ...],
            ) -> None:
                records = []
                for step, (loss, grad_norm) in enumerate(zip(losses, grad_norms)):
                    records.append(
                        json.dumps(
                            {
                                "step": step,
                                "rank": 0,
                                "metrics": {
                                    "loss_metrics/global_avg_loss": loss,
                                    "loss_metrics/global_max_loss": loss,
                                    "grad_norm": grad_norm,
                                },
                            }
                        )
                    )
                path.write_text("\n".join(records) + "\n", encoding="utf-8")

            write_metrics(reference, (2.0, 1.0, 0.5), (1.0, 0.001, 3.0))
            write_metrics(candidate, (2.0, 1.02, "NaN"), (1.0, 0.031, 3.0))
            summary_path = compare_training_metrics(
                reference_path=reference,
                candidate_path=candidate,
                output_directory=root / "output",
                spike_relative_threshold=0.4,
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(
                "candidate-nan-or-inf",
                summary["observation"]["diagnostic_symptom"],
            )
            self.assertEqual(
                {
                    "loss_metrics/global_avg_loss": 2,
                    "loss_metrics/global_max_loss": 2,
                },
                summary["observation"]["candidate_first_nonfinite_metrics"],
            )
            self.assertEqual(1, summary["loss"]["first_step_above_threshold"])
            self.assertEqual(2, summary["loss"]["candidate_nonfinite_step"])
            self.assertEqual([1, 2], summary["loss"]["reference_spike_steps"])
            self.assertEqual(0.0, summary["grad_norm"]["median_relative_error"])
            self.assertEqual(30.0, summary["grad_norm"]["max_relative_error"])
            self.assertEqual(1, summary["grad_norm"]["max_relative_error_step"])
            self.assertEqual(
                [1, 0, 2],
                [
                    item["step"]
                    for item in summary["grad_norm"][
                        "largest_relative_error_steps"
                    ]
                ],
            )
            prominent = summary["grad_norm"]["prominent_anomalies"]
            self.assertEqual(1, len(prominent))
            self.assertEqual(1, prominent[0]["step"])
            self.assertEqual(0.001, prominent[0]["reference"])
            self.assertEqual(0.031, prominent[0]["candidate"])
            self.assertEqual(30.0, prominent[0]["relative_error"])
            self.assertEqual(0, prominent[0]["previous"]["step"])
            self.assertEqual(2, prominent[0]["next"]["step"])
            for name in (
                "training_metrics_compare.csv",
                "training_observation.html",
                "loss.svg",
                "grad_norm.svg",
                "loss_relative_error.svg",
                "grad_norm_relative_error.svg",
                "loss_signed_difference.svg",
                "grad_norm_signed_difference.svg",
                "early_loss.svg",
                "early_loss_relative_error.svg",
                "grad_norm_signed_relative_error.svg",
            ):
                self.assertTrue((root / "output" / name).is_file())
            loss_error_chart = (
                root / "output" / "loss_relative_error.svg"
            ).read_text(encoding="utf-8")
            self.assertIn("Training step", loss_error_chart)
            self.assertIn("Relative error (%)", loss_error_chart)
            self.assertIn("Zero-error baseline", loss_error_chart)
            self.assertIn("Guidance 1%", loss_error_chart)
            self.assertIn("class=\"data-point\"", loss_error_chart)
            grad_error_chart = (
                root / "output" / "grad_norm_relative_error.svg"
            ).read_text(encoding="utf-8")
            self.assertIn("Gradient Norm Relative Error", grad_error_chart)
            self.assertIn("no universal acceptance threshold", grad_error_chart)
            self.assertFalse((root / "output" / "relative_error.svg").exists())
            interactive = (
                root / "output" / "training_observation.html"
            ).read_text(encoding="utf-8")
            self.assertIn("Training Observation", interactive)
            self.assertIn("Training Loss", interactive)
            self.assertIn("First Steps Loss", interactive)
            self.assertIn("First Steps Relative Error", interactive)
            self.assertIn("Whole-training Relative Error", interactive)
            self.assertIn("Grad Norm Relative Error", interactive)
            self.assertIn("NaN / Inf", interactive)
            self.assertIn("First-step inspection window", interactive)
            self.assertIn("After first guidance exceedance", interactive)
            self.assertIn('"showSymbol",true', interactive)
            self.assertIn('"interval",0', interactive)
            self.assertIn("dataZoom", interactive)
            self.assertIn("summary-table", interactive)
            self.assertIn("Interpretation", interactive)
            self.assertIn("font-size:42px", interactive)
            self.assertIn("font-size:27px", interactive)
            self.assertIn("border-collapse:collapse", interactive)
            self.assertIn("font-size:20px", interactive)
            self.assertNotIn("metric-card", interactive)
            self.assertNotIn('<script src="https://cdn', interactive)
            self.assertNotIn('<link rel="stylesheet" href="https://cdn', interactive)
            post_window = summary["loss"]["post_first_threshold_window"]
            self.assertEqual(1, post_window["first_step"])
            self.assertEqual(1, post_window["finite_step_count"])
            self.assertEqual(1.0, post_window["fraction_above_threshold"])
            early_window = summary["loss"]["early_window"]
            self.assertEqual(3, early_window["observed_step_count"])
            self.assertEqual(0, early_window["first_step"])
            self.assertEqual(1, early_window["first_step_above_threshold"])
            grad_signed_chart = (
                root / "output" / "grad_norm_signed_relative_error.svg"
            ).read_text(encoding="utf-8")
            self.assertIn("Precision upper guidance +5%", grad_signed_chart)
            self.assertIn("Precision lower guidance -5%", grad_signed_chart)
            self.assertEqual(
                "observed",
                summary["observation"]["nonfinite_analysis"]["candidate"][
                    "status"
                ],
            )

    def test_case_training_observation_is_cached(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            case_path = create_case(
                root,
                case_id="observation-001",
                title="Training observation",
                symptom="long-term-loss",
                topologies=("single",),
                repeat=1,
                notes="",
            )
            experiment = MIGRATION_CONFIG.storage_name
            config = _stage_scoped_config(
                MONITOR_CONFIG,
                MIGRATION_CONFIG,
                experiment,
            )
            run_root = root / config.run_root / config.output_relative_root
            for role, loss in (("reference", 1.0), ("candidate", 1.02)):
                path = (
                    run_root
                    / "single"
                    / config.operation_relative_root
                    / f"{role}-r1"
                    / "training_metrics.jsonl"
                )
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps(
                        {
                            "step": 0,
                            "rank": 0,
                            "metrics": {
                                "loss_metrics/global_avg_loss": loss,
                                "loss_metrics/global_max_loss": loss,
                                "grad_norm": 0.5,
                            },
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
            first = analyze_training_observation(
                root,
                case_id="observation-001",
                workflow="monitor",
                training_steps=100,
                loss_relative_threshold=0.01,
                grad_norm_relative_threshold=None,
                spike_relative_threshold=None,
                force=False,
            )
            second = analyze_training_observation(
                root,
                case_id="observation-001",
                workflow="monitor",
                training_steps=100,
                loss_relative_threshold=0.01,
                grad_norm_relative_threshold=None,
                spike_relative_threshold=None,
                force=False,
            )
            self.assertEqual(first, second)
            state = json.loads(
                (first / "observation_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual("completed", state["status"])

    @patch(
        "tests.glm5_2_mindstudio.accuracy_diagnostics.summarize_official_results",
        return_value={"verdict": "pass"},
    )
    @patch(
        "tests.glm5_2_mindstudio.accuracy_diagnostics._run_process",
    )
    @patch(
        "tests.glm5_2_mindstudio.accuracy_diagnostics.compare_command",
        return_value=["/opt/msprobe", "compare"],
    )
    @patch(
        "tests.glm5_2_mindstudio.accuracy_diagnostics.find_dump_compare_input",
    )
    @patch(
        "tests.glm5_2_mindstudio.accuracy_diagnostics.artifact_is_complete",
        return_value=True,
    )
    def test_repeat_compare_is_cached_and_keeps_both_captures(
        self,
        _complete,
        find_input,
        _command,
        run,
        _summarize,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            case_path = create_case(
                root,
                case_id="repeat-001",
                title="Repeat diagnosis",
                symptom="unstable",
                topologies=("single",),
                repeat=1,
                notes="",
            )
            experiment = MIGRATION_CONFIG.storage_name
            config = replace(
                MIGRATION_CONFIG,
                dump=replace(
                    MIGRATION_CONFIG.dump,
                    level="mix",
                    summary_mode="md5",
                ),
            )
            config = _stage_scoped_config(
                config,
                MIGRATION_CONFIG,
                experiment,
            )
            artifact_root = (
                root
                / config.artifact_root
                / config.output_relative_root
                / "single"
                / config.operation_relative_root
            )
            for repeat_value in (1, 2):
                artifact = artifact_root / f"candidate-r{repeat_value}"
                artifact.mkdir(parents=True)
                (artifact / "manifest.json").write_text(
                    json.dumps({"fixture_generation_id": "generation-a"}),
                    encoding="utf-8",
                )
            find_input.side_effect = lambda artifact, step: artifact / f"step{step}"

            first = compare_repeats(
                root,
                case_id="repeat-001",
                role="candidate",
                baseline_repeat=1,
                target_repeat=2,
            )
            second = compare_repeats(
                root,
                case_id="repeat-001",
                role="candidate",
                baseline_repeat=1,
                target_repeat=2,
            )

            self.assertEqual(first, second)
            self.assertEqual(2, run.call_count)
            self.assertTrue((first / "complete.json").is_file())
            state = json.loads(
                (first / "repeat_compare_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual("completed", state["status"])
            self.assertTrue((artifact_root / "candidate-r1").is_dir())
            self.assertTrue((artifact_root / "candidate-r2").is_dir())

    def test_case_advances_in_order_and_preserves_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evidence = root / "result.json"
            evidence.write_text("{}\n", encoding="utf-8")
            case_path = create_case(
                root,
                case_id="nan-001",
                title="GLM NaN diagnosis",
                symptom="nan-or-overflow",
                topologies=("single",),
                repeat=1,
                notes="",
            )
            value = json.loads(case_path.read_text(encoding="utf-8"))
            checklist = build_plan(value)
            self.assertEqual("checklist", checklist["stage"])
            commands = checklist["commands"]
            self.assertIn("--capture candidate", commands[1])
            self.assertIn("release_artifacts.py upload", commands[2])
            self.assertIn("release_artifacts.py download", commands[3])
            self.assertIn("--capture reference", commands[4])
            self.assertIn("--compare", commands[5])
            self.assertEqual(
                ["NPU: commands 1-3", "GPU: commands 4-7"],
                checklist["execution_hosts"],
            )
            self.assertTrue((case_path.parent / "README.md").is_file())

            record_stage(
                root,
                case_id="nan-001",
                stage="checklist",
                conclusion="pass",
                evidence=(str(evidence),),
                notes="reviewed",
                incident={},
            )
            value = json.loads(case_path.read_text(encoding="utf-8"))
            plan = build_plan(value)
            self.assertEqual("reproduce", plan["stage"])
            commands = "\n".join(plan["commands"])
            self.assertIn("--repeat 2", commands)
            self.assertIn("compare-repeats", commands)
            self.assertEqual(["result.json"], value["stages"]["checklist"]["evidence"])

    def test_stage_order_and_evidence_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            case_path = create_case(
                root,
                case_id="loss-001",
                title="Loss diagnosis",
                symptom="first-step-loss",
                topologies=("single",),
                repeat=1,
                notes="",
            )
            with self.assertRaisesRegex(ValueError, "complete stage 'checklist'"):
                record_stage(
                    root,
                    case_id="loss-001",
                    stage="reproduce",
                    conclusion="stable",
                    evidence=(str(root),),
                    notes="",
                    incident={},
                )
            with self.assertRaisesRegex(ValueError, "requires evidence"):
                record_stage(
                    root,
                    case_id="loss-001",
                    stage="checklist",
                    conclusion="pass",
                    evidence=(),
                    notes="",
                    incident={},
                )

            record_stage(
                root,
                case_id="loss-001",
                stage="checklist",
                conclusion="fail",
                evidence=(str(root),),
                notes="mismatch",
                incident={},
            )
            value = json.loads(case_path.read_text(encoding="utf-8"))
            self.assertEqual("checklist", build_plan(value)["stage"])
            with self.assertRaisesRegex(ValueError, "does not permit advancing"):
                record_stage(
                    root,
                    case_id="loss-001",
                    stage="reproduce",
                    conclusion="stable",
                    evidence=(str(root),),
                    notes="",
                    incident={},
                )

    def test_every_symptom_starts_with_uninstrumented_baseline(self) -> None:
        for case_id, symptom in (
            ("nan-002", "nan-or-overflow"),
            ("long-001", "long-term-loss"),
        ):
            with tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                evidence = root / "evidence"
                evidence.mkdir()
                path = create_case(
                    root,
                    case_id=case_id,
                    title=case_id,
                    symptom=symptom,
                    topologies=("single",),
                    repeat=1,
                    notes="",
                )
                for stage, conclusion in (
                    ("checklist", "pass"),
                    ("reproduce", "stable"),
                ):
                    record_stage(
                        root,
                        case_id=case_id,
                        stage=stage,
                        conclusion=conclusion,
                        evidence=(str(evidence),),
                        notes="",
                        incident={},
                    )
                value = json.loads(path.read_text(encoding="utf-8"))
                plan = build_plan(value)
                self.assertEqual("observe", plan["stage"])
                commands = "\n".join(plan["commands"])
                self.assertIn("--stage observation", commands)
                self.assertNotIn("--stage monitor", commands)
                self.assertNotIn("--stage dump", commands)

    def test_close_requires_supported_hypothesis_and_all_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evidence = root / "evidence.txt"
            evidence.write_text("evidence\n", encoding="utf-8")
            create_case(
                root,
                case_id="closed-001",
                title="Closed diagnosis",
                symptom="first-step-loss",
                topologies=("single",),
                repeat=1,
                notes="",
            )
            conclusions = {
                "checklist": "pass",
                "reproduce": "stable",
                "observe": "abnormal",
                "localize": "localized",
                "verify": "confirmed",
                "validate": "pass",
            }
            for stage, conclusion in conclusions.items():
                record_stage(
                    root,
                    case_id="closed-001",
                    stage=stage,
                    conclusion=conclusion,
                    evidence=(str(evidence),),
                    notes="",
                    incident={
                        "step": 0 if stage == "localize" else None,
                        "rank": 0 if stage == "localize" else None,
                        "phase": "forward" if stage == "localize" else None,
                        "module": "Module.layers.0" if stage == "localize" else None,
                    },
                )
            with self.assertRaisesRegex(ValueError, "supported hypothesis"):
                close_case(root, case_id="closed-001")
            add_hypothesis(
                root,
                case_id="closed-001",
                statement="GELU device implementation amplifies the input error",
                experiment="Move only GELU to CPU.",
            )
            record_hypothesis(
                root,
                case_id="closed-001",
                hypothesis_id=1,
                verdict="supported",
                evidence=(str(evidence),),
                notes="Loss aligned after the one-variable change.",
            )
            path = close_case(root, case_id="closed-001")
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual("closed", value["status"])


if __name__ == "__main__":
    unittest.main()
