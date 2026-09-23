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
from tests.glm5_2_mindstudio.migration_benchmark import CONFIG as MIGRATION_CONFIG
from tests.glm5_2_mindstudio.training_baseline_benchmark import (
    CONFIG as BASELINE_CONFIG,
)
from tests.glm5_2_mindstudio.training_monitor_benchmark import (
    CONFIG as MONITOR_CONFIG,
)
from tests.glm5_2_mindstudio.training_observation import compare_training_metrics
from tests.glm5_2_mindstudio.workflow import (
    _adopt_legacy_accuracy_storage,
    _fixture_directory,
    _paths,
    _stage_scoped_config,
    reset_selected_outputs,
)
from tests.glm5_2_precision.workflow import (
    _fixture_directory as _formal_fixture_directory,
)


class MindStudioDiagnosticsTest(unittest.TestCase):
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
            BASELINE_CONFIG.storage_name,
        )
        self.assertEqual(
            Path(MIGRATION_CONFIG.storage_name),
            CONFIG_CHECK_CONFIG.output_relative_root,
        )
        self.assertEqual(
            Path("diagnostics/configuration-check"),
            CONFIG_CHECK_CONFIG.operation_relative_root,
        )
        baseline = _stage_scoped_config(BASELINE_CONFIG, MIGRATION_CONFIG)
        self.assertEqual(
            Path(MIGRATION_CONFIG.storage_name),
            baseline.output_relative_root,
        )
        self.assertTrue(
            baseline.operation_relative_root.as_posix().startswith(
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

    def test_unified_accuracy_entry_defaults_to_baseline(self) -> None:
        stage, remaining = _select_stage(
            ["--capture", "candidate", "--topology", "single"]
        )
        self.assertEqual("baseline", stage)
        self.assertEqual(
            ["--capture", "candidate", "--topology", "single"], remaining
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
        self.assertTrue(short.output_subdirectory.startswith("captures/"))
        self.assertTrue(long.output_subdirectory.startswith("observations/monitor/"))
        self.assertNotEqual(short.fixture_subdirectory, long.fixture_subdirectory)

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

    def test_named_experiment_adopts_flattened_fixture_path(self) -> None:
        experiment = "glm5-debug-bf16-b64-seq128-seed61"
        scoped = _stage_scoped_config(
            MIGRATION_CONFIG,
            MIGRATION_CONFIG,
            experiment,
        )
        topology = scoped.candidate.topology
        formal = scoped.formal_fixture_config(topology)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            legacy = (
                root
                / formal.fixture_root
                / experiment
                / scoped.fixture_subdirectory.replace("inputs/", "fixtures/", 1)
            )
            legacy.mkdir(parents=True)
            (legacy / "fixture.json").write_text(
                json.dumps({"training": asdict(formal.training)}),
                encoding="utf-8",
            )

            _adopt_legacy_accuracy_storage(
                root,
                scoped,
                topologies=(topology,),
                legacy_storage_name=experiment,
            )
            destination = _formal_fixture_directory(root, formal)

            self.assertEqual(
                _fixture_directory(root, scoped, topology),
                destination,
            )
            self.assertTrue((destination / "fixture.json").is_file())
            self.assertFalse(legacy.exists())

    def test_named_experiment_adopts_operation_below_topology(self) -> None:
        experiment = "glm5-debug-bf16-b64-seq128-seed61"
        scoped = _stage_scoped_config(
            CONFIG_CHECK_CONFIG,
            MIGRATION_CONFIG,
            experiment,
        )
        topology = scoped.candidate.topology
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            legacy = (
                root
                / scoped.artifact_root
                / experiment
                / scoped.operation_relative_root
                / topology.slug
            )
            legacy.mkdir(parents=True)
            (legacy / "candidate-r1").mkdir()

            _adopt_legacy_accuracy_storage(
                root,
                scoped,
                topologies=(topology,),
                legacy_storage_name=experiment,
            )
            destination = (
                root
                / scoped.artifact_root
                / MIGRATION_CONFIG.storage_name
                / topology.slug
                / scoped.operation_relative_root
            )

            self.assertTrue((destination / "candidate-r1").is_dir())
            self.assertFalse(legacy.exists())

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

            def write_metrics(path: Path, losses: tuple[object, ...]) -> None:
                records = []
                for step, loss in enumerate(losses):
                    records.append(
                        json.dumps(
                            {
                                "step": step,
                                "rank": 0,
                                "metrics": {
                                    "loss_metrics/global_avg_loss": loss,
                                    "loss_metrics/global_max_loss": loss,
                                    "grad_norm": 1.0 + step,
                                },
                            }
                        )
                    )
                path.write_text("\n".join(records) + "\n", encoding="utf-8")

            write_metrics(reference, (2.0, 1.0, 0.5))
            write_metrics(candidate, (2.0, 1.02, "NaN"))
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
            for name in (
                "training_metrics_compare.csv",
                "loss.svg",
                "grad_norm.svg",
                "loss_relative_error.svg",
                "grad_norm_relative_error.svg",
            ):
                self.assertTrue((root / "output" / name).is_file())
            loss_error_chart = (
                root / "output" / "loss_relative_error.svg"
            ).read_text(encoding="utf-8")
            self.assertIn("Training step", loss_error_chart)
            self.assertIn("Relative error (%)", loss_error_chart)
            self.assertIn("Zero-error baseline", loss_error_chart)
            self.assertIn("Guidance 1%", loss_error_chart)
            grad_error_chart = (
                root / "output" / "grad_norm_relative_error.svg"
            ).read_text(encoding="utf-8")
            self.assertIn("Gradient Norm Relative Error", grad_error_chart)
            self.assertIn("no universal acceptance threshold", grad_error_chart)
            self.assertFalse((root / "output" / "relative_error.svg").exists())

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
