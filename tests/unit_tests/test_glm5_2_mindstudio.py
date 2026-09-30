# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""CPU-only contracts for the official MindStudio experiment family."""

from __future__ import annotations

from contextlib import redirect_stdout
import csv
from dataclasses import asdict, replace
import io
import json
import os
import shlex
import sys
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.glm5_2_common.topology import ParallelTopology
from tests.glm5_2_common.naming import config_digest
from tests.glm5_2_mindstudio.artifacts import (
    MindStudioArtifactError,
    artifact_is_complete,
    output_index,
    reusable_artifact_is_complete,
    validate_output_files,
    write_json,
)
from tests.glm5_2_mindstudio.config import (
    MindStudioExperimentConfig,
    MsProbeCompileConfig,
    MsProbeDumpConfig,
    MsProbeMonitorConfig,
)
from tests.glm5_2_mindstudio.msprobe_adapter import (
    aggregate_compile_csv,
    compare_command,
    config_check_compare_command,
    find_dump_compare_input,
    graph_visualize_command,
    overflow_check_command,
    precheck_command,
    precheck_compare_command,
    summarize_official_results,
    trend_data2db_command,
    validate_compile_report,
)
from tests.glm5_2_mindstudio.report import write_report_index
from tests.glm5_2_mindstudio.workflow import (
    _capture_is_complete,
    _capture_toolchain_compatibility,
    _compatible_experiment_digests,
    _comparison_directory,
    _comparison_is_current,
    _comparison_output_index,
    _comparison_toolchain_identity,
    _compile_rank_csv_files,
    _compatible_fixture_directory,
    _experiment_digest,
    _endpoint_display_labels,
    _fixture_directory,
    _legacy_stage_scoped_config,
    _intermediate_training_window_config,
    _move_legacy_path,
    _migrate_legacy_stage_layout,
    _paths,
    _stage_scoped_config,
    _validate_compare_options,
    _validate_compile_outputs,
    _validate_monitor_execution,
    _validate_monitor_outputs,
    _wait_for_active_capture,
    _write_launch_contract,
    capture_official,
    compare_official,
    compare_prechecks,
    reset_precheck_outputs,
    reset_compare_outputs,
    reset_selected_outputs,
    run_mindstudio_cli,
    run_precheck,
)
from tests.glm5_2_mindstudio.visualization import run_graph_visualization
from tests.glm5_2_precision.workflow import (
    FormalTrainingConfig,
    TrainingEndpoint,
)


def _experiment(
    *,
    workflow: str = "migration",
    reference_device: str = "cuda",
    candidate_device: str = "npu",
) -> MindStudioExperimentConfig:
    topology = ParallelTopology("single", 1)
    return MindStudioExperimentConfig(
        name="unit-test",
        workflow=workflow,  # type: ignore[arg-type]
        reference=TrainingEndpoint(
            "reference",
            reference_device,  # type: ignore[arg-type]
            "0",
            topology,
            repeats=1,
        ),
        candidate=TrainingEndpoint(
            "candidate",
            candidate_device,  # type: ignore[arg-type]
            "0",
            topology,
            repeats=1,
        ),
        training=FormalTrainingConfig(
            steps=2,
            local_batch_size=1,
            global_batch_size=1,
            sequence_length=8,
        ),
        dump=MsProbeDumpConfig(steps=(0, 1)),
    )


def _test_toolchain_compatibility() -> dict[str, object]:
    return _capture_toolchain_compatibility(_test_toolchain_identity())


def _test_execution_identity() -> dict[str, object]:
    source = {
        "commit": "source-a",
        "branch": "main",
        "status": "",
        "dirty_tree_sha256": "dirty-a",
    }
    return {
        "device_type": "npu",
        "packages": {
            "torch": "2.0",
            "torch_npu": "2.0",
            "torchtitan": "0.2",
            "torchtitanturbo": "0.1",
        },
        "sources": {
            "torchtitan_test": dict(source),
            "torchtitan": dict(source),
            "torchtitanturbo": dict(source),
        },
        "python": {"version": "3.12", "executable_sha256": "python-a"},
        "cann_version_files": ["cann-a"],
    }


def _test_toolchain_metadata(
    *,
    version: str = "1.2.3",
    commit: str = "commit-a",
) -> dict[str, object]:
    return {
        "profile_name": "mindstudio-standard",
        "lock": {"sha256": "lock-a"},
        "versions": {"msprobe": version},
        "sources": {
            "msprobe": {
                "official_url": "https://example.invalid/msprobe.git",
                "selected_ref": "master",
                "locked_commit": None,
                "locked_version": None,
                "commit": commit,
                "branch": "master",
                "tag": None,
                "dirty": False,
            }
        },
        "doctor": {
            "checks": {
                "msprobe": {
                    "executable": {"path": None},
                    "import": {"origin": None},
                }
            }
        },
        "resolved_toolchain": {
            "msprobe_source_commit": commit,
            "msprobe_artifact_identity": "artifact-a",
        },
    }


def _test_toolchain_identity() -> dict[str, object]:
    identity = _comparison_toolchain_identity(_test_toolchain_metadata())
    identity["execution_environment"] = _test_execution_identity()
    return identity


def _write_fixture_and_capture(
    root: Path,
    config: MindStudioExperimentConfig,
    role: str,
) -> Path:
    topology = config.candidate.topology
    fixture = _fixture_directory(root, config, topology)
    fixture.mkdir(parents=True, exist_ok=True)
    write_json(fixture / "fixture.json", {"generation_id": "fixture-a"})
    _, artifact, _ = _paths(
        root,
        config,
        topology,
        role,  # type: ignore[arg-type]
        1,
    )
    official = artifact / "official"
    for step in config.dump.steps:
        dump = official / f"step{step}" / "rank0" / "dump.json"
        dump.parent.mkdir(parents=True, exist_ok=True)
        dump.write_text(
            json.dumps({"step": step, "role": role}),
            encoding="utf-8",
        )
    digest = _experiment_digest(
        config,
        topology,
        role,  # type: ignore[arg-type]
    )
    write_json(
        artifact / "manifest.json",
        {
            "experiment_digest": digest,
            "fixture_generation_id": "fixture-a",
            "official_output": "official",
            "official_files": output_index(official),
            "toolchain_identity": _test_toolchain_identity(),
            "toolchain_compatibility": _test_toolchain_compatibility(),
            "input_contract": {"valid": True},
        },
    )
    write_json(artifact / "complete.json", {"status": "completed"})
    return artifact


class TestMindStudioConfig(unittest.TestCase):
    def test_graph_observation_labels_describe_actual_endpoints(self) -> None:
        from tests.glm5_2_mindstudio.graph_accuracy_benchmark import (
            _stage_configs,
        )

        config = _stage_configs(
            "npu",
            graph_backend="inductor",
            codegen_backend="ascend-triton",
        )[1]["observation"]

        self.assertEqual(
            (
                "NPU eager",
                "NPU graph (Inductor + Ascend Triton)",
            ),
            _endpoint_display_labels(config),
        )

    def test_stage_paths_are_grouped_below_the_training_profile(self) -> None:
        base = _experiment()
        expected_prefix = f"s2-{config_digest(asdict(base.training), length=8)}"
        stages = {
            workflow: _stage_scoped_config(
                replace(base, workflow=workflow),
                base,
                experiment_name="shared",
            )
            for workflow in ("config-check", "observation", "migration", "monitor")
        }

        for config in stages.values():
            self.assertEqual(
                Path(expected_prefix) / "inputs",
                Path(config.fixture_subdirectory or ""),
            )
            self.assertEqual(
                expected_prefix,
                Path(config.output_subdirectory or "").parts[0],
            )
        self.assertEqual(
            Path(expected_prefix) / "observations" / "training",
            Path(stages["observation"].output_subdirectory or ""),
        )
        self.assertEqual(
            "observations",
            Path(stages["monitor"].output_subdirectory or "").parts[1],
        )
        self.assertEqual(
            "monitor",
            Path(stages["monitor"].output_subdirectory or "").parts[2],
        )

    def test_legacy_stage_layout_moves_without_rewriting_outputs(self) -> None:
        base = _experiment()
        selected = replace(base, workflow="monitor")
        legacy = _legacy_stage_scoped_config(
            selected, base, experiment_name="shared"
        )
        current = _stage_scoped_config(
            selected, base, experiment_name="shared"
        )
        topology = base.candidate.topology
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            legacy_fixture = _fixture_directory(root, legacy, topology)
            legacy_fixture.mkdir(parents=True)
            (legacy_fixture / "fixture.json").write_bytes(b"fixture")
            for configured_root in (
                current.run_root,
                current.artifact_root,
                current.report_root,
            ):
                source = (
                    root
                    / configured_root
                    / current.output_relative_root
                    / topology.slug
                    / legacy.operation_relative_root
                )
                source.mkdir(parents=True)
                (source / "marker").write_bytes(configured_root.encode())

            _migrate_legacy_stage_layout(
                root, legacy, current, (topology,)
            )

            self.assertEqual(
                b"fixture",
                (_fixture_directory(root, current, topology) / "fixture.json").read_bytes(),
            )
            for configured_root in (
                current.run_root,
                current.artifact_root,
                current.report_root,
            ):
                destination = (
                    root
                    / configured_root
                    / current.output_relative_root
                    / topology.slug
                    / current.operation_relative_root
                    / "marker"
                )
                self.assertEqual(configured_root.encode(), destination.read_bytes())

    def test_legacy_migration_keeps_different_training_fixture_profiles(self) -> None:
        base = _experiment()
        legacy_stage = replace(base, workflow="config-check")
        legacy = replace(
            _legacy_stage_scoped_config(
                legacy_stage, base, experiment_name="shared"
            ),
            fixture_subdirectory="inputs/s2-legacy",
        )
        current_stage = replace(
            legacy_stage,
            fixture_training=replace(base.training, steps=500),
        )
        current = replace(
            _stage_scoped_config(
                current_stage, base, experiment_name="shared"
            ),
            fixture_subdirectory="s500-current/inputs",
        )
        topology = base.candidate.topology
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            legacy_fixture = _fixture_directory(root, legacy, topology)
            current_fixture = _fixture_directory(root, current, topology)
            legacy_fixture.mkdir(parents=True)
            current_fixture.mkdir(parents=True)
            (legacy_fixture / "token_generation.log").write_text("s2")
            (current_fixture / "token_generation.log").write_text("s500")

            _migrate_legacy_stage_layout(
                root, legacy, current, (topology,)
            )

            self.assertEqual(
                "s2", (legacy_fixture / "token_generation.log").read_text()
            )
            self.assertEqual(
                "s500", (current_fixture / "token_generation.log").read_text()
            )

    def test_monitor_reuses_normal_training_profile_not_short_run_profile(
        self,
    ) -> None:
        base = _experiment()
        monitor_training = replace(base.training, steps=27)
        monitor = _stage_scoped_config(
            replace(base, workflow="monitor", training=monitor_training),
            base,
            experiment_name="shared",
        )
        topology = base.candidate.topology
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            topology_root = (
                root
                / monitor.fixture_root
                / monitor.storage_name
                / topology.slug
            )
            for steps, profile in ((27, "s27-short"), (500, "s500-normal")):
                fixture = topology_root / profile / "inputs"
                fixture.mkdir(parents=True)
                training = asdict(monitor.training)
                training["steps"] = steps
                write_json(
                    fixture / "fixture.json",
                    {
                        "training": training,
                        "token_plan": {"steps": steps},
                    },
                )
            normal = (
                root
                / monitor.run_root
                / monitor.output_relative_root
                / topology.slug
                / "s500-normal"
                / "observations"
                / "training"
            )
            normal.mkdir(parents=True)

            selected = _compatible_fixture_directory(root, monitor, topology)

            self.assertEqual(topology_root / "s500-normal" / "inputs", selected)

    def test_intermediate_monitor_layout_and_html_are_migrated(self) -> None:
        base = _experiment()
        selected = replace(base, workflow="monitor")
        intermediate = _intermediate_training_window_config(
            selected, base, experiment_name="shared"
        )
        current = _stage_scoped_config(
            selected,
            base,
            experiment_name="shared",
            training_profile="s500-normal",
        )
        topology = base.candidate.topology
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for configured_root in (
                current.run_root,
                current.artifact_root,
                current.report_root,
            ):
                source = (
                    root
                    / configured_root
                    / current.output_relative_root
                    / topology.slug
                    / intermediate.operation_relative_root
                )
                source.mkdir(parents=True)
                (source / "marker").write_text(configured_root)
            collected = (
                root
                / current.report_root
                / current.output_relative_root
                / "html_reports"
                / topology.slug
                / intermediate.operation_relative_root
                / "report.html"
            )
            collected.parent.mkdir(parents=True)
            collected.write_text("html")

            _migrate_legacy_stage_layout(
                root, (intermediate,), current, (topology,)
            )

            for configured_root in (
                current.run_root,
                current.artifact_root,
                current.report_root,
            ):
                destination = (
                    root
                    / configured_root
                    / current.output_relative_root
                    / topology.slug
                    / current.operation_relative_root
                    / "marker"
                )
                self.assertEqual(configured_root, destination.read_text())
            destination = (
                root
                / current.report_root
                / current.output_relative_root
                / "html_reports"
                / "s500-normal"
                / "monitor"
                / current.operation_relative_root.name
                / f"{topology.slug}.html"
            )
            self.assertEqual("html", destination.read_text())
            self.assertFalse(
                (
                    root
                    / current.report_root
                    / current.output_relative_root
                    / "html_reports"
                    / topology.slug
                ).exists()
            )

    def test_force_compare_reset_removes_legacy_report_aliases(self) -> None:
        base = _experiment()
        selected = replace(base, workflow="monitor")
        legacy = _legacy_stage_scoped_config(
            selected, base, experiment_name="shared"
        )
        intermediate = _intermediate_training_window_config(
            selected, base, experiment_name="shared"
        )
        current = _stage_scoped_config(
            selected,
            base,
            experiment_name="shared",
            training_profile="s500-normal",
        )
        topology = base.candidate.topology
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            report_root = (
                root
                / current.report_root
                / current.output_relative_root
            )
            aliases = (
                report_root / "topologies" / f"{topology.slug}.html",
                report_root
                / "html_reports"
                / topology.slug
                / intermediate.operation_relative_root
                / "report.html",
                report_root
                / "operation_indexes"
                / legacy.operation_relative_root
                / "report.json",
            )
            for alias in aliases:
                alias.parent.mkdir(parents=True, exist_ok=True)
                alias.write_text("stale", encoding="utf-8")

            reset_compare_outputs(
                root,
                current,
                topologies=(topology,),
                repeat=1,
                legacy_configs=(legacy, intermediate),
            )

            self.assertTrue(all(not alias.exists() for alias in aliases))

    def test_stale_legacy_report_is_removed_when_canonical_exists(self) -> None:
        base = _experiment()
        selected = replace(base, workflow="monitor")
        intermediate = _intermediate_training_window_config(
            selected, base, experiment_name="shared"
        )
        current = _stage_scoped_config(
            selected,
            base,
            experiment_name="shared",
            training_profile="s500-normal",
        )
        topology = base.candidate.topology
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            legacy_report = (
                root
                / current.report_root
                / current.output_relative_root
                / "html_reports"
                / topology.slug
                / intermediate.operation_relative_root
                / "report.html"
            )
            canonical_report = (
                root
                / current.report_root
                / current.output_relative_root
                / "html_reports"
                / "s500-normal"
                / "monitor"
                / current.operation_relative_root.name
                / f"{topology.slug}.html"
            )
            legacy_report.parent.mkdir(parents=True)
            canonical_report.parent.mkdir(parents=True)
            legacy_report.write_text("stale", encoding="utf-8")
            canonical_report.write_text("current", encoding="utf-8")

            _migrate_legacy_stage_layout(
                root, (intermediate,), current, (topology,)
            )

            self.assertFalse(legacy_report.exists())
            self.assertEqual("current", canonical_report.read_text())

    def test_interrupted_legacy_migration_resumes_by_safe_merge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "legacy"
            destination = root / "current"
            (source / "nested").mkdir(parents=True)
            (destination / "nested").mkdir(parents=True)
            (source / "nested" / "same.json").write_text("same")
            (destination / "nested" / "same.json").write_text("same")
            (source / "legacy-only.csv").write_text("legacy")
            (destination / "current-only.html").write_text("current")

            _move_legacy_path(source, destination)

            self.assertFalse(source.exists())
            self.assertEqual(
                "same", (destination / "nested" / "same.json").read_text()
            )
            self.assertEqual(
                "legacy", (destination / "legacy-only.csv").read_text()
            )
            self.assertEqual(
                "current", (destination / "current-only.html").read_text()
            )

    def test_legacy_migration_rejects_different_same_name_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "legacy.json"
            destination = root / "current.json"
            source.write_text("legacy")
            destination.write_text("current")

            with self.assertRaisesRegex(FileExistsError, "conflicting entries"):
                _move_legacy_path(source, destination)

    def test_observation_accepts_existing_training_capture_identity(self) -> None:
        config = _experiment(workflow="observation")
        topology = config.candidate.topology
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = _write_fixture_and_capture(root, config, "candidate")
            manifest_path = artifact / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            legacy = _experiment_digest(
                replace(config, workflow="baseline"),  # type: ignore[arg-type]
                topology,
                "candidate",
            )
            manifest["experiment_digest"] = legacy
            write_json(manifest_path, manifest)

            self.assertTrue(
                _capture_is_complete(
                    artifact,
                    experiment_digests=_compatible_experiment_digests(
                        config, topology, "candidate"
                    ),
                    fixture_generation_id="fixture-a",
                )
            )

    def test_dump_config_is_the_official_precision_debugger_schema(self) -> None:
        dump = MsProbeDumpConfig(
            task="tensor",
            level="mix",
            steps=(0, 2),
            ranks=(0, 3),
            async_dump=True,
            scope=("Glm5TransformerBlock",),
            module_or_api_list=("torch.matmul",),
            tensor_list=("input", "output"),
            data_mode=("forward", "backward"),
            summary_mode="statistics",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "official"
            official = dump.official_config(path)

        self.assertEqual("tensor", official["task"])
        self.assertEqual("mix", official["level"])
        self.assertEqual([0, 2], official["step"])
        self.assertEqual([0, 3], official["rank"])
        self.assertTrue(official["async_dump"])
        self.assertEqual(["Glm5TransformerBlock"], official["tensor"]["scope"])
        self.assertEqual("statistics", official["tensor"]["summary_mode"])
        self.assertNotIn("tensor_list", official["tensor"])

    def test_dump_config_rejects_unsupported_async_combinations(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not support md5"):
            MsProbeDumpConfig(async_dump=True, summary_mode="md5")
        with self.assertRaisesRegex(ValueError, "requires a non-empty list"):
            MsProbeDumpConfig(task="tensor", async_dump=True)

    def test_dump_steps_are_sorted_unique_and_inside_training_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "sorted and unique"):
            MsProbeDumpConfig(steps=(1, 0))
        with self.assertRaisesRegex(ValueError, "sorted and unique"):
            MsProbeDumpConfig(ranks=(1, 0))
        with self.assertRaisesRegex(ValueError, "greater than every"):
            MindStudioExperimentConfig(
                **{
                    **_experiment().__dict__,
                    "training": FormalTrainingConfig(
                        steps=1,
                        local_batch_size=1,
                        global_batch_size=1,
                        sequence_length=8,
                    ),
                }
            )

    def test_output_roots_must_stay_below_the_repository(self) -> None:
        with self.assertRaisesRegex(ValueError, "repository-relative"):
            MindStudioExperimentConfig(
                **{**_experiment().__dict__, "artifact_root": "../outside"}
            )
        with self.assertRaisesRegex(ValueError, "repository-relative"):
            MindStudioExperimentConfig(
                **{**_experiment().__dict__, "report_root": str(Path.cwd())}
            )

    def test_identity_and_storage_separate_official_workflow_kinds(self) -> None:
        migration = _experiment()
        config_check = _experiment(workflow="config-check")
        compile_experiment = _experiment(
            workflow="compile",
            reference_device="npu",
            candidate_device="npu",
        )

        self.assertIn("migration-cuda-npu", migration.storage_name)
        self.assertIn("config-check-cuda-npu", config_check.storage_name)
        self.assertIn("compile-npu-inductor", compile_experiment.storage_name)
        self.assertNotEqual(migration.storage_name, config_check.storage_name)
        self.assertEqual("migration", migration.formal_fixture_config().kind)
        self.assertEqual(
            "self_consistency", compile_experiment.formal_fixture_config().kind
        )

    def test_compile_config_rejects_empty_backend_and_cross_device_pair(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            MsProbeCompileConfig(backend=" ")
        with self.assertRaisesRegex(ValueError, "one device type"):
            _experiment(workflow="compile")

    def test_monitor_config_matches_the_documented_v2_schema(self) -> None:
        monitor = MsProbeMonitorConfig(
            ranks=(0, 3),
            start_step=2,
            stop_step=12,
            step_interval=2,
            step_count_per_record=5,
            collect_times=20,
            module=True,
            optimizer=True,
            param=True,
            cc=True,
            module_targets=("layers.0", "router"),
            monitor_mbs_grad=True,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "monitor"
            official = monitor.official_config(output)

        self.assertEqual("pytorch", official["framework"])
        self.assertEqual([0, 3], official["rank"])
        self.assertEqual(2, official["start_step"])
        self.assertEqual(12, official["stop_step"])
        self.assertEqual(2, official["step_interval"])
        self.assertEqual(5, official["step_count_per_record"])
        self.assertEqual(20, official["collect_times"])
        self.assertEqual("csv", official["format"])
        self.assertFalse(official["patch_optimizer_step"])
        self.assertEqual(
            ["layers.0", "router"],
            official["monitors"]["module"]["targets"],
        )
        self.assertTrue(
            official["monitors"]["weight_grad"]["monitor_mbs_grad"]
        )
        self.assertTrue(
            official["monitors"]["optimizer"]["mv_distribution"]
        )
        self.assertTrue(
            official["monitors"]["param"]["param_distribution"]
        )

    def test_monitor_options_change_only_monitor_experiment_identity(self) -> None:
        migration = _experiment()
        changed_migration = replace(
            migration,
            monitor=replace(migration.monitor, step_interval=2),
        )
        self.assertEqual(migration.storage_name, changed_migration.storage_name)
        self.assertNotIn("monitor", migration.identity)

        monitor = _experiment(workflow="monitor")
        changed_monitor = replace(
            monitor,
            monitor=replace(monitor.monitor, step_interval=2),
        )
        self.assertNotEqual(monitor.storage_name, changed_monitor.storage_name)
        self.assertIn("monitor", monitor.identity)


class TestMindStudioArtifacts(unittest.TestCase):
    def test_complete_marker_binds_identity_generation_and_file_digests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact = Path(temporary_directory) / "capture"
            official = artifact / "official"
            official.mkdir(parents=True)
            result = official / "step0" / "rank0" / "dump.json"
            result.parent.mkdir(parents=True)
            result.write_text('{"value": 1}\n', encoding="utf-8")
            write_json(
                artifact / "manifest.json",
                {
                    "experiment_digest": "experiment-a",
                    "fixture_generation_id": "fixture-a",
                    "official_output": "official",
                    "official_files": output_index(official),
                },
            )
            write_json(artifact / "complete.json", {"status": "completed"})

            self.assertTrue(
                artifact_is_complete(
                    artifact,
                    experiment_digest="experiment-a",
                    fixture_generation_id="fixture-a",
                )
            )
            self.assertFalse(
                artifact_is_complete(
                    artifact,
                    experiment_digest="experiment-a",
                    fixture_generation_id="fixture-b",
                )
            )

            result.write_text('{"value": 2}\n', encoding="utf-8")
            self.assertFalse(
                artifact_is_complete(
                    artifact,
                    experiment_digest="experiment-a",
                    fixture_generation_id="fixture-a",
                )
            )

    def test_manifest_official_output_cannot_escape_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = root / "capture"
            external = root / "external"
            artifact.mkdir()
            external.mkdir()
            (external / "dump.json").write_text("{}", encoding="utf-8")
            write_json(
                artifact / "manifest.json",
                {
                    "experiment_digest": "experiment-a",
                    "fixture_generation_id": "fixture-a",
                    "official_output": "../external",
                    "official_files": output_index(external),
                },
            )
            write_json(artifact / "complete.json", {"status": "completed"})

            self.assertFalse(
                artifact_is_complete(
                    artifact,
                    experiment_digest="experiment-a",
                    fixture_generation_id="fixture-a",
                )
            )

    def test_reusable_capture_binds_its_own_digest_workflow_and_role(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact = Path(temporary_directory) / "capture"
            official = artifact / "official"
            official.mkdir(parents=True)
            (official / "result.json").write_text("{}", encoding="utf-8")
            manifest = {
                "workflow": "config-check",
                "role": "candidate",
                "experiment_digest": "eager-digest",
                "fixture_generation_id": "fixture-a",
                "official_output": "official",
                "official_files": output_index(official),
            }
            write_json(artifact / "manifest.json", manifest)
            write_json(
                artifact / "complete.json",
                {"status": "completed"},
            )

            self.assertTrue(
                reusable_artifact_is_complete(
                    artifact,
                    fixture_generation_id="fixture-a",
                    workflow="config-check",
                    role="candidate",
                )
            )
            self.assertFalse(
                reusable_artifact_is_complete(
                    artifact,
                    fixture_generation_id="fixture-a",
                    workflow="config-check",
                    role="reference",
                )
            )

            manifest["workflow"] = "baseline"
            write_json(artifact / "manifest.json", manifest)
            self.assertTrue(
                reusable_artifact_is_complete(
                    artifact,
                    fixture_generation_id="fixture-a",
                    workflow="observation",
                    role="candidate",
                )
            )
    def test_output_index_rejects_symbolic_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            official = Path(temporary_directory) / "official"
            official.mkdir()
            linked = official / "linked.json"
            linked.write_text("{}", encoding="utf-8")
            original_is_symlink = Path.is_symlink

            def report_test_link(path: Path) -> bool:
                if path == linked:
                    return True
                return original_is_symlink(path)

            with patch.object(Path, "is_symlink", new=report_test_link):
                with self.assertRaisesRegex(
                    MindStudioArtifactError,
                    "symbolic links are not supported",
                ):
                    output_index(official)

    def test_output_validation_and_dump_input_cover_distributed_ranks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact = Path(temporary_directory) / "artifact"
            for rank in range(2):
                rank_directory = artifact / "official" / "step3" / f"rank{rank}"
                rank_directory.mkdir(parents=True)
                (rank_directory / "dump.json").write_text("{}", encoding="utf-8")

            validate_output_files(artifact / "official", ("dump.json",), 2)
            self.assertEqual(
                artifact / "official" / "step3",
                find_dump_compare_input(artifact, 3),
            )


class TestMindStudioOfficialAdapter(unittest.TestCase):
    def test_compare_command_honors_msprobe_environment_override(self) -> None:
        configured = str(Path("configured") / "msprobe")
        with patch.dict(os.environ, {"TORCHTITAN_MSPROBE": configured}):
            with patch.object(Path, "is_file", return_value=True):
                with patch("os.access", return_value=True):
                    command = compare_command(
                        target=Path("target"),
                        golden=Path("golden"),
                        output=Path("report"),
                    )
        self.assertEqual(str(Path(configured).resolve()), command[0])

    @patch(
        "tests.glm5_2_mindstudio.msprobe_adapter.resolve_tool_executable",
        return_value="/opt/mindstudio/bin/msprobe",
    )
    def test_compare_commands_preserve_official_cli_contract(self, _which) -> None:
        command = compare_command(
            target=Path("target/dump.json"),
            golden=Path("golden/dump.json"),
            output=Path("report"),
            fuzzy_match=False,
            data_mapping=Path("data-mapping.json"),
            cell_mapping=Path("cell-mapping.json"),
            diff_analysis=True,
        )
        self.assertEqual(
            [
                "/opt/mindstudio/bin/msprobe",
                "compare",
                "-tp",
                str(Path("target/dump.json")),
                "-gp",
                str(Path("golden/dump.json")),
                "-o",
                str(Path("report")),
                "-dm",
                str(Path("data-mapping.json")),
                "-cm",
                str(Path("cell-mapping.json")),
                "-da",
            ],
            command,
        )
        self.assertEqual(
            [
                "/opt/mindstudio/bin/msprobe",
                "config_check",
                "-c",
                str(Path("golden.zip")),
                str(Path("target.zip")),
                "-o",
                str(Path("report")),
            ],
            config_check_compare_command(
                golden_pack=Path("golden.zip"),
                target_pack=Path("target.zip"),
                output=Path("report"),
            ),
        )

    @patch(
        "tests.glm5_2_mindstudio.msprobe_adapter.resolve_tool_executable",
        return_value="/opt/mindstudio/bin/msprobe",
    )
    def test_graph_visualize_command_preserves_official_cli_contract(
        self,
        _which,
    ) -> None:
        self.assertEqual(
            [
                "/opt/mindstudio/bin/msprobe",
                "graph_visualize",
                "-tp",
                str(Path("target")),
                "-gp",
                str(Path("golden")),
                "-o",
                str(Path("output")),
                "-fm",
                "-oc",
                "-tensor_log",
                "-progress_log",
            ],
            graph_visualize_command(
                target=Path("target"),
                golden=Path("golden"),
                output=Path("output"),
                fuzzy_match=True,
                overflow_check=True,
                tensor_log=True,
                progress_log=True,
            ),
        )

        self.assertEqual(
            [
                "/opt/mindstudio/bin/msprobe",
                "graph_visualize",
                "-tp",
                str(Path("target")),
                "-o",
                str(Path("output")),
                "-lm",
                str(Path("mapping.yaml")),
            ],
            graph_visualize_command(
                target=Path("target"),
                golden=None,
                output=Path("output"),
                layer_mapping=Path("mapping.yaml"),
            ),
        )

    @patch(
        "tests.glm5_2_mindstudio.msprobe_adapter.resolve_tool_executable",
        return_value="/opt/mindstudio/bin/msprobe",
    )
    def test_trend_data2db_command_preserves_official_cli_contract(
        self,
        _which,
    ) -> None:
        self.assertEqual(
            [
                "/opt/mindstudio/bin/msprobe",
                "data2db",
                "--data",
                str(Path("dump")),
                "--db",
                str(Path("trend")),
                "--format",
                "dump",
                "--micro_step",
                "false",
                "--process_num",
                "2",
                "--mapping",
                str(Path("mapping.json")),
            ],
            trend_data2db_command(
                data=Path("dump"),
                output=Path("trend"),
                data_format="dump",
                mapping=Path("mapping.json"),
                micro_step=False,
                process_num=2,
            ),
        )

    def test_dump_tasks_follow_official_constraints(self) -> None:
        for task in ("statistics", "tensor", "structure"):
            config = MsProbeDumpConfig(task=task)  # type: ignore[arg-type]
            self.assertEqual(task, config.official_config(Path("dump"))["task"])
        nan = MsProbeDumpConfig(task="nan_check", level="L1")
        self.assertEqual("nan_check", nan.official_config(Path("dump"))["task"])
        with self.assertRaisesRegex(ValueError, "requires dump level L1"):
            MsProbeDumpConfig(task="nan_check", level="L0")
        with self.assertRaisesRegex(ValueError, "dump task must be"):
            MsProbeDumpConfig(task="overflow_check")  # type: ignore[arg-type]

    @patch(
        "tests.glm5_2_mindstudio.msprobe_adapter.resolve_tool_executable",
        return_value="/opt/mindstudio/bin/msprobe",
    )
    def test_overflow_check_command_preserves_official_cli_contract(
        self,
        _which,
    ) -> None:
        self.assertEqual(
            [
                "/opt/mindstudio/bin/msprobe",
                "overflow_check",
                "-i",
                str(Path("dump/step0")),
                "-o",
                str(Path("report/overflow")),
            ],
            overflow_check_command(
                input_path=Path("dump/step0"),
                output=Path("report/overflow"),
            ),
        )

    def test_compare_options_enforce_official_mode_constraints(self) -> None:
        config = _experiment()
        mapping = Path("mapping.yaml")
        topology = ParallelTopology("tp2", 2, tensor_parallel_degree=2)
        common = {
            "diff_analysis": False,
            "xlsx": False,
        }

        with self.assertRaisesRegex(ValueError, "incompatible"):
            _validate_compare_options(
                config,
                topology,
                fuzzy_match=True,
                data_mapping=mapping,
                cell_mapping=None,
                tensor_log=False,
                **common,
            )
        with self.assertRaisesRegex(ValueError, "rank-by-rank"):
            _validate_compare_options(
                config,
                topology,
                fuzzy_match=False,
                data_mapping=mapping,
                cell_mapping=None,
                tensor_log=False,
                **common,
            )
        with self.assertRaisesRegex(ValueError, "L0"):
            _validate_compare_options(
                replace(config, dump=replace(config.dump, level="L1")),
                config.candidate.topology,
                fuzzy_match=False,
                data_mapping=None,
                cell_mapping=mapping,
                tensor_log=False,
                **common,
            )
        with self.assertRaisesRegex(ValueError, "tensor dump"):
            _validate_compare_options(
                config,
                config.candidate.topology,
                fuzzy_match=False,
                data_mapping=None,
                cell_mapping=None,
                tensor_log=True,
                **common,
            )

        _validate_compare_options(
            replace(config, dump=replace(config.dump, ranks=(0,))),
            topology,
            fuzzy_match=False,
            data_mapping=mapping,
            cell_mapping=None,
            tensor_log=False,
            **common,
        )

    @patch(
        "tests.glm5_2_mindstudio.msprobe_adapter.resolve_tool_executable",
        return_value="/opt/mindstudio/bin/msprobe",
    )
    def test_precheck_commands_preserve_official_cli_contract(self, _which) -> None:
        command = precheck_command(
            api_info=Path("capture/step0/rank0/dump.json"),
            output=Path("report/precheck"),
            device_ids=(0, 1),
            num_splits=16,
            jit_compile=True,
            save_error_data=True,
            filter_api=True,
            config_path=Path("config.json"),
            resume_csv=Path("accuracy_checking_result_previous.csv"),
        )
        self.assertEqual(
            [
                "/opt/mindstudio/bin/msprobe",
                "multi_acc_check",
                "-api_info",
                str(Path("capture/step0/rank0/dump.json")),
                "-o",
                str(Path("report/precheck")),
                "-j",
                "-n",
                "16",
                "-d",
                "0",
                "1",
                "-save_error_data",
                "-f",
                "-config",
                str(Path("config.json")),
                "-csv_path",
                str(Path("accuracy_checking_result_previous.csv")),
            ],
            command,
        )
        self.assertEqual(
            [
                "/opt/mindstudio/bin/msprobe",
                "api_precision_compare",
                "-npu",
                str(Path("npu_details.csv")),
                "-gpu",
                str(Path("gpu_details.csv")),
                "-o",
                str(Path("compare")),
            ],
            precheck_compare_command(
                npu_details=Path("npu_details.csv"),
                gpu_details=Path("gpu_details.csv"),
                output=Path("compare"),
            ),
        )

    def test_official_statuses_are_aggregated_without_recomputing_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            with (output / "compare.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.writer(stream)
                writer.writerow(("name", "Result"))
                writer.writerow(("module.0", "PASS"))
                writer.writerow(("module.1", "warning"))
                writer.writerow(("module.2", "SKIP"))

            summary = summarize_official_results(output)

            self.assertEqual("warning", summary["verdict"])
            self.assertEqual(
                {"pass": 1, "skip": 1, "warning": 1},
                summary["status_counts"],
            )
            self.assertEqual(["compare.csv"], summary["parsed_csv"])
            stored = json.loads(
                (output / "official_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["status_counts"], stored["status_counts"])

    def test_all_official_status_columns_contribute_to_the_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            (output / "precheck.csv").write_text(
                "API name,Forward Test Success,Backward Test Success\n"
                "torch.matmul,pass,error\n",
                encoding="utf-8",
            )

            summary = summarize_official_results(output)

            self.assertEqual("error", summary["verdict"])
            self.assertEqual(
                {"error": 1, "pass": 1},
                summary["status_counts"],
            )

    def test_config_checker_pass_check_column_is_an_official_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            (output / "summary.csv").write_text(
                "filename,pass_check\nweights,pass\ndataset,error\n",
                encoding="utf-8",
            )

            summary = summarize_official_results(output)

            self.assertEqual("error", summary["verdict"])
            self.assertEqual(
                {"error": 1, "pass": 1},
                summary["status_counts"],
            )

    def test_unparsed_decisive_result_prevents_a_false_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            (output / "parsed.csv").write_text(
                "name,status\nmodule.0,pass\n",
                encoding="utf-8",
            )
            (output / "unknown.csv").write_text(
                "name,status\nmodule.1,NOT_EVALUATED\n",
                encoding="utf-8",
            )

            summary = summarize_official_results(output)

            self.assertEqual("unparsed", summary["verdict"])
            self.assertEqual({"pass": 1}, summary["status_counts"])
            self.assertEqual(
                ["parsed.csv"],
                summary["parsed_decisive_result_files"],
            )
            self.assertEqual(
                ["unknown.csv"],
                summary["unparsed_decisive_result_files"],
            )
            self.assertEqual(
                ["parsed.csv", "unknown.csv"],
                summary["decisive_result_files"],
            )

    def test_compile_csv_aggregation_preserves_rank_and_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rank_files: list[Path] = []
            for rank in (1, 0):
                path = root / f"precision_rank{rank}.csv"
                path.write_text(
                    "name,status\nblock.%d,PASS\n" % rank,
                    encoding="utf-8",
                )
                rank_files.append(path)

            combined = aggregate_compile_csv(
                rank_files,
                output=root / "combined.csv",
                scenario="tp2-inductor",
            )

            with combined.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(["rank", "scenario", "name", "status"], rows[0])
            self.assertEqual(["0", "tp2-inductor", "block.0", "PASS"], rows[1])
            self.assertEqual(["1", "tp2-inductor", "block.1", "PASS"], rows[2])

    def test_compile_report_rejects_loss_only_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            report = root / "precision_rank0_part0.csv"
            report.write_text(
                "module_name,check_type,tensor_index,status,max_abs_diff,"
                "mean_abs_diff,max_rel_diff,shape,note\n"
                "LOSS,loss,0,PASS,N/A,N/A,N/A,compiled=1.0,single_pass mode\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                MindStudioArtifactError, "module forward and backward"
            ):
                validate_compile_report(report)

            with report.open("a", encoding="utf-8") as stream:
                stream.write(
                    "layers.0,fwd_output,0,PASS,0,0,0,(1,),\n"
                    "layers.0,grad_input,0,PASS,0,0,0,(1,),\n"
                )
            coverage = validate_compile_report(report)
            self.assertEqual(1, coverage["forward_rows"])
            self.assertEqual(1, coverage["backward_rows"])
            self.assertEqual(["layers.0"], coverage["module_names"])


class TestMindStudioReport(unittest.TestCase):
    def test_baseline_uses_project_comparison_directory(self) -> None:
        report = Path("report")
        self.assertEqual(
            report / "comparison",
            _comparison_directory(report, "observation"),
        )
        self.assertEqual(
            report / "official_compare",
            _comparison_directory(report, "migration"),
        )

    def test_baseline_report_presents_observation_evidence_not_msprobe_verdict(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            compare = root / "mindstudio_reports" / "experiment" / "single"
            compare.mkdir(parents=True)
            observation = {
                "step_count": 500,
                "endpoints": {
                    "reference_label": "GPU",
                    "candidate_label": "NPU",
                },
                "observation": {
                    "diagnostic_symptom": "later-window-loss-difference",
                    "first_step": 0,
                    "last_step": 499,
                    "reference_first_nonfinite_metrics": {},
                    "candidate_first_nonfinite_metrics": {},
                },
                "loss": {
                    "mean_relative_error": 0.0125,
                    "first_step_above_threshold": 137,
                },
                "grad_norm": {
                    "mean_relative_error": 0.02,
                    "median_relative_error": 0.004,
                    "prominent_anomalies": [
                        {
                            "step": 22,
                            "reference": 0.779,
                            "candidate": 23.496,
                            "absolute_difference": 22.717,
                            "candidate_to_reference_ratio": 30.1617,
                            "relative_error": 29.1617,
                            "loss_relative_error": 0.001,
                            "previous": {
                                "step": 21,
                                "relative_error": 0.005,
                            },
                            "next": {
                                "step": 23,
                                "relative_error": 0.006,
                            },
                        }
                    ],
                },
            }
            write_json(compare / "summary.json", observation)
            write_json(
                compare / "official_summary.json",
                {
                    "verdict": "later-window-loss-difference",
                    "status_counts": {},
                    "training_observation": observation,
                },
            )
            for name in (
                "training_metrics_compare.csv",
                "runtime.log",
            ):
                (compare / name).write_text(name, encoding="utf-8")
            for name in (
                "loss.svg",
                "grad_norm.svg",
                "loss_relative_error.svg",
                "grad_norm_relative_error.svg",
            ):
                (compare / name).write_text(
                    f'<svg><text>{name}</text></svg>', encoding="utf-8"
                )
            (compare / "training_observation.html").write_text(
                "<!doctype html><p>standalone interactive report</p>",
                encoding="utf-8",
            )
            report_directory = root / "mindstudio_reports" / "experiment"
            topology_report = (
                report_directory
                / "html_reports"
                / "s500-training"
                / "training"
                / "single.html"
            )

            output = write_report_index(
                repository_root=root,
                report_directory=report_directory,
                experiment_name="experiment",
                workflow="observation",
                rows=(
                    {
                        "topology": "single",
                        "verdict": "later-window-loss-difference",
                        "status_counts": {},
                        "official_result": str(compare / "official_summary.json"),
                        "runtime_log": str(compare / "runtime.log"),
                        "download_report": str(topology_report),
                    },
                ),
                supplemental_report_patterns=(),
            )

            report = json.loads(
                (report_directory / "report.json").read_text(encoding="utf-8")
            )
            markdown = (report_directory / "README.md").read_text(
                encoding="utf-8"
            )
            page = output.read_text(encoding="utf-8")
            topology_page = topology_report.read_text(encoding="utf-8")
            self.assertIsNone(report["delivery_verdict"])
            self.assertEqual(
                "later-window-loss-difference",
                report["training_observations"][0]["diagnostic_symptom"],
            )
            self.assertIn("loss.svg", markdown)
            self.assertIn("html_reports/s500-training/training", markdown)
            self.assertIn("html_reports/s500-training/training", page)
            self.assertNotIn("training_metrics_compare.csv", page)
            self.assertIn("training_metrics_compare.csv", topology_page)
            self.assertIn("Per-step metric comparison", topology_page)
            self.assertIn("<svg><text>loss.svg</text></svg>", topology_page)
            self.assertIn("Interactive training observation", topology_page)
            self.assertIn('scrolling="no"', topology_page)
            self.assertIn("documentElement.scrollHeight", topology_page)
            self.assertNotIn("single evidence", topology_page)
            self.assertIn("自动提取异常事实", topology_page)
            self.assertIn("step 22", topology_page)
            self.assertIn("GPU=0.779", topology_page)
            self.assertIn("NPU=23.496", topology_page)
            self.assertIn("Static SVG fallbacks", topology_page)
            self.assertIn("runtime.log", topology_page)
            self.assertNotIn("Official msProbe", page)
            self.assertNotIn("unparsed", page)

    def test_report_resolves_supplemental_evidence_and_marks_missing_globs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            supplemental = root / "precision_reports" / "experiment"
            supplemental.mkdir(parents=True)
            report_file = supplemental / "single.html"
            report_file.write_text("report", encoding="utf-8")
            report_directory = root / "mindstudio_reports" / "official"

            output = write_report_index(
                repository_root=root,
                report_directory=report_directory,
                experiment_name="official",
                workflow="migration",
                rows=(),
                supplemental_report_patterns=(
                    "precision_reports/experiment",
                    "performance_reports/missing-*",
                ),
            )

            report = json.loads(
                (report_directory / "report.json").read_text(encoding="utf-8")
            )
            self.assertTrue(output.is_file())
            self.assertEqual("available", report["supplemental_reports"][0]["status"])
            self.assertEqual("missing", report["supplemental_reports"][1]["status"])
            page = output.read_text(encoding="utf-8")
            self.assertIn("single.html", page)
            self.assertIn("Not synchronized", page)

    def test_config_check_report_embeds_workbook_and_repository_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "torchtitan-test"
            compare = root / "mindstudio_reports" / "experiment" / "official_compare"
            rank = compare / "rank0"
            rank.mkdir(parents=True)
            workbook_path = rank / "result.xlsx"
            workbook_path.write_bytes(b"official workbook")
            summary_path = compare / "official_summary.json"
            summary_path.write_text("{}", encoding="utf-8")
            runtime_log = compare / "runtime.log"
            runtime_log.write_text("complete", encoding="utf-8")
            report_directory = root / "mindstudio_reports" / "experiment"

            class FakeSheet:
                title = "summary"

                @staticmethod
                def iter_rows(*, values_only: bool):
                    self.assertTrue(values_only)
                    return iter(
                        (
                            ("filename", "pass_check"),
                            ("weights", "pass"),
                            ("dataset", "error"),
                        )
                    )

            fake_workbook = SimpleNamespace(
                worksheets=[FakeSheet()],
                close=lambda: None,
            )
            fake_openpyxl = SimpleNamespace(
                load_workbook=lambda *args, **kwargs: fake_workbook
            )
            with patch.dict(sys.modules, {"openpyxl": fake_openpyxl}):
                output = write_report_index(
                    repository_root=root,
                    report_directory=report_directory,
                    experiment_name="experiment",
                    workflow="config-check",
                    rows=(
                        {
                            "topology": "single",
                            "verdict": "error",
                            "status_counts": {"error": 1, "pass": 1},
                            "official_result": str(summary_path),
                            "runtime_log": str(runtime_log),
                        },
                    ),
                    supplemental_report_patterns=(),
                )

            page = output.read_text(encoding="utf-8")
            markdown = (report_directory / "README.md").read_text(
                encoding="utf-8"
            )
            source = (
                "torchtitan-test/mindstudio_reports/experiment/"
                "official_compare/rank0/result.xlsx"
            )
            self.assertIn("Config Check details", page)
            self.assertIn(source, page)
            self.assertIn("weights", page)
            self.assertIn("dataset", page)
            self.assertIn("error", page)
            self.assertIn(source, markdown)

    def test_report_indexes_official_derived_diagnostics_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            report_directory = root / "mindstudio_reports" / "official"
            diagnostics = (
                report_directory
                / "single"
                / "precision_precheck"
                / "compare-r1"
                / "precheck_report.html",
                report_directory / "single" / "graph-visualize-r1" / "README.md",
                report_directory / "single" / "graph-visualize-r1" / "index.json",
                report_directory / "single" / "monitor_analysis" / "monitor_report.html",
            )
            for diagnostic in diagnostics:
                diagnostic.parent.mkdir(parents=True, exist_ok=True)
                diagnostic.write_text("diagnostic", encoding="utf-8")

            output = write_report_index(
                repository_root=root,
                report_directory=report_directory,
                experiment_name="official",
                workflow="monitor",
                rows=(),
                supplemental_report_patterns=(),
            )

            report = json.loads(
                (report_directory / "report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                {
                    "graph_visualization_guide",
                    "graph_visualization_index",
                    "precision_precheck",
                    "training_monitor",
                },
                {
                    entry["kind"]
                    for entry in report["official_diagnostics"]
                },
            )
            markdown = (report_directory / "README.md").read_text(
                encoding="utf-8"
            )
            page = output.read_text(encoding="utf-8")
            self.assertIn("## Monitor diagnostic analyses", markdown)
            self.assertIn("## Related diagnostic evidence", markdown)
            self.assertNotIn("Official verdict", markdown)
            self.assertIn("## Supplemental long-run evidence", markdown)
            self.assertIn("Supplemental long-run evidence", page)
            self.assertIn("precheck_report.html", page)
            self.assertIn("monitor_report.html", page)

    def test_monitor_reports_are_collected_in_one_download_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            experiment = root / "mindstudio_reports" / "monitor-experiment"
            report_directory = (
                experiment
                / "html_reports"
                / "s27-training"
                / "monitor"
                / "profile"
            )
            source = root / "scoped" / "monitor_report.html"
            source.parent.mkdir(parents=True)
            source.write_text("self-contained monitor", encoding="utf-8")
            runtime_log = root / "scoped" / "runtime.log"
            runtime_log.write_text("done", encoding="utf-8")
            collected = (
                experiment
                / "html_reports"
                / "s27-training"
                / "monitor"
                / "profile"
                / "fsdp8.html"
            ).resolve()
            collected.parent.mkdir(parents=True)
            collected.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

            output = write_report_index(
                repository_root=root,
                report_directory=report_directory,
                experiment_name="monitor-experiment",
                workflow="monitor",
                rows=({
                    "topology": "fsdp8",
                    "verdict": "diagnostic-only",
                    "status_counts": {},
                    "official_result": str(source),
                    "runtime_log": str(runtime_log),
                    "download_report": str(collected),
                },),
                supplemental_report_patterns=(),
            )

            self.assertEqual("self-contained monitor", collected.read_text())
            self.assertIn("fsdp8.html", output.read_text())


class TestMindStudioLifecycle(unittest.TestCase):
    def test_duplicate_capture_waits_for_existing_owner(self) -> None:
        run = Path("run")
        runtime_log = run / "runtime.log"
        with (
            patch(
                "tests.glm5_2_mindstudio.workflow.active_run_pid",
                side_effect=(321, 321, None),
            ),
            patch("tests.glm5_2_mindstudio.workflow.time.sleep") as sleep,
            patch(
                "tests.glm5_2_mindstudio.workflow.print_runtime_log"
            ) as print_log,
        ):
            waited = _wait_for_active_capture(
                run,
                runtime_log=runtime_log,
            )

        self.assertTrue(waited)
        sleep.assert_called_once_with(1)
        print_log.assert_called_once_with(runtime_log)

    def test_portable_compatibility_ignores_install_paths_not_package_code(self):
        from tests.glm5_2_mindstudio.workflow import _portable_capture_compatibility
        a = {"msprobe_executable_sha256": "a", "installed_msprobe": {
            "record": {"sha256": "a"}, "console_scripts": [{"sha256": "a"}],
            "package_tree": {"sha256": "code"},
        }}
        b = json.loads(json.dumps(a))
        b["msprobe_executable_sha256"] = "b"
        b["installed_msprobe"]["record"] = {"sha256": "b"}
        b["installed_msprobe"]["console_scripts"] = []
        normalize = lambda value: _portable_capture_compatibility(value, Path("."))
        self.assertEqual(normalize(a), normalize(b))
        b["installed_msprobe"]["package_tree"]["sha256"] = "different code"
        self.assertNotEqual(normalize(a), normalize(b))
        self.assertIn("record", a["installed_msprobe"])

    def test_portable_compatibility_uses_runtime_content_not_report_status(self):
        from tests.glm5_2_mindstudio.workflow import _portable_capture_compatibility
        a = {"project_sources": {"torchtitan_test": {
            "commit": "a" * 40, "status": "?? mindstudio_reports/",
            "dirty_tree_sha256": "a",
        }}}
        b = {"project_sources": {"torchtitan_test": {
            "commit": "b" * 40, "status": " M tests/glm5_2_performance/explorations/reports/x.json",
            "dirty_tree_sha256": "b",
        }}}
        with patch("tests.glm5_2_mindstudio.workflow.subprocess.run") as run:
            run.return_value.stdout = "100644 blob abc\ttests/glm5_2_mindstudio/capture_training.py\n"
            normalize = lambda value: _portable_capture_compatibility(value, Path("."))
            self.assertEqual(normalize(a), normalize(b))
            b["project_sources"]["torchtitan_test"]["status"] = " M tests/glm5_2_mindstudio/capture_training.py"
            self.assertNotEqual(normalize(a), normalize(b))

    def setUp(self) -> None:
        self._toolchain_patch = patch(
            "tests.glm5_2_mindstudio.workflow._accuracy_toolchain_metadata",
            return_value=_test_toolchain_metadata(),
        )
        self._execution_patch = patch(
            "tests.glm5_2_mindstudio.workflow._capture_execution_identity",
            return_value=_test_execution_identity(),
        )
        self._toolchain_patch.start()
        self._execution_patch.start()

    def tearDown(self) -> None:
        self._execution_patch.stop()
        self._toolchain_patch.stop()

    def test_data_cli_resolves_root_before_writing_overview(self) -> None:
        config = _experiment()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            script = root / "tests" / "glm5_2_mindstudio" / "benchmark.py"
            script.parent.mkdir(parents=True)
            with (
                patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "7"}, clear=True),
                patch.object(sys, "argv", [
                    str(script), "--data", "--topology", "single",
                    "--dump-task", "statistics", "--level", "mix",
                    "--dump-steps", "0,1", "--repeat", "1",
                ]),
                patch(
                    "tests.glm5_2_mindstudio.workflow.prepare_shared_fixture",
                    return_value=root / "fixture",
                ) as prepare,
                redirect_stdout(io.StringIO()),
            ):
                run_mindstudio_cli(config, str(script))
            self.assertEqual(prepare.call_args.args[0], root)
            overviews = list((root / config.run_root).glob("*/experiment.json"))
            self.assertEqual(1, len(overviews))
            overview = overviews[0].parent
            self.assertTrue((overview / "README.md").is_file())
            self.assertTrue((overview / "experiment.json").is_file())

    def test_force_dry_run_cli_actions_are_read_only(self) -> None:
        config = _experiment()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            script = root / "tests" / "glm5_2_mindstudio" / "benchmark.py"
            script.parent.mkdir(parents=True)

            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "benchmark.py",
                        "--data",
                        "--data-device",
                        "cuda",
                        "--force",
                        "--dry-run",
                    ],
                ),
                patch(
                    "tests.glm5_2_mindstudio.workflow.reset_selected_outputs"
                ) as reset,
                patch(
                    "tests.glm5_2_mindstudio.workflow.prepare_shared_fixture"
                ) as prepare,
                redirect_stdout(io.StringIO()),
            ):
                run_mindstudio_cli(config, str(script))
            reset.assert_not_called()
            prepare.assert_not_called()

            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "benchmark.py",
                        "--capture",
                        "candidate",
                        "--force",
                        "--dry-run",
                    ],
                ),
                patch(
                    "tests.glm5_2_mindstudio.workflow.reset_selected_outputs"
                ) as reset,
                patch(
                    "tests.glm5_2_mindstudio.workflow.capture_official",
                    return_value=root / "planned-artifact",
                ) as capture,
                redirect_stdout(io.StringIO()),
            ):
                run_mindstudio_cli(config, str(script))
            reset.assert_not_called()
            self.assertTrue(capture.call_args.kwargs["dry_run"])

            planned_row = {
                "topology": "single",
                "verdict": "planned",
                "status_counts": {},
                "official_files": [],
                "official_result": str(root / "planned.json"),
                "runtime_log": str(root / "planned.log"),
            }
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "benchmark.py",
                        "--compare",
                        "--force",
                        "--dry-run",
                    ],
                ),
                patch(
                    "tests.glm5_2_mindstudio.workflow.reset_compare_outputs"
                ) as reset,
                patch(
                    "tests.glm5_2_mindstudio.workflow.compare_official",
                    return_value=planned_row,
                ) as compare,
                patch(
                    "tests.glm5_2_mindstudio.workflow.write_report_index"
                ) as report,
                redirect_stdout(io.StringIO()),
            ):
                run_mindstudio_cli(config, str(script))
            reset.assert_not_called()
            report.assert_not_called()
            self.assertTrue(compare.call_args.kwargs["dry_run"])

    def test_compile_input_selection_excludes_virtual_stage_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact = Path(temporary_directory)
            official = artifact / "official"
            official.mkdir()
            merged = official / "precision_rank0.csv"
            part = official / "precision_rank0_part0.csv"
            merged.touch()
            part.touch()

            self.assertEqual([merged], _compile_rank_csv_files(artifact, 1))

    def test_compile_output_validation_requires_nonzero_policy_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            official = Path(temporary_directory)
            part = official / "precision_rank0_part0.csv"
            part.write_text(
                "module_name,check_type,tensor_index,status,max_abs_diff,"
                "mean_abs_diff,max_rel_diff,shape,note\n"
                "LOSS,loss,0,PASS,N/A,N/A,N/A,compiled=1.0,single_pass mode\n"
                "layers.0,fwd_output,0,PASS,0,0,0,(1,),\n"
                "layers.0,grad_input,0,PASS,0,0,0,(1,),\n",
                encoding="utf-8",
            )
            merged = official / "precision_rank0.csv"
            merged.write_bytes(part.read_bytes())
            decisive_rows = validate_compile_report(part)
            coverage_path = official / "compile_coverage_rank0.json"
            coverage = {
                "schema": "torchtitan.glm5_2.msprobe_compile_coverage",
                "schema_version": 1,
                "rank": 0,
                "policy": {"name": "glm5-block", "children_depth": 1},
                "parts": [
                    {
                        "part": 0,
                        "expected_glm_block_count": 1,
                        "expected_glm_block_names": ["layers.0"],
                        "report": part.name,
                        "decisive_rows": decisive_rows,
                    }
                ],
            }
            write_json(coverage_path, coverage)

            _validate_compile_outputs(
                official,
                world_size=1,
                policy="glm5-block",
                children_depth=1,
            )
            coverage["parts"][0]["expected_glm_block_count"] = 0
            coverage["parts"][0]["expected_glm_block_names"] = []
            write_json(coverage_path, coverage)
            with self.assertRaisesRegex(
                MindStudioArtifactError, "coverage must be nonzero"
            ):
                _validate_compile_outputs(
                    official,
                    world_size=1,
                    policy="glm5-block",
                    children_depth=1,
                )

    def test_launch_contract_uses_posix_shell_quoting(self) -> None:
        endpoint = _experiment().candidate
        command = ("python", "argument with space", "$(touch should-not-run)")
        environment = {
            endpoint.visible_devices_env: endpoint.visible_devices,
            "TORCHTITAN_DEVICE": endpoint.torchtitan_device,
            "LOG_RANK": "0",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "launch.sh"
            _write_launch_contract(
                path,
                command=command,
                environment=environment,
                endpoint=endpoint,
            )
            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(shlex.join(command), lines[-1])

    def test_compare_toolchain_identity_tracks_msprobe_binary_and_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            executable = Path(temporary_directory) / "msprobe"
            executable.write_text("version-one", encoding="utf-8")
            metadata = {
                "profile_name": "mindstudio-standard",
                "lock": {"sha256": "lock-a"},
                "versions": {"msprobe": "1.2.3"},
                "sources": {
                    "msprobe": {
                        "official_url": "https://example.invalid/msprobe.git",
                        "selected_ref": "master",
                        "locked_commit": None,
                        "locked_version": None,
                        "commit": "commit-a",
                        "branch": "master",
                        "tag": None,
                        "dirty": False,
                    }
                },
                "doctor": {
                    "checks": {
                        "msprobe": {
                            "executable": {"path": str(executable)},
                            "import": {"origin": str(executable)},
                        }
                    }
                },
            }
            first = _comparison_toolchain_identity(metadata)
            executable.write_text("version-two", encoding="utf-8")
            second = _comparison_toolchain_identity(metadata)

        self.assertEqual("lock-a", first["lock_sha256"])
        self.assertEqual("commit-a", first["msprobe_source"]["commit"])
        self.assertNotEqual(
            config_digest(first),
            config_digest(second),
        )

    def test_compare_cache_binds_every_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            result = output / "compare.csv"
            result.write_text("name,status\nblock,PASS\n", encoding="utf-8")
            write_json(output / "official_summary.json", {"verdict": "pass"})
            state = output / "comparison_state.json"
            write_json(
                state,
                {
                    "comparison_digest": "comparison-a",
                    "output_files": _comparison_output_index(output),
                },
            )
            self.assertTrue(_comparison_is_current(state, "comparison-a"))

            result.write_text("name,status\nblock,ERROR\n", encoding="utf-8")
            self.assertFalse(_comparison_is_current(state, "comparison-a"))

    def test_migration_compares_every_selected_dump_step(self) -> None:
        config = _experiment()
        topology = config.candidate.topology
        metadata = _test_toolchain_metadata()

        def emit_compare_csv(command, *, log_path, **_kwargs) -> None:
            output = Path(command[command.index("-o") + 1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "compare.csv").write_text(
                "name,status\nblock,PASS\n", encoding="utf-8"
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("completed\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_fixture_and_capture(root, config, "reference")
            _write_fixture_and_capture(root, config, "candidate")
            with (
                patch(
                    "tests.glm5_2_mindstudio.workflow._accuracy_toolchain_metadata",
                    return_value=metadata,
                ),
                patch(
                    "tests.glm5_2_mindstudio.msprobe_adapter.resolve_tool_executable",
                    return_value="/opt/mindstudio/bin/msprobe",
                ),
                patch(
                    "tests.glm5_2_mindstudio.workflow._run_process",
                    side_effect=emit_compare_csv,
                ) as run,
            ):
                result = compare_official(
                    root,
                    config,
                    topology=topology,
                    repeat=1,
                    fuzzy_match=False,
                    data_mapping=None,
                    cell_mapping=None,
                    diff_analysis=False,
                    tensor_log=False,
                    xlsx=False,
                    force=False,
                )

            _, _, report = _paths(root, config, topology, "reference", 1)
            compare = report / "official_compare"
            state = json.loads(
                (compare / "comparison_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(2, run.call_count)
            self.assertTrue((compare / "step0" / "invocation.json").is_file())
            self.assertTrue((compare / "step1" / "invocation.json").is_file())
            self.assertEqual({"pass": 2}, result["status_counts"])
            self.assertEqual(metadata, state["toolchain"])

    @patch("tests.glm5_2_mindstudio.workflow._fixture_manifest")
    @patch("tests.glm5_2_mindstudio.workflow.resolve_fixture_inputs")
    def test_capture_dry_run_is_read_only_and_redacts_endpoint_values(
        self, resolve_inputs, fixture_manifest
    ) -> None:
        config = _experiment()
        secret = "do-not-print-this-secret"
        config = replace(
            config,
            candidate=replace(
                config.candidate,
                environment={"PRIVATE_TOKEN": secret},
            ),
        )
        fixture_manifest.return_value = {
            "generation_id": "fixture-a",
            "training": asdict(config.training),
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            resolve_inputs.return_value = (
                root / "fixture" / "checkpoint",
                root / "fixture" / "tokens.json",
            )
            stream = io.StringIO()
            with redirect_stdout(stream):
                artifact = capture_official(
                    root,
                    config,
                    topology=config.candidate.topology,
                    role="candidate",
                    repeat=1,
                    dry_run=True,
                )

            run, expected_artifact, report = _paths(
                root, config, config.candidate.topology, "candidate", 1
            )
            output = stream.getvalue()
            plan = json.loads(output)
            self.assertEqual(expected_artifact, artifact)
            self.assertFalse(run.exists())
            self.assertFalse(expected_artifact.exists())
            self.assertFalse(report.exists())
            self.assertNotIn(secret, output)
            self.assertIn("PRIVATE_TOKEN", plan["endpoint_environment_keys"])
            self.assertEqual(
                run / "training_metrics.jsonl",
                Path(
                    plan["environment_overrides"][
                        "GLM5_MINDSTUDIO_METRICS_PATH"
                    ]
                ),
            )
            self.assertTrue(
                any(
                    value.endswith("-candidate-r1")
                    for value in plan["command"]
                    if value.startswith("--rdzv_id=")
                )
            )

    @patch("tests.glm5_2_mindstudio.workflow._fixture_manifest")
    @patch("tests.glm5_2_mindstudio.workflow.resolve_fixture_inputs")
    def test_monitor_short_execution_preserves_parent_lr_schedule(
        self, resolve_inputs, fixture_manifest
    ) -> None:
        base = _experiment()
        config = replace(
            base,
            workflow="monitor",
            training=replace(base.training, steps=27),
        )
        fixture_training = asdict(config.training)
        fixture_training["steps"] = 500
        fixture_manifest.return_value = {
            "generation_id": "fixture-a",
            "training": fixture_training,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            resolve_inputs.return_value = (
                root / "fixture" / "checkpoint",
                root / "fixture" / "tokens.json",
            )
            stream = io.StringIO()
            with redirect_stdout(stream):
                capture_official(
                    root,
                    config,
                    topology=config.candidate.topology,
                    role="candidate",
                    repeat=1,
                    dry_run=True,
                )

            plan = json.loads(stream.getvalue())
            self.assertIn("--training.steps=27", plan["command"])
            self.assertIn("--lr_scheduler.total_steps=500", plan["command"])

    @patch("tests.glm5_2_mindstudio.workflow._fixture_manifest")
    @patch("tests.glm5_2_mindstudio.workflow.resolve_fixture_inputs")
    def test_dump_short_execution_preserves_parent_lr_schedule(
        self, resolve_inputs, fixture_manifest
    ) -> None:
        base = _experiment()
        config = replace(
            base,
            training=replace(base.training, steps=22),
            dump=replace(base.dump, steps=(21,)),
        )
        fixture_training = asdict(config.training)
        fixture_training["steps"] = 500
        fixture_manifest.return_value = {
            "generation_id": "fixture-a",
            "training": fixture_training,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            resolve_inputs.return_value = (
                root / "fixture" / "checkpoint",
                root / "fixture" / "tokens.json",
            )
            stream = io.StringIO()
            with redirect_stdout(stream):
                capture_official(
                    root,
                    config,
                    topology=config.candidate.topology,
                    role="candidate",
                    repeat=1,
                    dry_run=True,
                )

            plan = json.loads(stream.getvalue())
            self.assertIn("--training.steps=22", plan["command"])
            self.assertIn("--lr_scheduler.total_steps=500", plan["command"])

    @patch("tests.glm5_2_mindstudio.workflow._fixture_manifest")
    @patch("tests.glm5_2_mindstudio.workflow.resolve_fixture_inputs")
    def test_completed_capture_is_not_recollected_after_msprobe_build_changes(
        self,
        resolve_inputs,
        fixture_manifest,
    ) -> None:
        config = _experiment()
        topology = config.candidate.topology
        fixture_manifest.return_value = {
            "generation_id": "fixture-a",
            "checkpoint_sha256": "checkpoint-a",
            "token_plan": {"sha256": "tokens-a"},
            "training": asdict(config.training),
        }

        def emit_dump(_command, *, environment, log_path, **_kwargs) -> None:
            official = Path(environment["GLM5_MINDSTUDIO_OUTPUT"])
            self.assertTrue(official.is_dir())
            for step in config.dump.steps:
                dump = official / f"step{step}" / "rank0" / "dump.json"
                dump.parent.mkdir(parents=True, exist_ok=True)
                dump.write_text("{}\n", encoding="utf-8")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("completed\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            resolve_inputs.return_value = (
                root / "fixture" / "checkpoint",
                root / "fixture" / "tokens.json",
            )
            metadata_a = _test_toolchain_metadata()
            input_contract = {
                "valid": True,
                "validated_global_ranks": [0],
                "token_plan_step_series_sha256": "tokens-a",
            }
            with (
                patch(
                    "tests.glm5_2_mindstudio.workflow._accuracy_toolchain_metadata",
                    return_value=metadata_a,
                ),
                patch(
                    "tests.glm5_2_mindstudio.workflow._run_process",
                    side_effect=emit_dump,
                ) as run,
                patch(
                    "tests.glm5_2_mindstudio.workflow.load_token_plan",
                    return_value=object(),
                ) as load_plan,
                patch(
                    "tests.glm5_2_mindstudio.workflow.validate_runtime_input_contract",
                    return_value=input_contract,
                ) as validate_contract,
            ):
                artifact = capture_official(
                    root,
                    config,
                    topology=topology,
                    role="candidate",
                    repeat=1,
                )

            # A complete member must be reusable even if the current checkout
            # can no longer resolve or import the capture toolchain. The saved
            # identity describes the historical capture; it is not a trigger
            # for an implicit new generation.
            with patch(
                "tests.glm5_2_mindstudio.workflow._accuracy_toolchain_metadata",
                side_effect=AssertionError(
                    "completed capture probed current toolchain"
                ),
            ):
                capture_official(
                    root,
                    config,
                    topology=topology,
                    role="candidate",
                    repeat=1,
                )
                capture_official(
                    root,
                    config,
                    topology=topology,
                    role="candidate",
                    repeat=1,
                )

            manifest = json.loads(
                (artifact / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(1, run.call_count)
            self.assertEqual(1, load_plan.call_count)
            self.assertEqual(1, validate_contract.call_count)
            self.assertEqual(input_contract, manifest["input_contract"])
            self.assertEqual(
                "1.2.3",
                manifest["toolchain_compatibility"]["msprobe_version"],
            )

    def test_compare_allows_manually_managed_capture_toolchains(self) -> None:
        config = _experiment()
        topology = config.candidate.topology
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_fixture_and_capture(root, config, "reference")
            candidate = _write_fixture_and_capture(root, config, "candidate")
            manifest_path = candidate / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["toolchain_compatibility"]["msprobe_version"] = "9.9.9"
            write_json(manifest_path, manifest)

            with (
                redirect_stdout(io.StringIO()),
                patch("tests.glm5_2_mindstudio.msprobe_adapter._msprobe_executable", return_value="msprobe"),
            ):
                compare_official(
                    root,
                    config,
                    topology=topology,
                    repeat=1,
                    fuzzy_match=False,
                    data_mapping=None,
                    cell_mapping=None,
                    diff_analysis=False,
                    tensor_log=False,
                    xlsx=False,
                    force=False,
                    dry_run=True,
                )

    def test_compare_refuses_an_active_attempt(self) -> None:
        config = _experiment()
        topology = config.candidate.topology
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_fixture_and_capture(root, config, "reference")
            _write_fixture_and_capture(root, config, "candidate")
            _, _, report = _paths(root, config, topology, "reference", 1)
            compare = report / "official_compare"
            write_json(
                compare / "compare_run_state.json",
                {"status": "running", "pid": 987654},
            )
            with (
                patch(
                    "tests.glm5_2_mindstudio.workflow._accuracy_toolchain_metadata",
                    return_value=_test_toolchain_metadata(),
                ),
                patch(
                    "tests.glm5_2_common.cli.process_is_running",
                    return_value=True,
                ),
                self.assertRaisesRegex(RuntimeError, "still active"),
            ):
                compare_official(
                    root,
                    config,
                    topology=topology,
                    repeat=1,
                    fuzzy_match=False,
                    data_mapping=None,
                    cell_mapping=None,
                    diff_analysis=False,
                    tensor_log=False,
                    xlsx=False,
                    force=True,
                )

    @patch(
        "tests.glm5_2_mindstudio.msprobe_adapter.resolve_tool_executable",
        return_value="/opt/mindstudio/bin/msprobe",
    )
    def test_failed_compare_invalidates_only_its_workflow_index(self, _which) -> None:
        config = _experiment()
        topology = config.candidate.topology
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_fixture_and_capture(root, config, "reference")
            _write_fixture_and_capture(root, config, "candidate")
            aggregate = root / config.report_root / config.storage_name
            experiment_report = aggregate / f"{config.storage_name}.html"
            experiment_report.parent.mkdir(parents=True, exist_ok=True)
            experiment_report.write_text("preserve", encoding="utf-8")
            stale = (
                aggregate
                / "html_reports"
                / f"{config.storage_name}.html"
            )
            stale.parent.mkdir(parents=True, exist_ok=True)
            stale.write_text("stale", encoding="utf-8")

            with (
                patch(
                    "tests.glm5_2_mindstudio.workflow._run_process",
                    side_effect=RuntimeError("compare failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "compare failed"),
            ):
                compare_official(
                    root,
                    config,
                    topology=topology,
                    repeat=1,
                    fuzzy_match=False,
                    data_mapping=None,
                    cell_mapping=None,
                    diff_analysis=False,
                    tensor_log=False,
                    xlsx=False,
                    force=False,
                )

            self.assertFalse(stale.exists())
            self.assertTrue(experiment_report.exists())

    @staticmethod
    def _emit_precheck_csv(command, **_kwargs) -> None:
        output = Path(command[command.index("-o") + 1])
        output.mkdir(parents=True, exist_ok=True)
        if command[1] in {"acc_check", "multi_acc_check"}:
            if "-csv_path" in command:
                result = Path(command[command.index("-csv_path") + 1])
                details = result.with_name(
                    result.name.replace(
                        "accuracy_checking_result_",
                        "accuracy_checking_details_",
                        1,
                    )
                )
                result.write_text(
                    "API name,Forward Test Success,Backward Test Success\n"
                    "torch.matmul,pass,pass\n",
                    encoding="utf-8",
                )
                details.write_text(
                    "API name,Status\ntorch.matmul,pass\n",
                    encoding="utf-8",
                )
                return
            device_arguments = []
            if command[1] == "multi_acc_check":
                for value in command[command.index("-d") + 1 :]:
                    if value.startswith("-"):
                        break
                    device_arguments.append(value)
            destinations = (
                (output / "device0", output / "device1")
                if len(device_arguments) > 1
                else (output,)
            )
            for index, destination in enumerate(destinations):
                destination.mkdir(parents=True, exist_ok=True)
                result_path = (
                    destination
                    / f"accuracy_checking_result_20260101{index}.csv"
                )
                result_path.write_text(
                    "API name,Forward Test Success,Backward Test Success\n"
                    "torch.matmul,pass,pass\n",
                    encoding="utf-8",
                )
                details_path = (
                    destination
                    / f"accuracy_checking_details_20260101{index}.csv"
                )
                details_path.write_text(
                    "API name,Status\ntorch.matmul,pass\n",
                    encoding="utf-8",
                )
        elif command[1] == "api_precision_compare":
            (output / "api_precision_compare_result_20260101.csv").write_text(
                "API name,Forward Test Success,Backward Test Success\n"
                "torch.matmul,pass,pass\n",
                encoding="utf-8",
            )
            (output / "api_precision_compare_details_20260101.csv").write_text(
                "API name,Status\ntorch.matmul,pass\n",
                encoding="utf-8",
            )
        else:
            raise AssertionError(f"unexpected command: {command}")

    @patch(
        "tests.glm5_2_mindstudio.msprobe_adapter.resolve_tool_executable",
        return_value="/opt/mindstudio/bin/msprobe",
    )
    def test_precheck_is_generation_safe_and_stays_out_of_capture_artifact(
        self,
        _which,
    ) -> None:
        base = _experiment()
        config = replace(base, dump=replace(base.dump, level="L1"))
        topology = config.candidate.topology
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = _write_fixture_and_capture(root, config, "reference")
            with patch(
                "tests.glm5_2_mindstudio.workflow._run_process",
                side_effect=self._emit_precheck_csv,
            ) as run:
                report = run_precheck(
                    root,
                    config,
                    topology=topology,
                    role="reference",
                    repeat=1,
                    device_ids=(0,),
                    num_splits=None,
                    jit_compile=False,
                    save_error_data=False,
                    filter_api=False,
                    config_path=None,
                )
                run_precheck(
                    root,
                    config,
                    topology=topology,
                    role="reference",
                    repeat=1,
                    device_ids=(0,),
                    num_splits=None,
                    jit_compile=False,
                    save_error_data=False,
                    filter_api=False,
                    config_path=None,
                )

            self.assertEqual(2, run.call_count)
            self.assertTrue((report / "step0" / "rank0" / "complete.json").is_file())
            self.assertTrue((report / "step1" / "rank0" / "complete.json").is_file())
            artifact_parts = Path(config.artifact_root).parts
            self.assertTrue(
                any(
                    report.parts[index : index + len(artifact_parts)]
                    == artifact_parts
                    for index in range(len(report.parts) - len(artifact_parts) + 1)
                )
            )
            self.assertFalse((artifact / "precision_precheck").exists())

            reset_precheck_outputs(
                root,
                config,
                topologies=(topology,),
                owner="reference",
                repeat=1,
            )
            self.assertFalse(report.exists())
            self.assertTrue(artifact.exists())

    @patch(
        "tests.glm5_2_mindstudio.msprobe_adapter.resolve_tool_executable",
        return_value="/opt/mindstudio/bin/msprobe",
    )
    def test_precheck_rerun_invalidates_dependent_report(self, _which) -> None:
        base = _experiment()
        config = replace(
            base,
            dump=replace(base.dump, level="L1", steps=(0,)),
        )
        topology = config.candidate.topology
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_fixture_and_capture(root, config, "reference")
            with patch(
                "tests.glm5_2_mindstudio.workflow._run_process",
                side_effect=self._emit_precheck_csv,
            ):
                precheck = run_precheck(
                    root,
                    config,
                    topology=topology,
                    role="reference",
                    repeat=1,
                    device_ids=(0,),
                    num_splits=None,
                    jit_compile=False,
                    save_error_data=False,
                    filter_api=False,
                    config_path=None,
                )
                dependent = (
                    root
                    / config.report_root
                    / config.storage_name
                    / topology.slug
                    / "precision_precheck"
                    / "compare-r1"
                    / "precheck_report.html"
                )
                dependent.parent.mkdir(parents=True, exist_ok=True)
                dependent.write_text("stale", encoding="utf-8")
                (precheck / "step0" / "rank0" / "complete.json").unlink()
                run_precheck(
                    root,
                    config,
                    topology=topology,
                    role="reference",
                    repeat=1,
                    device_ids=(0,),
                    num_splits=None,
                    jit_compile=False,
                    save_error_data=False,
                    filter_api=False,
                    config_path=None,
                )

            self.assertFalse(dependent.exists())

    @patch(
        "tests.glm5_2_mindstudio.msprobe_adapter.resolve_tool_executable",
        return_value="/opt/mindstudio/bin/msprobe",
    )
    def test_precheck_dry_run_does_not_replace_stale_output(self, _which) -> None:
        base = _experiment()
        config = replace(
            base,
            dump=replace(base.dump, level="L1", steps=(0,)),
        )
        topology = config.candidate.topology
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_fixture_and_capture(root, config, "reference")
            precheck = (
                root
                / config.artifact_root
                / config.storage_name
                / topology.slug
                / "precision_precheck"
                / "reference-r1"
            )
            member = precheck / "step0" / "rank0"
            member.mkdir(parents=True)
            evidence = member / "old-evidence.txt"
            evidence.write_text("keep", encoding="utf-8")
            with (
                patch(
                    "tests.glm5_2_mindstudio.workflow._run_process"
                ) as run,
                patch(
                    "tests.glm5_2_mindstudio.workflow._accuracy_toolchain_metadata",
                    return_value={"versions": {"msprobe": "26.1.0"}},
                ),
            ):
                result = run_precheck(
                    root,
                    config,
                    topology=topology,
                    role="reference",
                    repeat=1,
                    device_ids=(0,),
                    num_splits=None,
                    jit_compile=False,
                    save_error_data=False,
                    filter_api=False,
                    config_path=None,
                    dry_run=True,
                )

            run.assert_not_called()
            self.assertEqual(precheck, result)
            self.assertEqual("keep", evidence.read_text(encoding="utf-8"))
            self.assertFalse((member / "invocation.json").exists())

    @patch(
        "tests.glm5_2_mindstudio.msprobe_adapter.resolve_tool_executable",
        return_value="/opt/mindstudio/bin/msprobe",
    )
    def test_precheck_compare_maps_roles_by_endpoint_device_type(
        self,
        _which,
    ) -> None:
        base = _experiment(reference_device="npu", candidate_device="cuda")
        config = replace(base, dump=replace(base.dump, level="L1"))
        topology = config.candidate.topology
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_fixture_and_capture(root, config, "reference")
            _write_fixture_and_capture(root, config, "candidate")
            with patch(
                "tests.glm5_2_mindstudio.workflow._run_process",
                side_effect=self._emit_precheck_csv,
            ):
                for role in ("reference", "candidate"):
                    run_precheck(
                        root,
                        config,
                        topology=topology,
                        role=role,
                        repeat=1,
                        device_ids=(0,),
                        num_splits=None,
                        jit_compile=False,
                        save_error_data=False,
                        filter_api=False,
                        config_path=None,
                    )
                report = compare_prechecks(
                    root,
                    config,
                    topology=topology,
                    repeat=1,
                )

            invocation = json.loads(
                (report / "step0" / "rank0" / "invocation.json").read_text(
                    encoding="utf-8"
                )
            )["command"]
            report_parts = Path(config.report_root).parts
            self.assertTrue(
                any(
                    report.parts[index : index + len(report_parts)]
                    == report_parts
                    for index in range(len(report.parts) - len(report_parts) + 1)
                )
            )
            npu_path = invocation[invocation.index("-npu") + 1]
            gpu_path = invocation[invocation.index("-gpu") + 1]
            self.assertIn("reference-r1", npu_path)
            self.assertIn("candidate-r1", gpu_path)
            self.assertTrue(
                (report / "step1" / "rank0" / "complete.json").is_file()
            )
            html = (report / "precheck_report.html").read_text(encoding="utf-8")
            self.assertIn("does <strong>not</strong> compare", html)
            self.assertIn("api_precision_compare", html)
            index = json.loads(
                (report / "precheck_index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(2, len(index["entries"]))
            self.assertIn(
                "does not compare the two complete training processes directly",
                index["meaning"],
            )

    @patch(
        "tests.glm5_2_mindstudio.msprobe_adapter.resolve_tool_executable",
        return_value="/opt/mindstudio/bin/msprobe",
    )
    def test_multi_device_precheck_is_not_implicitly_paired(self, _which) -> None:
        base = _experiment()
        config = replace(
            base,
            dump=replace(base.dump, level="L1", steps=(0,)),
        )
        topology = config.candidate.topology
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_fixture_and_capture(root, config, "reference")
            _write_fixture_and_capture(root, config, "candidate")
            with patch(
                "tests.glm5_2_mindstudio.workflow._run_process",
                side_effect=self._emit_precheck_csv,
            ):
                run_precheck(
                    root,
                    config,
                    topology=topology,
                    role="reference",
                    repeat=1,
                    device_ids=(0, 1),
                    num_splits=8,
                    jit_compile=False,
                    save_error_data=False,
                    filter_api=False,
                    config_path=None,
                )
                run_precheck(
                    root,
                    config,
                    topology=topology,
                    role="candidate",
                    repeat=1,
                    device_ids=(0,),
                    num_splits=None,
                    jit_compile=False,
                    save_error_data=False,
                    filter_api=False,
                    config_path=None,
                )

            with self.assertRaisesRegex(
                RuntimeError,
                "requires exactly one accuracy_checking_details CSV",
            ):
                compare_prechecks(
                    root,
                    config,
                    topology=topology,
                    repeat=1,
                )

    @patch(
        "tests.glm5_2_mindstudio.msprobe_adapter.resolve_tool_executable",
        return_value="/opt/mindstudio/bin/msprobe",
    )
    def test_precheck_resume_is_staged_and_bound_to_one_member(self, _which) -> None:
        base = _experiment()
        config = replace(
            base,
            dump=replace(base.dump, level="L1", steps=(0,)),
        )
        topology = config.candidate.topology
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_fixture_and_capture(root, config, "reference")
            with patch(
                "tests.glm5_2_mindstudio.workflow._run_process",
                side_effect=self._emit_precheck_csv,
            ) as run:
                report = run_precheck(
                    root,
                    config,
                    topology=topology,
                    role="reference",
                    repeat=1,
                    device_ids=(0,),
                    num_splits=None,
                    jit_compile=False,
                    save_error_data=False,
                    filter_api=False,
                    config_path=None,
                )
                member = report / "step0" / "rank0"
                resume = next(
                    (member / "official").rglob(
                        "accuracy_checking_result_*.csv"
                    )
                )
                details = resume.with_name(
                    resume.name.replace(
                        "accuracy_checking_result_",
                        "accuracy_checking_details_",
                        1,
                    )
                )
                original_result = resume.read_text(encoding="utf-8")
                original_details = details.read_text(encoding="utf-8")
                run_precheck(
                    root,
                    config,
                    topology=topology,
                    role="reference",
                    repeat=1,
                    device_ids=(0,),
                    num_splits=None,
                    jit_compile=False,
                    save_error_data=False,
                    filter_api=False,
                    config_path=None,
                    resume_csv=resume,
                )

            self.assertEqual(2, run.call_count)
            self.assertEqual(original_result, resume.read_text(encoding="utf-8"))
            self.assertEqual(original_details, details.read_text(encoding="utf-8"))
            invocation = json.loads(
                (member / "invocation.json").read_text(encoding="utf-8")
            )["command"]
            staged = Path(invocation[invocation.index("-csv_path") + 1])
            self.assertEqual(member / "official" / resume.name, staged)
            manifest = json.loads(
                (member / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                resume.name,
                manifest["identity"]["options"]["resume"]["result"]["name"],
            )

        multi_member = replace(
            config,
            dump=replace(config.dump, steps=(0, 1)),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_fixture_and_capture(root, multi_member, "reference")
            resume = root / "accuracy_checking_result_previous.csv"
            details = root / "accuracy_checking_details_previous.csv"
            resume.write_text("result", encoding="utf-8")
            details.write_text("details", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "exactly one step/rank member",
            ):
                run_precheck(
                    root,
                    multi_member,
                    topology=topology,
                    role="reference",
                    repeat=1,
                    device_ids=(0,),
                    num_splits=None,
                    jit_compile=False,
                    save_error_data=False,
                    filter_api=False,
                    config_path=None,
                    resume_csv=resume,
                )

    @patch(
        "tests.glm5_2_mindstudio.msprobe_adapter.resolve_tool_executable",
        return_value="/opt/mindstudio/bin/msprobe",
    )
    def test_precheck_cache_binds_execution_toolchain_version(self, _which) -> None:
        base = _experiment()
        config = replace(
            base,
            dump=replace(base.dump, level="L1", steps=(0,)),
        )
        topology = config.candidate.topology
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_fixture_and_capture(root, config, "reference")
            with (
                patch(
                    "tests.glm5_2_mindstudio.workflow._run_process",
                    side_effect=self._emit_precheck_csv,
                ) as run,
                patch(
                    "tests.glm5_2_mindstudio.workflow._accuracy_toolchain_metadata",
                    side_effect=(
                        _test_toolchain_metadata(),
                        _test_toolchain_metadata(
                            version="26.1.0", commit="commit-b"
                        ),
                    ),
                ),
            ):
                report = run_precheck(
                    root,
                    config,
                    topology=topology,
                    role="reference",
                    repeat=1,
                    device_ids=(0,),
                    num_splits=None,
                    jit_compile=False,
                    save_error_data=False,
                    filter_api=False,
                    config_path=None,
                )
                with self.assertRaisesRegex(
                    MindStudioArtifactError, "does not match the selected capture"
                ):
                    run_precheck(
                        root,
                        config,
                        topology=topology,
                        role="reference",
                        repeat=1,
                        device_ids=(0,),
                        num_splits=None,
                        jit_compile=False,
                        save_error_data=False,
                        filter_api=False,
                        config_path=None,
                    )

            self.assertEqual(1, run.call_count)
            manifest = json.loads(
                (report / "step0" / "rank0" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                "1.2.3",
                manifest["toolchain"]["versions"]["msprobe"],
            )

    @patch(
        "tests.glm5_2_mindstudio.msprobe_adapter.resolve_tool_executable",
        return_value="/opt/mindstudio/bin/msprobe",
    )
    def test_precheck_compare_cache_binds_execution_toolchain_version(
        self,
        _which,
    ) -> None:
        base = _experiment()
        config = replace(
            base,
            dump=replace(base.dump, level="L1", steps=(0,)),
        )
        topology = config.candidate.topology
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_fixture_and_capture(root, config, "reference")
            _write_fixture_and_capture(root, config, "candidate")
            with (
                patch(
                    "tests.glm5_2_mindstudio.workflow._run_process",
                    side_effect=self._emit_precheck_csv,
                ),
                patch(
                    "tests.glm5_2_mindstudio.workflow._accuracy_toolchain_metadata",
                    return_value=_test_toolchain_metadata(),
                ),
            ):
                for role in ("reference", "candidate"):
                    run_precheck(
                        root,
                        config,
                        topology=topology,
                        role=role,
                        repeat=1,
                        device_ids=(0,),
                        num_splits=None,
                        jit_compile=False,
                        save_error_data=False,
                        filter_api=False,
                        config_path=None,
                    )
            with (
                patch(
                    "tests.glm5_2_mindstudio.workflow._run_process",
                    side_effect=self._emit_precheck_csv,
                ) as run,
                patch(
                    "tests.glm5_2_mindstudio.workflow._accuracy_toolchain_metadata",
                    side_effect=(
                        _test_toolchain_metadata(),
                        _test_toolchain_metadata(
                            version="26.1.0", commit="commit-b"
                        ),
                    ),
                ),
            ):
                report = compare_prechecks(
                    root,
                    config,
                    topology=topology,
                    repeat=1,
                )
                with self.assertRaisesRegex(
                    MindStudioArtifactError, "incompatible with the reference"
                ):
                    compare_prechecks(
                        root,
                        config,
                        topology=topology,
                        repeat=1,
                    )

            self.assertEqual(1, run.call_count)
            manifest = json.loads(
                (report / "step0" / "rank0" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                "1.2.3",
                manifest["toolchain"]["versions"]["msprobe"],
            )

    def test_force_reset_removes_only_selected_role_and_its_stale_archives(self) -> None:
        config = _experiment()
        topology = config.candidate.topology
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = _fixture_directory(root, config, topology)
            reference_run, reference_artifact, report = _paths(
                root, config, topology, "reference", 1
            )
            candidate_run, candidate_artifact, _ = _paths(
                root, config, topology, "candidate", 1
            )
            for path in (
                fixture,
                reference_run,
                reference_artifact,
                candidate_run,
                candidate_artifact,
                report,
            ):
                path.mkdir(parents=True)
                (path / "evidence.txt").write_text("old", encoding="utf-8")
            stale_archive = candidate_run.with_name(
                f".{candidate_run.name}.previous-20260101-000000"
            )
            stale_archive.mkdir()
            aggregate = root / config.report_root / config.storage_name
            topology_report = aggregate / f"{topology.slug}.html"
            topology_report.parent.mkdir(parents=True, exist_ok=True)
            topology_report.write_text("stale", encoding="utf-8")

            reset_selected_outputs(
                root,
                config,
                topologies=(topology,),
                role="candidate",
                include_fixture=False,
            )

            self.assertTrue(fixture.is_dir())
            self.assertTrue(reference_run.is_dir())
            self.assertTrue(reference_artifact.is_dir())
            self.assertFalse(candidate_run.exists())
            self.assertFalse(candidate_artifact.exists())
            self.assertFalse(report.exists())
            self.assertFalse(stale_archive.exists())
            self.assertTrue(topology_report.exists())

    @patch(
        "tests.glm5_2_common.cli.process_is_running",
        return_value=True,
    )
    def test_force_reset_refuses_to_delete_an_active_capture(self, _running) -> None:
        config = _experiment()
        topology = config.candidate.topology
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            candidate_run, _, _ = _paths(
                root, config, topology, "candidate", 1
            )
            candidate_run.mkdir(parents=True)
            state = {
                "status": "running",
                "pid": os.getpid() + 1000,
            }
            (candidate_run / "run_state.json").write_text(
                json.dumps(state), encoding="utf-8"
            )

            with self.assertRaisesRegex(RuntimeError, "still active"):
                reset_selected_outputs(
                    root,
                    config,
                    topologies=(topology,),
                    role="candidate",
                    include_fixture=False,
                )
            self.assertTrue(candidate_run.is_dir())

    @patch(
        "tests.glm5_2_common.cli.process_is_running",
        return_value=True,
    )
    def test_force_reset_refuses_to_delete_an_active_precheck(self, _running) -> None:
        config = _experiment()
        topology = config.candidate.topology
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            precheck = (
                root
                / config.artifact_root
                / config.storage_name
                / topology.slug
                / "precision_precheck"
                / "candidate-r1"
                / "step0"
                / "rank0"
            )
            precheck.mkdir(parents=True)
            write_json(
                precheck / "precheck_state.json",
                {
                    "status": "running",
                    "pid": os.getpid() + 1000,
                },
            )

            with self.assertRaisesRegex(RuntimeError, "still active"):
                reset_precheck_outputs(
                    root,
                    config,
                    topologies=(topology,),
                    owner="candidate",
                    repeat=1,
                )
            self.assertTrue(precheck.is_dir())

    def test_monitor_output_validation_accepts_timestamp_subdirectories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "official"
            csv_path = output / "rank_0" / "20260901_120000" / "weight_grad.csv"
            csv_path.parent.mkdir(parents=True)
            csv_path.write_text("step,norm\n0,1.0\n", encoding="utf-8")

            _validate_monitor_outputs(
                output,
                configured_ranks=(0,),
                world_size=1,
            )

            with self.assertRaisesRegex(RuntimeError, "rank CSV"):
                _validate_monitor_outputs(
                    output,
                    configured_ranks=(0, 1),
                    world_size=2,
                )

    def test_monitor_execution_limits_compiled_module_hooks_and_invalid_rank(
        self,
    ) -> None:
        config = _experiment(workflow="monitor")
        topology = config.candidate.topology
        with self.assertRaisesRegex(ValueError, "outside"):
            _validate_monitor_execution(
                replace(
                    config,
                    monitor=replace(config.monitor, ranks=(1,)),
                ),
                topology,
                config.candidate,
            )
        compiled_endpoint = replace(
            config.candidate,
            extra_args=("--compile.enable",),
        )
        _validate_monitor_execution(
            config,
            topology,
            compiled_endpoint,
        )
        with self.assertRaisesRegex(ValueError, "module Monitor hooks"):
            _validate_monitor_execution(
                replace(
                    config,
                    monitor=replace(config.monitor, module=True),
                ),
                topology,
                compiled_endpoint,
            )

    @staticmethod
    def _emit_graph_database(command, **kwargs) -> None:
        output = Path(command[command.index("-o") + 1])
        output.mkdir(parents=True, exist_ok=True)
        (output / "compare_20260901.vis.db").write_bytes(b"database")
        Path(kwargs["log_path"]).write_text("graph complete\n", encoding="utf-8")

    @patch(
        "tests.glm5_2_mindstudio.msprobe_adapter.resolve_tool_executable",
        return_value="/opt/mindstudio/bin/msprobe",
    )
    def test_graph_visualization_is_generation_safe_and_indexed(
        self,
        _which,
    ) -> None:
        config = replace(
            _experiment(),
            dump=MsProbeDumpConfig(level="L0", steps=(0,)),
        )
        config = replace(
            config,
            experiment_storage_name="accuracy-experiment",
            output_subdirectory=(
                "s500-parent/dump/statistics-L0-statistics-profile"
            ),
            fixture_subdirectory="s500-parent/inputs",
        )
        topology = config.candidate.topology
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for role in ("reference", "candidate"):
                artifact = _write_fixture_and_capture(root, config, role)
                construct = artifact / "official" / "step0" / "rank0" / "construct.json"
                construct.write_text('{"Module.model": []}\n', encoding="utf-8")
                manifest_path = artifact / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["official_files"] = output_index(artifact / "official")
                write_json(manifest_path, manifest)

            with (
                patch(
                    "tests.glm5_2_mindstudio.visualization._toolchain",
                    return_value=(
                        {"versions": {"msprobe": "26.1.0"}},
                        {"msprobe": "26.1.0"},
                    ),
                ),
                patch(
                    "tests.glm5_2_mindstudio.workflow._run_process",
                    side_effect=self._emit_graph_database,
                ) as run,
            ):
                report = run_graph_visualization(
                    root,
                    config,
                    topology=topology,
                    repeat=1,
                    overflow_check=True,
                    progress_log=True,
                )
                same_report = run_graph_visualization(
                    root,
                    config,
                    topology=topology,
                    repeat=1,
                    overflow_check=True,
                    progress_log=True,
                )

            self.assertEqual(1, run.call_count)
            self.assertEqual(report, same_report)
            index = json.loads((report / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(1, len(index["databases"]))
            self.assertIn("tensorboard --logdir", index["tensorboard_command"])
            self.assertTrue((report / "README.md").is_file())
            artifact = (
                root
                / config.artifact_root
                / config.storage_name
                / topology.slug
                / config.operation_relative_root
                / "graph-visualize-r1"
            )
            self.assertTrue(
                (artifact / "official" / "compare_20260901.vis.db").is_file()
            )
            self.assertTrue((artifact / "complete.json").is_file())


if __name__ == "__main__":
    unittest.main()
