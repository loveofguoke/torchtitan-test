# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import tarfile
import tempfile
import unittest
from pathlib import Path

from release_artifacts import (
    EXPERIMENT_ROOTS,
    create_archive,
    find_experiment_paths,
)


class TestReleaseArtifacts(unittest.TestCase):
    def test_release_roots_cover_every_experiment_family(self) -> None:
        expected = {
            "parity_fixtures",
            "parity_artifacts",
            "parity_reports",
            "precision_fixtures",
            "precision_artifacts",
            "precision_runs",
            "precision_reports",
            "performance_runs",
            "performance_artifacts",
            "performance_reports",
            "performance_dynamic",
            "stability_fixtures",
            "stability_artifacts",
            "stability_runs",
            "stability_reports",
            "checkpoint_fixtures",
            "checkpoint_artifacts",
            "checkpoint_runs",
            "checkpoint_reports",
            "combination_artifacts",
            "combination_runs",
            "combination_reports",
            "combination_reports/precision",
            "graph_debug_runs",
            "smoke_runs",
            "train_runs",
        }

        self.assertEqual(expected, set(EXPERIMENT_ROOTS))

    def test_archive_collects_outputs_and_explicit_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            experiment = "npu-eager-inductor"
            fixture = "self-npu-bf16-random-s10-b8-seq32-seed61"
            artifact = root / "combination_artifacts" / experiment
            report = root / "combination_reports" / f"{experiment}.html"
            fixture_path = root / "precision_fixtures" / fixture
            artifact.mkdir(parents=True)
            report.parent.mkdir(parents=True)
            fixture_path.mkdir(parents=True)
            (artifact / "manifest.json").write_text("{}", encoding="utf-8")
            report.write_text("report", encoding="utf-8")
            (fixture_path / "fixture.json").write_text("{}", encoding="utf-8")
            archive_path = root / "release.tar.gz"

            create_archive(
                root,
                experiment,
                archive_path,
                include=(fixture,),
            )

            with tarfile.open(archive_path, "r:gz") as archive:
                names = set(archive.getnames())
            self.assertIn(
                f"combination_artifacts/{experiment}/manifest.json",
                names,
            )
            self.assertIn(f"combination_reports/{experiment}.html", names)
            self.assertIn(
                f"precision_fixtures/{fixture}/fixture.json",
                names,
            )

    def test_find_experiment_paths_supports_scoped_performance_outputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            experiment = "npu-tp8-bf16-r1-abcd1234"
            run = root / "performance_runs" / "8-card" / "tp8" / experiment
            artifact = (
                root
                / "performance_artifacts"
                / "8-card"
                / "tp8"
                / experiment
            )
            report = (
                root
                / "performance_reports"
                / "8-card"
                / "tp8"
                / f"{experiment}.html"
            )
            run.mkdir(parents=True)
            artifact.mkdir(parents=True)
            report.parent.mkdir(parents=True)
            report.write_text("report", encoding="utf-8")

            self.assertEqual(
                {run, artifact, report},
                set(find_experiment_paths(root, experiment)),
            )

    def test_find_does_not_descend_into_another_experiment_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            experiment = "npu-target"
            unrelated = root / "performance_runs" / "8-card" / "tp8" / "other"
            fake_nested = unrelated / "profiler" / experiment
            unrelated.mkdir(parents=True)
            fake_nested.mkdir(parents=True)
            (unrelated / "manifest.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                find_experiment_paths(root, experiment)

    def test_archive_collects_scoped_performance_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            experiment = "npu-fsdp8-bf16-r1-abcd1234"
            run = root / "performance_runs" / "8-card" / "fsdp8" / experiment
            artifact = (
                root
                / "performance_artifacts"
                / "8-card"
                / "fsdp8"
                / experiment
            )
            report = (
                root
                / "performance_reports"
                / "8-card"
                / "fsdp8"
                / f"{experiment}.html"
            )
            run.mkdir(parents=True)
            artifact.mkdir(parents=True)
            report.parent.mkdir(parents=True)
            (run / "runtime.log").write_text("run", encoding="utf-8")
            (artifact / "manifest.json").write_text("{}", encoding="utf-8")
            report.write_text("report", encoding="utf-8")
            archive_path = root / "release.tar.gz"

            create_archive(root, experiment, archive_path)

            with tarfile.open(archive_path, "r:gz") as archive:
                names = set(archive.getnames())
            self.assertIn(
                f"performance_runs/8-card/fsdp8/{experiment}/runtime.log",
                names,
            )
            self.assertIn(
                f"performance_artifacts/8-card/fsdp8/{experiment}/manifest.json",
                names,
            )
            self.assertIn(
                f"performance_reports/8-card/fsdp8/{experiment}.html",
                names,
            )


if __name__ == "__main__":
    unittest.main()
