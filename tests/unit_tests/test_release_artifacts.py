# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import tarfile
import tempfile
import unittest
from pathlib import Path

from release_artifacts import EXPERIMENT_ROOTS, create_archive


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


if __name__ == "__main__":
    unittest.main()
