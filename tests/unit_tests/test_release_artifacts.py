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
            visualization = (
                run
                / "graph_visualization"
                / "rank_0"
                / "torch_trace"
                / "dedicated_log_torch_trace.log"
            )
            visualization.parent.mkdir(parents=True)
            visualization.write_text("trace", encoding="utf-8")
            memory_timeline = (
                run
                / "trainer_output"
                / "profiling"
                / "traces"
                / "memory_timeline"
                / "rank_0_memory_timeline.html"
            )
            memory_timeline.parent.mkdir(parents=True)
            memory_timeline.write_text("memory", encoding="utf-8")
            official_flamegraph = (
                run
                / "trainer_output"
                / "profiling"
                / "traces"
                / "mindstudio_flamegraphs"
                / "rank_0"
                / "flamegraph.html"
            )
            official_flamegraph.parent.mkdir(parents=True)
            official_flamegraph.write_text("flamegraph", encoding="utf-8")
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
                "performance_runs/8-card/fsdp8/"
                f"{experiment}/graph_visualization/rank_0/torch_trace/"
                "dedicated_log_torch_trace.log",
                names,
            )
            self.assertIn(
                "performance_runs/8-card/fsdp8/"
                f"{experiment}/trainer_output/profiling/traces/"
                "memory_timeline/rank_0_memory_timeline.html",
                names,
            )
            self.assertIn(
                "performance_runs/8-card/fsdp8/"
                f"{experiment}/trainer_output/profiling/traces/"
                "mindstudio_flamegraphs/rank_0/flamegraph.html",
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

    def test_analysis_archive_keeps_visualizations_without_raw_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            experiment = "npu-tp8-bf16-r1-analysis"
            run = root / "performance_runs" / "8-card" / "tp8" / experiment
            artifact = (
                root / "performance_artifacts" / "8-card" / "tp8" / experiment
            )
            report = (
                root
                / "performance_reports"
                / "8-card"
                / "tp8"
                / f"{experiment}.html"
            )
            fixture = root / "precision_fixtures" / experiment

            parsed_root = (
                run
                / "trainer_output"
                / "profiling"
                / "traces"
                / "rank_0_ascend_pt"
                / "ASCEND_PROFILER_OUTPUT"
            )
            parsed_root.mkdir(parents=True)
            parsed = (
                run
                / "trainer_output"
                / "profiling"
                / "traces"
                / "rank_0_ascend_pt"
                / "ASCEND_PROFILER_OUTPUT"
                / "trace_view.json"
            )
            parsed.write_text("{}", encoding="utf-8")
            raw = (
                run
                / "trainer_output"
                / "profiling"
                / "traces"
                / "rank_0_ascend_pt"
                / "PROF_000001"
                / "device"
                / "raw.bin"
            )
            raw.parent.mkdir(parents=True)
            raw.write_bytes(b"raw")
            graph = run / "graph_visualization" / "tlparse" / "index.html"
            graph.parent.mkdir(parents=True)
            graph.write_text("graph", encoding="utf-8")
            flamegraph = (
                run
                / "trainer_output"
                / "profiling"
                / "traces"
                / "mindstudio_flamegraphs"
                / "rank_0"
                / "flamegraph.html"
            )
            flamegraph.parent.mkdir(parents=True)
            flamegraph.write_text("flame", encoding="utf-8")
            memory = (
                run
                / "trainer_output"
                / "profiling"
                / "traces"
                / "memory_timeline"
                / "rank_0_memory_timeline.html"
            )
            memory.parent.mkdir(parents=True)
            memory.write_text("memory", encoding="utf-8")
            tensorboard = (
                run / "trainer_output" / "tensorboard" / "events.out.tfevents.test"
            )
            tensorboard.parent.mkdir(parents=True)
            tensorboard.write_bytes(b"events")
            advisor = run / "advisor" / "analysis_output" / "advice.csv"
            advisor.parent.mkdir(parents=True)
            advisor.write_text("advice", encoding="utf-8")
            (run / "runtime.log").write_text("run", encoding="utf-8")
            artifact.mkdir(parents=True)
            (artifact / "analysis.json").write_text("{}", encoding="utf-8")
            report.parent.mkdir(parents=True)
            report.write_text("report", encoding="utf-8")
            fixture.mkdir(parents=True)
            (fixture / "checkpoint.distcp").write_bytes(b"checkpoint")

            archive_path = root / "analysis.tar.gz"
            create_archive(root, experiment, archive_path, content="analysis")

            with tarfile.open(archive_path, "r:gz") as archive:
                names = set(archive.getnames())
            self.assertIn(parsed.relative_to(root).as_posix(), names)
            self.assertIn(graph.relative_to(root).as_posix(), names)
            self.assertIn(flamegraph.relative_to(root).as_posix(), names)
            self.assertIn(memory.relative_to(root).as_posix(), names)
            self.assertIn(tensorboard.relative_to(root).as_posix(), names)
            self.assertIn(advisor.relative_to(root).as_posix(), names)
            self.assertIn((run / "runtime.log").relative_to(root).as_posix(), names)
            self.assertIn(
                (artifact / "analysis.json").relative_to(root).as_posix(), names
            )
            self.assertIn(report.relative_to(root).as_posix(), names)
            self.assertNotIn(raw.relative_to(root).as_posix(), names)
            self.assertNotIn(
                (fixture / "checkpoint.distcp").relative_to(root).as_posix(), names
            )


if __name__ == "__main__":
    unittest.main()
