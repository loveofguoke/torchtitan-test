import json
import os
import tempfile
import unittest
from pathlib import Path

from tests.glm5_2_performance.documentation import card_scope, write_run_readme
from tests.glm5_2_performance.explorations.tools.summarize_topologies import (
    _baseline_row,
    _latest_runs,
)
from tests.glm5_2_performance.workflow import (
    _exploration_directory,
    _scoped_parent,
)


class TestTopologyExplorationSummary(unittest.TestCase):
    def _write_run(
        self,
        root: Path,
        *,
        name: str,
        topology: str,
        replicate: int,
        mtime: int,
    ) -> Path:
        directory = root / name
        directory.mkdir()
        manifest = {
            "run_name": name,
            "device": "npu",
            "topology": topology,
            "preset": "distributed",
            "visible_devices": [str(index) for index in range(8)],
            "config": {
                "steps": 20,
                "replicate": replicate,
                "profiler_enabled": False,
            },
        }
        analysis = {
            "profile_phases": {
                "comparison": {
                    "baseline_median_step_seconds": 0.2,
                    "baseline_median_throughput_tps": 5120.0,
                    "baseline_job_throughput_tps": 40960.0,
                },
                "phases": {
                    "steady": {
                        "summary": {
                            "time_metrics/end_to_end(s)": {
                                "median": 0.2,
                                "p90": 0.22,
                                "max": 0.25,
                            },
                            "memory/max_active(GiB)": {"max": 0.75},
                        }
                    }
                },
            }
        }
        manifest_path = directory / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        (directory / "analysis.json").write_text(
            json.dumps(analysis), encoding="utf-8"
        )
        os.utime(manifest_path, ns=(mtime, mtime))
        return directory

    def test_latest_successful_replicate_is_scoped_to_world_size(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_run(
                root,
                name="ddp8-r1",
                topology="ddp8",
                replicate=1,
                mtime=1,
            )
            latest = self._write_run(
                root,
                name="ddp8-r2",
                topology="ddp8",
                replicate=2,
                mtime=2,
            )
            self._write_run(
                root,
                name="fsdp4-r3",
                topology="fsdp4",
                replicate=3,
                mtime=3,
            )

            runs = _latest_runs(
                root, steps=20, replicate=None, world_size=8
            )

            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["manifest_path"].parent, latest)

    def test_explicit_replicate_remains_selectable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            selected = self._write_run(
                root,
                name="ddp8-r1",
                topology="ddp8",
                replicate=1,
                mtime=1,
            )
            self._write_run(
                root,
                name="ddp8-r2",
                topology="ddp8",
                replicate=2,
                mtime=2,
            )

            runs = _latest_runs(root, steps=20, replicate=1, world_size=8)

            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["manifest_path"].parent, selected)

    def test_baseline_row_derives_stability_and_memory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            directory = self._write_run(
                root,
                name="ddp8-r2",
                topology="ddp8",
                replicate=2,
                mtime=1,
            )
            run = {
                "manifest_path": directory / "manifest.json",
                "manifest": json.loads(
                    (directory / "manifest.json").read_text(encoding="utf-8")
                ),
                "analysis": json.loads(
                    (directory / "analysis.json").read_text(encoding="utf-8")
                ),
            }

            row = _baseline_row(run)

            self.assertEqual(row["replicate"], 2)
            self.assertAlmostEqual(row["step_ms"], 200.0)
            self.assertAlmostEqual(row["step_p90_ms"], 220.0)
            self.assertAlmostEqual(row["step_max_ms"], 250.0)
            self.assertAlmostEqual(row["p90_over_median_pct"], 10.0)
            self.assertAlmostEqual(row["memory_gib"], 0.75)


class TestExplorationDocumentation(unittest.TestCase):
    def test_scoped_paths_follow_card_count_and_topology(self):
        root = Path("/repo")

        self.assertEqual(card_scope("fsdp8"), "8-card")
        self.assertEqual(
            _scoped_parent(root, "performance_runs", "fsdp8"),
            root / "performance_runs" / "8-card" / "fsdp8",
        )
        self.assertEqual(
            _exploration_directory(root, "fsdp8", "run-id"),
            root
            / "tests"
            / "glm5_2_performance"
            / "explorations"
            / "runs"
            / "8-card"
            / "fsdp8"
            / "run-id",
        )

    def test_run_readme_contains_process_results_and_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory) / "run-id"
            directory.mkdir()
            manifest = {
                "run_name": "run-id",
                "device": "npu",
                "topology": "ddp2",
                "preset": "distributed",
                "visible_devices": ["1", "2"],
                "command": ["torchrun", "--nproc_per_node=2", "train.py"],
                "config": {
                    "steps": 20,
                    "local_batch_size": 8,
                    "global_batch_size": 64,
                    "sequence_length": 128,
                    "mixed_precision_param": "bfloat16",
                    "mixed_precision_reduce": "float32",
                    "profiler_enabled": False,
                    "replicate": 1,
                },
            }
            analysis = {
                "profile_phases": {
                    "comparison": {
                        "baseline_median_step_seconds": 0.5,
                        "baseline_median_throughput_tps": 2048.0,
                        "baseline_job_throughput_tps": 4096.0,
                    }
                }
            }
            (directory / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (directory / "analysis.json").write_text(
                json.dumps(analysis), encoding="utf-8"
            )
            (directory / "command_history.jsonl").write_text(
                json.dumps(
                    {
                        "action": "probe",
                        "topology": "ddp2",
                        "shell_command": "python profiler_benchmark.py --probe",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            output = write_run_readme(directory)
            text = output.read_text(encoding="utf-8")

            self.assertIn("python profiler_benchmark.py --probe", text)
            self.assertIn("torchrun --nproc_per_node=2 train.py", text)
            self.assertIn("4,096.00 tok/s", text)
            self.assertIn("[analysis.json](analysis.json)", text)


if __name__ == "__main__":
    unittest.main()
