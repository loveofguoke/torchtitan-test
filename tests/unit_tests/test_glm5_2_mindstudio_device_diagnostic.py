# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.glm5_2_mindstudio.artifacts import output_index
from tests.glm5_2_mindstudio.device_diagnostic.diagnostic_benchmark import (
    generation_complete,
    prepare_generation,
    run_command,
)


class DeviceDiagnosticLifecycleTest(unittest.TestCase):
    def _complete_generation(
        self,
        root: Path,
        digest: str,
    ) -> tuple[Path, Path, Path]:
        run = root / "mindstudio_runs/performance/device_diagnostic/run"
        artifact = root / "mindstudio_artifacts/performance/device_diagnostic/run"
        report = root / "mindstudio_reports/performance/device_diagnostic/run"
        run.mkdir(parents=True)
        (run / "runtime.log").write_text("done\n", encoding="utf-8")
        official = artifact / "official"
        official.mkdir(parents=True)
        (official / "inventory.txt").write_text("ok\n", encoding="utf-8")
        artifact.joinpath("manifest.json").write_text(
            json.dumps(
                {
                    "experiment_digest": digest,
                    "official_files": output_index(official),
                }
            ),
            encoding="utf-8",
        )
        artifact.joinpath("complete.json").write_text(
            json.dumps({"status": "completed", "experiment_digest": digest}),
            encoding="utf-8",
        )
        report.mkdir(parents=True)
        (report / "summary.json").write_text("{}\n", encoding="utf-8")
        (report / "README.md").write_text("done\n", encoding="utf-8")
        return run, artifact, report

    def test_completed_generation_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run, artifact, report = self._complete_generation(root, "digest-a")
            self.assertTrue(
                generation_complete(
                    artifact,
                    report,
                    experiment_digest="digest-a",
                )
            )
            self.assertFalse(
                prepare_generation(
                    run,
                    artifact,
                    report,
                    experiment_digest="digest-a",
                    force=False,
                )
            )
            self.assertTrue(run.is_dir())

    def test_incomplete_generation_is_archived_for_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run = root / "mindstudio_runs/performance/device_diagnostic/run"
            artifact = root / "mindstudio_artifacts/performance/device_diagnostic/run"
            report = root / "mindstudio_reports/performance/device_diagnostic/run"
            for path in (run, artifact, report):
                path.mkdir(parents=True)
            self.assertTrue(
                prepare_generation(
                    run,
                    artifact,
                    report,
                    experiment_digest="digest-a",
                    force=False,
                )
            )
            self.assertFalse(run.exists())
            self.assertEqual(1, len(list(run.parent.glob(".run.previous-*"))))

    @patch("tests.glm5_2_common.cli.process_is_running", return_value=True)
    def test_active_generation_refuses_retry(self, _running) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run = root / "mindstudio_runs/performance/device_diagnostic/run"
            artifact = root / "mindstudio_artifacts/performance/device_diagnostic/run"
            report = root / "mindstudio_reports/performance/device_diagnostic/run"
            run.mkdir(parents=True)
            (run / "run_state.json").write_text(
                json.dumps({"status": "running", "pid": os.getpid()}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "still active"):
                prepare_generation(
                    run,
                    artifact,
                    report,
                    experiment_digest="digest-a",
                    force=False,
                )

    @patch("subprocess.run")
    def test_subprocess_failure_identifies_runtime_log(self, run) -> None:
        run.return_value.returncode = 9
        run.return_value.stdout = ""
        run.return_value.stderr = "failed\n"
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            log_path = root / "mindstudio_runs/performance/device_diagnostic/runtime.log"
            log_path.parent.mkdir(parents=True)
            with log_path.open("w", encoding="utf-8") as log:
                with self.assertRaisesRegex(Exception, "runtime log") as caught:
                    run_command(
                        ["false"],
                        root=root,
                        log=log,
                        output=root / "stdout.txt",
                    )
            self.assertIn(str(log_path.resolve()), str(caught.exception))


if __name__ == "__main__":
    unittest.main()
