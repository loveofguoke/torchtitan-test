# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.glm5_2_parity.contracts import (
    GLM5_PARITY_CAPTURE_SUITE,
    GLM5_PARITY_FIXTURE_SUITE,
    GLM5_PARITY_SUITE_VERSION,
)
from tests.glm5_2_parity.workflow import (
    _config_digest,
    OfflineEndpointConfig,
    OfflineParityConfig,
    run_offline_cli,
)


def _write_artifact(
    path: Path,
    *,
    suite: str,
    scenario_config_digest: str,
    fixture_digest: str,
    status: str = "success",
) -> None:
    path.mkdir(parents=True)
    manifest = {
        "suite": suite,
        "suite_version": GLM5_PARITY_SUITE_VERSION,
        "status": status,
        "configuration": {
            "scenario_config_digest": scenario_config_digest,
        },
        "fixture_digest": fixture_digest,
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    (path / "manifest.json").write_bytes(manifest_bytes)
    complete = {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    (path / "complete.json").write_text(
        json.dumps(complete),
        encoding="utf-8",
    )


def _config() -> OfflineParityConfig:
    return OfflineParityConfig(
        actual=OfflineEndpointConfig(
            name="actual",
            endpoint="titan:fp32",
            device_type="cuda",
            visible_device="0",
            visible_devices_env="CUDA_VISIBLE_DEVICES",
            artifact_name="actual",
        ),
        expected=OfflineEndpointConfig(
            name="expected",
            endpoint="titan:fp32",
            device_type="cuda",
            visible_device="1",
            visible_devices_env="CUDA_VISIBLE_DEVICES",
            artifact_name="expected",
        ),
    )


class TestParityWorkflowRerun(unittest.TestCase):
    def test_force_data_resets_every_dependent_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config()
            scenario = "scenario"
            stale_paths = (
                root / config.fixture_root / scenario,
                root / config.artifact_root / scenario,
                root / config.report_root / scenario,
                root / config.log_root / scenario,
            )
            for stale in stale_paths:
                stale.mkdir(parents=True)
                (stale / "old-generation").write_text("old", encoding="utf-8")

            def complete_stage(**kwargs) -> None:
                kwargs["expected_output"].mkdir(parents=True)
                (kwargs["expected_output"] / "new-generation").write_text(
                    "new",
                    encoding="utf-8",
                )

            with (
                patch(
                    "tests.glm5_2_parity.workflow._repo_root",
                    return_value=root,
                ),
                patch(
                    "tests.glm5_2_parity.workflow._run_parity_stage",
                    side_effect=complete_stage,
                ),
                patch("sys.argv", ["scenario.py", "--data", "--force"]),
            ):
                run_offline_cli(config, "scenario.py")

            fixture = root / config.fixture_root / scenario / config.fixture_name
            self.assertTrue((fixture / "new-generation").is_file())
            self.assertFalse(
                any((path / "old-generation").exists() for path in stale_paths)
            )

    def test_completed_fixture_is_reused_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config()
            scenario = "scenario"
            fixture = root / config.fixture_root / scenario / config.fixture_name
            _write_artifact(
                fixture,
                suite=GLM5_PARITY_FIXTURE_SUITE,
                scenario_config_digest=_config_digest(config, scenario),
                fixture_digest="fixture-a",
            )

            with (
                patch(
                    "tests.glm5_2_parity.workflow._repo_root",
                    return_value=root,
                ),
                patch(
                    "tests.glm5_2_parity.workflow._run_parity_stage"
                ) as run_stage,
                patch("sys.argv", ["scenario.py", "--data"]),
            ):
                run_offline_cli(config, "scenario.py")

            run_stage.assert_not_called()

    def test_failed_capture_is_archived_and_retried_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config()
            scenario = "scenario"
            digest = _config_digest(config, scenario)
            fixture = root / config.fixture_root / scenario / config.fixture_name
            actual = (
                root
                / config.artifact_root
                / scenario
                / config.actual.artifact_name
            )
            report = root / config.report_root / scenario / config.report_name
            _write_artifact(
                fixture,
                suite=GLM5_PARITY_FIXTURE_SUITE,
                scenario_config_digest=digest,
                fixture_digest="fixture-a",
            )
            _write_artifact(
                actual,
                suite=GLM5_PARITY_CAPTURE_SUITE,
                scenario_config_digest=digest,
                fixture_digest="fixture-a",
                status="failed",
            )
            report.parent.mkdir(parents=True)
            report.write_text("stale report", encoding="utf-8")

            def complete_stage(**kwargs) -> None:
                kwargs["expected_output"].mkdir(parents=True)

            with (
                patch(
                    "tests.glm5_2_parity.workflow._repo_root",
                    return_value=root,
                ),
                patch(
                    "tests.glm5_2_parity.workflow._run_parity_stage",
                    side_effect=complete_stage,
                ),
                patch("sys.argv", ["scenario.py", "--actual-capture"]),
            ):
                run_offline_cli(config, "scenario.py")

            self.assertTrue(actual.is_dir())
            self.assertTrue(
                any(actual.parent.glob(f".{actual.name}.previous-*"))
            )
            self.assertFalse(report.exists())
            self.assertTrue(
                any(report.parent.glob(f".{report.name}.previous-*"))
            )

    def test_completed_capture_for_current_fixture_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config()
            scenario = "scenario"
            digest = _config_digest(config, scenario)
            fixture = root / config.fixture_root / scenario / config.fixture_name
            actual = (
                root
                / config.artifact_root
                / scenario
                / config.actual.artifact_name
            )
            _write_artifact(
                fixture,
                suite=GLM5_PARITY_FIXTURE_SUITE,
                scenario_config_digest=digest,
                fixture_digest="fixture-a",
            )
            _write_artifact(
                actual,
                suite=GLM5_PARITY_CAPTURE_SUITE,
                scenario_config_digest=digest,
                fixture_digest="fixture-a",
            )

            with (
                patch(
                    "tests.glm5_2_parity.workflow._repo_root",
                    return_value=root,
                ),
                patch(
                    "tests.glm5_2_parity.workflow._run_parity_stage"
                ) as run_stage,
                patch("sys.argv", ["scenario.py", "--actual-capture"]),
            ):
                run_offline_cli(config, "scenario.py")

            run_stage.assert_not_called()

    def test_completed_capture_from_another_fixture_requires_force(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config()
            scenario = "scenario"
            digest = _config_digest(config, scenario)
            fixture = root / config.fixture_root / scenario / config.fixture_name
            actual = (
                root
                / config.artifact_root
                / scenario
                / config.actual.artifact_name
            )
            _write_artifact(
                fixture,
                suite=GLM5_PARITY_FIXTURE_SUITE,
                scenario_config_digest=digest,
                fixture_digest="fixture-current",
            )
            _write_artifact(
                actual,
                suite=GLM5_PARITY_CAPTURE_SUITE,
                scenario_config_digest=digest,
                fixture_digest="fixture-old",
            )

            with (
                patch(
                    "tests.glm5_2_parity.workflow._repo_root",
                    return_value=root,
                ),
                patch("sys.argv", ["scenario.py", "--actual-capture"]),
            ):
                with self.assertRaisesRegex(
                    FileExistsError,
                    "different fixture",
                ):
                    run_offline_cli(config, "scenario.py")


if __name__ == "__main__":
    unittest.main()
