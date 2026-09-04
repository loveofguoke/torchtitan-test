from __future__ import annotations

import argparse
from pathlib import Path
import tarfile
import unittest

from release_artifacts import _analysis_archive_filter
from tests.glm5_2_common.topology import standard_topologies
from tests.glm5_2_mindstudio.specialized_tuning import (
    memory_command,
    operator_command,
)


class MindStudioSpecializedTuningTest(unittest.TestCase):
    def test_release_filter_keeps_processed_specialized_outputs(self) -> None:
        retained = (
            "mindstudio_runs/operator/run/operator_profile/OPPROF/output.csv",
            "mindstudio_runs/operator/run/operator_profile/visualize_data.bin",
            "mindstudio_runs/memory/1-card/single/run/memory_profile/result.db",
        )
        for name in retained:
            with self.subTest(name=name):
                self.assertIsNotNone(_analysis_archive_filter(tarfile.TarInfo(name)))

        raw = tarfile.TarInfo(
            "mindstudio_runs/operator/run/operator_profile/dump/raw.bin"
        )
        self.assertIsNone(_analysis_archive_filter(raw))

    def test_operator_command_preserves_official_argument_order(self) -> None:
        args = argparse.Namespace(
            executable="msopprof",
            mode="onboard",
            soc_version=None,
            metrics="Default",
            kernel_name="TopK*",
            launch_count=3,
            launch_skip=2,
            kill=True,
            core_id=None,
            dump=False,
            tool_arg=[],
            config_file=None,
            application=["--", "python", "probe.py"],
        )
        command = operator_command(args, Path("/tmp/operator"))
        self.assertEqual(command[0], "msopprof")
        self.assertIn(f"--output={Path('/tmp/operator')}", command)
        self.assertIn("--kernel-name=TopK*", command)
        self.assertEqual(command[-2:], ["python", "probe.py"])

    def test_simulator_places_mode_before_options(self) -> None:
        args = argparse.Namespace(
            executable="msopprof",
            mode="simulator",
            soc_version="Ascend910B2",
            metrics="Default",
            kernel_name=None,
            launch_count=1,
            launch_skip=0,
            kill=False,
            core_id=None,
            dump=False,
            tool_arg=[],
            config_file="operator.json",
            application=[],
        )
        command = operator_command(args, Path("/tmp/operator"))
        self.assertEqual(command[:3], ["msopprof", "simulator", "--soc-version=Ascend910B2"])
        self.assertTrue(command[-1].startswith("--config="))

    def test_memory_command_wraps_existing_training_launcher(self) -> None:
        args = argparse.Namespace(
            executable="msmemscope",
            mem_device="npu",
            level="op",
            events="alloc,free,launch,traceback",
            call_stack="python:20,c:20",
            collect_mode="immediate",
            analysis="leaks,decompose",
            data_format="db",
            capture_steps=None,
            tool_arg=[],
            steps=10,
            local_batch_size=8,
            global_batch_size=64,
            sequence_length=128,
            seed=61,
        )
        command = memory_command(
            args, Path("/tmp/memory"), standard_topologies()["single"]
        )
        separator = command.index("--")
        self.assertEqual(command[0], "msmemscope")
        self.assertIn("--analysis=leaks,decompose", command[:separator])
        self.assertEqual(command[separator + 1], "bash")
        self.assertTrue(command[separator + 2].endswith("run_train.sh"))


if __name__ == "__main__":
    unittest.main()
