# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import importlib.machinery
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests.glm5_2_mindstudio import toolchain
from tools import bootstrap_mindstudio_toolchain as bootstrap


class TestMindStudioToolchain(unittest.TestCase):
    def test_lock_records_only_official_source_selections(self):
        lock = toolchain.load_toolchain_lock()

        self.assertEqual(lock["schema_version"], 1)
        self.assertEqual(lock["profile_name"], "glm5_2_mindstudio_standard")
        self.assertEqual(
            set(lock["compatibility"]),
            {"cann", "driver", "firmware", "torch", "torch_npu"},
        )
        for name, entry in lock["tools"].items():
            with self.subTest(tool=name):
                source = entry["source"]
                self.assertTrue(source["url"].startswith("https://gitcode.com/Ascend/"))
                self.assertIsNone(source["commit"])
                self.assertIn(source["ref"]["kind"], {"branch", "tag", "commit"})
                self.assertTrue(source["version"])

    def test_paths_resolve_from_the_external_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = toolchain.tool_source_path("msprof_analyze", root)
            paths = toolchain.toolchain_paths(root)

        self.assertEqual(path, (root / "msprof-analyze").resolve())
        self.assertEqual(paths["msprobe"], str((root / "msprobe").resolve()))
        self.assertEqual(
            paths["msinsight_flamegraph"],
            str(
                (
                    root
                    / "msinsight"
                    / "scripts"
                    / "flame_graph"
                    / "flamegraph.py"
                ).resolve()
            ),
        )

    def test_doctor_reports_a_complete_mocked_toolchain(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cann = root / "cann"
            cann.mkdir()
            (cann / "version.info").write_text(
                "Version=8.5.0\npackage_name=CANN\n", encoding="utf-8"
            )
            flamegraph = root / "flamegraph.py"
            flamegraph.write_text("pass\n", encoding="utf-8")

            executable_paths = {
                name: str((root / "bin" / name).resolve())
                for name in ("msprobe", "msprof", "msprof-analyze", "npu-smi")
            }

            def find_spec(module_name):
                return importlib.machinery.ModuleSpec(
                    module_name,
                    loader=None,
                    origin=str(root / f"{module_name}.py"),
                )

            versions = {
                "mindstudio-probe": "26.0.0",
                "torch": "2.7.1",
                "torch-npu": "2.7.1.post4",
                "msprof-analyze": "26.0.0",
            }

            def distribution_version(name):
                if name not in versions:
                    raise toolchain.importlib_metadata.PackageNotFoundError(name)
                return versions[name]

            def which(name, path=None):
                del path
                return executable_paths.get(name)

            completed = subprocess.CompletedProcess(
                ["tool"], 0, stdout="version 26.0.0\n", stderr=""
            )
            environment = {
                "PATH": str(root / "bin"),
                "ASCEND_HOME_PATH": str(cann),
                "TORCHTITAN_MSINSIGHT_FLAMEGRAPH": str(flamegraph),
            }
            with (
                mock.patch.object(
                    toolchain.importlib.util,
                    "find_spec",
                    side_effect=find_spec,
                ),
                mock.patch.object(
                    toolchain.importlib_metadata,
                    "version",
                    side_effect=distribution_version,
                ),
                mock.patch.object(
                    toolchain.shutil, "which", side_effect=which
                ),
                mock.patch.object(
                    toolchain.subprocess, "run", return_value=completed
                ) as run,
            ):
                report = toolchain.doctor_report(root, environ=environment)

        self.assertTrue(report["ok"])
        self.assertEqual(report["required_failures"], [])
        self.assertEqual(
            report["checks"]["framework"]["torch_npu"]["version"],
            "2.7.1.post4",
        )
        self.assertEqual(
            report["checks"]["cann"]["roots"][0]["version_files"][0][
                "values"
            ]["Version"],
            "8.5.0",
        )
        self.assertTrue(report["checks"]["msinsight"]["ok"])
        self.assertIsInstance(json.dumps(report), str)
        for call in run.call_args_list:
            self.assertFalse(call.kwargs["check"])

    def test_doctor_keeps_optional_msinsight_out_of_required_failures(self):
        def missing_distribution(name):
            raise toolchain.importlib_metadata.PackageNotFoundError(name)

        with (
            mock.patch.object(
                toolchain.importlib.util, "find_spec", return_value=None
            ),
            mock.patch.object(
                toolchain.importlib_metadata,
                "version",
                side_effect=missing_distribution,
            ),
            mock.patch.object(toolchain.shutil, "which", return_value=None),
            mock.patch.object(toolchain.subprocess, "run") as run,
        ):
            report = toolchain.doctor_report(environ={})

        self.assertFalse(report["ok"])
        self.assertEqual(
            set(report["required_failures"]),
            {"msprobe", "framework", "cann", "msprof", "msprof-analyze"},
        )
        self.assertFalse(report["checks"]["msinsight"]["required"])
        run.assert_not_called()

    def test_full_doctor_requires_msprof_analyze_import_and_cli(self):
        def package_status(module, _distributions):
            return {
                "module": module,
                "importable": module != "msprof_analyze",
                "ok": module != "msprof_analyze",
                "origin": None,
                "distribution": None,
                "version": None,
                "error": None,
            }

        executable = {
            "required": True,
            "ok": True,
            "path": "tool",
            "reported_version": "26.0.0",
        }
        source = {"available": True}
        with (
            mock.patch.object(
                toolchain,
                "python_package_metadata",
                side_effect=package_status,
            ),
            mock.patch.object(
                toolchain,
                "_executable_status",
                return_value=executable,
            ),
            mock.patch.object(
                toolchain,
                "_cann_status",
                return_value={"required": True, "ok": True},
            ),
            mock.patch.object(
                toolchain,
                "_msinsight_status",
                return_value={"required": False, "ok": False},
            ),
            mock.patch.object(
                toolchain,
                "tool_source_metadata",
                return_value=source,
            ),
        ):
            report = toolchain.doctor_report(
                toolchain_root=Path("external"),
                environ={},
                scope="full",
            )

        self.assertFalse(report["ok"])
        self.assertIn("msprof-analyze", report["required_failures"])
        self.assertTrue(
            report["checks"]["msprof-analyze"]["executable"]["ok"]
        )
        self.assertFalse(
            report["checks"]["msprof-analyze"]["import"]["ok"]
        )

    def test_accuracy_scopes_separate_gpu_and_npu_readiness(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            def find_spec(module_name):
                if module_name not in {"msprobe", "torch"}:
                    return None
                return importlib.machinery.ModuleSpec(
                    module_name,
                    loader=None,
                    origin=str(root / f"{module_name}.py"),
                )

            def distribution_version(name):
                versions = {
                    "mindstudio-probe": "26.0.0",
                    "torch": "2.7.1",
                }
                if name not in versions:
                    raise toolchain.importlib_metadata.PackageNotFoundError(name)
                return versions[name]

            def which(name, path=None):
                del path
                return str(root / "msprobe") if name == "msprobe" else None

            completed = subprocess.CompletedProcess(
                ["msprobe", "--help"], 0, "usage", ""
            )
            with (
                mock.patch.object(
                    toolchain.importlib.util,
                    "find_spec",
                    side_effect=find_spec,
                ),
                mock.patch.object(
                    toolchain.importlib_metadata,
                    "version",
                    side_effect=distribution_version,
                ),
                mock.patch.object(
                    toolchain.shutil, "which", side_effect=which
                ),
                mock.patch.object(
                    toolchain.subprocess, "run", return_value=completed
                ),
            ):
                gpu = toolchain.doctor_report(
                    environ={}, scope="accuracy-gpu"
                )
                npu = toolchain.doctor_report(
                    environ={}, profile="accuracy_npu"
                )

        self.assertTrue(gpu["ok"])
        self.assertEqual(gpu["scope"], "accuracy-gpu")
        self.assertFalse(gpu["checks"]["framework"]["torch_npu"]["required"])
        self.assertFalse(gpu["checks"]["cann"]["required"])
        self.assertFalse(gpu["checks"]["msprof"]["required"])
        self.assertFalse(gpu["checks"]["msprof-analyze"]["required"])
        self.assertFalse(npu["ok"])
        self.assertEqual(npu["scope"], "accuracy-npu")
        self.assertEqual(
            set(npu["required_failures"]), {"framework", "cann"}
        )

    def test_performance_scopes_split_capture_and_offline_analysis(self):
        mode = {"value": "capture"}

        def package_status(module, _distributions):
            if mode["value"] == "capture":
                ok = module in {"torch", "torch_npu"}
            elif mode["value"] == "analysis":
                ok = module == "msprof_analyze"
            else:
                ok = module in {"torch", "torch_npu", "msprof_analyze"}
            return {
                "module": module,
                "importable": ok,
                "ok": ok,
                "origin": None,
                "distribution": None,
                "version": None,
                "error": None,
            }

        def executable_status(command, _arguments, *, required, **_kwargs):
            if mode["value"] == "capture":
                ok = command == "msprof"
            elif mode["value"] == "analysis":
                ok = command == "msprof-analyze"
            else:
                ok = command in {"msprof", "msprof-analyze"}
            return {
                "required": required,
                "ok": ok,
                "path": command if ok else None,
                "reported_version": "26.0.0" if ok else None,
            }

        def cann_status(_environment, *, required):
            return {
                "required": required,
                "ok": mode["value"] in {"capture", "full"},
            }

        with (
            mock.patch.object(
                toolchain,
                "python_package_metadata",
                side_effect=package_status,
            ),
            mock.patch.object(
                toolchain,
                "_executable_status",
                side_effect=executable_status,
            ),
            mock.patch.object(
                toolchain,
                "_cann_status",
                side_effect=cann_status,
            ),
            mock.patch.object(
                toolchain,
                "_msinsight_status",
                return_value={"required": False, "ok": False},
            ),
            mock.patch.object(
                toolchain,
                "tool_source_metadata",
                return_value={"available": True},
            ),
        ):
            capture = toolchain.doctor_report(
                toolchain_root=Path("external"),
                environ={},
                scope="performance-capture",
            )
            torch_npu_capture = toolchain.doctor_report(
                toolchain_root=Path("external"),
                environ={},
                scope="performance-torch-npu-capture",
            )
            mode["value"] = "analysis"
            analysis = toolchain.doctor_report(
                toolchain_root=Path("external"),
                environ={},
                scope="performance_analysis",
            )
            mode["value"] = "full"
            performance = toolchain.doctor_report(
                toolchain_root=Path("external"),
                environ={},
                scope="performance",
            )

        self.assertTrue(capture["ok"])
        self.assertTrue(capture["checks"]["msprof"]["required"])
        self.assertFalse(capture["checks"]["msprof-analyze"]["required"])
        self.assertTrue(capture["checks"]["framework"]["required"])
        self.assertTrue(capture["checks"]["cann"]["required"])
        self.assertTrue(torch_npu_capture["ok"])
        self.assertTrue(
            torch_npu_capture["checks"]["framework"]["required"]
        )
        self.assertTrue(torch_npu_capture["checks"]["cann"]["required"])
        self.assertFalse(torch_npu_capture["checks"]["msprof"]["required"])
        self.assertFalse(
            torch_npu_capture["checks"]["msprof-analyze"]["required"]
        )
        self.assertTrue(analysis["ok"])
        self.assertFalse(analysis["checks"]["framework"]["required"])
        self.assertFalse(analysis["checks"]["cann"]["required"])
        self.assertFalse(analysis["checks"]["msprof"]["required"])
        self.assertTrue(analysis["checks"]["msprof-analyze"]["required"])
        self.assertTrue(performance["ok"])
        self.assertTrue(performance["checks"]["framework"]["required"])
        self.assertTrue(performance["checks"]["cann"]["required"])
        self.assertTrue(performance["checks"]["msprof"]["required"])
        self.assertTrue(performance["checks"]["msprof-analyze"]["required"])

    def test_source_metadata_records_resolved_git_state(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "msprobe"
            (source / ".git").mkdir(parents=True)

            def git_run(command, **kwargs):
                del kwargs
                arguments = command[3:]
                if arguments == ["status", "--porcelain"]:
                    return subprocess.CompletedProcess(command, 0, " M file\n", "")
                if arguments == ["rev-parse", "HEAD"]:
                    return subprocess.CompletedProcess(command, 0, "abc123\n", "")
                if arguments == ["branch", "--show-current"]:
                    return subprocess.CompletedProcess(command, 0, "master\n", "")
                return subprocess.CompletedProcess(command, 1, "", "")

            with mock.patch.object(
                toolchain.subprocess, "run", side_effect=git_run
            ):
                metadata = toolchain.tool_source_metadata("msprobe", root)

        self.assertEqual(metadata["commit"], "abc123")
        self.assertEqual(metadata["branch"], "master")
        self.assertTrue(metadata["dirty"])
        self.assertIsNone(metadata["locked_commit"])

    def test_msprobe_installation_identity_hashes_tree_record_and_cli(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            site_packages = root / "Lib" / "site-packages"
            package = site_packages / "msprobe"
            dist_info = site_packages / "mindstudio_probe-26.0.0.dist-info"
            scripts = root / "Scripts"
            package.mkdir(parents=True)
            dist_info.mkdir()
            scripts.mkdir()
            origin = package / "__init__.py"
            origin.write_bytes(b"VERSION = 1\n")
            record = dist_info / "RECORD"
            record.write_text("msprobe/__init__.py,,\n", encoding="utf-8")
            executable = scripts / "msprobe.exe"
            executable.write_bytes(b"console-launcher")

            class FakeDistribution:
                files = (
                    Path("msprobe/__init__.py"),
                    Path("mindstudio_probe-26.0.0.dist-info/RECORD"),
                    Path("../../Scripts/msprobe.exe"),
                )
                entry_points = (
                    SimpleNamespace(name="msprobe", group="console_scripts"),
                )

                @staticmethod
                def locate_file(path):
                    return site_packages / path

            spec = importlib.machinery.ModuleSpec(
                "msprobe",
                loader=None,
                origin=str(origin),
            )
            with (
                mock.patch.object(
                    toolchain.importlib.util,
                    "find_spec",
                    return_value=spec,
                ),
                mock.patch.object(
                    toolchain.importlib_metadata,
                    "version",
                    return_value="26.0.0",
                ),
                mock.patch.object(
                    toolchain.importlib_metadata,
                    "distribution",
                    return_value=FakeDistribution(),
                ),
            ):
                first = toolchain.python_package_metadata(
                    "msprobe", ("mindstudio-probe",)
                )["installation_identity"]
                origin.write_bytes(b"VERSION = 2\n")
                second = toolchain.python_package_metadata(
                    "msprobe", ("mindstudio-probe",)
                )["installation_identity"]
                record.write_text("changed-record\n", encoding="utf-8")
                third = toolchain.python_package_metadata(
                    "msprobe", ("mindstudio-probe",)
                )["installation_identity"]

        self.assertTrue(first["ok"])
        self.assertTrue(first["origin_in_distribution"])
        self.assertEqual(first["console_scripts"][0]["path"], str(executable))
        self.assertNotEqual(
            first["package_tree"]["sha256"],
            second["package_tree"]["sha256"],
        )
        self.assertEqual(
            second["package_tree"]["sha256"],
            third["package_tree"]["sha256"],
        )
        self.assertNotEqual(
            second["record"]["sha256"],
            third["record"]["sha256"],
        )

    def test_resolved_manifest_validates_wheel_checkout_import_and_cli(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            source = root / "msprobe"
            wheel = source / "artifacts" / "mindstudio_probe-26.0.0.whl"
            wheel.parent.mkdir(parents=True)
            package = root / "installed" / "msprobe"
            package.mkdir(parents=True)
            origin = package / "__init__.py"
            origin.write_bytes(b"VERSION = 1\n")
            record = root / "installed" / "mindstudio_probe.dist-info" / "RECORD"
            record.parent.mkdir()
            record.write_text("installed-record\n", encoding="utf-8")
            executable = root / "bin" / "msprobe"
            executable.parent.mkdir()
            executable.write_bytes(b"console-launcher")
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("msprobe/__init__.py", b"VERSION = 1\n")
                archive.writestr(
                    "mindstudio_probe-26.0.0.dist-info/RECORD",
                    b"wheel-record\n",
                )
            lock = toolchain.load_toolchain_lock()
            lock_sha256 = hashlib.sha256(toolchain.LOCK_PATH.read_bytes()).hexdigest()
            resolved = {
                "schema_version": 1,
                "status": "completed",
                "profile_name": lock["profile_name"],
                "lock_sha256": lock_sha256,
                "dry_run": False,
                "destination": str(root),
                "python_executable": sys.executable,
                "tools": {
                    "msprobe": {
                        "source": {
                            "path": str(source),
                            "official_url": lock["tools"]["msprobe"]["source"]["url"],
                            "ref": lock["tools"]["msprobe"]["source"]["ref"],
                            "commit": "abc123",
                        },
                        "build": {
                            "strategy": "build-wheel",
                            "artifact": str(wheel),
                            "artifact_identity": toolchain._file_identity(wheel),
                        },
                    }
                },
            }
            resolved_path = root / "toolchain.resolved.json"
            resolved_path.write_text(json.dumps(resolved), encoding="utf-8")
            installation = {
                "ok": True,
                "record": {"path": str(record), **toolchain._file_identity(record)},
                "package_tree": toolchain._package_tree_identity(origin, "msprobe"),
                "origin_in_distribution": True,
                "console_scripts": [
                    {
                        "name": "msprobe",
                        "path": str(executable),
                        **toolchain._file_identity(executable),
                    }
                ],
                "error": None,
            }
            report = {
                "runtime": {"python_executable": sys.executable},
                "checks": {
                    "msprobe": {
                        "required": True,
                        "import": {"installation_identity": installation},
                        "executable": {
                            "path": str(executable),
                            "identity": toolchain._file_identity(executable),
                        },
                    },
                    "source_checkouts": {
                        "tools": {
                            "msprobe": {
                                "path": str(source),
                                "available": True,
                                "commit": "abc123",
                            }
                        }
                    },
                },
            }
            valid = toolchain._validate_resolved_toolchain(
                resolved_path,
                resolved_root=root,
                lock=lock,
                lock_sha256=lock_sha256,
                report=report,
            )
            expected_tree = toolchain._portable_tree_identity(
                installation["package_tree"]
            )
            report["checks"]["msprobe"]["executable"]["path"] = str(
                root / "bin" / "other-msprobe"
            )
            changed_cli = toolchain._validate_resolved_toolchain(
                resolved_path,
                resolved_root=root,
                lock=lock,
                lock_sha256=lock_sha256,
                report=report,
            )
            report["checks"]["msprobe"]["executable"]["path"] = str(executable)
            resolved["profile_name"] = "different-profile"
            resolved_path.write_text(json.dumps(resolved), encoding="utf-8")
            changed_profile = toolchain._validate_resolved_toolchain(
                resolved_path,
                resolved_root=root,
                lock=lock,
                lock_sha256=lock_sha256,
                report=report,
            )
            resolved["profile_name"] = lock["profile_name"]
            resolved["lock_sha256"] = "different-lock"
            resolved_path.write_text(json.dumps(resolved), encoding="utf-8")
            changed_lock = toolchain._validate_resolved_toolchain(
                resolved_path,
                resolved_root=root,
                lock=lock,
                lock_sha256=lock_sha256,
                report=report,
            )
            resolved["lock_sha256"] = lock_sha256
            resolved_path.write_text(json.dumps(resolved), encoding="utf-8")
            origin.write_bytes(b"VERSION = 2\n")
            installation["package_tree"] = toolchain._package_tree_identity(
                origin, "msprobe"
            )
            changed_install = toolchain._validate_resolved_toolchain(
                resolved_path,
                resolved_root=root,
                lock=lock,
                lock_sha256=lock_sha256,
                report=report,
            )
            report["checks"]["source_checkouts"]["tools"]["msprobe"][
                "commit"
            ] = "different"
            changed_checkout = toolchain._validate_resolved_toolchain(
                resolved_path,
                resolved_root=root,
                lock=lock,
                lock_sha256=lock_sha256,
                report=report,
            )

        self.assertTrue(valid["valid"], valid["validation_errors"])
        self.assertEqual(
            valid["msprobe_artifact_identity"]["package_tree"],
            expected_tree,
        )
        self.assertFalse(changed_cli["valid"])
        self.assertIn(
            "current msprobe CLI is not owned by the imported distribution",
            changed_cli["validation_errors"],
        )
        self.assertFalse(changed_profile["valid"])
        self.assertIn(
            "profile_name does not match the current lock",
            changed_profile["validation_errors"],
        )
        self.assertFalse(changed_lock["valid"])
        self.assertIn(
            "lock_sha256 does not match the current lock",
            changed_lock["validation_errors"],
        )
        self.assertFalse(changed_install["valid"])
        self.assertIn(
            "current msprobe package tree does not match the resolved wheel",
            changed_install["validation_errors"],
        )
        self.assertFalse(changed_checkout["valid"])
        self.assertIn(
            "source checkout commit does not match for msprobe",
            changed_checkout["validation_errors"],
        )

    def test_collect_metadata_is_json_serializable(self):
        def missing_distribution(name):
            raise toolchain.importlib_metadata.PackageNotFoundError(name)

        with (
            mock.patch.object(
                toolchain.importlib.util, "find_spec", return_value=None
            ),
            mock.patch.object(
                toolchain.importlib_metadata,
                "version",
                side_effect=missing_distribution,
            ),
            mock.patch.object(toolchain.shutil, "which", return_value=None),
        ):
            metadata = toolchain.collect_toolchain_metadata(environ={})

        self.assertEqual(len(metadata["lock"]["sha256"]), 64)
        self.assertIn("doctor", metadata)
        self.assertIsInstance(json.dumps(metadata), str)


class TestMindStudioBootstrap(unittest.TestCase):
    def test_repository_destination_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = Path(temporary_directory) / "repo"
            repository.mkdir()
            with self.assertRaisesRegex(ValueError, "outside the repository"):
                bootstrap.validate_destination(
                    repository / "vendor", repository_root=repository
                )
            sibling = bootstrap.validate_destination(
                repository.parent / "mindstudio", repository_root=repository
            )

        self.assertEqual(sibling, (repository.parent / "mindstudio").resolve())

    def test_dry_run_plans_default_builds_without_subprocesses_or_writes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            repository = base / "repo"
            repository.mkdir()
            destination = base / "external-toolchain"
            with mock.patch.object(bootstrap.subprocess, "run") as run:
                plan = bootstrap.bootstrap_toolchain(
                    destination,
                    python_executable="python-test",
                    dry_run=True,
                    repository_root=repository,
                )

            commands = [record["command"] for record in plan["commands"]]

        self.assertFalse(destination.exists())
        run.assert_not_called()
        self.assertEqual(plan["selected_tools"], ["msprobe", "msprof-analyze"])
        self.assertTrue(
            any(command[:2] == ["git", "clone"] for command in commands)
        )
        self.assertTrue(
            any(
                len(command) > 1 and command[1].endswith("build.py")
                for command in commands
            )
        )
        self.assertTrue(
            any("include-mod=tb_graph_ascend" in command for command in commands)
        )
        self.assertTrue(any("--editable" in command for command in commands))
        self.assertTrue(
            any("mindstudio_probe*.whl" in command[-1] for command in commands)
        )
        self.assertTrue(all(not record["executed"] for record in plan["commands"]))

    def test_optional_source_only_tool_can_be_planned(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            repository = base / "repo"
            repository.mkdir()
            plan = bootstrap.bootstrap_toolchain(
                base / "external",
                tools=["msinsight"],
                dry_run=True,
                repository_root=repository,
            )

        self.assertEqual(plan["selected_tools"], ["msinsight"])
        self.assertEqual(plan["tools"]["msinsight"]["build"]["strategy"], "source-only")

    def test_msprobe_build_runs_build_py_then_installs_the_wheel(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            repository = base / "repo"
            repository.mkdir()
            destination = base / "external"
            source = destination / "msprobe"
            (source / ".git").mkdir(parents=True)
            (source / "build.py").write_text("pass\n", encoding="utf-8")
            wheel = source / "artifacts" / "mindstudio_probe-26.0.0.whl"
            wheel.parent.mkdir()
            wheel.write_bytes(b"wheel")

            def run_command(command, **kwargs):
                if command[-3:] == ["remote", "get-url", "origin"]:
                    stdout = "https://gitcode.com/Ascend/msprobe.git\n"
                elif command[-2:] == ["rev-parse", "HEAD"]:
                    stdout = "abc123\n"
                elif command[-2:] == ["rev-parse", "origin/master"]:
                    stdout = "abc123\n"
                elif len(command) > 1 and command[1].endswith("build.py"):
                    wheel.write_bytes(b"fresh-wheel")
                    stdout = ""
                else:
                    stdout = ""
                return subprocess.CompletedProcess(command, 0, stdout, "")

            with mock.patch.object(
                bootstrap.subprocess, "run", side_effect=run_command
            ) as run:
                result = bootstrap.bootstrap_toolchain(
                    destination,
                    tools=["msprobe"],
                    python_executable="python-test",
                    repository_root=repository,
                )

            resolved_manifest = destination / "toolchain.resolved.json"
            self.assertTrue(resolved_manifest.is_file())
            resolved = json.loads(resolved_manifest.read_text(encoding="utf-8"))
            self.assertEqual(
                resolved["tools"]["msprobe"]["source"]["commit"],
                "abc123",
            )

            commands = [call.args[0] for call in run.call_args_list]

        build_index = next(
            index
            for index, command in enumerate(commands)
            if len(command) > 1 and command[1].endswith("build.py")
        )
        self.assertIn("include-mod=tb_graph_ascend", commands[build_index])
        install_index = next(
            index
            for index, command in enumerate(commands)
            if command[-1].endswith("mindstudio_probe-26.0.0.whl")
        )
        self.assertLess(build_index, install_index)
        self.assertIn("--force-reinstall", commands[install_index])
        self.assertEqual(result["tools"]["msprobe"]["source"]["commit"], "abc123")
        for call in run.call_args_list:
            self.assertIs(call.kwargs["shell"], False)

    def test_msprobe_build_rejects_an_unchanged_stale_wheel(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            repository = base / "repo"
            repository.mkdir()
            destination = base / "external"
            source = destination / "msprobe"
            (source / ".git").mkdir(parents=True)
            (source / "build.py").write_text("pass\n", encoding="utf-8")
            wheel = source / "artifacts" / "mindstudio_probe-old.whl"
            wheel.parent.mkdir()
            wheel.write_bytes(b"stale-wheel")

            def run_command(command, **kwargs):
                del kwargs
                if command[-3:] == ["remote", "get-url", "origin"]:
                    stdout = "https://gitcode.com/Ascend/msprobe.git\n"
                elif command[-2:] == ["rev-parse", "HEAD"]:
                    stdout = "abc123\n"
                elif command[-2:] == ["rev-parse", "origin/master"]:
                    stdout = "abc123\n"
                else:
                    stdout = ""
                return subprocess.CompletedProcess(command, 0, stdout, "")

            with mock.patch.object(
                bootstrap.subprocess, "run", side_effect=run_command
            ):
                with self.assertRaisesRegex(
                    FileNotFoundError, "refusing to install a stale artifact"
                ):
                    bootstrap.bootstrap_toolchain(
                        destination,
                        tools=["msprobe"],
                        python_executable="python-test",
                        repository_root=repository,
                    )

    def test_msprof_analyze_installs_declared_requirements_editable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            repository = base / "repo"
            repository.mkdir()
            destination = base / "external"
            source = destination / "msprof-analyze"
            (source / ".git").mkdir(parents=True)
            requirements = source / "requirements" / "build.txt"
            requirements.parent.mkdir()
            requirements.write_text("pandas\n", encoding="utf-8")

            def run_command(command, **kwargs):
                if command[-3:] == ["remote", "get-url", "origin"]:
                    stdout = "https://gitcode.com/Ascend/msprof-analyze.git\n"
                elif command[-2:] == ["rev-parse", "HEAD"]:
                    stdout = "def456\n"
                elif command[-2:] == ["rev-parse", "origin/master"]:
                    stdout = "def456\n"
                else:
                    stdout = ""
                return subprocess.CompletedProcess(command, 0, stdout, "")

            with mock.patch.object(
                bootstrap.subprocess, "run", side_effect=run_command
            ) as run:
                result = bootstrap.bootstrap_toolchain(
                    destination,
                    tools=["msprof_analyze"],
                    python_executable="python-test",
                    repository_root=repository,
                )

            commands = [call.args[0] for call in run.call_args_list]

        self.assertIn(
            [
                "python-test",
                "-m",
                "pip",
                "install",
                "-r",
                str(requirements.resolve()),
            ],
            commands,
        )
        self.assertIn(
            [
                "python-test",
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--editable",
                ".",
            ],
            commands,
        )
        self.assertEqual(
            result["tools"]["msprof-analyze"]["source"]["commit"], "def456"
        )
        for call in run.call_args_list:
            self.assertIs(call.kwargs["shell"], False)

    def test_resolved_manifest_merges_tools_instead_of_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            repository = base / "repo"
            repository.mkdir()
            destination = (base / "external").resolve()
            destination.mkdir()
            lock = toolchain.load_toolchain_lock()
            lock_sha256 = hashlib.sha256(toolchain.LOCK_PATH.read_bytes()).hexdigest()
            previous = {
                "schema_version": 1,
                "status": "completed",
                "profile_name": lock["profile_name"],
                "lock_sha256": lock_sha256,
                "dry_run": False,
                "destination": str(destination),
                "python_executable": "python-test",
                "selected_tools": ["msinsight"],
                "updated_tools": ["msinsight"],
                "tools": {
                    "msinsight": {
                        "source": {"commit": "old"},
                        "build": {"strategy": "source-only"},
                    }
                },
                "commands": [{"command": ["old"], "executed": True}],
            }
            resolved_path = destination / "toolchain.resolved.json"
            resolved_path.write_text(json.dumps(previous), encoding="utf-8")
            with (
                mock.patch.object(
                    bootstrap,
                    "_sync_source",
                    return_value={"commit": "new"},
                ),
                mock.patch.object(
                    bootstrap,
                    "_build_tool",
                    return_value={"strategy": "build-wheel"},
                ),
            ):
                result = bootstrap.bootstrap_toolchain(
                    destination,
                    tools=["msprobe"],
                    python_executable="python-test",
                    repository_root=repository,
                )
            written = json.loads(resolved_path.read_text(encoding="utf-8"))

            self.assertEqual(set(result["tools"]), {"msinsight", "msprobe"})
            self.assertEqual(
                set(written["selected_tools"]), {"msinsight", "msprobe"}
            )
            self.assertEqual(written["updated_tools"], ["msprobe"])
            self.assertEqual(written["commands"], previous["commands"])
            self.assertEqual(written["status"], "completed")
            self.assertFalse(resolved_path.with_suffix(".json.tmp").exists())

    def test_failed_bootstrap_does_not_write_a_success_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            repository = base / "repo"
            repository.mkdir()
            destination = base / "external"
            with mock.patch.object(
                bootstrap,
                "_sync_source",
                side_effect=RuntimeError("sync failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "sync failed"):
                    bootstrap.bootstrap_toolchain(
                        destination,
                        tools=["msprobe"],
                        python_executable="python-test",
                        repository_root=repository,
                    )

            self.assertFalse((destination / "toolchain.resolved.json").exists())

    def test_existing_checkout_with_wrong_origin_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            repository = base / "repo"
            repository.mkdir()
            destination = base / "external"
            (destination / "msprobe" / ".git").mkdir(parents=True)
            remote = subprocess.CompletedProcess(
                ["git"], 0, "https://example.invalid/msprobe.git\n", ""
            )
            with mock.patch.object(
                bootstrap.subprocess, "run", return_value=remote
            ):
                with self.assertRaisesRegex(ValueError, "unexpected origin"):
                    bootstrap.bootstrap_toolchain(
                        destination,
                        tools=["msprobe"],
                        repository_root=repository,
                    )

    def test_clean_local_branch_ahead_of_official_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            repository = base / "repo"
            repository.mkdir()
            destination = base / "external"
            source = destination / "msprobe"
            (source / ".git").mkdir(parents=True)

            def run_command(command, **_kwargs):
                if command[-3:] == ["remote", "get-url", "origin"]:
                    stdout = "https://gitcode.com/Ascend/msprobe.git\n"
                elif command[-2:] == ["rev-parse", "HEAD"]:
                    stdout = "local-ahead\n"
                elif command[-2:] == ["rev-parse", "origin/master"]:
                    stdout = "official-head\n"
                else:
                    stdout = ""
                return subprocess.CompletedProcess(command, 0, stdout, "")

            with (
                mock.patch.object(
                    bootstrap.subprocess,
                    "run",
                    side_effect=run_command,
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "refusing non-official branch state",
                ),
            ):
                bootstrap.bootstrap_toolchain(
                    destination,
                    tools=["msprobe"],
                    repository_root=repository,
                )


if __name__ == "__main__":
    unittest.main()
