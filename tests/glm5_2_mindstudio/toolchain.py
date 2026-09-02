# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inspect the external MindStudio toolchain used by GLM-5.2 workflows."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence


LOCK_PATH = Path(__file__).with_name("toolchain.lock.json")
TOOLCHAIN_ROOT_ENV = "TORCHTITAN_MINDSTUDIO_TOOLCHAIN_ROOT"
DOCTOR_SCOPES = (
    "full",
    "accuracy-gpu",
    "accuracy-npu",
    "compare",
    "performance",
    "performance-capture",
    "performance-torch-npu-capture",
    "performance-analysis",
)

_DOCTOR_SCOPE_ALIASES = {
    "full": "full",
    "accuracy-gpu": "accuracy-gpu",
    "accuracy_gpu": "accuracy-gpu",
    "gpu-accuracy": "accuracy-gpu",
    "accuracy-npu": "accuracy-npu",
    "accuracy_npu": "accuracy-npu",
    "npu-accuracy": "accuracy-npu",
    "compare": "compare",
    "offline-compare": "compare",
    "performance": "performance",
    "profiler": "performance",
    "performance-capture": "performance-capture",
    "performance_capture": "performance-capture",
    "profiler-capture": "performance-capture",
    "performance-torch-npu-capture": "performance-torch-npu-capture",
    "performance_torch_npu_capture": "performance-torch-npu-capture",
    "torch-npu-profiler-capture": "performance-torch-npu-capture",
    "torch_npu_profiler_capture": "performance-torch-npu-capture",
    "performance-analysis": "performance-analysis",
    "performance_analysis": "performance-analysis",
    "offline-analysis": "performance-analysis",
}

_TOOL_ALIASES = {
    "mindstudio-probe": "msprobe",
    "msprobe": "msprobe",
    "msprof": "msprof",
    "msprof_analyze": "msprof-analyze",
    "msprof-analyze": "msprof-analyze",
    "mindstudio-insight": "msinsight",
    "msinsight": "msinsight",
}

_CANN_ENVIRONMENT_VARIABLES = (
    "ASCEND_HOME_PATH",
    "ASCEND_TOOLKIT_HOME",
    "ASCEND_OPP_PATH",
    "ASCEND_AICPU_PATH",
)


def load_toolchain_lock(path: Path | str | None = None) -> dict[str, Any]:
    """Load and minimally validate the repository-owned toolchain lock."""

    lock_path = Path(path).resolve() if path is not None else LOCK_PATH
    data = json.loads(lock_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(
            f"unsupported MindStudio toolchain schema: "
            f"{data.get('schema_version')!r}"
        )
    if not isinstance(data.get("profile_name"), str):
        raise ValueError("toolchain lock must define profile_name")
    tools = data.get("tools")
    if not isinstance(tools, dict) or not tools:
        raise ValueError("toolchain lock must define tools")
    for name, entry in tools.items():
        if not isinstance(entry, dict):
            raise ValueError(f"tool entry must be an object: {name}")
        source = entry.get("source")
        if not isinstance(source, dict):
            raise ValueError(f"tool source must be an object: {name}")
        if not source.get("url") or not source.get("directory"):
            raise ValueError(f"tool source is incomplete: {name}")
        if source.get("path") not in {None, source["directory"]}:
            raise ValueError(f"tool source path and directory disagree: {name}")
    return data


def canonical_tool_name(name: str) -> str:
    """Return the lock key for a supported tool name or alias."""

    try:
        return _TOOL_ALIASES[name.strip().lower()]
    except KeyError as error:
        raise KeyError(f"unknown MindStudio tool: {name}") from error


def canonical_doctor_scope(scope: str) -> str:
    """Normalize a doctor readiness scope while retaining stable JSON names."""

    try:
        return _DOCTOR_SCOPE_ALIASES[scope.strip().lower()]
    except KeyError as error:
        raise ValueError(
            f"unknown MindStudio doctor scope {scope!r}; "
            f"choose one of {', '.join(DOCTOR_SCOPES)}"
        ) from error


def resolve_toolchain_root(
    root: Path | str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    """Resolve an explicit or environment-configured external source root."""

    environment = os.environ if environ is None else environ
    value: Path | str | None = root
    if value is None:
        value = environment.get(TOOLCHAIN_ROOT_ENV)
    if value is None or not str(value).strip():
        return None
    return Path(value).expanduser().resolve()


def tool_source_path(
    name: str,
    root: Path | str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    """Resolve a tool checkout path without requiring the path to exist."""

    toolchain_root = resolve_toolchain_root(root, environ=environ)
    if toolchain_root is None:
        return None
    tool_name = canonical_tool_name(name)
    lock = load_toolchain_lock()
    try:
        directory = lock["tools"][tool_name]["source"]["directory"]
    except KeyError as error:
        raise KeyError(f"tool is not present in the lock: {tool_name}") from error
    source_path = (toolchain_root / str(directory)).resolve()
    try:
        source_path.relative_to(toolchain_root)
    except ValueError as error:
        raise ValueError(
            f"tool source directory escapes toolchain root: {directory}"
        ) from error
    return source_path


def toolchain_paths(
    root: Path | str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str | None]:
    """Return JSON-ready source and script paths for workflow manifests."""

    lock = load_toolchain_lock()
    paths: dict[str, str | None] = {}
    for name in lock["tools"]:
        path = tool_source_path(name, root, environ=environ)
        paths[name] = str(path) if path is not None else None
    msinsight = tool_source_path("msinsight", root, environ=environ)
    flamegraph = (
        msinsight / "scripts" / "flame_graph" / "flamegraph.py"
        if msinsight is not None
        else None
    )
    paths["msinsight_flamegraph"] = (
        str(flamegraph) if flamegraph is not None else None
    )
    return paths


def _distribution_version(
    distributions: Sequence[str],
) -> tuple[str | None, str | None, str | None]:
    errors: list[str] = []
    for distribution in distributions:
        try:
            return distribution, importlib_metadata.version(distribution), None
        except importlib_metadata.PackageNotFoundError:
            continue
        except Exception as error:  # pragma: no cover - defensive metadata API
            errors.append(f"{distribution}: {error!r}")
    return None, None, "; ".join(errors) or None


def _file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"size": path.stat().st_size, "sha256": digest.hexdigest()}


def _tree_identity(
    entries: Sequence[tuple[str, int, str]],
) -> dict[str, Any]:
    normalized = sorted(entries)
    digest = hashlib.sha256()
    for relative_path, size, sha256 in normalized:
        digest.update(
            json.dumps(
                {
                    "path": relative_path,
                    "size": size,
                    "sha256": sha256,
                },
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return {
        "file_count": len(normalized),
        "total_size": sum(size for _, size, _ in normalized),
        "sha256": digest.hexdigest(),
    }


def _package_tree_identity(origin: Path, module: str) -> dict[str, Any]:
    package_name = module.split(".", 1)[0]
    if origin.is_dir():
        package_root = origin
    elif origin.name == "__init__.py":
        package_root = origin.parent
    else:
        package_root = origin
    files = (
        [package_root]
        if package_root.is_file()
        else sorted(package_root.rglob("*"))
    )
    entries: list[tuple[str, int, str]] = []
    for path in files:
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        relative = (
            path.name
            if package_root.is_file()
            else f"{package_name}/{path.relative_to(package_root).as_posix()}"
        )
        identity = _file_identity(path)
        entries.append((relative, identity["size"], identity["sha256"]))
    return {
        "root": str(package_root.resolve()),
        **_tree_identity(entries),
    }


def _distribution_installation_identity(
    distribution_name: str,
    *,
    module: str,
    origin: Path | None,
) -> dict[str, Any]:
    """Hash the installed package tree and its installer RECORD."""

    try:
        distribution = importlib_metadata.distribution(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return {
            "ok": False,
            "distribution": distribution_name,
            "error": "distribution metadata is missing",
        }
    except Exception as error:  # pragma: no cover - defensive metadata API
        return {
            "ok": False,
            "distribution": distribution_name,
            "error": repr(error),
        }
    files = distribution.files
    if files is None:
        return {
            "ok": False,
            "distribution": distribution_name,
            "error": "distribution file inventory is missing",
        }
    record: dict[str, Any] | None = None
    installed_paths: set[Path] = set()
    console_scripts: list[dict[str, Any]] = []
    entry_point_names = {
        entry_point.name.lower()
        for entry_point in distribution.entry_points
        if entry_point.group == "console_scripts"
    }
    for entry in files:
        relative = Path(str(entry).replace("\\", "/"))
        path = Path(distribution.locate_file(entry)).resolve()
        if path.is_file():
            installed_paths.add(path)
        if (
            relative.name == "RECORD"
            and relative.parent.name.endswith(".dist-info")
            and path.is_file()
        ):
            record = {"path": str(path), **_file_identity(path)}
        base_name = relative.name.lower()
        if any(
            base_name in {name, f"{name}.exe", f"{name}-script.py"}
            for name in entry_point_names
        ) and path.is_file():
            console_scripts.append(
                {
                    "name": relative.name,
                    "path": str(path),
                    **_file_identity(path),
                }
            )
    recorded_script_paths = {item["path"] for item in console_scripts}
    scripts_root = Path(sysconfig.get_path("scripts")).resolve()
    for name in entry_point_names:
        for candidate_name in (name, f"{name}.exe", f"{name}-script.py"):
            candidate = (scripts_root / candidate_name).resolve()
            if candidate.is_file() and str(candidate) not in recorded_script_paths:
                console_scripts.append(
                    {
                        "name": candidate.name,
                        "path": str(candidate),
                        **_file_identity(candidate),
                    }
                )
    origin_in_distribution = bool(
        origin is not None and origin.resolve() in installed_paths
    )
    if origin is None or not origin.exists():
        package_tree = None
        package_error = "import origin is missing"
    else:
        try:
            package_tree = _package_tree_identity(origin.resolve(), module)
            package_error = None
        except OSError as error:
            package_tree = None
            package_error = repr(error)
    errors: list[str] = []
    if record is None:
        errors.append("installer RECORD is missing")
    if not origin_in_distribution:
        errors.append("import origin is not owned by the selected distribution")
    if package_tree is None:
        errors.append(package_error or "package tree could not be hashed")
    return {
        "ok": not errors,
        "distribution": distribution_name,
        "record": record,
        "package_tree": package_tree,
        "origin_in_distribution": origin_in_distribution,
        "console_scripts": sorted(
            console_scripts,
            key=lambda item: (item["name"], item["path"]),
        ),
        "error": "; ".join(errors) or None,
    }


def python_package_metadata(
    module: str,
    distributions: Sequence[str],
) -> dict[str, Any]:
    """Inspect import and distribution metadata without importing the module."""

    error: str | None = None
    origin: str | None = None
    try:
        spec = importlib.util.find_spec(module)
    except Exception as import_error:  # pragma: no cover - broken import hooks
        spec = None
        error = repr(import_error)
    if spec is not None:
        if spec.origin and spec.origin not in {"built-in", "frozen"}:
            origin = str(Path(spec.origin).resolve())
        elif spec.submodule_search_locations:
            locations = list(spec.submodule_search_locations)
            origin = str(Path(locations[0]).resolve()) if locations else None
    distribution, version, version_error = _distribution_version(distributions)
    installation_identity = (
        _distribution_installation_identity(
            distribution,
            module=module,
            origin=Path(origin) if origin is not None else None,
        )
        if distribution is not None and module == "msprobe"
        else None
    )
    return {
        "module": module,
        "importable": spec is not None,
        "ok": spec is not None,
        "origin": origin,
        "distribution": distribution,
        "version": version,
        "installation_identity": installation_identity,
        "error": error or version_error,
    }


def _bounded_output(value: str | None, limit: int = 4096) -> str:
    text = value or ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... output truncated ..."


def _resolve_executable(
    command: str,
    environment_variable: str | None,
    environ: Mapping[str, str],
) -> tuple[str | None, str | None]:
    configured = (
        environ.get(environment_variable) if environment_variable else None
    )
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            return None, f"{environment_variable} is not executable: {candidate}"
        return str(candidate), None
    executable = shutil.which(command, path=environ.get("PATH"))
    if executable is None:
        return None, f"{command} is not on PATH"
    return str(Path(executable).resolve()), None


def resolve_tool_executable(
    command: str,
    *,
    environment_variable: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve one configured official CLI with the same rule as doctor."""

    environment = os.environ if environ is None else environ
    path, error = _resolve_executable(
        command,
        environment_variable,
        environment,
    )
    if path is None:
        raise RuntimeError(error or f"could not resolve {command}")
    return path


def _executable_status(
    command: str,
    probe_arguments: Sequence[str],
    *,
    environment_variable: str | None,
    required: bool,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    path, error = _resolve_executable(command, environment_variable, environ)
    status: dict[str, Any] = {
        "required": required,
        "ok": False,
        "command": command,
        "environment_variable": environment_variable,
        "path": path,
        "identity": None,
        "identity_error": None,
        "probe": None,
        "return_code": None,
        "reported_version": None,
        "stdout": "",
        "stderr": "",
        "error": error,
    }
    if path is None:
        return status
    try:
        status["identity"] = _file_identity(Path(path))
    except OSError as identity_error:
        status["identity_error"] = repr(identity_error)
    probe = [path, *probe_arguments]
    status["probe"] = probe
    try:
        result = subprocess.run(
            probe,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as probe_error:
        status["error"] = repr(probe_error)
        return status
    stdout = _bounded_output(result.stdout)
    stderr = _bounded_output(result.stderr)
    first_line = next(
        (
            line.strip()
            for line in (stdout or stderr).splitlines()
            if line.strip()
        ),
        None,
    )
    status.update(
        {
            "ok": result.returncode == 0,
            "return_code": result.returncode,
            "reported_version": first_line,
            "stdout": stdout,
            "stderr": stderr,
            "error": None if result.returncode == 0 else "probe failed",
        }
    )
    return status


def _parse_version_file(path: Path) -> dict[str, str]:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    values: dict[str, str] = {}
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        separator = "=" if "=" in stripped else ":" if ":" in stripped else None
        if separator is None:
            continue
        key, value = (part.strip() for part in stripped.split(separator, 1))
        if "version" in key.lower() or key.lower() in {
            "package_name",
            "package",
        }:
            values[key] = value
    return values


def _cann_version_paths(root: Path) -> list[Path]:
    candidates = [
        root / "version.info",
        root / "compiler" / "version.info",
        root / "ascend_toolkit_install.info",
    ]
    try:
        candidates.extend(root.glob("*-linux/ascend_toolkit_install.info"))
    except OSError:
        pass
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen and resolved.is_file():
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _cann_status(
    environ: Mapping[str, str], *, required: bool = True
) -> dict[str, Any]:
    variables = {
        name: environ[name]
        for name in _CANN_ENVIRONMENT_VARIABLES
        if environ.get(name)
    }
    roots: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for name, value in variables.items():
        path = Path(value).expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        version_files = [
            {"path": str(candidate), "values": _parse_version_file(candidate)}
            for candidate in _cann_version_paths(path)
        ]
        roots.append(
            {
                "environment_variable": name,
                "path": str(path),
                "exists": path.exists(),
                "version_files": version_files,
            }
        )
    ok = bool(variables) and any(root["exists"] for root in roots)
    return {
        "required": required,
        "ok": ok,
        "environment": variables,
        "roots": roots,
        "error": None if ok else "no existing CANN environment root was found",
    }


def _git_value(source: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(source), *arguments],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def tool_source_metadata(
    name: str,
    root: Path | str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return lock selection and resolved Git metadata for one checkout."""

    tool_name = canonical_tool_name(name)
    lock_entry = load_toolchain_lock()["tools"][tool_name]
    path = tool_source_path(tool_name, root, environ=environ)
    result: dict[str, Any] = {
        "name": tool_name,
        "official_url": lock_entry["source"]["url"],
        "selected_ref": lock_entry["source"].get("ref"),
        "locked_commit": lock_entry["source"].get("commit"),
        "locked_version": lock_entry["source"].get("version"),
        "path": str(path) if path is not None else None,
        "available": bool(path is not None and path.is_dir()),
        "commit": None,
        "branch": None,
        "tag": None,
        "dirty": None,
    }
    if path is None or not (path / ".git").exists():
        return result
    status = _git_value(path, "status", "--porcelain")
    result.update(
        {
            "commit": _git_value(path, "rev-parse", "HEAD"),
            "branch": _git_value(path, "branch", "--show-current"),
            "tag": _git_value(path, "describe", "--tags", "--exact-match", "HEAD"),
            "dirty": bool(status),
        }
    )
    return result


def _msinsight_status(
    root: Path | str | None,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    configured = environ.get("TORCHTITAN_MSINSIGHT_FLAMEGRAPH")
    source = tool_source_path("msinsight", root, environ=environ)
    if configured:
        script = Path(configured).expanduser().resolve()
        if script.is_dir():
            script = script / "flamegraph.py"
    elif source is not None:
        script = source / "scripts" / "flame_graph" / "flamegraph.py"
    else:
        script = None
    ok = bool(script is not None and script.is_file())
    return {
        "required": False,
        "ok": ok,
        "environment_variable": "TORCHTITAN_MSINSIGHT_FLAMEGRAPH",
        "path": str(script) if script is not None else None,
        "source_path": str(source) if source is not None else None,
        "error": None if ok else "optional MindStudio Insight script not found",
    }


def doctor_report(
    toolchain_root: Path | str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    scope: str = "full",
    profile: str | None = None,
) -> dict[str, Any]:
    """Return a read-only, JSON-serializable toolchain health report."""

    environment = os.environ if environ is None else environ
    lock = load_toolchain_lock()
    root = resolve_toolchain_root(toolchain_root, environ=environment)
    if profile is not None:
        profile_scope = canonical_doctor_scope(profile)
        if scope != "full" and canonical_doctor_scope(scope) != profile_scope:
            raise ValueError("doctor scope and profile aliases disagree")
        scope = profile_scope
    else:
        scope = canonical_doctor_scope(scope)
    requires_npu = scope in {
        "full",
        "accuracy-npu",
        "performance",
        "performance-capture",
        "performance-torch-npu-capture",
    }
    requires_framework = scope != "performance-analysis"
    requires_msprof = scope in {
        "full",
        "performance",
    }
    requires_msprof_analyze = scope in {
        "full",
        "performance",
        "performance-analysis",
    }
    requires_msprobe = scope not in {
        "performance",
        "performance-capture",
        "performance-torch-npu-capture",
        "performance-analysis",
    }

    msprobe_import = python_package_metadata(
        "msprobe", ("mindstudio-probe", "mindstudio_probe")
    )
    msprobe_executable = _executable_status(
        "msprobe",
        ("--help",),
        environment_variable="TORCHTITAN_MSPROBE",
        required=requires_msprobe,
        environ=environment,
    )
    torch = python_package_metadata("torch", ("torch",))
    torch_npu = python_package_metadata("torch_npu", ("torch-npu", "torch_npu"))
    msprof = _executable_status(
        "msprof",
        ("--version",),
        environment_variable="TORCHTITAN_MSPROF",
        required=requires_msprof,
        environ=environment,
    )
    msprof_analyze_import = python_package_metadata(
        "msprof_analyze", ("msprof-analyze",)
    )
    msprof_analyze = _executable_status(
        "msprof-analyze",
        ("--help",),
        environment_variable="TORCHTITAN_MSPROF_ANALYZE",
        required=requires_msprof_analyze,
        environ=environment,
    )
    npu_smi = _executable_status(
        "npu-smi",
        ("info",),
        environment_variable=None,
        required=False,
        environ=environment,
    )

    source_tools = {
        name: tool_source_metadata(name, root, environ=environment)
        for name in lock["tools"]
    }
    default_source_tools = [
        name
        for name, entry in lock["tools"].items()
        if entry.get("bootstrap", {}).get("default")
    ]
    torch["required"] = requires_framework
    torch_npu["required"] = requires_npu
    checks: dict[str, dict[str, Any]] = {
        "msprobe": {
            "required": requires_msprobe,
            "ok": msprobe_executable["ok"] and msprobe_import["ok"],
            "executable": msprobe_executable,
            "import": msprobe_import,
        },
        "framework": {
            "required": requires_framework,
            "ok": (not requires_framework)
            or (torch["ok"] and (torch_npu["ok"] or not requires_npu)),
            "torch": torch,
            "torch_npu": torch_npu,
        },
        "cann": _cann_status(environment, required=requires_npu),
        "msprof": msprof,
        "msprof-analyze": {
            "required": requires_msprof_analyze,
            "ok": msprof_analyze["ok"] and msprof_analyze_import["ok"],
            "executable": msprof_analyze,
            "import": msprof_analyze_import,
        },
        "msinsight": _msinsight_status(root, environment),
        "npu-smi": npu_smi,
        "source_checkouts": {
            "required": False,
            "ok": bool(root is not None)
            and all(source_tools[name]["available"] for name in default_source_tools),
            "tools": source_tools,
        },
    }
    required_failures = [
        name
        for name, check in checks.items()
        if check.get("required") and not check.get("ok")
    ]
    return {
        "schema_version": 1,
        "profile_name": lock["profile_name"],
        "scope": scope,
        "ok": not required_failures,
        "required_failures": required_failures,
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "toolchain_root": str(root) if root is not None else None,
        "checks": checks,
    }


def _wheel_package_identity(path: Path, module: str) -> dict[str, Any]:
    package_name = module.split(".", 1)[0]
    entries: list[tuple[str, int, str]] = []
    record_sha256: str | None = None
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            relative = member.filename.replace("\\", "/")
            if member.is_dir():
                continue
            if relative.endswith(".dist-info/RECORD"):
                record_sha256 = hashlib.sha256(archive.read(member)).hexdigest()
            if not (
                relative == f"{package_name}.py"
                or relative.startswith(f"{package_name}/")
            ):
                continue
            relative_path = Path(relative)
            if "__pycache__" in relative_path.parts or relative_path.suffix in {
                ".pyc",
                ".pyo",
            }:
                continue
            payload = archive.read(member)
            entries.append(
                (
                    relative,
                    len(payload),
                    hashlib.sha256(payload).hexdigest(),
                )
            )
    if not entries:
        raise ValueError(f"wheel contains no {package_name} package files: {path}")
    if record_sha256 is None:
        raise ValueError(f"wheel contains no installer RECORD: {path}")
    return {
        "package_tree": _tree_identity(entries),
        "record_sha256": record_sha256,
    }


def _portable_tree_identity(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        key: value.get(key)
        for key in ("file_count", "total_size", "sha256")
    }


def _resolved_executable_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        return candidate.resolve()
    executable = shutil.which(value)
    return Path(executable).resolve() if executable is not None else None


def _validate_resolved_toolchain(
    resolved_path: Path,
    *,
    resolved_root: Path,
    lock: dict[str, Any],
    lock_sha256: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Validate a bootstrap result against the code that will actually run."""

    resolved_bytes = resolved_path.read_bytes()
    resolved_data = json.loads(resolved_bytes)
    errors: list[str] = []
    if not isinstance(resolved_data, dict):
        raise ValueError("MindStudio resolved manifest must be a JSON object")
    if resolved_data.get("schema_version") != 1:
        errors.append("schema_version does not match")
    if resolved_data.get("status") != "completed":
        errors.append("status is not completed")
    if resolved_data.get("dry_run") is not False:
        errors.append("dry_run manifest cannot identify an installed toolchain")
    if resolved_data.get("profile_name") != lock["profile_name"]:
        errors.append("profile_name does not match the current lock")
    if resolved_data.get("lock_sha256") != lock_sha256:
        errors.append("lock_sha256 does not match the current lock")
    destination = resolved_data.get("destination")
    if not isinstance(destination, str) or Path(destination).resolve() != resolved_root:
        errors.append("destination does not match the configured toolchain root")

    checks = report.get("checks", {})
    source_checkouts = checks.get("source_checkouts", {})
    current_sources = source_checkouts.get("tools", {})
    tools = resolved_data.get("tools")
    if not isinstance(tools, dict):
        errors.append("tools must be an object")
        tools = {}
    for name, result in tools.items():
        if name not in lock["tools"]:
            errors.append(f"resolved tool is absent from the current lock: {name}")
            continue
        if not isinstance(result, dict):
            errors.append(f"resolved tool result is not an object: {name}")
            continue
        resolved_source = result.get("source")
        current_source = (
            current_sources.get(name) if isinstance(current_sources, dict) else None
        )
        if not isinstance(resolved_source, dict) or not isinstance(
            current_source, dict
        ):
            errors.append(f"source metadata is missing for {name}")
            continue
        expected_source = lock["tools"][name]["source"]
        if resolved_source.get("official_url") != expected_source.get("url"):
            errors.append(f"official source URL does not match for {name}")
        if resolved_source.get("ref") != expected_source.get("ref"):
            errors.append(f"selected source ref does not match for {name}")
        resolved_source_path = resolved_source.get("path")
        current_source_path = current_source.get("path")
        if (
            not isinstance(resolved_source_path, str)
            or not isinstance(current_source_path, str)
            or Path(resolved_source_path).resolve()
            != Path(current_source_path).resolve()
        ):
            errors.append(f"source checkout path does not match for {name}")
        if not current_source.get("available"):
            errors.append(f"source checkout is unavailable for {name}")
        if not resolved_source.get("commit") or (
            resolved_source.get("commit") != current_source.get("commit")
        ):
            errors.append(f"source checkout commit does not match for {name}")

    msprobe_required = bool(checks.get("msprobe", {}).get("required"))
    artifact_identity: dict[str, Any] | None = None
    installation_identity: dict[str, Any] | None = None
    executable_identity: dict[str, Any] | None = None
    msprobe_source_commit: str | None = None
    if msprobe_required:
        msprobe_result = tools.get("msprobe")
        if not isinstance(msprobe_result, dict):
            errors.append("resolved manifest has no msprobe result")
        else:
            source = msprobe_result.get("source")
            build = msprobe_result.get("build")
            if isinstance(source, dict):
                msprobe_source_commit = source.get("commit")
            if not isinstance(build, dict) or build.get("strategy") != "build-wheel":
                errors.append("resolved msprobe build is not a wheel build")
            else:
                artifact = build.get("artifact")
                recorded_artifact = build.get("artifact_identity")
                source_path = source.get("path") if isinstance(source, dict) else None
                if not isinstance(artifact, str) or not isinstance(
                    recorded_artifact, dict
                ):
                    errors.append("resolved msprobe wheel identity is missing")
                else:
                    artifact_path = Path(artifact).resolve()
                    if not artifact_path.is_file():
                        errors.append("resolved msprobe wheel is missing")
                    elif not isinstance(
                        source_path, str
                    ) or not artifact_path.is_relative_to(Path(source_path).resolve()):
                        errors.append("resolved msprobe wheel escapes its checkout")
                    else:
                        actual_artifact = _file_identity(artifact_path)
                        if actual_artifact != recorded_artifact:
                            errors.append("resolved msprobe wheel hash does not match")
                        try:
                            wheel_identity = _wheel_package_identity(
                                artifact_path, "msprobe"
                            )
                        except (OSError, ValueError, zipfile.BadZipFile) as error:
                            errors.append(f"resolved msprobe wheel is invalid: {error}")
                        else:
                            artifact_identity = {
                                **actual_artifact,
                                **wheel_identity,
                            }

        expected_python = _resolved_executable_path(
            resolved_data.get("python_executable")
        )
        current_python = Path(report["runtime"]["python_executable"]).resolve()
        if expected_python is None or expected_python != current_python:
            errors.append("bootstrap Python does not match the current interpreter")

        msprobe = checks.get("msprobe", {})
        imported = msprobe.get("import", {}) if isinstance(msprobe, dict) else {}
        executable = (
            msprobe.get("executable", {}) if isinstance(msprobe, dict) else {}
        )
        candidate_installation = (
            imported.get("installation_identity")
            if isinstance(imported, dict)
            else None
        )
        if not isinstance(
            candidate_installation, dict
        ) or not candidate_installation.get("ok"):
            errors.append("current msprobe installation identity is unavailable")
        else:
            installation_identity = candidate_installation
            installed_tree = _portable_tree_identity(
                candidate_installation.get("package_tree")
            )
            wheel_tree = (
                _portable_tree_identity(artifact_identity.get("package_tree"))
                if artifact_identity is not None
                else None
            )
            if installed_tree is None or installed_tree != wheel_tree:
                errors.append(
                    "current msprobe package tree does not match the resolved wheel"
                )
            executable_path = executable.get("path")
            executable_identity = executable.get("identity")
            matching_scripts = [
                script
                for script in candidate_installation.get("console_scripts", [])
                if isinstance(script, dict)
                and script.get("path") == executable_path
                and {
                    "size": script.get("size"),
                    "sha256": script.get("sha256"),
                }
                == executable_identity
            ]
            if not matching_scripts:
                errors.append(
                    "current msprobe CLI is not owned by the imported distribution"
                )

    return {
        "manifest_sha256": hashlib.sha256(resolved_bytes).hexdigest(),
        "profile_name": resolved_data.get("profile_name"),
        "lock_sha256": resolved_data.get("lock_sha256"),
        "valid": not errors,
        "validation_errors": errors,
        "msprobe_source_commit": msprobe_source_commit,
        "msprobe_artifact_identity": artifact_identity,
        "msprobe_installation_identity": installation_identity,
        "msprobe_executable_identity": executable_identity,
    }


def collect_toolchain_metadata(
    toolchain_root: Path | str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    scope: str = "full",
    profile: str | None = None,
) -> dict[str, Any]:
    """Collect lock, path, version, and doctor metadata for run manifests."""

    lock_bytes = LOCK_PATH.read_bytes()
    lock_sha256 = hashlib.sha256(lock_bytes).hexdigest()
    lock = load_toolchain_lock()
    report = doctor_report(
        toolchain_root,
        environ=environ,
        scope=scope,
        profile=profile,
    )
    checks = report["checks"]
    resolved_root = resolve_toolchain_root(toolchain_root, environ=environ)
    resolved_toolchain: dict[str, Any] | None = None
    if resolved_root is not None:
        resolved_path = resolved_root / "toolchain.resolved.json"
        if resolved_path.is_file():
            resolved_toolchain = _validate_resolved_toolchain(
                resolved_path,
                resolved_root=resolved_root,
                lock=lock,
                lock_sha256=lock_sha256,
                report=report,
            )
            if not resolved_toolchain["valid"]:
                raise ValueError(
                    "MindStudio resolved manifest does not match the current "
                    "toolchain: "
                    + "; ".join(resolved_toolchain["validation_errors"])
                )
    versions = {
        "torch": checks["framework"]["torch"].get("version"),
        "torch_npu": checks["framework"]["torch_npu"].get("version"),
        "msprobe": checks["msprobe"]["import"].get("version"),
        "msprof": checks["msprof"].get("reported_version"),
        "msprof-analyze": checks["msprof-analyze"]["import"].get("version"),
    }
    return {
        "schema_version": 1,
        "profile_name": report["profile_name"],
        "scope": report["scope"],
        "lock": {
            "path": str(LOCK_PATH.resolve()),
            "sha256": lock_sha256,
        },
        "toolchain_root": report["toolchain_root"],
        "compatibility": lock["compatibility"],
        "paths": toolchain_paths(toolchain_root, environ=environ),
        "sources": checks["source_checkouts"]["tools"],
        "versions": versions,
        "installations": {
            "msprobe": checks["msprobe"]["import"].get(
                "installation_identity"
            ),
        },
        "doctor": report,
        "resolved_toolchain": resolved_toolchain,
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect the GLM-5.2 MindStudio toolchain"
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("doctor", "metadata"),
        default="doctor",
    )
    parser.add_argument(
        "--scope",
        "--profile",
        dest="doctor_scope",
        choices=DOCTOR_SCOPES,
        default="full",
        help="required readiness scope; all checks are always reported",
    )
    parser.add_argument(
        "--toolchain-root",
        type=Path,
        help=f"external source root (or set {TOOLCHAIN_ROOT_ENV})",
    )
    parser.add_argument("--compact", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    if args.command == "metadata":
        result = collect_toolchain_metadata(
            args.toolchain_root, scope=args.doctor_scope
        )
        exit_code = 0
    else:
        result = doctor_report(args.toolchain_root, scope=args.doctor_scope)
        exit_code = 0 if result["ok"] else 1
    print(
        json.dumps(
            result,
            ensure_ascii=True,
            indent=None if args.compact else 2,
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
