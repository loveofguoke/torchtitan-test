# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bootstrap reviewed MindStudio source tools outside this repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tests.glm5_2_mindstudio.toolchain import (  # noqa: E402
    LOCK_PATH,
    canonical_tool_name,
    load_toolchain_lock,
    tool_source_path,
)


def validate_destination(
    destination: Path | str,
    *,
    repository_root: Path | str = REPOSITORY_ROOT,
) -> Path:
    """Resolve an external destination and reject repository descendants."""

    root = Path(destination).expanduser().resolve()
    repository = Path(repository_root).resolve()
    if root == repository or repository in root.parents:
        raise ValueError(
            "MindStudio tool sources must be stored outside the repository: "
            f"{root}"
        )
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(f"toolchain destination is not a directory: {root}")
    return root


def _display_command(command: Sequence[str]) -> str:
    return shlex.join(str(part) for part in command)


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path | None,
    dry_run: bool,
    records: list[dict[str, Any]],
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    normalized = [str(part) for part in command]
    record: dict[str, Any] = {
        "command": normalized,
        "cwd": str(cwd) if cwd is not None else None,
        "executed": not dry_run,
    }
    records.append(record)
    print(f"+ {_display_command(normalized)}")
    if dry_run:
        return subprocess.CompletedProcess(normalized, 0, "", "")
    result = subprocess.run(
        normalized,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        check=False,
        shell=False,
    )
    record["return_code"] = result.returncode
    if capture_output:
        record["stdout"] = result.stdout or ""
        record["stderr"] = result.stderr or ""
    if result.returncode:
        raise subprocess.CalledProcessError(
            result.returncode,
            normalized,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result


def _normalized_git_url(value: str) -> str:
    normalized = value.strip().replace("\\", "/").rstrip("/")
    if normalized.lower().endswith(".git"):
        normalized = normalized[:-4]
    return normalized.lower()


def _source_ref(entry: dict[str, Any]) -> tuple[str, str]:
    ref = entry["source"].get("ref") or {
        "kind": "branch",
        "value": "master",
    }
    kind = str(ref.get("kind", "branch"))
    value = str(ref.get("value", "master"))
    if kind not in {"branch", "tag", "commit"}:
        raise ValueError(f"unsupported source ref kind: {kind}")
    if not value:
        raise ValueError("source ref value must not be empty")
    return kind, value


def _sync_source(
    name: str,
    entry: dict[str, Any],
    source: Path,
    *,
    dry_run: bool,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    url = str(entry["source"]["url"])
    ref_kind, ref_value = _source_ref(entry)
    existed = source.exists()
    if not existed:
        clone = ["git", "clone"]
        if ref_kind in {"branch", "tag"}:
            clone.extend(["--branch", ref_value, "--single-branch"])
        elif ref_kind == "commit":
            clone.append("--no-checkout")
        clone.extend([url, str(source)])
        _run_command(
            clone,
            cwd=source.parent,
            dry_run=dry_run,
            records=records,
        )
        if ref_kind == "commit":
            _run_command(
                ["git", "-C", str(source), "fetch", "origin", ref_value],
                cwd=None,
                dry_run=dry_run,
                records=records,
            )
            _run_command(
                ["git", "-C", str(source), "checkout", "--detach", ref_value],
                cwd=None,
                dry_run=dry_run,
                records=records,
            )
    else:
        if not (source / ".git").exists():
            raise ValueError(f"existing tool path is not a Git checkout: {source}")
        if not dry_run:
            remote = _run_command(
                ["git", "-C", str(source), "remote", "get-url", "origin"],
                cwd=None,
                dry_run=False,
                records=records,
                capture_output=True,
            ).stdout.strip()
            if _normalized_git_url(remote) != _normalized_git_url(url):
                raise ValueError(
                    f"unexpected origin for {name}: {remote!r}; expected {url!r}"
                )
            status = _run_command(
                ["git", "-C", str(source), "status", "--porcelain"],
                cwd=None,
                dry_run=False,
                records=records,
                capture_output=True,
            ).stdout.strip()
            if status:
                raise RuntimeError(
                    f"refusing to update dirty tool checkout: {source}"
                )
        if ref_kind == "branch":
            _run_command(
                ["git", "-C", str(source), "fetch", "origin"],
                cwd=None,
                dry_run=dry_run,
                records=records,
            )
            _run_command(
                ["git", "-C", str(source), "checkout", ref_value],
                cwd=None,
                dry_run=dry_run,
                records=records,
            )
            _run_command(
                [
                    "git",
                    "-C",
                    str(source),
                    "merge",
                    "--ff-only",
                    f"origin/{ref_value}",
                ],
                cwd=None,
                dry_run=dry_run,
                records=records,
            )
        elif ref_kind == "tag":
            _run_command(
                ["git", "-C", str(source), "fetch", "--tags", "origin"],
                cwd=None,
                dry_run=dry_run,
                records=records,
            )
            _run_command(
                ["git", "-C", str(source), "checkout", "--detach", ref_value],
                cwd=None,
                dry_run=dry_run,
                records=records,
            )
        else:
            _run_command(
                ["git", "-C", str(source), "fetch", "origin", ref_value],
                cwd=None,
                dry_run=dry_run,
                records=records,
            )
            _run_command(
                ["git", "-C", str(source), "checkout", "--detach", ref_value],
                cwd=None,
                dry_run=dry_run,
                records=records,
            )

    commit = None
    if not dry_run:
        commit = _run_command(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            cwd=None,
            dry_run=False,
            records=records,
            capture_output=True,
        ).stdout.strip()
        if ref_kind == "branch":
            official_commit = _run_command(
                [
                    "git",
                    "-C",
                    str(source),
                    "rev-parse",
                    f"origin/{ref_value}",
                ],
                cwd=None,
                dry_run=False,
                records=records,
                capture_output=True,
            ).stdout.strip()
            if commit != official_commit:
                raise RuntimeError(
                    f"refusing non-official branch state for {name}: "
                    f"HEAD={commit}, origin/{ref_value}={official_commit}"
                )
    return {
        "path": str(source),
        "official_url": url,
        "ref": {"kind": ref_kind, "value": ref_value},
        "existed": existed,
        "commit": commit or None,
    }


def _safe_source_child(source: Path, relative_path: str) -> Path:
    path = (source / relative_path).resolve()
    try:
        path.relative_to(source.resolve())
    except ValueError as error:
        raise ValueError(f"source path escapes checkout: {relative_path}") from error
    return path


def _artifact_signature(path: Path) -> tuple[int, str]:
    """Identify a built artifact independently of timestamp granularity."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return path.stat().st_size, digest.hexdigest()


def _build_msprobe(
    entry: dict[str, Any],
    source: Path,
    *,
    python_executable: str,
    dry_run: bool,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    bootstrap = entry["bootstrap"]
    for requirement in bootstrap.get("build_requirements", []):
        _run_command(
            [python_executable, "-m", "pip", "install", str(requirement)],
            cwd=source,
            dry_run=dry_run,
            records=records,
        )
    build_script = str(bootstrap.get("build_script", "build.py"))
    _safe_source_child(source, build_script)
    build_arguments = [
        str(value) for value in bootstrap.get("build_arguments", [])
    ]
    artifact_glob = str(
        bootstrap.get("artifact_glob", "artifacts/mindstudio_probe*.whl")
    )
    artifact_pattern = Path(artifact_glob)
    if artifact_pattern.is_absolute() or ".." in artifact_pattern.parts:
        raise ValueError(f"unsafe msprobe artifact glob: {artifact_glob}")
    previous_artifacts = (
        {
            path.resolve(): _artifact_signature(path)
            for path in source.glob(artifact_glob)
            if path.is_file()
        }
        if not dry_run
        else {}
    )
    _run_command(
        [python_executable, build_script, *build_arguments],
        cwd=source,
        dry_run=dry_run,
        records=records,
    )
    if dry_run:
        wheel: Path | str = source / artifact_glob
    else:
        candidates = [
            path.resolve()
            for path in source.glob(artifact_glob)
            if path.is_file()
        ]
        fresh_candidates = [
            path
            for path in candidates
            if previous_artifacts.get(path) != _artifact_signature(path)
        ]
        if not fresh_candidates:
            raise FileNotFoundError(
                "msprobe build did not create or update a wheel matching "
                f"{artifact_glob}; refusing to install a stale artifact"
            )
        wheel = max(
            fresh_candidates,
            key=lambda path: path.stat().st_mtime_ns,
        )
    _run_command(
        [
            python_executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            str(wheel),
        ],
        cwd=source,
        dry_run=dry_run,
        records=records,
    )
    artifact_identity = None
    if not dry_run:
        artifact_size, artifact_sha256 = _artifact_signature(Path(wheel))
        artifact_identity = {
            "size": artifact_size,
            "sha256": artifact_sha256,
        }
    return {
        "strategy": "build-wheel",
        "artifact": str(wheel),
        "artifact_identity": artifact_identity,
    }


def _build_msprof_analyze(
    entry: dict[str, Any],
    source: Path,
    *,
    python_executable: str,
    dry_run: bool,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    bootstrap = entry["bootstrap"]
    preferred = _safe_source_child(
        source, str(bootstrap.get("requirements", "requirements/build.txt"))
    )
    fallback = _safe_source_child(
        source, str(bootstrap.get("fallback_requirements", "requirements.txt"))
    )
    if dry_run:
        requirements = preferred
    elif preferred.is_file():
        requirements = preferred
    elif fallback.is_file():
        requirements = fallback
    else:
        raise FileNotFoundError(
            "msprof-analyze source has neither the declared build requirements "
            f"nor fallback requirements: {preferred}, {fallback}"
        )
    _run_command(
        [python_executable, "-m", "pip", "install", "-r", str(requirements)],
        cwd=source,
        dry_run=dry_run,
        records=records,
    )
    _run_command(
        [
            python_executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--editable",
            ".",
        ],
        cwd=source,
        dry_run=dry_run,
        records=records,
    )
    return {
        "strategy": "editable",
        "requirements": str(requirements),
    }


def _build_tool(
    name: str,
    entry: dict[str, Any],
    source: Path,
    *,
    python_executable: str,
    dry_run: bool,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    strategy = entry["bootstrap"].get("strategy")
    if name == "msprobe" and strategy == "build-wheel":
        return _build_msprobe(
            entry,
            source,
            python_executable=python_executable,
            dry_run=dry_run,
            records=records,
        )
    if name == "msprof-analyze" and strategy == "editable":
        return _build_msprof_analyze(
            entry,
            source,
            python_executable=python_executable,
            dry_run=dry_run,
            records=records,
        )
    if strategy == "source-only":
        return {"strategy": "source-only"}
    raise ValueError(f"unsupported bootstrap strategy for {name}: {strategy}")


def _selected_tools(
    lock: dict[str, Any],
    tools: Sequence[str] | None,
    *,
    include_optional: bool,
) -> list[str]:
    entries = lock["tools"]
    if tools:
        selected = [canonical_tool_name(name) for name in tools]
    else:
        selected = [
            name
            for name, entry in entries.items()
            if entry.get("bootstrap", {}).get("default")
        ]
    if include_optional:
        selected.extend(
            name
            for name, entry in entries.items()
            if entry.get("bootstrap", {}).get("enabled")
            and not entry.get("required", False)
        )
    unique: list[str] = []
    for name in selected:
        if name not in entries:
            raise KeyError(f"tool is not present in the lock: {name}")
        if not entries[name].get("bootstrap", {}).get("enabled"):
            raise ValueError(
                f"source bootstrap is disabled for {name}: "
                f"{entries[name]['bootstrap'].get('note', '')}"
            )
        if name not in unique:
            unique.append(name)
    return unique


def _load_resolved_manifest_for_merge(
    path: Path,
    *,
    root: Path,
    lock: dict[str, Any],
    lock_sha256: str,
    python_executable: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("existing resolved manifest must be a JSON object")
    expected = {
        "schema_version": 1,
        "status": "completed",
        "profile_name": lock["profile_name"],
        "lock_sha256": lock_sha256,
        "dry_run": False,
        "destination": str(root),
        "python_executable": python_executable,
    }
    mismatches = [
        key for key, value in expected.items() if data.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "existing resolved manifest cannot be merged because these fields "
            f"do not match: {', '.join(mismatches)}"
        )
    if not isinstance(data.get("tools"), dict):
        raise ValueError("existing resolved manifest tools must be an object")
    if not isinstance(data.get("commands"), list):
        raise ValueError("existing resolved manifest commands must be a list")
    return data


def _write_resolved_manifest(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def bootstrap_toolchain(
    destination: Path | str,
    *,
    tools: Sequence[str] | None = None,
    include_optional: bool = False,
    python_executable: str = sys.executable,
    dry_run: bool = False,
    repository_root: Path | str = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Clone or fast-forward selected tools, then build or expose them."""

    root = validate_destination(destination, repository_root=repository_root)
    lock = load_toolchain_lock()
    lock_sha256 = hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest()
    selected = _selected_tools(
        lock, tools, include_optional=include_optional
    )
    resolved_manifest = root / "toolchain.resolved.json"
    previous = (
        _load_resolved_manifest_for_merge(
            resolved_manifest,
            root=root,
            lock=lock,
            lock_sha256=lock_sha256,
            python_executable=python_executable,
        )
        if not dry_run
        else None
    )
    if not dry_run:
        root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for name in selected:
        entry = lock["tools"][name]
        source = tool_source_path(name, root)
        if source is None:  # pragma: no cover - explicit root is always present
            raise RuntimeError(f"could not resolve source path for {name}")
        if not dry_run:
            source.parent.mkdir(parents=True, exist_ok=True)
        sync = _sync_source(
            name,
            entry,
            source,
            dry_run=dry_run,
            records=records,
        )
        build = _build_tool(
            name,
            entry,
            source,
            python_executable=python_executable,
            dry_run=dry_run,
            records=records,
        )
        results[name] = {"source": sync, "build": build}
    previous_tools = previous["tools"] if previous is not None else {}
    merged_tools = {**previous_tools, **results}
    previous_commands = previous["commands"] if previous is not None else []
    result = {
        "schema_version": 1,
        "status": "planned" if dry_run else "completed",
        "profile_name": lock["profile_name"],
        "lock_sha256": lock_sha256,
        "dry_run": dry_run,
        "destination": str(root),
        "python_executable": python_executable,
        "selected_tools": list(merged_tools),
        "updated_tools": selected,
        "tools": merged_tools,
        "commands": [*previous_commands, *records],
    }
    if not dry_run:
        result["resolved_manifest"] = str(resolved_manifest)
        _write_resolved_manifest(resolved_manifest, result)
    return result


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bootstrap MindStudio source tools outside torchtitan-test"
    )
    parser.add_argument(
        "--root",
        "--destination",
        "--toolchain-root",
        dest="destination",
        type=Path,
        required=True,
        help="external directory that will contain official source checkouts",
    )
    parser.add_argument(
        "--tool",
        action="append",
        help="tool to bootstrap; repeat for multiple tools",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="also clone optional source-only tools such as msinsight",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=sys.executable,
        help="Python interpreter whose environment receives Python tools",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the complete plan without creating paths or running commands",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    try:
        result = bootstrap_toolchain(
            args.destination,
            tools=args.tool,
            include_optional=args.include_optional,
            python_executable=args.python_executable,
            dry_run=args.dry_run,
        )
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
