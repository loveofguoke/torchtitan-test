# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Integrity metadata for official MindStudio outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


class MindStudioArtifactError(RuntimeError):
    """Raised when official output is incomplete or belongs to another run."""


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_index(root: Path) -> list[dict[str, Any]]:
    """Index official files without rewriting or interpreting their contents."""

    if not root.is_dir():
        raise MindStudioArtifactError(f"official output directory is missing: {root}")
    if root.is_symlink():
        raise MindStudioArtifactError(
            f"official output directory cannot be a symbolic link: {root}"
        )
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        # A linked file or directory can make an apparently relative manifest
        # hash data outside the capture root. Official artifacts are portable
        # snapshots, so links are rejected instead of being followed.
        if path.is_symlink():
            raise MindStudioArtifactError(
                f"symbolic links are not supported in official output: {path}"
            )
        if path.is_file():
            files.append(path)
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]


def validate_output_files(root: Path, patterns: Iterable[str], minimum: int) -> None:
    selected_patterns = tuple(patterns)
    matches = {
        path.resolve()
        for pattern in selected_patterns
        for path in root.rglob(pattern)
        if path.is_file()
    }
    if len(matches) < minimum:
        raise MindStudioArtifactError(
            f"official output has {len(matches)} matching files, expected at "
            f"least {minimum}: root={root}, patterns={selected_patterns}"
        )


def _official_output_root(artifact_directory: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise MindStudioArtifactError("official_output must be a relative directory")
    relative = Path(value)
    if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
        raise MindStudioArtifactError("official_output escapes the artifact directory")
    artifact_root = artifact_directory.resolve()
    official_root = (artifact_root / relative).resolve()
    try:
        official_root.relative_to(artifact_root)
    except ValueError as error:
        raise MindStudioArtifactError(
            "official_output escapes the artifact directory"
        ) from error
    return official_root


def artifact_is_complete(
    artifact_directory: Path,
    *,
    experiment_digest: str,
    fixture_generation_id: str,
) -> bool:
    marker = artifact_directory / "complete.json"
    manifest_path = artifact_directory / "manifest.json"
    if not marker.is_file() or not manifest_path.is_file():
        return False
    try:
        complete = json.loads(marker.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        official_root = _official_output_root(
            artifact_directory, manifest["official_output"]
        )
        current_index = output_index(official_root)
    except (KeyError, OSError, TypeError, ValueError, MindStudioArtifactError):
        return False
    return (
        complete.get("status") == "completed"
        and manifest.get("experiment_digest") == experiment_digest
        and manifest.get("fixture_generation_id") == fixture_generation_id
        and manifest.get("official_files") == current_index
    )
