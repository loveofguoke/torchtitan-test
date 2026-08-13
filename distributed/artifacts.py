"""Checksummed fixtures and JSON run artifacts for distributed comparisons."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterator


SCHEMA_NAME = "torchtitan.distributed-run"
SCHEMA_VERSION = 1
FIXTURE_SCHEMA_NAME = "torchtitan.distributed-fixture"
FIXTURE_SCHEMA_VERSION = 1


class ArtifactError(RuntimeError):
    """Raised when a fixture or capture is corrupt or incompatible."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def json_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            yield path


def file_manifest(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": file_digest(path),
        }
        for path in iter_files(root)
    ]


def fixture_digest(manifest: dict[str, Any]) -> str:
    identity = {
        "schema": manifest.get("schema"),
        "schema_version": manifest.get("schema_version"),
        "scenario_digest": manifest.get("scenario_digest"),
        "files": manifest.get("files"),
    }
    return json_digest(identity)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ArtifactError(f"JSON artifact must contain an object: {path}")
    return value


def validate_fixture(path: Path, scenario_digest: str) -> dict[str, Any]:
    manifest = read_json(path / "manifest.json")
    if manifest.get("schema") != FIXTURE_SCHEMA_NAME:
        raise ArtifactError(f"unsupported fixture schema in {path}")
    if manifest.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise ArtifactError(f"unsupported fixture schema version in {path}")
    if manifest.get("scenario_digest") != scenario_digest:
        raise ArtifactError(
            "fixture scenario differs from the requested scenario: "
            f"{manifest.get('scenario_digest')} != {scenario_digest}"
        )
    expected = manifest.get("files")
    actual = file_manifest(path)
    if expected != actual:
        raise ArtifactError(
            "fixture files failed exact checksum validation; do not run with a "
            "partially copied or modified fixture"
        )
    expected_digest = manifest.get("fixture_digest")
    actual_digest = fixture_digest(manifest)
    if expected_digest != actual_digest:
        raise ArtifactError(
            f"fixture digest mismatch: {expected_digest} != {actual_digest}"
        )
    return manifest


def runtime_metadata() -> dict[str, str]:
    metadata = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
    }
    for package in ("torch", "torch-npu", "torchtitan", "torchtitanturbo"):
        try:
            metadata[package] = version(package)
        except PackageNotFoundError:
            metadata[package] = "unavailable"
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "ASCEND_RT_VISIBLE_DEVICES",
        "WORLD_SIZE",
    ):
        if key in os.environ:
            metadata[key] = os.environ[key]
    return metadata


def validate_capture(
    path: Path,
    *,
    scenario_digest: str,
    expected_fixture_digest: str,
) -> dict[str, Any]:
    capture = read_json(path)
    if capture.get("schema") != SCHEMA_NAME:
        raise ArtifactError(f"unsupported capture schema in {path}")
    if capture.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError(f"unsupported capture schema version in {path}")
    if capture.get("status") != "success":
        raise ArtifactError(f"capture did not succeed: {path}")
    if capture.get("scenario_digest") != scenario_digest:
        raise ArtifactError(f"capture scenario mismatch: {path}")
    if capture.get("fixture_digest") != expected_fixture_digest:
        raise ArtifactError(f"capture fixture mismatch: {path}")
    return capture
