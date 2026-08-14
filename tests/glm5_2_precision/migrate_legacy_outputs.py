#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Migrate legacy formal-precision outputs to the compact storage layout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any


LEGACY_SCENARIO = re.compile(
    r"^.*?-(?P<kind>migration|self-consistency)-"
    r"(?P<reference_device>cuda|npu)-(?P<reference_topology>.+?)-vs-"
    r"(?P<candidate_device>cuda|npu)-(?P<candidate_topology>.+?)-"
    r"(?P<precision>mixed-bfloat16|mixed-float32|full-bf16)-"
    r"(?P<checkpoint>random-seed|converged)-steps(?P<steps>\d+)-"
    r"global-batch(?P<batch>\d+)-seq(?P<sequence>\d+)-seed(?P<seed>\d+)$"
)
CAPTURE_NAME = re.compile(
    r"^(?P<role>reference|candidate)-.*?(?:run-|r)(?P<repeat>\d+)$"
)
ROOT_NAMES = (
    "precision_artifacts",
    "precision_fixtures",
    "precision_runs",
    "precision_reports",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_tree(path: Path) -> None:
    """Remove a tree, including legacy paths beyond the Windows MAX_PATH limit."""

    value = str(path.resolve())
    if value.startswith("\\\\"):
        value = "\\\\?\\UNC\\" + value[2:]
    elif not value.startswith("\\\\?\\"):
        value = "\\\\?\\" + value
    shutil.rmtree(value)


def compact_scenario_name(legacy_name: str) -> str:
    match = LEGACY_SCENARIO.fullmatch(legacy_name)
    if match is None:
        raise ValueError(f"unrecognized legacy scenario name: {legacy_name}")
    values = match.groupdict()
    if values["kind"] == "migration":
        if values["reference_topology"] != values["candidate_topology"]:
            topology = (
                f"{values['reference_topology']}-{values['candidate_topology']}"
            )
        else:
            topology = values["reference_topology"]
        experiment = (
            f"migration-{values['reference_device']}-"
            f"{values['candidate_device']}-{topology}"
        )
    else:
        experiment = f"self-{values['reference_device']}"
    precision = {
        "mixed-bfloat16": "bf16",
        "mixed-float32": "fp32",
        "full-bf16": "full-bf16",
    }[values["precision"]]
    checkpoint = (
        "random" if values["checkpoint"] == "random-seed" else "converged"
    )
    return (
        f"{experiment}-{precision}-{checkpoint}-s{values['steps']}-"
        f"b{values['batch']}-seq{values['sequence']}-seed{values['seed']}"
    )


def _compact_capture_name(
    name: str,
    *,
    candidate_topology: str | None = None,
) -> str:
    match = CAPTURE_NAME.fullmatch(name)
    if match is None:
        raise ValueError(f"unrecognized legacy capture name: {name}")
    role = match.group("role")
    prefix = candidate_topology if role == "candidate" and candidate_topology else role
    return f"{prefix}-r{match.group('repeat')}"


def _merge_path(source: Path, destination: Path, *, apply: bool) -> None:
    if source.is_dir():
        if apply:
            destination.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            _merge_path(child, destination / child.name, apply=apply)
        if apply:
            source.rmdir()
        return
    if destination.exists():
        if not destination.is_file() or _digest(source) != _digest(destination):
            raise FileExistsError(
                f"refusing to overwrite different output: {destination}"
            )
        if apply:
            source.unlink()
        return
    if apply:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), destination)


def _clean_artifact(capture: Path, *, apply: bool) -> None:
    manifest_path = capture / "manifest.json"
    required = ("manifest.json", "metrics.jsonl", "training_contract.json")
    if not all((capture / name).is_file() for name in required):
        raise FileNotFoundError(f"incomplete precision artifact: {capture}")
    manifest = _read_json(manifest_path)
    keep = ("metrics.jsonl", "training_contract.json")
    manifest["files"] = {name: _digest(capture / name) for name in keep}
    if apply:
        _write_json(manifest_path, manifest)
        attachments = capture / "attachments"
        if attachments.exists():
            # Legacy runtime logs and raw metric copies are redundant. Current
            # compact input-contract summaries are part of the formal artifact.
            for child in attachments.iterdir():
                if child.name != "input_contract.json":
                    _remove_tree(child) if child.is_dir() else child.unlink()
            summary = attachments / "input_contract.json"
            if summary.is_file():
                manifest = _read_json(manifest_path)
                manifest["files"]["attachments/input_contract.json"] = _digest(
                    summary
                )
                _write_json(manifest_path, manifest)
            elif attachments.exists() and not any(attachments.iterdir()):
                attachments.rmdir()


def repair_artifact_manifests(root: Path, *, apply: bool) -> int:
    """Rebuild portable artifact file checksums without touching metric data."""

    num_repaired = 0
    artifact_root = root / "precision_artifacts"
    if not artifact_root.is_dir():
        return num_repaired
    for manifest_path in artifact_root.glob("*/*/manifest.json"):
        capture = manifest_path.parent
        required = ("metrics.jsonl", "training_contract.json")
        if not all((capture / name).is_file() for name in required):
            continue
        manifest = _read_json(manifest_path)
        files = {name: _digest(capture / name) for name in required}
        input_contract = capture / "attachments" / "input_contract.json"
        if input_contract.is_file():
            files["attachments/input_contract.json"] = _digest(input_contract)
        manifest["files"] = files
        print(f"{'REPAIR' if apply else 'WOULD REPAIR'} {manifest_path}")
        if apply:
            _write_json(manifest_path, manifest)
        num_repaired += 1
    return num_repaired


def _migrate_artifact_scenario(
    source: Path,
    destination: Path,
    *,
    apply: bool,
    candidate_topology: str | None,
) -> None:
    captures = [item for item in source.iterdir() if item.is_dir()]
    for capture in captures:
        compact_name = _compact_capture_name(
            capture.name,
            candidate_topology=candidate_topology,
        )
        _clean_artifact(capture, apply=apply)
        _merge_path(capture, destination / compact_name, apply=apply)
    if apply:
        source.rmdir()


def _migrate_fixture_scenario(
    source: Path,
    destination: Path,
    *,
    apply: bool,
) -> None:
    manifest_path = source / "fixture.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"fixture manifest not found: {manifest_path}")
    manifest = _read_json(manifest_path)
    checkpoint = source / manifest["checkpoint_relative_path"]
    if not checkpoint.is_dir():
        completed_checkpoint = destination / "checkpoint"
        completed_manifest = destination / "fixture.json"
        if apply and completed_checkpoint.is_dir() and completed_manifest.is_file():
            _remove_tree(source)
            return
        raise FileNotFoundError(f"fixture checkpoint not found: {checkpoint}")
    if apply:
        destination.mkdir(parents=True, exist_ok=True)
        _merge_path(checkpoint, destination / "checkpoint", apply=True)
        manifest["checkpoint_relative_path"] = "checkpoint"
        _write_json(destination / "fixture.json", manifest)
        _remove_tree(source)


def _migrate_regular_scenario(
    source: Path,
    destination: Path,
    *,
    apply: bool,
    candidate_topology: str | None,
) -> None:
    for item in list(source.iterdir()):
        target_name = item.name
        if item.is_dir() and CAPTURE_NAME.fullmatch(item.name):
            target_name = _compact_capture_name(
                item.name,
                candidate_topology=candidate_topology,
            )
        _merge_path(item, destination / target_name, apply=apply)
    if apply:
        source.rmdir()


def migrate(root: Path, *, apply: bool) -> list[tuple[Path, Path]]:
    changes: list[tuple[Path, Path]] = []
    for root_name in ROOT_NAMES:
        output_root = root / root_name
        if not output_root.is_dir():
            continue
        for source in list(output_root.iterdir()):
            if not source.is_dir() or LEGACY_SCENARIO.fullmatch(source.name) is None:
                continue
            match = LEGACY_SCENARIO.fullmatch(source.name)
            assert match is not None
            values = match.groupdict()
            candidate_topology = (
                values["candidate_topology"]
                if values["kind"] == "self-consistency"
                else None
            )
            destination = output_root / compact_scenario_name(source.name)
            changes.append((source, destination))
            print(f"{'MIGRATE' if apply else 'WOULD MIGRATE'} {source}")
            print(f"  -> {destination}")
            if root_name == "precision_artifacts":
                _migrate_artifact_scenario(
                    source,
                    destination,
                    apply=apply,
                    candidate_topology=candidate_topology,
                )
            elif root_name == "precision_fixtures":
                _migrate_fixture_scenario(source, destination, apply=apply)
            else:
                _migrate_regular_scenario(
                    source,
                    destination,
                    apply=apply,
                    candidate_topology=candidate_topology,
                )
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the migration; without this flag, only print the plan",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="torchtitan-test repository root",
    )
    parser.add_argument(
        "--repair-artifacts",
        action="store_true",
        help="recompute portable artifact checksums after moving across filesystems",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    changes = migrate(root, apply=args.apply)
    num_repaired = (
        repair_artifact_manifests(root, apply=args.apply)
        if args.repair_artifacts
        else 0
    )
    if not changes:
        print("No legacy precision output directories found.")
    if args.repair_artifacts and not num_repaired:
        print("No precision artifact manifests found to repair.")
    if changes and not args.apply:
        print("Dry run only. Re-run with --apply after reviewing the paths above.")


if __name__ == "__main__":
    main()
