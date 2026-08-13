#!/usr/bin/env python3
"""Restore the deduplicated four-card NPU bundle into a runnable output root."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from distributed.artifacts import read_json, validate_capture, validate_fixture


COMMON_FILES = (
    Path("checkpoint/__0_0.distcp"),
    Path("data/data.json"),
    Path("hf_assets/tokenizer.json"),
    Path("hf_assets/tokenizer_config.json"),
)


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def materialize(bundle: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    fixtures_output = output / "fixtures"
    captures_output = output / "captures"
    fixtures_output.mkdir(parents=True)
    captures_output.mkdir(parents=True)

    fixture_count = 0
    for fixture_source in sorted((bundle / "fixtures").iterdir()):
        if not fixture_source.is_dir():
            continue
        fixture_destination = fixtures_output / fixture_source.name
        shutil.copytree(fixture_source, fixture_destination)
        for relative_path in COMMON_FILES:
            _link_or_copy(
                bundle / "common" / relative_path,
                fixture_destination / relative_path,
            )
        manifest = read_json(fixture_destination / "manifest.json")
        validate_fixture(fixture_destination, manifest["scenario_digest"])
        fixture_count += 1

    capture_count = 0
    for capture_source in sorted((bundle / "captures").glob("*_npu.json")):
        capture_destination = captures_output / capture_source.name
        shutil.copy2(capture_source, capture_destination)
        scenario_id = capture_source.name.removesuffix("_npu.json")
        fixture_manifest = read_json(
            fixtures_output / scenario_id / "manifest.json"
        )
        validate_capture(
            capture_destination,
            scenario_digest=fixture_manifest["scenario_digest"],
            expected_fixture_digest=fixture_manifest["fixture_digest"],
        )
        capture_count += 1

    history_source = bundle / "captures" / "history"
    if history_source.is_dir():
        shutil.copytree(history_source, captures_output / "history")

    print(f"materialized {fixture_count} fixtures and {capture_count} captures")
    print(f"output: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    materialize(Path(__file__).resolve().parent, arguments.output.resolve())


if __name__ == "__main__":
    main()
