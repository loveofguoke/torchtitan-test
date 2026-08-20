# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Small CLI helpers shared by independent experiment entry points."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


def archive_previous_output(path: Path) -> Path | None:
    """Move stale experiment output aside without overwriting prior evidence."""

    if not path.exists():
        return None
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    destination = path.with_name(f".{path.name}.previous-{timestamp}")
    suffix = 1
    while destination.exists():
        destination = path.with_name(
            f".{path.name}.previous-{timestamp}-{suffix}"
        )
        suffix += 1
    path.rename(destination)
    return destination


def replace_topology(argv: Sequence[str], topology: str) -> list[str]:
    """Return argv with exactly one topology selection."""

    result: list[str] = []
    skip_next = False
    for index, argument in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if argument == "--topology":
            if index + 1 >= len(argv):
                raise ValueError("--topology requires a value")
            skip_next = True
            continue
        if argument.startswith("--topology="):
            continue
        result.append(argument)
    result.append(f"--topology={topology}")
    return result


def run_all_topologies(
    script_path: str,
    *,
    argv: Sequence[str],
    topology_names: Sequence[str],
) -> int:
    """Run a lifecycle experiment once per topology in deterministic order."""

    for topology in topology_names:
        command = [
            sys.executable,
            str(Path(script_path).resolve()),
            *replace_topology(argv, topology),
        ]
        print(f"Starting topology suite member: {topology}", flush=True)
        process = subprocess.run(command, check=False)
        if process.returncode:
            return process.returncode
    return 0
