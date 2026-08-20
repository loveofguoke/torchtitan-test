# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Small CLI helpers shared by independent experiment entry points."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _replace_topology(argv: Sequence[str], topology: str) -> list[str]:
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
            *_replace_topology(argv, topology),
        ]
        print(f"Starting topology suite member: {topology}", flush=True)
        process = subprocess.run(command, check=False)
        if process.returncode:
            return process.returncode
    return 0
