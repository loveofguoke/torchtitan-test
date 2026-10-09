"""Run baseline/candidate benchmarks in a balanced interleaved order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from typing import Sequence


def balanced_order(repeats: int) -> list[str]:
    """Alternate AB/BA pairs so wall-clock drift does not favor one side."""

    return [
        role
        for pair in range(repeats)
        for role in (("reference", "candidate") if pair % 2 == 0 else ("candidate", "reference"))
    ]


def run_interleaved(
    *,
    reference: Sequence[str],
    candidate: Sequence[str],
    repeats: int,
    output: Path,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    commands = {"reference": list(reference), "candidate": list(candidate)}
    attempts = []
    for ordinal, role in enumerate(balanced_order(repeats), start=1):
        repeat = (ordinal + 1) // 2
        command = [*commands[role], "--replicate", str(repeat)]
        log = output / f"{ordinal:02d}-{role}-r{repeat}.log"
        started = time.time()
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                command,
                text=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        attempt = {
            "ordinal": ordinal,
            "role": role,
            "replicate": repeat,
            "command": command,
            "return_code": result.returncode,
            "started_unix": started,
            "duration_seconds": time.time() - started,
            "log": str(log),
        }
        attempts.append(attempt)
        (output / "schedule.json").write_text(
            json.dumps({"attempts": attempts}, indent=2) + "\n", encoding="utf-8"
        )
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)
    return {"status": "completed", "attempts": attempts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-command-json", required=True)
    parser.add_argument("--candidate-command-json", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    run_interleaved(
        reference=json.loads(args.reference_command_json),
        candidate=json.loads(args.candidate_command_json),
        repeats=args.repeats,
        output=args.output,
    )


if __name__ == "__main__":
    main()
