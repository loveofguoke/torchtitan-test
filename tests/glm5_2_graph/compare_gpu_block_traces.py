#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Compare CUDA eager and independently compiled Block 0 JSONL traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TraceKey = tuple[int, str, str]


def _read_trace(path: Path) -> dict[TraceKey, dict[str, Any]]:
    records: dict[TraceKey, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["step"]), str(row["name"]), str(row["kind"]))
            if key in records:
                raise ValueError(f"duplicate trace key {key} in {path}:{line_number}")
            records[key] = row
    if not records:
        raise ValueError(f"empty trace: {path}")
    return records


def _compare_pair(
    left_name: str,
    left: dict[TraceKey, dict[str, Any]],
    right_name: str,
    right: dict[TraceKey, dict[str, Any]],
) -> dict[str, Any]:
    left_keys = set(left)
    right_keys = set(right)
    common = sorted(left_keys & right_keys)
    rows = []
    for key in common:
        left_row = left[key]
        right_row = right[key]
        rows.append(
            {
                "step": key[0],
                "name": key[1],
                "kind": key[2],
                "exact": left_row["sha256"] == right_row["sha256"],
                "mean_abs_delta": abs(left_row["mean"] - right_row["mean"]),
                "max_abs_delta": abs(
                    left_row["max_abs"] - right_row["max_abs"]
                ),
                "l2_abs_delta": abs(left_row["l2"] - right_row["l2"]),
            }
        )
    mismatches = [row for row in rows if not row["exact"]]
    return {
        "left": left_name,
        "right": right_name,
        "left_records": len(left),
        "right_records": len(right),
        "common_records": len(common),
        "missing_from_left": [list(key) for key in sorted(right_keys - left_keys)],
        "missing_from_right": [list(key) for key in sorted(left_keys - right_keys)],
        "exact_records": len(rows) - len(mismatches),
        "mismatch_records": len(mismatches),
        "first_mismatches": mismatches[:20],
        "max_mean_abs_delta": max(
            (row["mean_abs_delta"] for row in rows), default=0.0
        ),
        "max_l2_abs_delta": max(
            (row["l2_abs_delta"] for row in rows), default=0.0
        ),
    }


def compare_traces(paths: dict[str, Path]) -> dict[str, Any]:
    traces = {name: _read_trace(path) for name, path in paths.items()}
    pairs = (
        ("eager-r1", "eager-r2"),
        ("inductor-r1", "inductor-r2"),
        ("eager-r1", "inductor-r1"),
        ("eager-r1", "inductor-r2"),
    )
    return {
        "schema": "torchtitan.glm5_2.gpu_block_trace_comparison",
        "paths": {name: str(path) for name, path in paths.items()},
        "comparisons": [
            _compare_pair(left, traces[left], right, traces[right])
            for left, right in pairs
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("eager-r1", "eager-r2", "inductor-r1", "inductor-r2"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        name: getattr(args, name.replace("-", "_"))
        for name in ("eager-r1", "eager-r2", "inductor-r1", "inductor-r2")
    }
    result = compare_traces(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for comparison in result["comparisons"]:
        print(
            f"{comparison['left']} vs {comparison['right']}: "
            f"exact={comparison['exact_records']}/"
            f"{comparison['common_records']}, "
            f"mismatch={comparison['mismatch_records']}, "
            f"max_mean_delta={comparison['max_mean_abs_delta']:.9g}, "
            f"max_l2_delta={comparison['max_l2_abs_delta']:.9g}"
        )
        for row in comparison["first_mismatches"][:5]:
            print(
                "  first mismatch: "
                f"step={row['step']} name={row['name']} kind={row['kind']} "
                f"mean_delta={row['mean_abs_delta']:.9g} "
                f"l2_delta={row['l2_abs_delta']:.9g}"
            )
    print(f"Wrote comparison: {args.output}")


if __name__ == "__main__":
    main()
