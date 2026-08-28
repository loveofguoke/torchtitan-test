#!/usr/bin/env python3
"""Compare effective top-k indices for one controlled indexer variant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != 1:
        raise ValueError(f"expected one indexer record in {path}, got {len(rows)}")
    return rows[0]


def compare_indexer_control_traces(paths: dict[str, Path]) -> dict:
    traces = {name: _read(path) for name, path in paths.items()}
    pairs = (
        ("eager-r1", "eager-r2"),
        ("inductor-r1", "inductor-r2"),
        ("eager-r1", "inductor-r1"),
        ("eager-r1", "inductor-r2"),
    )
    comparisons = []
    for left_name, right_name in pairs:
        left = traces[left_name]
        right = traces[right_name]
        mismatches = sum(
            a != b
            for a, b in zip(left["values"], right["values"], strict=True)
        )
        comparisons.append(
            {
                "left": left_name,
                "right": right_name,
                "exact": left["sha256"] == right["sha256"],
                "value_mismatch_count": mismatches,
            }
        )
    return {
        "schema": "torchtitan.glm5_2.gpu_indexer_control_comparison",
        "variant": traces["eager-r1"]["variant"],
        "paths": {name: str(path) for name, path in paths.items()},
        "comparisons": comparisons,
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
    result = compare_indexer_control_traces(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"variant={result['variant']}")
    for comparison in result["comparisons"]:
        print(
            f"  {comparison['left']} vs {comparison['right']}: "
            f"exact={comparison['exact']} "
            f"value_mismatches={comparison['value_mismatch_count']}"
        )
    print(f"Wrote comparison: {args.output}")


if __name__ == "__main__":
    main()
