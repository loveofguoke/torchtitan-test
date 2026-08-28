#!/usr/bin/env python3
"""Compare ordered Layer 1 shared-index attention traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_graph.gpu_shared_attention_layer_trace import (  # noqa: E402
    SHARED_ATTENTION_STAGE_NAMES,
)


def _read_trace(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            name = str(row["name"])
            if name in records:
                raise ValueError(f"duplicate stage {name} in {path}:{line_number}")
            records[name] = row
    if not records:
        raise ValueError(f"empty trace: {path}")
    return records


def _compare_pair(left_name, left, right_name, right):
    rows = []
    for index, stage in enumerate(SHARED_ATTENTION_STAGE_NAMES):
        left_row = left.get(stage)
        right_row = right.get(stage)
        row = {
            "stage": stage,
            "stage_index": index,
            "present_left": left_row is not None,
            "present_right": right_row is not None,
            "exact": False,
        }
        if left_row is not None and right_row is not None:
            row.update(
                exact=left_row["sha256"] == right_row["sha256"],
                mean_abs_delta=abs(left_row["mean"] - right_row["mean"]),
                l2_abs_delta=abs(left_row["l2"] - right_row["l2"]),
            )
        rows.append(row)
    return {
        "left": left_name,
        "right": right_name,
        "records": rows,
        "exact_records": sum(row["exact"] for row in rows),
        "compared_records": len(rows),
        "first_divergence": next(
            (row for row in rows if not row["exact"]), None
        ),
    }


def compare_shared_attention_traces(paths: dict[str, Path]) -> dict[str, Any]:
    traces = {name: _read_trace(path) for name, path in paths.items()}
    pairs = (
        ("eager-r1", "eager-r2"),
        ("inductor-r1", "inductor-r2"),
        ("eager-r1", "inductor-r1"),
        ("eager-r1", "inductor-r2"),
    )
    return {
        "schema": "torchtitan.glm5_2.gpu_shared_attention_trace_comparison",
        "paths": {name: str(path) for name, path in paths.items()},
        "stage_order": list(SHARED_ATTENTION_STAGE_NAMES),
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
    result = compare_shared_attention_traces(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for comparison in result["comparisons"]:
        print(
            f"{comparison['left']} vs {comparison['right']}: "
            f"exact={comparison['exact_records']}/"
            f"{comparison['compared_records']}"
        )
        first = comparison["first_divergence"]
        if first is None:
            print("  all Layer 1 attention stages exact")
        else:
            print(
                f"  first={first['stage']} "
                f"mean_delta={first.get('mean_abs_delta', float('nan')):.9g} "
                f"l2_delta={first.get('l2_abs_delta', float('nan')):.9g}"
            )
            if comparison["left"] == "eager-r1":
                for row in comparison["records"]:
                    print(
                        f"  {row['stage']}: exact={row['exact']} "
                        f"mean_delta={row.get('mean_abs_delta', float('nan')):.9g} "
                        f"l2_delta={row.get('l2_abs_delta', float('nan')):.9g}"
                    )
    print(f"Wrote comparison: {args.output}")


if __name__ == "__main__":
    main()
