#!/usr/bin/env python3
"""Compare eager and cold-Inductor all-layer boundary traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


TraceKey = tuple[int, str, str]
_LAYER_NAME = re.compile(r"^layer\.(\d+)\.(input|output)$")


def _read_trace(path: Path) -> dict[TraceKey, dict[str, Any]]:
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["step"]), str(row["kind"]), str(row["name"]))
            if key in records:
                raise ValueError(f"duplicate trace key {key} in {path}:{line_number}")
            if _LAYER_NAME.fullmatch(key[2]) is None:
                raise ValueError(f"invalid layer name {key[2]} in {path}:{line_number}")
            records[key] = row
    if not records:
        raise ValueError(f"empty trace: {path}")
    return records


def _ordered_names(records: dict[TraceKey, dict[str, Any]], kind: str) -> list[str]:
    layer_ids = sorted(
        {
            int(match.group(1))
            for _, row_kind, name in records
            if row_kind == kind
            for match in [_LAYER_NAME.fullmatch(name)]
            if match is not None
        }
    )
    if kind == "forward":
        return [
            f"layer.{layer_id}.{boundary}"
            for layer_id in layer_ids
            for boundary in ("input", "output")
        ]
    return [
        f"layer.{layer_id}.{boundary}"
        for layer_id in reversed(layer_ids)
        for boundary in ("output", "input")
    ]


def _compare_pair(left_name, left, right_name, right):
    steps = sorted({key[0] for key in left} | {key[0] for key in right})
    rows = []
    first_divergences = []
    for step in steps:
        for kind in ("forward", "backward_grad"):
            first = None
            names = _ordered_names(left | right, kind)
            for order, name in enumerate(names):
                key = (step, kind, name)
                left_row = left.get(key)
                right_row = right.get(key)
                row = {
                    "step": step,
                    "kind": kind,
                    "name": name,
                    "order": order,
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
                if not row["exact"] and first is None:
                    first = row
            first_divergences.append(
                {"step": step, "kind": kind, "first_divergence": first}
            )
    return {
        "left": left_name,
        "right": right_name,
        "records": rows,
        "exact_records": sum(row["exact"] for row in rows),
        "compared_records": len(rows),
        "first_divergences": first_divergences,
    }


def compare_layer_boundary_traces(paths: dict[str, Path]) -> dict[str, Any]:
    traces = {name: _read_trace(path) for name, path in paths.items()}
    pairs = (
        ("eager-r1", "eager-r2"),
        ("inductor-r1", "inductor-r2"),
        ("eager-r1", "inductor-r1"),
        ("eager-r1", "inductor-r2"),
    )
    return {
        "schema": "torchtitan.glm5_2.gpu_layer_boundary_comparison",
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
    result = compare_layer_boundary_traces(paths)
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
        for item in comparison["first_divergences"]:
            first = item["first_divergence"]
            if first is None:
                print(f"  step={item['step']} {item['kind']}: all layers exact")
            else:
                print(
                    f"  step={item['step']} {item['kind']}: "
                    f"first={first['name']} "
                    f"mean_delta={first.get('mean_abs_delta', float('nan')):.9g} "
                    f"l2_delta={first.get('l2_abs_delta', float('nan')):.9g}"
                )
    print(f"Wrote comparison: {args.output}")


if __name__ == "__main__":
    main()
