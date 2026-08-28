#!/usr/bin/env python3
"""Compare CUDA eager and cold-Inductor Block 0 internal-stage traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_graph.gpu_internal_block_trace import (
    BACKWARD_STAGE_NAMES,
    STAGE_NAMES,
)


TraceKey = tuple[int, str, str]


def _read_trace(path: Path) -> dict[TraceKey, dict[str, Any]]:
    records: dict[TraceKey, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                int(row["step"]),
                str(row.get("kind", "forward")),
                str(row["name"]),
            )
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
    steps = sorted({key[0] for key in left} | {key[0] for key in right})
    kinds = ["forward"]
    if any(key[1] == "backward_grad" for key in left | right):
        kinds.append("backward_grad")
    rows = []
    first_divergence_by_step = []
    for step in steps:
        for kind in kinds:
            first_divergence = None
            stage_names = (
                STAGE_NAMES if kind == "forward" else BACKWARD_STAGE_NAMES
            )
            for stage_index, stage in enumerate(stage_names):
                key = (step, kind, stage)
                left_row = left.get(key)
                right_row = right.get(key)
                if left_row is None or right_row is None:
                    row = {
                        "step": step,
                        "kind": kind,
                        "stage": stage,
                        "stage_index": stage_index,
                        "present_left": left_row is not None,
                        "present_right": right_row is not None,
                        "exact": False,
                    }
                else:
                    row = {
                        "step": step,
                        "kind": kind,
                        "stage": stage,
                        "stage_index": stage_index,
                        "present_left": True,
                        "present_right": True,
                        "exact": left_row["sha256"] == right_row["sha256"],
                        "mean_abs_delta": abs(
                            left_row["mean"] - right_row["mean"]
                        ),
                        "max_abs_delta": abs(
                            left_row["max_abs"] - right_row["max_abs"]
                        ),
                        "l2_abs_delta": abs(
                            left_row["l2"] - right_row["l2"]
                        ),
                    }
                rows.append(row)
                if not row["exact"] and first_divergence is None:
                    first_divergence = row
            first_divergence_by_step.append(
                {
                    "step": step,
                    "kind": kind,
                    "first_divergence": first_divergence,
                }
            )
    return {
        "left": left_name,
        "right": right_name,
        "records": rows,
        "exact_records": sum(row["exact"] for row in rows),
        "compared_records": len(rows),
        "first_divergence_by_step": first_divergence_by_step,
    }


def compare_internal_traces(paths: dict[str, Path]) -> dict[str, Any]:
    traces = {name: _read_trace(path) for name, path in paths.items()}
    pairs = (
        ("eager-r1", "eager-r2"),
        ("inductor-r1", "inductor-r2"),
        ("eager-r1", "inductor-r1"),
        ("eager-r1", "inductor-r2"),
    )
    return {
        "schema": "torchtitan.glm5_2.gpu_internal_trace_comparison",
        "paths": {name: str(path) for name, path in paths.items()},
        "stage_order": list(STAGE_NAMES),
        "backward_stage_order": list(BACKWARD_STAGE_NAMES),
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
    result = compare_internal_traces(paths)
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
        for item in comparison["first_divergence_by_step"]:
            row = item["first_divergence"]
            if row is None:
                print(
                    f"  step={item['step']} {item['kind']}: all stages exact"
                )
            else:
                print(
                    f"  step={item['step']} {item['kind']}: "
                    f"first={row['stage']} "
                    f"mean_delta={row.get('mean_abs_delta', float('nan')):.9g} "
                    f"l2_delta={row.get('l2_abs_delta', float('nan')):.9g}"
                )
    print(f"Wrote comparison: {args.output}")


if __name__ == "__main__":
    main()
