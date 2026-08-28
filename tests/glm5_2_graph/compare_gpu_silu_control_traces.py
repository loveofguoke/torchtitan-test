#!/usr/bin/env python3
"""Compare Block 0 forward/backward and parameter gradients for SiLU control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TraceKey = tuple[int, str, str]
ORDERED_KEYS: tuple[TraceKey, ...] = (
    (1, "block_input", "forward"),
    (1, "block_output", "forward"),
    (1, "block_output", "backward_grad"),
    (1, "parameter.feed_forward.w2.weight", "gradient_pre_clip"),
    (1, "parameter.feed_forward.w1.weight", "gradient_pre_clip"),
    (1, "parameter.feed_forward.w3.weight", "gradient_pre_clip"),
    (1, "parameter.ffn_norm.weight", "gradient_pre_clip"),
    (1, "parameter.attention.wo.weight", "gradient_pre_clip"),
    (1, "parameter.attention.wq_b.weight", "gradient_pre_clip"),
    (1, "parameter.attention.q_norm.weight", "gradient_pre_clip"),
    (1, "parameter.attention.wq_a.weight", "gradient_pre_clip"),
    (1, "parameter.attention.wkv_b.weight", "gradient_pre_clip"),
    (1, "parameter.attention.kv_norm.weight", "gradient_pre_clip"),
    (1, "parameter.attention.wkv_a.weight", "gradient_pre_clip"),
    (1, "parameter.attention_norm.weight", "gradient_pre_clip"),
    (1, "block_input", "backward_grad"),
)


def _read(path: Path) -> dict[TraceKey, dict[str, Any]]:
    records = {}
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


def _compare(
    left_name: str,
    left: dict[TraceKey, dict[str, Any]],
    right_name: str,
    right: dict[TraceKey, dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    for key in ORDERED_KEYS:
        left_row = left.get(key)
        right_row = right.get(key)
        if left_row is None or right_row is None:
            rows.append(
                {
                    "step": key[0],
                    "name": key[1],
                    "kind": key[2],
                    "present_left": left_row is not None,
                    "present_right": right_row is not None,
                    "exact": False,
                }
            )
            continue
        rows.append(
            {
                "step": key[0],
                "name": key[1],
                "kind": key[2],
                "present_left": True,
                "present_right": True,
                "exact": left_row["sha256"] == right_row["sha256"],
                "mean_abs_delta": abs(left_row["mean"] - right_row["mean"]),
                "max_abs_delta": abs(
                    left_row["max_abs"] - right_row["max_abs"]
                ),
                "l2_abs_delta": abs(left_row["l2"] - right_row["l2"]),
            }
        )
    return {
        "left": left_name,
        "right": right_name,
        "records": rows,
        "exact_records": sum(row["exact"] for row in rows),
        "compared_records": len(rows),
    }


def compare_control_traces(paths: dict[str, Path]) -> dict[str, Any]:
    traces = {name: _read(path) for name, path in paths.items()}
    pairs = (
        ("eager-r1", "eager-r2"),
        ("inductor-r1", "inductor-r2"),
        ("eager-r1", "inductor-r1"),
        ("eager-r1", "inductor-r2"),
    )
    return {
        "schema": "torchtitan.glm5_2.gpu_silu_symmetric_control",
        "paths": {name: str(path) for name, path in paths.items()},
        "comparisons": [
            _compare(left, traces[left], right, traces[right])
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
    result = compare_control_traces(paths)
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
        if comparison["left"] != "eager-r1":
            continue
        for row in comparison["records"]:
            print(
                f"  {row['name']} {row['kind']}: exact={row['exact']} "
                f"mean_delta={row.get('mean_abs_delta', float('nan')):.9g} "
                f"l2_delta={row.get('l2_abs_delta', float('nan')):.9g}"
            )
    print(f"Wrote comparison: {args.output}")


if __name__ == "__main__":
    main()
