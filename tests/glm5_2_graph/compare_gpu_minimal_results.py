#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Summarize two cold-cache CUDA minimal Inductor regression runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    metadata = next(row for row in rows if row.get("type") == "metadata")
    results = {row["name"]: row for row in rows if "name" in row}
    if not results:
        raise ValueError(f"no minimal results in {path}")
    return metadata, results


def compare_minimal_results(first: Path, second: Path) -> dict[str, Any]:
    first_metadata, first_rows = _read(first)
    second_metadata, second_rows = _read(second)
    if first_metadata["dtype"] != second_metadata["dtype"]:
        raise ValueError("minimal result dtypes differ")
    if set(first_rows) != set(second_rows):
        raise ValueError("minimal result endpoints differ")
    rows = []
    for name in sorted(first_rows):
        first_row = first_rows[name]
        second_row = second_rows[name]
        rows.append(
            {
                "name": name,
                "repeat_1_exact_to_eager": first_row["exact"],
                "repeat_2_exact_to_eager": second_row["exact"],
                "repeat_1_mismatch_count": first_row["mismatch_count"],
                "repeat_2_mismatch_count": second_row["mismatch_count"],
                "repeat_1_max_abs_diff": first_row["max_abs_diff"],
                "repeat_2_max_abs_diff": second_row["max_abs_diff"],
                "cold_eager_reproducible": (
                    first_row["eager_sha256"] == second_row["eager_sha256"]
                ),
                "cold_inductor_reproducible": (
                    first_row["inductor_sha256"]
                    == second_row["inductor_sha256"]
                ),
            }
        )
    return {
        "schema": "torchtitan.glm5_2.gpu_minimal_comparison",
        "dtype": first_metadata["dtype"],
        "first": str(first),
        "second": str(second),
        "metadata": first_metadata,
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_minimal_results(args.first, args.second)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"dtype={result['dtype']} "
        f"tf32={result['metadata']['allow_tf32_matmul']} "
        "bf16_reduced_reduction="
        f"{result['metadata']['allow_bf16_reduced_precision_reduction']}"
    )
    for row in result["results"]:
        print(
            f"{row['name']}: "
            f"eager_exact={row['repeat_1_exact_to_eager']}/"
            f"{row['repeat_2_exact_to_eager']} "
            f"mismatch={row['repeat_1_mismatch_count']}/"
            f"{row['repeat_2_mismatch_count']} "
            f"max_abs={row['repeat_1_max_abs_diff']:.9g}/"
            f"{row['repeat_2_max_abs_diff']:.9g} "
            f"cold_inductor_repro={row['cold_inductor_reproducible']}"
        )
    print(f"Wrote comparison: {args.output}")


if __name__ == "__main__":
    main()
