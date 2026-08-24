#!/usr/bin/env python3
"""Locate the first GPU/NPU sampled-trace mismatch in hook execution order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_precision.compare_sampled_trace import _metric


def _records(path: Path, step: int, phase: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if (
            value.get("record_type") == "tensor"
            and value.get("step") == step
            and value.get("phase") == phase
        ):
            records.append(value)
    return records


def locate(
    gpu_dir: Path,
    npu_dir: Path,
    *,
    step: int,
    phase: str,
    limit: int,
    output: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    first_by_rank: list[dict[str, Any]] = []
    for gpu_path in sorted(gpu_dir.glob("rank-*.jsonl")):
        npu_path = npu_dir / gpu_path.name
        if not npu_path.is_file():
            raise FileNotFoundError(npu_path)
        gpu_records = _records(gpu_path, step, phase)
        npu_records = _records(npu_path, step, phase)
        npu_by_key = {record["key"]: record for record in npu_records}
        rank_rows: list[dict[str, Any]] = []
        previous_exact: dict[str, Any] | None = None
        for execution_index, gpu in enumerate(gpu_records):
            npu = npu_by_key.get(gpu["key"])
            if npu is None:
                row = {
                    "rank": gpu["rank"],
                    "execution_index": execution_index,
                    "key": gpu["key"],
                    "missing": "npu",
                }
            else:
                metric = _metric(gpu, npu)
                if metric.get("exact"):
                    previous_exact = {
                        "execution_index": execution_index,
                        "key": gpu["key"],
                        "kind": gpu["kind"],
                    }
                    continue
                row = {
                    "rank": gpu["rank"],
                    "execution_index": execution_index,
                    "key": gpu["key"],
                    "kind": gpu["kind"],
                    **metric,
                }
            if not rank_rows:
                row["previous_exact"] = previous_exact
            rank_rows.append(row)
        if rank_rows:
            first_by_rank.append(rank_rows[0])
            rows.extend(rank_rows[:limit])

    payload = {
        "schema": "torchtitan.glm5_2.first_trace_divergence",
        "schema_version": 1,
        "gpu": str(gpu_dir.resolve()),
        "npu": str(npu_dir.resolve()),
        "step": step,
        "phase": phase,
        "first_by_rank": first_by_rank,
        "mismatches": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("rank\texec_index\trel_l2\tmax_abs\tnorm_rel\tkey")
    for row in first_by_rank:
        print(
            f"{row['rank']}\t{row['execution_index']}\t"
            f"{row.get('sample_relative_l2', float('nan')):.6e}\t"
            f"{row.get('sample_max_absolute', float('nan')):.6e}\t"
            f"{row.get('norm_relative', float('nan')):.6e}\t{row['key']}"
        )
    print(f"report: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--npu", type=Path, required=True)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--phase", default="forward")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    locate(
        args.gpu,
        args.npu,
        step=args.step,
        phase=args.phase,
        limit=args.limit,
        output=args.output,
    )


if __name__ == "__main__":
    main()
