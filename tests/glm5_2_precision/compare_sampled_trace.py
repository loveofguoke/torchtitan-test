#!/usr/bin/env python3
"""Compare GPU and NPU sampled trace directories and summarize split points."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PHASE_ORDER = (
    "parameter_before",
    "input",
    "forward_input",
    "forward",
    "forward_diagnostic",
    "loss",
    "backward_output_grad",
    "gradient_post_reduce_pre_clip",
    "gradient_post_reduce_post_clip",
    "optimizer_state_before",
    "optimizer_state_after",
    "parameter_after",
    "parameter_delta",
)


def _load(
    directory: Path,
) -> tuple[dict[int, dict[str, Any]], dict[tuple[Any, ...], dict[str, Any]]]:
    manifests: dict[int, dict[str, Any]] = {}
    records: dict[tuple[Any, ...], dict[str, Any]] = {}
    for path in sorted(directory.glob("rank-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if value["record_type"] == "manifest":
                manifests[int(value["rank"])] = value
            elif value["record_type"] == "tensor":
                key = (
                    int(value["rank"]),
                    int(value["step"]),
                    value["phase"],
                    value["key"],
                )
                records[key] = value
    if not manifests:
        raise FileNotFoundError(f"no rank traces found in {directory}")
    return manifests, records


def _metric(gpu: dict[str, Any], npu: dict[str, Any]) -> dict[str, Any]:
    if gpu["shape"] != npu["shape"] or gpu["sample_indices"] != npu["sample_indices"]:
        return {"comparable": False}
    expected = [float(item) for item in gpu["sample"]]
    actual = [float(item) for item in npu["sample"]]
    delta = [candidate - reference for reference, candidate in zip(expected, actual)]
    squared_ref = sum(item * item for item in expected)
    squared_delta = sum(item * item for item in delta)
    max_abs = max((abs(item) for item in delta), default=0.0)
    mean_abs = sum(abs(item) for item in delta) / max(len(delta), 1)
    relative_l2 = math.sqrt(squared_delta) / max(math.sqrt(squared_ref), 1e-12)
    result = {
        "comparable": True,
        "exact": gpu["sha256"] == npu["sha256"],
        "sample_max_absolute": max_abs,
        "sample_mean_absolute": mean_abs,
        "sample_relative_l2": relative_l2,
        "norm_relative": (
            (float(npu.get("norm", 0.0)) - float(gpu.get("norm", 0.0)))
            / max(abs(float(gpu.get("norm", 0.0))), 1e-12)
        ),
    }
    gpu_rows = gpu.get("row_samples")
    npu_rows = npu.get("row_samples")
    if (
        isinstance(gpu_rows, list)
        and isinstance(npu_rows, list)
        and gpu.get("row_sample_indices") == npu.get("row_sample_indices")
        and len(gpu_rows) == len(npu_rows)
    ):
        ordered_matches = 0
        set_matches = 0
        jaccard_sum = 0.0
        for gpu_row, npu_row in zip(gpu_rows, npu_rows):
            ordered_matches += gpu_row == npu_row
            gpu_counts = Counter(gpu_row)
            npu_counts = Counter(npu_row)
            intersection = sum((gpu_counts & npu_counts).values())
            union = sum((gpu_counts | npu_counts).values())
            set_matches += gpu_counts == npu_counts
            jaccard_sum += intersection / max(union, 1)
        row_count = max(len(gpu_rows), 1)
        result.update(
            {
                "unordered_last_dim_exact": (
                    gpu.get("unordered_last_dim_sha256")
                    == npu.get("unordered_last_dim_sha256")
                ),
                "sampled_row_ordered_match_fraction": (
                    ordered_matches / row_count
                ),
                "sampled_row_set_match_fraction": set_matches / row_count,
                "sampled_row_mean_jaccard": jaccard_sum / row_count,
            }
        )
    return result


def compare(gpu_dir: Path, npu_dir: Path, output: Path) -> None:
    gpu_manifests, gpu = _load(gpu_dir)
    npu_manifests, npu = _load(npu_dir)
    if set(gpu_manifests) != set(npu_manifests):
        raise ValueError("GPU/NPU trace rank sets differ")
    for rank in gpu_manifests:
        for field in ("world_size", "steps", "sample_count", "detail"):
            if (
                field not in gpu_manifests[rank]
                and field not in npu_manifests[rank]
            ):
                continue
            if gpu_manifests[rank][field] != npu_manifests[rank][field]:
                raise ValueError(
                    f"GPU/NPU manifest mismatch: rank={rank}, field={field}"
                )

    keys = sorted(set(gpu) | set(npu))
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for key in keys:
        if key not in gpu or key not in npu:
            missing.append(
                {"identity": key, "missing": "gpu" if key not in gpu else "npu"}
            )
            continue
        rank, step, phase, tensor_key = key
        rows.append(
            {
                "rank": rank,
                "step": step,
                "phase": phase,
                "key": tensor_key,
                "kind": gpu[key]["kind"],
                **_metric(gpu[key], npu[key]),
            }
        )

    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("comparable"):
            grouped[(row["step"], row["phase"])].append(row)
    summary: list[dict[str, Any]] = []
    for (step, phase), items in grouped.items():
        worst = max(items, key=lambda item: item["sample_relative_l2"])
        phase_summary = {
            "step": step,
            "phase": phase,
            "count": len(items),
            "exact_count": sum(bool(item["exact"]) for item in items),
            "mean_sample_relative_l2": sum(
                item["sample_relative_l2"] for item in items
            )
            / len(items),
            "max_sample_relative_l2": worst["sample_relative_l2"],
            "worst_rank": worst["rank"],
            "worst_key": worst["key"],
        }
        semantic_items = [
            item
            for item in items
            if "sampled_row_set_match_fraction" in item
        ]
        if semantic_items:
            phase_summary.update(
                {
                    "unordered_last_dim_exact_count": sum(
                        bool(item["unordered_last_dim_exact"])
                        for item in semantic_items
                    ),
                    "integer_semantic_count": len(semantic_items),
                    "mean_sampled_row_ordered_match_fraction": sum(
                        item["sampled_row_ordered_match_fraction"]
                        for item in semantic_items
                    )
                    / len(semantic_items),
                    "mean_sampled_row_set_match_fraction": sum(
                        item["sampled_row_set_match_fraction"]
                        for item in semantic_items
                    )
                    / len(semantic_items),
                    "mean_sampled_row_jaccard": sum(
                        item["sampled_row_mean_jaccard"]
                        for item in semantic_items
                    )
                    / len(semantic_items),
                }
            )
        summary.append(phase_summary)
    phase_index = {phase: index for index, phase in enumerate(PHASE_ORDER)}
    summary.sort(
        key=lambda item: (
            item["step"],
            phase_index.get(item["phase"], 999),
            item["phase"],
        )
    )
    top = sorted(
        (row for row in rows if row.get("comparable")),
        key=lambda item: item["sample_relative_l2"],
        reverse=True,
    )[:50]
    index_semantics = sorted(
        (
            row
            for row in rows
            if "sampled_row_set_match_fraction" in row
            and "indexer" in row["key"]
        ),
        key=lambda item: (
            item["sampled_row_set_match_fraction"],
            item["sampled_row_mean_jaccard"],
            item["sampled_row_ordered_match_fraction"],
        ),
    )
    payload = {
        "schema": "torchtitan.glm5_2.sampled_trace_comparison",
        "schema_version": 1,
        "gpu": str(gpu_dir),
        "npu": str(npu_dir),
        "missing": missing,
        "summary": summary,
        "top_differences": top,
        "index_semantics": index_semantics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("step\tphase\tcount\texact\tmean_rel_l2\tmax_rel_l2\tworst_rank\tworst_key")
    for item in summary:
        print(
            f"{item['step']}\t{item['phase']}\t{item['count']}\t"
            f"{item['exact_count']}\t{item['mean_sample_relative_l2']:.6e}\t"
            f"{item['max_sample_relative_l2']:.6e}\t{item['worst_rank']}\t"
            f"{item['worst_key']}"
        )
    print(f"report: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--npu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.gpu.resolve(), args.npu.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
