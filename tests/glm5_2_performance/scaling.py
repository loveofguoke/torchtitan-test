"""Cross-topology strong/weak scaling analysis for profiler-off runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .analysis import read_metrics, summarize_profile_phases


def analyze_scaling(points: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(points, key=lambda point: int(point["world_size"]))
    if not ordered:
        return {"status": "not_available", "reason": "no runs supplied"}
    sequence_lengths = {int(point["sequence_length"]) for point in ordered}
    global_batches = {int(point["global_batch_size"]) for point in ordered}
    per_device_batches = {
        int(point["global_batch_size"]) / int(point["world_size"])
        for point in ordered
    }
    if len(sequence_lengths) != 1:
        mode = "not_comparable"
    elif len(global_batches) == 1:
        mode = "strong"
    elif len(per_device_batches) == 1:
        mode = "weak"
    else:
        mode = "not_comparable"
    baseline = ordered[0]
    rows = []
    for point in ordered:
        size_ratio = int(point["world_size"]) / int(baseline["world_size"])
        throughput_ratio = (
            float(point["job_throughput_tps"])
            / float(baseline["job_throughput_tps"])
        )
        rows.append(
            {
                **point,
                "speedup": throughput_ratio,
                "strong_scaling_efficiency_percent": (
                    throughput_ratio / size_ratio * 100 if mode == "strong" else None
                ),
                "weak_scaling_efficiency_percent": (
                    float(baseline["median_step_seconds"])
                    / float(point["median_step_seconds"])
                    * 100
                    if mode == "weak"
                    else None
                ),
            }
        )
    return {
        "schema": "torchtitan.glm5_2.scaling.v1",
        "status": "observed" if mode != "not_comparable" else "not_comparable",
        "mode": mode,
        "rows": rows,
        "interpretation": (
            "Strong scaling keeps global work fixed; efficiency measures speedup "
            "relative to device-count growth."
            if mode == "strong"
            else "Weak scaling keeps work per device fixed; efficiency measures "
            "step-time retention."
            if mode == "weak"
            else "Runs change both global and per-device work, so no scaling "
            "efficiency is reported."
        ),
    }


def point_from_run(run_directory: Path) -> dict[str, Any]:
    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    phases = summarize_profile_phases(
        read_metrics(run_directory / "metrics.jsonl"),
        config=config,
        profiler_environment=manifest.get("profiler_environment"),
    )
    comparison = phases["comparison"]
    return {
        "run": str(run_directory),
        "topology": manifest["topology"],
        "world_size": int(comparison["world_size"]),
        "global_batch_size": int(config["global_batch_size"]),
        "sequence_length": int(config["sequence_length"]),
        "median_step_seconds": float(comparison["baseline_median_step_seconds"]),
        "job_throughput_tps": float(comparison["baseline_job_throughput_tps"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = analyze_scaling([point_from_run(path) for path in args.runs])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Scaling report: {args.output}")


if __name__ == "__main__":
    main()
