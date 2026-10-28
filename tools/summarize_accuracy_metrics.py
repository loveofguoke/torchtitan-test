#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Print compact GPU/NPU training-accuracy metrics for a PR table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any


LOSS_KEY = "loss_metrics/global_avg_loss"
GRAD_NORM_KEY = "grad_norm"


@dataclass(frozen=True)
class MetricSummary:
    reference_final: float
    candidate_final: float
    final_signed_difference: float
    mean_signed_difference: float
    mean_absolute_difference: float
    max_absolute_difference: float
    max_absolute_difference_step: int
    mean_relative_error: float
    max_relative_error: float
    max_relative_error_step: int


def _finite_float(value: Any, *, path: Path, line_number: int, key: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{path}:{line_number}: {key} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{path}:{line_number}: {key} must be numeric"
        ) from error
    if not math.isfinite(result):
        raise ValueError(f"{path}:{line_number}: {key} is not finite: {value!r}")
    return result


def _read_metrics(path: Path) -> dict[int, dict[str, float]]:
    records: dict[int, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: record must be a JSON object")
            metrics = payload.get("metrics", payload)
            if not isinstance(metrics, dict):
                raise ValueError(f"{path}:{line_number}: metrics must be a JSON object")
            if LOSS_KEY not in metrics or GRAD_NORM_KEY not in metrics:
                continue
            if "step" not in payload:
                raise ValueError(f"{path}:{line_number}: record has no step")
            step = int(payload["step"])
            if step in records:
                raise ValueError(f"{path}:{line_number}: duplicate step {step}")
            records[step] = {
                "loss": _finite_float(
                    metrics[LOSS_KEY], path=path, line_number=line_number, key=LOSS_KEY
                ),
                "grad_norm": _finite_float(
                    metrics[GRAD_NORM_KEY],
                    path=path,
                    line_number=line_number,
                    key=GRAD_NORM_KEY,
                ),
            }
    if not records:
        raise ValueError(
            f"{path}: no records contain both {LOSS_KEY!r} and {GRAD_NORM_KEY!r}"
        )
    return records


def _relative_error(reference: float, candidate: float) -> float:
    if reference == 0.0:
        return 0.0 if candidate == 0.0 else math.inf
    return abs(candidate - reference) / abs(reference)


def _summarize(
    reference: dict[int, float], candidate: dict[int, float]
) -> MetricSummary:
    steps = sorted(reference)
    signed = [candidate[step] - reference[step] for step in steps]
    absolute = [abs(value) for value in signed]
    relative = [
        _relative_error(reference[step], candidate[step]) for step in steps
    ]
    if not all(math.isfinite(value) for value in relative):
        raise ValueError("relative error is non-finite because a reference value is zero")
    max_absolute_index = max(range(len(steps)), key=absolute.__getitem__)
    max_relative_index = max(range(len(steps)), key=relative.__getitem__)
    final_step = steps[-1]
    return MetricSummary(
        reference_final=reference[final_step],
        candidate_final=candidate[final_step],
        final_signed_difference=signed[-1],
        mean_signed_difference=fmean(signed),
        mean_absolute_difference=fmean(absolute),
        max_absolute_difference=absolute[max_absolute_index],
        max_absolute_difference_step=steps[max_absolute_index],
        mean_relative_error=fmean(relative),
        max_relative_error=relative[max_relative_index],
        max_relative_error_step=steps[max_relative_index],
    )


def _series(records: dict[int, dict[str, float]], name: str) -> dict[int, float]:
    return {step: values[name] for step, values in records.items()}


def _number(value: float) -> str:
    return f"{value:.10g}"


def _markdown_row(name: str, summary: MetricSummary) -> str:
    return " | ".join(
        (
            name,
            _number(summary.reference_final),
            _number(summary.candidate_final),
            _number(summary.final_signed_difference),
            _number(summary.mean_signed_difference),
            _number(summary.mean_absolute_difference),
            _number(summary.max_absolute_difference),
            str(summary.max_absolute_difference_step),
            f"{summary.mean_relative_error:.6%}",
            f"{summary.max_relative_error:.6%}",
            str(summary.max_relative_error_step),
        )
    )


def _discover_pairs(root: Path, profile: str | None) -> dict[str, tuple[Path, Path]]:
    candidates: dict[str, list[tuple[Path, Path]]] = {}
    for gpu_path in sorted(root.rglob("reference-r1/training_metrics.jsonl")):
        relative = gpu_path.relative_to(root)
        if profile is not None and profile not in relative.parts:
            continue
        npu_path = gpu_path.parent.parent / "candidate-r1" / "training_metrics.jsonl"
        if not npu_path.is_file():
            continue
        topology = relative.parts[0]
        candidates.setdefault(topology, []).append((gpu_path, npu_path))
    if not candidates:
        suffix = f" for profile {profile!r}" if profile is not None else ""
        raise ValueError(f"no reference-r1/candidate-r1 metric pairs found under {root}{suffix}")

    pairs: dict[str, tuple[Path, Path]] = {}
    for topology, topology_candidates in candidates.items():
        canonical_suffixes = {
            (
                "observations",
                "training",
                profile,
                "reference-r1",
                "training_metrics.jsonl",
            ),
            (
                profile,
                "observations",
                "training",
                "reference-r1",
                "training_metrics.jsonl",
            ),
        }
        canonical = [
            pair
            for pair in topology_candidates
            if profile is not None
            and pair[0].relative_to(root).parts[1:] in canonical_suffixes
        ]
        if len(canonical) == 1:
            pairs[topology] = canonical[0]
            continue
        if len(topology_candidates) > 1:
            rendered = "\n  ".join(str(pair[0]) for pair in topology_candidates)
            raise ValueError(
                f"multiple non-canonical GPU/NPU metric pairs found for topology "
                f"{topology!r}:\n  {rendered}"
            )
        pairs[topology] = topology_candidates[0]
    return pairs


def _summarize_pair(
    gpu_path: Path, npu_path: Path, expected_steps: int
) -> dict[str, Any]:
    gpu = _read_metrics(gpu_path)
    npu = _read_metrics(npu_path)
    if set(gpu) != set(npu):
        raise ValueError(
            f"GPU/NPU steps differ for {gpu_path.parent.parent}: "
            f"missing_gpu={sorted(set(npu) - set(gpu))}, "
            f"missing_npu={sorted(set(gpu) - set(npu))}"
        )
    steps = sorted(gpu)
    if len(steps) != expected_steps:
        raise ValueError(
            f"{gpu_path.parent.parent}: expected {expected_steps} aligned steps, "
            f"found {len(steps)} ({steps[0]}..{steps[-1]})"
        )
    return {
        "step_count": len(steps),
        "first_step": steps[0],
        "last_step": steps[-1],
        "gpu_path": str(gpu_path),
        "npu_path": str(npu_path),
        "loss": _summarize(_series(gpu, "loss"), _series(npu, "loss")),
        "grad_norm": _summarize(
            _series(gpu, "grad_norm"), _series(npu, "grad_norm")
        ),
    }


def _write_loss_curves(
    pairs: dict[str, tuple[Path, Path]], topologies: list[str], output: Path
) -> None:
    missing = sorted(set(topologies) - set(pairs))
    if missing:
        raise ValueError(f"curve topologies not found: {missing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("topology", "step", "gpu_loss", "npu_loss", "difference"))
        for topology in topologies:
            gpu_path, npu_path = pairs[topology]
            gpu = _read_metrics(gpu_path)
            npu = _read_metrics(npu_path)
            if set(gpu) != set(npu):
                raise ValueError(f"GPU/NPU steps differ for curve topology {topology!r}")
            for step in sorted(gpu):
                gpu_loss = gpu[step]["loss"]
                npu_loss = npu[step]["loss"]
                writer.writerow((topology, step, gpu_loss, npu_loss, npu_loss - gpu_loss))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--gpu", type=Path, help="GPU training_metrics.jsonl")
    inputs.add_argument(
        "--root",
        type=Path,
        help="experiment root; discover every reference-r1/candidate-r1 topology pair",
    )
    parser.add_argument("--npu", type=Path, help="NPU training_metrics.jsonl")
    parser.add_argument(
        "--profile",
        help="training profile to select while scanning --root, for example s500-fb08c375",
    )
    parser.add_argument(
        "--expected-steps",
        type=int,
        default=500,
        help="required aligned step count (default: 500)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="optional path for the same compact summary as JSON",
    )
    parser.add_argument(
        "--curve-output",
        type=Path,
        help="optional CSV path for selected per-step GPU/NPU loss curves",
    )
    parser.add_argument(
        "--curve-topologies",
        default="single,fsdp8",
        help="comma-separated topologies written by --curve-output (default: single,fsdp8)",
    )
    args = parser.parse_args()

    if args.root is not None:
        if args.npu is not None:
            parser.error("--npu cannot be used with --root")
        pairs = _discover_pairs(args.root, args.profile)
    else:
        if args.npu is None:
            parser.error("--npu is required with --gpu")
        if args.profile is not None:
            parser.error("--profile can only be used with --root")
        pairs = {"selected": (args.gpu, args.npu)}

    summaries = {
        topology: _summarize_pair(gpu_path, npu_path, args.expected_steps)
        for topology, (gpu_path, npu_path) in pairs.items()
    }
    if args.curve_output is not None:
        curve_topologies = [
            value.strip() for value in args.curve_topologies.split(",") if value.strip()
        ]
        if not curve_topologies:
            parser.error("--curve-topologies must contain at least one topology")
        _write_loss_curves(pairs, curve_topologies, args.curve_output)
    output = {
        "expected_steps": args.expected_steps,
        "difference_definition": "NPU - GPU",
        "topologies": {
            topology: {
                **{key: value for key, value in summary.items() if key not in {"loss", "grad_norm"}},
                "loss": asdict(summary["loss"]),
                "grad_norm": asdict(summary["grad_norm"]),
            }
            for topology, summary in summaries.items()
        },
    }

    print(f"topologies: {len(summaries)} ({', '.join(summaries)})")
    print(f"required aligned steps per topology: {args.expected_steps}")
    print("difference: NPU - GPU")
    print()
    print(
        "topology | metric | GPU final | NPU final | final diff | mean diff | mean abs diff | "
        "max abs diff | max abs step | mean rel error | max rel error | max rel step"
    )
    print("---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:")
    for topology, summary in summaries.items():
        print(f"{topology} | {_markdown_row('loss', summary['loss'])}")
        print(f"{topology} | {_markdown_row('grad_norm', summary['grad_norm'])}")
    print()
    print("JSON:")
    print(json.dumps(output, indent=2, sort_keys=True))

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
