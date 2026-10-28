#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Print compact GPU/NPU training-accuracy metrics for a PR table."""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=Path, required=True, help="GPU training_metrics.jsonl")
    parser.add_argument("--npu", type=Path, required=True, help="NPU training_metrics.jsonl")
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
    args = parser.parse_args()

    gpu = _read_metrics(args.gpu)
    npu = _read_metrics(args.npu)
    if set(gpu) != set(npu):
        raise ValueError(
            "GPU/NPU steps differ: "
            f"missing_gpu={sorted(set(npu) - set(gpu))}, "
            f"missing_npu={sorted(set(gpu) - set(npu))}"
        )
    steps = sorted(gpu)
    if len(steps) != args.expected_steps:
        raise ValueError(
            f"expected {args.expected_steps} aligned steps, found {len(steps)} "
            f"({steps[0]}..{steps[-1]})"
        )

    loss = _summarize(_series(gpu, "loss"), _series(npu, "loss"))
    grad_norm = _summarize(
        _series(gpu, "grad_norm"), _series(npu, "grad_norm")
    )
    output = {
        "step_count": len(steps),
        "first_step": steps[0],
        "last_step": steps[-1],
        "difference_definition": "NPU - GPU",
        "loss": asdict(loss),
        "grad_norm": asdict(grad_norm),
    }

    print(f"steps: {len(steps)} ({steps[0]}..{steps[-1]})")
    print("difference: NPU - GPU")
    print()
    print(
        "metric | GPU final | NPU final | final diff | mean diff | mean abs diff | "
        "max abs diff | max abs step | mean rel error | max rel error | max rel step"
    )
    print("---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:")
    print(_markdown_row("loss", loss))
    print(_markdown_row("grad_norm", grad_norm))
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
