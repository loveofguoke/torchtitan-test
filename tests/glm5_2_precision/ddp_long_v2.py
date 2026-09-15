# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Offline DDP long-run convergence-curve assessment.

This module deliberately consumes existing formal precision artifacts.  It does
not launch training, mutate fixtures, or require accelerator dependencies.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

from .artifacts import PrecisionArtifactReader


@dataclass(frozen=True)
class DdpLongV2Config:
    """Project guardrails for overall loss-curve convergence equivalence."""

    curve_area_relative_limit: float = 0.02
    final_mean_relative_limit: float = 0.02
    smoothed_correlation_minimum: float = 0.99
    sustained_window_relative_limit: float = 0.03
    warmup_steps: int = 100
    late_fraction: float = 0.20
    window_size: int = 100
    sustained_window_size: int = 500
    minimum_steps: int = 5000

    def __post_init__(self) -> None:
        if self.curve_area_relative_limit < 0.0:
            raise ValueError("curve_area_relative_limit must be nonnegative")
        if self.final_mean_relative_limit < 0.0:
            raise ValueError("final_mean_relative_limit must be nonnegative")
        if not -1.0 <= self.smoothed_correlation_minimum <= 1.0:
            raise ValueError("smoothed_correlation_minimum must be in [-1, 1]")
        if self.sustained_window_relative_limit < 0.0:
            raise ValueError("sustained_window_relative_limit must be nonnegative")
        if self.warmup_steps < 1:
            raise ValueError("warmup_steps must be positive")
        if not 0.0 < self.late_fraction <= 1.0:
            raise ValueError("late_fraction must be in (0, 1]")
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        if self.sustained_window_size < 1:
            raise ValueError("sustained_window_size must be positive")
        if self.minimum_steps <= self.warmup_steps:
            raise ValueError("minimum_steps must be greater than warmup_steps")


@dataclass(frozen=True)
class Criterion:
    name: str
    passed: bool
    observed: str
    required: str
    category: str
    explanation: str


@dataclass(frozen=True)
class SeriesDiagnostics:
    count: int
    mean_absolute_relative_error: float
    signed_mean_relative_error: float
    p95_absolute_relative_error: float
    p99_absolute_relative_error: float
    maximum_absolute_relative_error: float


@dataclass(frozen=True)
class CurveDiagnostics:
    area_relative_error: float
    final_mean_relative_error: float
    smoothed_correlation: float
    maximum_sustained_window_relative_error: float
    worst_window_start_step: int
    worst_window_end_step: int


@dataclass(frozen=True)
class RepeatCurveDiagnostics:
    maximum_area_relative_error: float
    maximum_final_mean_relative_error: float
    minimum_smoothed_correlation: float
    maximum_sustained_window_relative_error: float


@dataclass(frozen=True)
class DdpLongV2Result:
    status: str
    criteria: tuple[Criterion, ...]
    inconclusive_reasons: tuple[str, ...]
    steps: tuple[int, ...]
    loss: SeriesDiagnostics
    grad_norm: SeriesDiagnostics
    curve: CurveDiagnostics
    gpu_repeat_curve: RepeatCurveDiagnostics
    npu_repeat_curve: RepeatCurveDiagnostics
    gpu_repeat_loss_mare: float
    npu_repeat_loss_mare: float
    gpu_repeat_grad_norm_mare: float
    npu_repeat_grad_norm_mare: float
    gpu_loss_bitwise_reproducible: bool
    npu_loss_bitwise_reproducible: bool
    gpu_grad_norm_bitwise_reproducible: bool
    npu_grad_norm_bitwise_reproducible: bool
    config: DdpLongV2Config


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires observations")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _relative_errors(
    reference: Sequence[float],
    candidate: Sequence[float],
    *,
    epsilon: float = 1e-8,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if len(reference) != len(candidate) or not reference:
        raise ValueError("series must have the same nonzero length")
    absolute: list[float] = []
    signed: list[float] = []
    for left, right in zip(reference, candidate):
        denominator = max(abs(left), epsilon)
        signed_value = (right - left) / denominator
        signed.append(signed_value)
        absolute.append(abs(signed_value))
    return tuple(absolute), tuple(signed)


def _diagnostics(
    reference: Sequence[float], candidate: Sequence[float]
) -> SeriesDiagnostics:
    absolute, signed = _relative_errors(reference, candidate)
    return SeriesDiagnostics(
        count=len(absolute),
        mean_absolute_relative_error=sum(absolute) / len(absolute),
        signed_mean_relative_error=sum(signed) / len(signed),
        p95_absolute_relative_error=_quantile(absolute, 0.95),
        p99_absolute_relative_error=_quantile(absolute, 0.99),
        maximum_absolute_relative_error=max(absolute),
    )


def _mean_curve(curves: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not curves:
        raise ValueError("at least one curve is required")
    width = len(curves[0])
    if any(len(curve) != width for curve in curves):
        raise ValueError("repeat curves have different lengths")
    return tuple(sum(values) / len(values) for values in zip(*curves))


def _maximum_pairwise_mare(curves: Sequence[Sequence[float]]) -> float:
    values = [
        _diagnostics(curves[left], curves[right]).mean_absolute_relative_error
        for left in range(len(curves))
        for right in range(left + 1, len(curves))
    ]
    if not values:
        raise ValueError("at least two repeats are required")
    return max(values)


def _all_exact(curves: Sequence[Sequence[float]]) -> bool:
    first = tuple(float(value).hex() for value in curves[0])
    return all(
        tuple(float(value).hex() for value in curve) == first
        for curve in curves[1:]
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires observations")
    return sum(values) / len(values)


def _relative_mean_error(
    reference: Sequence[float],
    candidate: Sequence[float],
    *,
    epsilon: float = 1e-8,
) -> float:
    if len(reference) != len(candidate) or not reference:
        raise ValueError("series must have the same nonzero length")
    reference_mean = _mean(reference)
    candidate_mean = _mean(candidate)
    return abs(candidate_mean - reference_mean) / max(abs(reference_mean), epsilon)


def _moving_average(values: Sequence[float], window_size: int) -> tuple[float, ...]:
    if not values:
        raise ValueError("moving average requires observations")
    width = min(window_size, len(values))
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)
    return tuple(
        (prefix[end] - prefix[end - width]) / width
        for end in range(width, len(values) + 1)
    )


def _pearson_correlation(
    reference: Sequence[float], candidate: Sequence[float]
) -> float:
    if len(reference) != len(candidate) or not reference:
        raise ValueError("series must have the same nonzero length")
    if len(reference) == 1:
        return 1.0
    reference_mean = _mean(reference)
    candidate_mean = _mean(candidate)
    reference_centered = tuple(value - reference_mean for value in reference)
    candidate_centered = tuple(value - candidate_mean for value in candidate)
    covariance = sum(
        left * right for left, right in zip(reference_centered, candidate_centered)
    )
    reference_norm = math.sqrt(sum(value * value for value in reference_centered))
    candidate_norm = math.sqrt(sum(value * value for value in candidate_centered))
    if reference_norm <= 1e-12 and candidate_norm <= 1e-12:
        return 1.0
    if reference_norm <= 1e-12 or candidate_norm <= 1e-12:
        return 0.0
    return max(-1.0, min(1.0, covariance / (reference_norm * candidate_norm)))


def _curve_diagnostics(
    reference: Sequence[float],
    candidate: Sequence[float],
    steps: Sequence[int],
    config: DdpLongV2Config,
) -> CurveDiagnostics:
    if len(reference) != len(candidate) or len(reference) != len(steps) or not steps:
        raise ValueError("curve values and steps must have the same nonzero length")

    late_count = max(1, math.ceil(len(reference) * config.late_fraction))
    retained_start = min(config.warmup_steps, len(reference) - 1)
    retained_reference = reference[retained_start:]
    retained_candidate = candidate[retained_start:]
    smoothed_reference = _moving_average(retained_reference, config.window_size)
    smoothed_candidate = _moving_average(retained_candidate, config.window_size)

    sustained_width = min(config.sustained_window_size, len(retained_reference))
    reference_prefix = [0.0]
    candidate_prefix = [0.0]
    for reference_value, candidate_value in zip(
        retained_reference, retained_candidate
    ):
        reference_prefix.append(reference_prefix[-1] + reference_value)
        candidate_prefix.append(candidate_prefix[-1] + candidate_value)

    worst_error = -1.0
    worst_offset = 0
    for offset in range(0, len(retained_reference) - sustained_width + 1):
        end = offset + sustained_width
        reference_mean = (
            reference_prefix[end] - reference_prefix[offset]
        ) / sustained_width
        candidate_mean = (
            candidate_prefix[end] - candidate_prefix[offset]
        ) / sustained_width
        error = abs(candidate_mean - reference_mean) / max(abs(reference_mean), 1e-8)
        if error > worst_error:
            worst_error = error
            worst_offset = offset

    worst_start = retained_start + worst_offset
    worst_end = worst_start + sustained_width - 1
    return CurveDiagnostics(
        area_relative_error=_relative_mean_error(reference, candidate),
        final_mean_relative_error=_relative_mean_error(
            reference[-late_count:], candidate[-late_count:]
        ),
        smoothed_correlation=_pearson_correlation(
            smoothed_reference, smoothed_candidate
        ),
        maximum_sustained_window_relative_error=worst_error,
        worst_window_start_step=steps[worst_start],
        worst_window_end_step=steps[worst_end],
    )


def _repeat_curve_diagnostics(
    curves: Sequence[Sequence[float]],
    steps: Sequence[int],
    config: DdpLongV2Config,
) -> RepeatCurveDiagnostics:
    pairwise = [
        _curve_diagnostics(curves[left], curves[right], steps, config)
        for left in range(len(curves))
        for right in range(left + 1, len(curves))
    ]
    if not pairwise:
        raise ValueError("at least two repeats are required")
    return RepeatCurveDiagnostics(
        maximum_area_relative_error=max(item.area_relative_error for item in pairwise),
        maximum_final_mean_relative_error=max(
            item.final_mean_relative_error for item in pairwise
        ),
        minimum_smoothed_correlation=min(
            item.smoothed_correlation for item in pairwise
        ),
        maximum_sustained_window_relative_error=max(
            item.maximum_sustained_window_relative_error for item in pairwise
        ),
    )


def _series(
    readers: Sequence[PrecisionArtifactReader], name: str
) -> tuple[tuple[int, ...], tuple[tuple[float, ...], ...]]:
    mappings: list[Mapping[int, float]] = []
    for reader in readers:
        if name == "loss":
            mappings.append(reader.loss_series())
        elif name == "grad_norm":
            mappings.append(reader.grad_norm_series())
        else:
            raise ValueError(f"unsupported series: {name}")
    steps = tuple(sorted(mappings[0]))
    if any(tuple(sorted(mapping)) != steps for mapping in mappings[1:]):
        raise ValueError(f"{name} repeat steps do not match")
    return steps, tuple(tuple(mapping[step] for step in steps) for mapping in mappings)


def _validate_contracts(readers: Sequence[PrecisionArtifactReader]) -> None:
    expected = readers[0].training_contract
    for reader in readers[1:]:
        if reader.training_contract != expected:
            raise ValueError(
                "GPU/NPU artifacts use different training contracts; comparison is invalid"
            )
    topology = expected.get("topology", {})
    replicate = int(topology.get("data_parallel_replicate_degree", 1))
    shard = int(topology.get("data_parallel_shard_degree", 1))
    tensor = int(topology.get("tensor_parallel_degree", 1))
    pipeline = int(topology.get("pipeline_parallel_degree", 1))
    expert = int(topology.get("expert_parallel_degree", 1))
    if replicate <= 1 or (shard, tensor, pipeline, expert) != (1, 1, 1, 1):
        raise ValueError(
            "ddp-long-v2 requires a pure multi-rank DDP topology "
            "(replicate>1; shard=tp=pp=ep=1)"
        )


def _criterion(
    *,
    name: str,
    observed: float,
    maximum: float,
    category: str,
    explanation: str,
) -> Criterion:
    return Criterion(
        name=name,
        passed=observed <= maximum,
        observed=f"{observed:.6%}",
        required=f"<= {maximum:.3%}",
        category=category,
        explanation=explanation,
    )


def compare_ddp_long_v2(
    gpu_artifacts: Sequence[str | Path],
    npu_artifacts: Sequence[str | Path],
    *,
    config: DdpLongV2Config = DdpLongV2Config(),
) -> DdpLongV2Result:
    """Compare existing GPU/NPU DDP artifacts without rerunning training."""

    if len(gpu_artifacts) < 2 or len(npu_artifacts) < 2:
        raise ValueError("ddp-long-v2 requires at least two GPU and two NPU artifacts")
    gpu = tuple(PrecisionArtifactReader(path) for path in gpu_artifacts)
    npu = tuple(PrecisionArtifactReader(path) for path in npu_artifacts)
    _validate_contracts((*gpu, *npu))

    gpu_loss_steps, gpu_loss_curves = _series(gpu, "loss")
    npu_loss_steps, npu_loss_curves = _series(npu, "loss")
    gpu_grad_steps, gpu_grad_curves = _series(gpu, "grad_norm")
    npu_grad_steps, npu_grad_curves = _series(npu, "grad_norm")
    if not (
        gpu_loss_steps == npu_loss_steps == gpu_grad_steps == npu_grad_steps
    ):
        raise ValueError("GPU/NPU loss and grad-norm steps do not match")

    steps = gpu_loss_steps
    all_values = [
        value
        for curves in (
            gpu_loss_curves,
            npu_loss_curves,
            gpu_grad_curves,
            npu_grad_curves,
        )
        for curve in curves
        for value in curve
    ]
    finite = all(math.isfinite(value) for value in all_values)

    gpu_repeat_loss = _maximum_pairwise_mare(gpu_loss_curves)
    npu_repeat_loss = _maximum_pairwise_mare(npu_loss_curves)
    gpu_repeat_grad = _maximum_pairwise_mare(gpu_grad_curves)
    npu_repeat_grad = _maximum_pairwise_mare(npu_grad_curves)
    gpu_loss_mean = _mean_curve(gpu_loss_curves)
    npu_loss_mean = _mean_curve(npu_loss_curves)
    gpu_grad_mean = _mean_curve(gpu_grad_curves)
    npu_grad_mean = _mean_curve(npu_grad_curves)

    # Replace non-finite values only to finish a structured failure report.
    safe_gpu_loss = tuple(value if math.isfinite(value) else 0.0 for value in gpu_loss_mean)
    safe_npu_loss = tuple(value if math.isfinite(value) else 0.0 for value in npu_loss_mean)
    safe_gpu_grad = tuple(value if math.isfinite(value) else 0.0 for value in gpu_grad_mean)
    safe_npu_grad = tuple(value if math.isfinite(value) else 0.0 for value in npu_grad_mean)
    loss = _diagnostics(safe_gpu_loss, safe_npu_loss)
    grad_norm = _diagnostics(safe_gpu_grad, safe_npu_grad)
    curve = _curve_diagnostics(safe_gpu_loss, safe_npu_loss, steps, config)
    gpu_repeat_curve = _repeat_curve_diagnostics(
        gpu_loss_curves, steps, config
    )
    npu_repeat_curve = _repeat_curve_diagnostics(
        npu_loss_curves, steps, config
    )

    criteria = [
        Criterion(
            name="Finite loss and grad norm",
            passed=finite,
            observed="all finite" if finite else "NaN or Inf found",
            required="all observations finite",
            category="hard_gate",
            explanation="NaN or Inf is a hard precision failure.",
        ),
        _criterion(
            name="Whole-run loss AUC relative difference",
            observed=curve.area_relative_error,
            maximum=config.curve_area_relative_limit,
            category="convergence_core",
            explanation=(
                "Compares the total area under both loss curves instead of "
                "requiring individual optimizer steps to match."
            ),
        ),
        _criterion(
            name="Final-window mean loss relative difference",
            observed=curve.final_mean_relative_error,
            maximum=config.final_mean_relative_limit,
            category="convergence_core",
            explanation=f"Checks the mean loss over the final {config.late_fraction:.0%}.",
        ),
        Criterion(
            name="Smoothed loss-curve correlation",
            passed=(
                curve.smoothed_correlation
                >= config.smoothed_correlation_minimum
            ),
            observed=f"{curve.smoothed_correlation:.8f}",
            required=f">= {config.smoothed_correlation_minimum:.5f}",
            category="convergence_core",
            explanation=(
                f"Uses a {config.window_size}-step moving average after the "
                f"first {config.warmup_steps} steps."
            ),
        ),
        _criterion(
            name="Maximum sustained-window mean loss difference",
            observed=curve.maximum_sustained_window_relative_error,
            maximum=config.sustained_window_relative_limit,
            category="convergence_core",
            explanation=(
                f"Checks every rolling {config.sustained_window_size}-step window; "
                f"the worst is step {curve.worst_window_start_step}.."
                f"{curve.worst_window_end_step}."
            ),
        ),
    ]

    inconclusive_reasons: list[str] = []
    if len(steps) < config.minimum_steps:
        inconclusive_reasons.append(
            f"only {len(steps)} steps were captured; {config.minimum_steps} are required"
        )
    for platform, repeat in (
        ("GPU", gpu_repeat_curve),
        ("NPU", npu_repeat_curve),
    ):
        if repeat.maximum_area_relative_error > config.curve_area_relative_limit:
            inconclusive_reasons.append(
                f"{platform} repeat loss AUC difference "
                f"{repeat.maximum_area_relative_error:.6%} exceeds "
                f"{config.curve_area_relative_limit:.3%}"
            )
        if (
            repeat.maximum_final_mean_relative_error
            > config.final_mean_relative_limit
        ):
            inconclusive_reasons.append(
                f"{platform} repeat final-window mean loss difference "
                f"{repeat.maximum_final_mean_relative_error:.6%} exceeds "
                f"{config.final_mean_relative_limit:.3%}"
            )
        if (
            repeat.minimum_smoothed_correlation
            < config.smoothed_correlation_minimum
        ):
            inconclusive_reasons.append(
                f"{platform} repeat smoothed loss correlation "
                f"{repeat.minimum_smoothed_correlation:.8f} is below "
                f"{config.smoothed_correlation_minimum:.5f}"
            )
        if (
            repeat.maximum_sustained_window_relative_error
            > config.sustained_window_relative_limit
        ):
            inconclusive_reasons.append(
                f"{platform} repeat sustained-window mean loss difference "
                f"{repeat.maximum_sustained_window_relative_error:.6%} exceeds "
                f"{config.sustained_window_relative_limit:.3%}"
            )

    if inconclusive_reasons:
        status = "INCONCLUSIVE"
    elif all(item.passed for item in criteria):
        status = "PASS"
    else:
        status = "FAIL"

    return DdpLongV2Result(
        status=status,
        criteria=tuple(criteria),
        inconclusive_reasons=tuple(inconclusive_reasons),
        steps=steps,
        loss=loss,
        grad_norm=grad_norm,
        curve=curve,
        gpu_repeat_curve=gpu_repeat_curve,
        npu_repeat_curve=npu_repeat_curve,
        gpu_repeat_loss_mare=gpu_repeat_loss,
        npu_repeat_loss_mare=npu_repeat_loss,
        gpu_repeat_grad_norm_mare=gpu_repeat_grad,
        npu_repeat_grad_norm_mare=npu_repeat_grad,
        gpu_loss_bitwise_reproducible=_all_exact(gpu_loss_curves),
        npu_loss_bitwise_reproducible=_all_exact(npu_loss_curves),
        gpu_grad_norm_bitwise_reproducible=_all_exact(gpu_grad_curves),
        npu_grad_norm_bitwise_reproducible=_all_exact(npu_grad_curves),
        config=config,
    )


def _write_report(result: DdpLongV2Result, output_directory: Path) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "ddp_long_v2_summary.json"
    payload = asdict(result)
    payload["step_range"] = [result.steps[0], result.steps[-1]]
    payload.pop("steps")
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# DDP long-run convergence alignment V2",
        "",
        f"Result: **{result.status}**",
        "",
        f"Steps: {result.steps[0]}..{result.steps[-1]} ({len(result.steps)} observations)",
        "",
        "| Criterion | Result | Observed | Required | Category |",
        "|---|---:|---:|---:|---|",
    ]
    for item in result.criteria:
        lines.append(
            f"| {item.name} | {'PASS' if item.passed else 'FAIL'} | "
            f"{item.observed} | {item.required} | {item.category} |"
        )
    lines.extend(
        [
            "",
            "## Pointwise diagnostics (non-gating)",
            "",
            f"- Raw loss MARE: {result.loss.mean_absolute_relative_error:.6%}",
            f"- Raw loss P95 relative error: {result.loss.p95_absolute_relative_error:.6%}",
            f"- Raw loss P99 relative error: {result.loss.p99_absolute_relative_error:.6%}",
            f"- Raw loss maximum relative error: {result.loss.maximum_absolute_relative_error:.6%}",
            f"- Grad-norm MARE: {result.grad_norm.mean_absolute_relative_error:.6%}",
            "",
            "## Repeat stability",
            "",
            (
                "- GPU repeat maximum loss AUC difference: "
                f"{result.gpu_repeat_curve.maximum_area_relative_error:.6%}"
            ),
            (
                "- NPU repeat maximum loss AUC difference: "
                f"{result.npu_repeat_curve.maximum_area_relative_error:.6%}"
            ),
            (
                "- GPU repeat minimum smoothed correlation: "
                f"{result.gpu_repeat_curve.minimum_smoothed_correlation:.8f}"
            ),
            (
                "- NPU repeat minimum smoothed correlation: "
                f"{result.npu_repeat_curve.minimum_smoothed_correlation:.8f}"
            ),
            f"- GPU loss MARE: {result.gpu_repeat_loss_mare:.6%}",
            f"- NPU loss MARE: {result.npu_repeat_loss_mare:.6%}",
            f"- GPU grad-norm MARE: {result.gpu_repeat_grad_norm_mare:.6%}",
            f"- NPU grad-norm MARE: {result.npu_repeat_grad_norm_mare:.6%}",
            f"- GPU loss bitwise reproducible: {result.gpu_loss_bitwise_reproducible}",
            f"- NPU loss bitwise reproducible: {result.npu_loss_bitwise_reproducible}",
            "",
            (
                "Raw pointwise loss error, grad norm, and bitwise equality are "
                "diagnostic in V2 and do not independently decide PASS/FAIL."
            ),
        ]
    )
    if result.inconclusive_reasons:
        lines.extend(["", "## Inconclusive reasons", ""])
        lines.extend(f"- {reason}" for reason in result.inconclusive_reasons)
    markdown_path = output_directory / "ddp_long_v2_report.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-score existing GPU/NPU DDP artifacts using overall loss-curve "
            "convergence equivalence."
        )
    )
    parser.add_argument("--gpu-artifact", action="append", required=True, type=Path)
    parser.add_argument("--npu-artifact", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--minimum-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--sustained-window-size", type=int, default=500)
    parser.add_argument("--curve-area-relative-limit", type=float, default=0.02)
    parser.add_argument("--final-mean-relative-limit", type=float, default=0.02)
    parser.add_argument("--smoothed-correlation-minimum", type=float, default=0.99)
    parser.add_argument(
        "--sustained-window-relative-limit", type=float, default=0.03
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = compare_ddp_long_v2(
            args.gpu_artifact,
            args.npu_artifact,
            config=DdpLongV2Config(
                minimum_steps=args.minimum_steps,
                warmup_steps=args.warmup_steps,
                window_size=args.window_size,
                sustained_window_size=args.sustained_window_size,
                curve_area_relative_limit=args.curve_area_relative_limit,
                final_mean_relative_limit=args.final_mean_relative_limit,
                smoothed_correlation_minimum=args.smoothed_correlation_minimum,
                sustained_window_relative_limit=(
                    args.sustained_window_relative_limit
                ),
            ),
        )
        json_path, markdown_path = _write_report(result, args.output_dir)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ddp-long-v2: {error}", file=sys.stderr)
        return 2
    print(f"DDP long-run convergence V2: {result.status}")
    print(f"JSON: {json_path}")
    print(f"Report: {markdown_path}")
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[result.status]


if __name__ == "__main__":
    raise SystemExit(main())
