# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Offline MindStudio-aligned DDP long-run precision assessment.

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
from typing import Any, Mapping, Sequence

from .artifacts import PrecisionArtifactReader


@dataclass(frozen=True)
class DdpLongV2Config:
    """Project guardrails around MindStudio's public one-percent loss signal."""

    relative_loss_limit: float = 0.01
    pointwise_p99_limit: float = 0.05
    warmup_steps: int = 100
    late_fraction: float = 0.20
    window_size: int = 100
    consecutive_bad_windows: int = 3
    minimum_steps: int = 5000

    def __post_init__(self) -> None:
        if self.relative_loss_limit < 0.0:
            raise ValueError("relative_loss_limit must be nonnegative")
        if self.pointwise_p99_limit < 0.0:
            raise ValueError("pointwise_p99_limit must be nonnegative")
        if self.warmup_steps < 1:
            raise ValueError("warmup_steps must be positive")
        if not 0.0 < self.late_fraction <= 1.0:
            raise ValueError("late_fraction must be in (0, 1]")
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        if self.consecutive_bad_windows < 1:
            raise ValueError("consecutive_bad_windows must be positive")
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
class DdpLongV2Result:
    status: str
    criteria: tuple[Criterion, ...]
    inconclusive_reasons: tuple[str, ...]
    steps: tuple[int, ...]
    loss: SeriesDiagnostics
    grad_norm: SeriesDiagnostics
    gpu_repeat_loss_mare: float
    npu_repeat_loss_mare: float
    gpu_repeat_grad_norm_mare: float
    npu_repeat_grad_norm_mare: float
    gpu_loss_bitwise_reproducible: bool
    npu_loss_bitwise_reproducible: bool
    gpu_grad_norm_bitwise_reproducible: bool
    npu_grad_norm_bitwise_reproducible: bool
    bad_window_streak: int
    bad_windows: tuple[dict[str, Any], ...]
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

    early_count = min(config.warmup_steps, len(steps))
    retained_start = min(config.warmup_steps, len(steps))
    retained_gpu = safe_gpu_loss[retained_start:]
    retained_npu = safe_npu_loss[retained_start:]
    if not retained_gpu:
        retained_gpu = safe_gpu_loss
        retained_npu = safe_npu_loss
    late_count = max(1, math.ceil(len(retained_gpu) * config.late_fraction))

    first_mare = _diagnostics(safe_gpu_loss[:1], safe_npu_loss[:1]).mean_absolute_relative_error
    early_mare = _diagnostics(
        safe_gpu_loss[:early_count], safe_npu_loss[:early_count]
    ).mean_absolute_relative_error
    retained_diagnostics = _diagnostics(retained_gpu, retained_npu)
    late_mare = _diagnostics(
        retained_gpu[-late_count:], retained_npu[-late_count:]
    ).mean_absolute_relative_error

    bad_windows: list[dict[str, Any]] = []
    bad_streak = 0
    maximum_bad_streak = 0
    for offset in range(0, len(retained_gpu), config.window_size):
        left = retained_gpu[offset : offset + config.window_size]
        right = retained_npu[offset : offset + config.window_size]
        if len(left) < config.window_size:
            break
        mare = _diagnostics(left, right).mean_absolute_relative_error
        if mare > config.relative_loss_limit:
            bad_streak += 1
            bad_windows.append(
                {
                    "start_step": steps[retained_start + offset],
                    "end_step": steps[retained_start + offset + len(left) - 1],
                    "mare": mare,
                }
            )
        else:
            bad_streak = 0
        maximum_bad_streak = max(maximum_bad_streak, bad_streak)

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
            name="First-step loss MARE",
            observed=first_mare,
            maximum=config.relative_loss_limit,
            category="mindstudio_core",
            explanation="Checks the first-step difference with a relative metric.",
        ),
        _criterion(
            name=f"First {early_count} steps loss MARE",
            observed=early_mare,
            maximum=config.relative_loss_limit,
            category="mindstudio_core",
            explanation="Covers the early-training window instead of silently dropping it.",
        ),
        _criterion(
            name="Post-warmup loss MARE",
            observed=retained_diagnostics.mean_absolute_relative_error,
            maximum=config.relative_loss_limit,
            category="mindstudio_core",
            explanation="The public MindStudio long-stable signal uses one-percent mean error.",
        ),
        _criterion(
            name="Final-window loss MARE",
            observed=late_mare,
            maximum=config.relative_loss_limit,
            category="drift_guardrail",
            explanation=f"Checks the final {config.late_fraction:.0%} of retained steps.",
        ),
        _criterion(
            name="Post-warmup pointwise relative-error P99",
            observed=retained_diagnostics.p99_absolute_relative_error,
            maximum=config.pointwise_p99_limit,
            category="spike_guardrail",
            explanation="Controls spikes without deleting the largest errors.",
        ),
        Criterion(
            name="Sustained bad-window streak",
            passed=maximum_bad_streak < config.consecutive_bad_windows,
            observed=f"{maximum_bad_streak} consecutive bad windows",
            required=f"< {config.consecutive_bad_windows}",
            category="drift_guardrail",
            explanation=(
                f"A bad window is {config.window_size} complete steps with loss MARE "
                f"> {config.relative_loss_limit:.3%}."
            ),
        ),
    ]

    inconclusive_reasons: list[str] = []
    if len(steps) < config.minimum_steps:
        inconclusive_reasons.append(
            f"only {len(steps)} steps were captured; {config.minimum_steps} are required"
        )
    for platform, value in (("GPU", gpu_repeat_loss), ("NPU", npu_repeat_loss)):
        if value > config.relative_loss_limit:
            inconclusive_reasons.append(
                f"{platform} repeat loss MARE {value:.6%} exceeds "
                f"{config.relative_loss_limit:.3%}"
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
        gpu_repeat_loss_mare=gpu_repeat_loss,
        npu_repeat_loss_mare=npu_repeat_loss,
        gpu_repeat_grad_norm_mare=gpu_repeat_grad,
        npu_repeat_grad_norm_mare=npu_repeat_grad,
        gpu_loss_bitwise_reproducible=_all_exact(gpu_loss_curves),
        npu_loss_bitwise_reproducible=_all_exact(npu_loss_curves),
        gpu_grad_norm_bitwise_reproducible=_all_exact(gpu_grad_curves),
        npu_grad_norm_bitwise_reproducible=_all_exact(npu_grad_curves),
        bad_window_streak=maximum_bad_streak,
        bad_windows=tuple(bad_windows),
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
        "# DDP long-run precision V2",
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
            "## Repeat stability",
            "",
            f"- GPU loss MARE: {result.gpu_repeat_loss_mare:.6%}",
            f"- NPU loss MARE: {result.npu_repeat_loss_mare:.6%}",
            f"- GPU grad-norm MARE: {result.gpu_repeat_grad_norm_mare:.6%}",
            f"- NPU grad-norm MARE: {result.npu_repeat_grad_norm_mare:.6%}",
            f"- GPU loss bitwise reproducible: {result.gpu_loss_bitwise_reproducible}",
            f"- NPU loss bitwise reproducible: {result.npu_loss_bitwise_reproducible}",
            "",
            "Grad norm is diagnostic in V2 and does not independently decide PASS/FAIL.",
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
        description="Re-score existing GPU/NPU DDP artifacts with the long-run V2 standard."
    )
    parser.add_argument("--gpu-artifact", action="append", required=True, type=Path)
    parser.add_argument("--npu-artifact", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--minimum-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=100)
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
            ),
        )
        json_path, markdown_path = _write_report(result, args.output_dir)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ddp-long-v2: {error}", file=sys.stderr)
        return 2
    print(f"DDP long-run V2: {result.status}")
    print(f"JSON: {json_path}")
    print(f"Report: {markdown_path}")
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[result.status]


if __name__ == "__main__":
    raise SystemExit(main())
