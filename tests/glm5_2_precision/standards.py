# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Configurable numerical standards for formal training comparisons.

Metric computation is independent from pass/fail policy. Migration compares
equivalent GPU/NPU executions with tolerance-based loss and grad-norm criteria;
self-consistency can additionally request exact repeat checks but uses the same
formal tolerance when BF16 distributed order makes bitwise identity unrealistic.
Warmup removal, outlier policy, quantiles, and customer reference limits remain
data/configuration rather than being hidden inside report rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Mapping, Sequence


DEFAULT_QUANTILES = (0.5, 0.9, 0.99, 0.999, 0.9999)


@dataclass(frozen=True)
class ErrorMetrics:
    """Four documented errors plus diagnostics used by the HTML report."""

    count: int
    normal: float
    absolute: float
    relative_normal: float
    relative_absolute: float
    max_absolute: float
    quantiles: dict[float, float]
    absolute_errors: tuple[float, ...] = field(repr=False)
    relative_errors: tuple[float, ...] = field(repr=False)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantiles require at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"invalid quantile probability: {probability}")
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def compute_error_metrics(
    reference: Sequence[float],
    candidate: Sequence[float],
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> ErrorMetrics:
    """Compute signed/absolute and relative errors against the reference.

    For aligned values ``x_i`` (reference) and ``y_i`` (candidate), errors are
    ``e_i = x_i - y_i``. ``normal`` is mean signed error, ``absolute`` is mean
    ``|e_i|``, and relative variants divide each observation by ``x_i`` before
    averaging. Zero reference values are rejected because their relative error
    is undefined rather than silently hidden behind an epsilon.
    """

    if len(reference) != len(candidate):
        raise ValueError(
            "reference and candidate series must have the same length: "
            f"{len(reference)} != {len(candidate)}"
        )
    if not reference:
        raise ValueError("error metrics require at least one observation")

    errors = tuple(left - right for left, right in zip(reference, candidate))
    absolute_errors = tuple(abs(value) for value in errors)
    relative_errors: list[float] = []
    relative_absolute_errors: list[float] = []
    for index, (left, error) in enumerate(zip(reference, errors)):
        if left == 0.0:
            raise ValueError(
                "relative metrics are undefined when a reference value is zero "
                f"(observation {index})"
            )
        relative_errors.append(error / left)
        # This follows the internal standard: only the numerator is absolute.
        # Loss and global grad norm reference values are nonnegative.
        relative_absolute_errors.append(abs(error) / left)

    count = len(errors)
    return ErrorMetrics(
        count=count,
        normal=sum(errors) / count,
        absolute=sum(absolute_errors) / count,
        relative_normal=sum(relative_errors) / count,
        relative_absolute=sum(relative_absolute_errors) / count,
        max_absolute=max(absolute_errors),
        quantiles={
            probability: _quantile(absolute_errors, probability)
            for probability in quantiles
        },
        absolute_errors=absolute_errors,
        relative_errors=tuple(relative_errors),
    )


@dataclass(frozen=True)
class AnyOfErrorLimit:
    """Pass when any configured metric limit passes."""

    absolute: float | None = None
    relative_absolute: float | None = None
    max_absolute: float | None = None

    def __post_init__(self) -> None:
        limits = (self.absolute, self.relative_absolute, self.max_absolute)
        if all(limit is None for limit in limits):
            raise ValueError("at least one error limit must be configured")
        if any(limit is not None and limit < 0.0 for limit in limits):
            raise ValueError("error limits must be nonnegative")


@dataclass(frozen=True)
class RangeLimit:
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if self.minimum > self.maximum:
            raise ValueError("range minimum cannot exceed maximum")


@dataclass(frozen=True)
class QuantileLimit:
    probability: float
    maximum: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("quantile probability must be in [0, 1]")
        if self.maximum < 0.0:
            raise ValueError("quantile maximum must be nonnegative")


@dataclass(frozen=True)
class MigrationStandard:
    """GPU/NPU migration criteria derived from the internal guidance."""

    first_loss: AnyOfErrorLimit = field(
        default_factory=lambda: AnyOfErrorLimit(
            absolute=0.005,
            relative_absolute=0.005,
        )
    )
    all_loss: AnyOfErrorLimit = field(
        default_factory=lambda: AnyOfErrorLimit(
            absolute=0.01,
            relative_absolute=0.01,
        )
    )
    grad_norm_relative_normal: RangeLimit = field(
        default_factory=lambda: RangeLimit(-0.05, 0.05)
    )
    # Tail limits are retained for diagnostic reporting. They do not decide
    # the formal migration result.
    loss_quantiles: tuple[QuantileLimit, ...] = ()
    warmup_steps: int = 0
    outlier_fraction: float = 0.0
    minimum_observations: int = 1000
    required_reference_repeats: int = 2
    required_candidate_repeats: int = 2

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be nonnegative")
        if not 0.0 <= self.outlier_fraction < 1.0:
            raise ValueError("outlier_fraction must be in [0, 1)")
        if self.minimum_observations < 1:
            raise ValueError("minimum_observations must be positive")
        if self.required_reference_repeats < 2:
            raise ValueError("reference reproducibility requires at least two runs")
        if self.required_candidate_repeats < 2:
            raise ValueError("candidate reproducibility requires at least two runs")


@dataclass(frozen=True)
class SelfConsistencyStandard:
    """Self-consistency is exact unless a change affects determinism."""

    randomness_impacted: bool = True
    migration_fallback: MigrationStandard = field(default_factory=MigrationStandard)
    required_reference_repeats: int = 2
    required_candidate_repeats: int = 2

    def __post_init__(self) -> None:
        if self.required_reference_repeats < 2:
            raise ValueError("reference reproducibility requires at least two runs")
        if self.required_candidate_repeats < 2:
            raise ValueError("candidate reproducibility requires at least two runs")


@dataclass(frozen=True)
class PrecisionStandard:
    migration: MigrationStandard = field(default_factory=MigrationStandard)
    self_consistency: SelfConsistencyStandard = field(
        default_factory=SelfConsistencyStandard
    )


def customer3_strict_standard() -> PrecisionStandard:
    """Return the customer-3 cumulative profile with diagnostic tail limits."""

    migration = MigrationStandard(
        all_loss=AnyOfErrorLimit(absolute=0.005),
        loss_quantiles=(
            QuantileLimit(0.99, 0.01),
            QuantileLimit(0.999, 0.01),
            QuantileLimit(0.9999, 0.05),
        ),
        warmup_steps=100,
        minimum_observations=1000,
    )
    return PrecisionStandard(
        migration=migration,
        self_consistency=SelfConsistencyStandard(migration_fallback=migration),
    )


@dataclass(frozen=True)
class CriterionResult:
    name: str
    passed: bool
    observed: str
    required: str
    explanation: str


@dataclass(frozen=True)
class SeriesEvaluation:
    name: str
    reference: tuple[float, ...]
    candidate: tuple[float, ...]
    steps: tuple[int, ...]
    metrics: ErrorMetrics


@dataclass(frozen=True)
class PairEvaluation:
    passed: bool
    criteria: tuple[CriterionResult, ...]
    loss: SeriesEvaluation
    grad_norm: SeriesEvaluation
    dropped_steps: tuple[int, ...]


def _format_limit(limit: AnyOfErrorLimit) -> str:
    values: list[str] = []
    if limit.absolute is not None:
        values.append(f"mean absolute <= {limit.absolute:g}")
    if limit.relative_absolute is not None:
        values.append(f"mean absolute relative <= {limit.relative_absolute:.3%}")
    if limit.max_absolute is not None:
        values.append(f"max absolute <= {limit.max_absolute:g}")
    return " OR ".join(values)


def _evaluate_any_of(
    metrics: ErrorMetrics,
    limit: AnyOfErrorLimit,
    *,
    name: str,
    explanation: str,
) -> CriterionResult:
    outcomes: list[bool] = []
    observed: list[str] = []
    if limit.absolute is not None:
        outcomes.append(metrics.absolute <= limit.absolute)
        observed.append(f"mean absolute={metrics.absolute:.9g}")
    if limit.relative_absolute is not None:
        outcomes.append(metrics.relative_absolute <= limit.relative_absolute)
        observed.append(f"mean absolute relative={metrics.relative_absolute:.6%}")
    if limit.max_absolute is not None:
        outcomes.append(metrics.max_absolute <= limit.max_absolute)
        observed.append(f"max absolute={metrics.max_absolute:.9g}")
    return CriterionResult(
        name=name,
        passed=any(outcomes),
        observed="; ".join(observed),
        required=_format_limit(limit),
        explanation=explanation,
    )


def _align_series(
    reference: Mapping[int, float],
    candidate: Mapping[int, float],
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...]]:
    if set(reference) != set(candidate):
        missing_candidate = sorted(set(reference) - set(candidate))
        missing_reference = sorted(set(candidate) - set(reference))
        raise ValueError(
            "metric steps differ; "
            f"missing from candidate={missing_candidate}, "
            f"missing from reference={missing_reference}"
        )
    steps = tuple(sorted(reference))
    return (
        steps,
        tuple(reference[step] for step in steps),
        tuple(candidate[step] for step in steps),
    )


def _drop_warmup(
    steps: tuple[int, ...],
    reference: tuple[float, ...],
    candidate: tuple[float, ...],
    warmup_steps: int,
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...]]:
    if warmup_steps >= len(steps):
        raise ValueError("warmup removes every observation")
    return (
        steps[warmup_steps:],
        reference[warmup_steps:],
        candidate[warmup_steps:],
    )


def _drop_largest_outliers(
    steps: tuple[int, ...],
    reference: tuple[float, ...],
    candidate: tuple[float, ...],
    fraction: float,
) -> tuple[
    tuple[int, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[int, ...],
]:
    num_drop = math.floor(len(steps) * fraction)
    if num_drop == 0:
        return steps, reference, candidate, ()
    ranked = sorted(
        range(len(steps)),
        key=lambda index: abs(reference[index] - candidate[index]),
        reverse=True,
    )
    dropped_indices = set(ranked[:num_drop])
    kept_indices = [index for index in range(len(steps)) if index not in dropped_indices]
    return (
        tuple(steps[index] for index in kept_indices),
        tuple(reference[index] for index in kept_indices),
        tuple(candidate[index] for index in kept_indices),
        tuple(sorted(steps[index] for index in dropped_indices)),
    )


def evaluate_migration_pair(
    *,
    reference_loss: Mapping[int, float],
    candidate_loss: Mapping[int, float],
    reference_grad_norm: Mapping[int, float],
    candidate_grad_norm: Mapping[int, float],
    standard: MigrationStandard,
) -> PairEvaluation:
    """Evaluate one reference/candidate run pair using the formal criteria."""

    raw_loss_steps, raw_reference_loss, raw_candidate_loss = _align_series(
        reference_loss, candidate_loss
    )
    first_loss_metrics = compute_error_metrics(
        raw_reference_loss[:1], raw_candidate_loss[:1]
    )

    loss_steps, reference_loss_values, candidate_loss_values = _drop_warmup(
        raw_loss_steps,
        raw_reference_loss,
        raw_candidate_loss,
        standard.warmup_steps,
    )
    grad_steps, reference_grad_values, candidate_grad_values = _align_series(
        reference_grad_norm, candidate_grad_norm
    )
    grad_steps, reference_grad_values, candidate_grad_values = _drop_warmup(
        grad_steps,
        reference_grad_values,
        candidate_grad_values,
        standard.warmup_steps,
    )
    if loss_steps != grad_steps:
        raise ValueError("loss and grad norm observations use different steps")

    (
        filtered_steps,
        filtered_reference_loss,
        filtered_candidate_loss,
        dropped_steps,
    ) = _drop_largest_outliers(
        loss_steps,
        reference_loss_values,
        candidate_loss_values,
        standard.outlier_fraction,
    )
    keep_steps = set(filtered_steps)
    filtered_reference_grad = tuple(
        value for step, value in zip(grad_steps, reference_grad_values) if step in keep_steps
    )
    filtered_candidate_grad = tuple(
        value for step, value in zip(grad_steps, candidate_grad_values) if step in keep_steps
    )

    probabilities = set(DEFAULT_QUANTILES)
    probabilities.update(limit.probability for limit in standard.loss_quantiles)
    loss_metrics = compute_error_metrics(
        filtered_reference_loss,
        filtered_candidate_loss,
        quantiles=tuple(sorted(probabilities)),
    )
    grad_metrics = compute_error_metrics(
        filtered_reference_grad,
        filtered_candidate_grad,
    )

    criteria: list[CriterionResult] = [
        CriterionResult(
            name="Observation count",
            passed=len(raw_loss_steps) >= standard.minimum_observations,
            observed=f"{len(raw_loss_steps)} steps",
            required=f">= {standard.minimum_observations} steps",
            explanation="The observation window includes warmup and retained steps.",
        ),
        _evaluate_any_of(
            first_loss_metrics,
            standard.first_loss,
            name="First loss",
            explanation="The first logged training step is checked before warmup removal.",
        ),
        _evaluate_any_of(
            loss_metrics,
            standard.all_loss,
            name="All retained loss steps",
            explanation="The configured warmup and outlier policy is applied first.",
        ),
        CriterionResult(
            name="Global grad norm",
            passed=(
                standard.grad_norm_relative_normal.minimum
                <= grad_metrics.relative_normal
                <= standard.grad_norm_relative_normal.maximum
            ),
            observed=f"mean relative={grad_metrics.relative_normal:.6%}",
            required=(
                f"[{standard.grad_norm_relative_normal.minimum:.3%}, "
                f"{standard.grad_norm_relative_normal.maximum:.3%}]"
            ),
            explanation=(
                "Global grad norm may fluctuate, so signed mean relative error is used."
            ),
        ),
    ]
    return PairEvaluation(
        passed=all(criterion.passed for criterion in criteria),
        criteria=tuple(criteria),
        loss=SeriesEvaluation(
            name="loss",
            reference=filtered_reference_loss,
            candidate=filtered_candidate_loss,
            steps=filtered_steps,
            metrics=loss_metrics,
        ),
        grad_norm=SeriesEvaluation(
            name="global grad norm",
            reference=filtered_reference_grad,
            candidate=filtered_candidate_grad,
            steps=filtered_steps,
            metrics=grad_metrics,
        ),
        dropped_steps=dropped_steps,
    )


def evaluate_exact_pair(
    *,
    reference_loss: Mapping[int, float],
    candidate_loss: Mapping[int, float],
    reference_grad_norm: Mapping[int, float],
    candidate_grad_norm: Mapping[int, float],
) -> PairEvaluation:
    """Evaluate self-consistency when the change must preserve exact numerics."""

    loss_steps, reference_loss_values, candidate_loss_values = _align_series(
        reference_loss, candidate_loss
    )
    grad_steps, reference_grad_values, candidate_grad_values = _align_series(
        reference_grad_norm, candidate_grad_norm
    )
    if loss_steps != grad_steps:
        raise ValueError("loss and grad norm observations use different steps")
    loss_metrics = compute_error_metrics(reference_loss_values, candidate_loss_values)
    grad_metrics = compute_error_metrics(reference_grad_values, candidate_grad_values)
    loss_exact = series_digest(reference_loss) == series_digest(candidate_loss)
    grad_exact = series_digest(reference_grad_norm) == series_digest(candidate_grad_norm)
    criteria = (
        CriterionResult(
            name="Loss zero error",
            passed=loss_exact,
            observed=f"max absolute={loss_metrics.max_absolute:.9g}",
            required="bitwise-identical float series",
            explanation="Self-consistency without a randomness change requires zero loss error.",
        ),
        CriterionResult(
            name="Global grad norm zero error",
            passed=grad_exact,
            observed=f"max absolute={grad_metrics.max_absolute:.9g}",
            required="bitwise-identical float series",
            explanation=(
                "Self-consistency without a randomness change requires zero grad norm error."
            ),
        ),
    )
    return PairEvaluation(
        passed=loss_exact and grad_exact,
        criteria=criteria,
        loss=SeriesEvaluation(
            name="loss",
            reference=reference_loss_values,
            candidate=candidate_loss_values,
            steps=loss_steps,
            metrics=loss_metrics,
        ),
        grad_norm=SeriesEvaluation(
            name="global grad norm",
            reference=reference_grad_values,
            candidate=candidate_grad_values,
            steps=grad_steps,
            metrics=grad_metrics,
        ),
        dropped_steps=(),
    )


def series_digest(series: Mapping[int, float]) -> str:
    """Return a stable digest for bitwise reproducibility checks."""

    payload = json.dumps(
        [[step, float.hex(value)] for step, value in sorted(series.items())],
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def exact_series_match(left: Mapping[int, float], right: Mapping[int, float]) -> bool:
    return series_digest(left) == series_digest(right)
