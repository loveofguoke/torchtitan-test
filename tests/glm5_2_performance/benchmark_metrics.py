"""Statistics and validity rules for profiler-off training benchmarks."""

from __future__ import annotations

import math
import random
import statistics
from typing import Any, Iterable


def percentile(values: Iterable[float], fraction: float) -> float | None:
    samples = sorted(float(value) for value in values)
    if not samples:
        return None
    position = (len(samples) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return samples[lower]
    weight = position - lower
    return samples[lower] * (1 - weight) + samples[upper] * weight


def metric_statistics(values: Iterable[float]) -> dict[str, float | int | None]:
    """Return ordinary, tail, and robust statistics for one metric series."""

    samples = [float(value) for value in values]
    if not samples:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
            "last": None,
            "sample_stddev": None,
            "cv_percent": None,
            "mad": None,
            "iqr": None,
            "drift_percent_per_100_steps": None,
        }
    mean = statistics.fmean(samples)
    median = statistics.median(samples)
    sample_stddev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    mad = statistics.median(abs(value - median) for value in samples)
    q25 = percentile(samples, 0.25)
    q75 = percentile(samples, 0.75)

    drift = 0.0
    if len(samples) > 1 and median:
        center = (len(samples) - 1) / 2.0
        denominator = sum((index - center) ** 2 for index in range(len(samples)))
        if denominator:
            slope = sum(
                (index - center) * (value - mean)
                for index, value in enumerate(samples)
            ) / denominator
            drift = slope / abs(median) * 100.0 * 100.0

    return {
        "count": len(samples),
        "mean": mean,
        "median": median,
        "p90": percentile(samples, 0.90),
        "p95": percentile(samples, 0.95),
        "p99": percentile(samples, 0.99),
        "min": min(samples),
        "max": max(samples),
        "last": samples[-1],
        "sample_stddev": sample_stddev,
        "cv_percent": sample_stddev / abs(mean) * 100.0 if mean else None,
        "mad": mad,
        "iqr": float(q75 - q25) if q25 is not None and q75 is not None else None,
        "drift_percent_per_100_steps": drift,
    }


def bootstrap_median_ci(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 61,
) -> list[float] | None:
    """Deterministic percentile-bootstrap CI over independent run summaries."""

    samples = [float(value) for value in values]
    if len(samples) < 2:
        return None
    generator = random.Random(seed)
    medians = sorted(
        statistics.median(generator.choices(samples, k=len(samples)))
        for _ in range(resamples)
    )
    alpha = (1.0 - confidence) / 2.0
    lower = percentile(medians, alpha)
    upper = percentile(medians, 1.0 - alpha)
    assert lower is not None and upper is not None
    return [lower, upper]


def numerical_validity(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Preserve explicit non-finite evidence without invalidating old records."""

    records = list(records)
    failures = [
        {
            "step": int(record["step"]),
            "metrics": sorted(str(name) for name in record["nonfinite_metrics"]),
        }
        for record in records
        if record.get("nonfinite_metrics")
    ]
    numerical_markers = ("loss", "grad_norm", "gradient_norm")
    observed = sorted(
        {
            str(name)
            for record in records
            for name in record.get("metrics", {})
            if any(marker in str(name).lower() for marker in numerical_markers)
        }
    )
    if failures:
        status = "invalid"
        reasons = ["non-finite training metrics were recorded"]
    elif observed:
        status = "valid"
        reasons = []
    else:
        status = "not_available"
        reasons = ["loss and gradient-norm evidence is absent"]
    return {
        "status": status,
        "reasons": reasons,
        "observed_metrics": observed,
        "nonfinite_events": failures,
    }


def detect_steady_state(
    steps: Iterable[int],
    step_seconds: Iterable[float],
    *,
    minimum_window: int = 10,
    maximum_cv_percent: float = 2.0,
    maximum_absolute_drift_percent_per_100_steps: float = 1.0,
) -> dict[str, Any]:
    """Find the earliest suffix satisfying the project steady-state policy."""

    step_values = [int(step) for step in steps]
    time_values = [float(value) for value in step_seconds]
    if len(step_values) != len(time_values):
        raise ValueError("step indices and step-time samples must have equal length")
    policy = {
        "minimum_window": minimum_window,
        "maximum_cv_percent": maximum_cv_percent,
        "maximum_absolute_drift_percent_per_100_steps": (
            maximum_absolute_drift_percent_per_100_steps
        ),
    }
    if len(time_values) < minimum_window:
        return {
            "status": "not_available",
            "reason": "fewer step-time samples than the minimum steady window",
            "policy": policy,
            "selected_steps": [],
            "statistics": None,
        }
    latest = None
    for start in range(0, len(time_values) - minimum_window + 1):
        selected = time_values[start:]
        summary = metric_statistics(selected)
        latest = (start, summary)
        cv = summary["cv_percent"]
        drift = summary["drift_percent_per_100_steps"]
        if (
            cv is not None
            and drift is not None
            and cv <= maximum_cv_percent
            and abs(drift) <= maximum_absolute_drift_percent_per_100_steps
        ):
            return {
                "status": "reached",
                "reason": None,
                "policy": policy,
                "selected_steps": step_values[start:],
                "statistics": summary,
            }
    assert latest is not None
    start, summary = latest
    return {
        "status": "not_reached",
        "reason": "no suffix satisfied both CV and drift thresholds",
        "policy": policy,
        "selected_steps": step_values[start:],
        "statistics": summary,
    }


__all__ = [
    "bootstrap_median_ci",
    "detect_steady_state",
    "metric_statistics",
    "numerical_validity",
    "percentile",
]
