# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Interactive whole-training observation report."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

from tests.glm5_2_common.reporting import echarts_line, metric_cards, save_panel_report


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4%}"


def _number(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6g}"


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
    return sum(values) / len(values) if values else None


def _values(rows: Sequence[dict[str, Any]], key: str, *, percent: bool = False) -> list[float | None]:
    scale = 100.0 if percent else 1.0
    return [float(row[key]) * scale if math.isfinite(float(row[key])) else None for row in rows]


def write_observation_report(
    *,
    output_directory: Path,
    rows: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> Path:
    loss = summary["loss"]
    grad = summary["grad_norm"]
    early = loss["early_window"]
    first_exceeded = loss["first_step_above_threshold"]
    nonfinite = summary["observation"]["nonfinite_analysis"]
    steps = [int(row["step"]) for row in rows]
    areas: list[tuple[str, int, int, str]] = []
    if early["first_step"] is not None:
        areas.append(("First-step inspection window", early["first_step"], early["last_step"], "#2563eb"))
    if first_exceeded is not None:
        areas.append(("After first guidance exceedance", first_exceeded, steps[-1], "#f59e0b"))
    loss_threshold_pct = float(loss["guidance_relative_threshold"]) * 100.0
    grad_threshold = grad["diagnostic_relative_threshold"]
    cards = metric_cards(
        [
            ("Compared steps", str(summary["step_count"]), f"{steps[0]} through {steps[-1]}"),
            ("Loss mean relative error", _percent(loss["mean_relative_error"]), f"guidance {loss_threshold_pct:g}%"),
            ("First Loss exceedance", "none" if first_exceeded is None else str(first_exceeded), "diagnostic trigger, not an automatic verdict"),
            ("Grad Norm mean relative error", _percent(grad["mean_relative_error"]), "diagnostic signal"),
            ("Reference non-finite", str(nonfinite["reference"]["metric_count"]), "metrics with NaN/Inf"),
            ("Candidate non-finite", str(nonfinite["candidate"]["metric_count"]), "metrics with NaN/Inf"),
        ]
    )
    charts = [
        echarts_line(
            title="Training Loss",
            subtitle=f"Mean relative error {_percent(loss['mean_relative_error'])}; first guidance exceedance: {first_exceeded if first_exceeded is not None else 'none'}",
            x_values=steps,
            series=[
                ("GPU reference", _values(rows, "reference_loss"), "#2563eb"),
                ("NPU candidate", _values(rows, "candidate_loss"), "#dc2626"),
            ],
            y_name="Loss",
            mark_areas=areas,
        ),
        echarts_line(
            title="First-step Loss Relative Error",
            subtitle=f"Window {early['first_step']}..{early['last_step']}: mean {_percent(early['mean_relative_error'])}, max {_percent(early['max_relative_error'])}",
            x_values=steps,
            series=[("Loss relative error", _values(rows, "loss_relative_error", percent=True), "#7c3aed")],
            y_name="Relative error (%)",
            mark_lines=[("Zero baseline", 0.0, "#475569"), (f"Guidance {loss_threshold_pct:g}%", loss_threshold_pct, "#f59e0b")],
            mark_areas=areas[:1],
        ),
        echarts_line(
            title="Whole-training Loss Relative Error",
            subtitle=f"Post-first-exceedance mean {_percent(loss['post_first_threshold_window']['mean_relative_error']) if first_exceeded is not None else 'N/A'}",
            x_values=steps,
            series=[("Loss relative error", _values(rows, "loss_relative_error", percent=True), "#7c3aed")],
            y_name="Relative error (%)",
            mark_lines=[("Zero baseline", 0.0, "#475569"), (f"Guidance {loss_threshold_pct:g}%", loss_threshold_pct, "#f59e0b")],
            mark_areas=areas,
        ),
        echarts_line(
            title="Loss Signed Difference",
            subtitle=f"Mean signed difference {_number(_mean(rows, 'loss_signed_difference'))}; candidate - reference",
            x_values=steps,
            series=[("Signed difference", _values(rows, "loss_signed_difference"), "#7c3aed")],
            y_name="NPU - GPU",
            mark_lines=[("Zero baseline", 0.0, "#475569")],
        ),
        echarts_line(
            title="Gradient Norm",
            subtitle=f"Mean relative error {_percent(grad['mean_relative_error'])}; inspect direction and persistence, not only magnitude",
            x_values=steps,
            series=[
                ("GPU reference", _values(rows, "reference_grad_norm"), "#2563eb"),
                ("NPU candidate", _values(rows, "candidate_grad_norm"), "#dc2626"),
            ],
            y_name="L2 norm",
        ),
        echarts_line(
            title="Gradient Norm Relative Error",
            subtitle="Diagnostic evidence; no universal acceptance threshold is asserted",
            x_values=steps,
            series=[("Grad Norm relative error", _values(rows, "grad_norm_relative_error", percent=True), "#059669")],
            y_name="Relative error (%)",
            mark_lines=[("Zero baseline", 0.0, "#475569")] + ([("Configured guidance", float(grad_threshold) * 100.0, "#f59e0b")] if grad_threshold is not None else []),
        ),
        echarts_line(
            title="Gradient Norm Signed Difference",
            subtitle=f"Mean signed difference {_number(_mean(rows, 'grad_norm_signed_difference'))}; zero line exposes persistent bias",
            x_values=steps,
            series=[("Signed difference", _values(rows, "grad_norm_signed_difference"), "#059669")],
            y_name="NPU - GPU",
            mark_lines=[("Zero baseline", 0.0, "#475569")],
        ),
        echarts_line(
            title="Gradient Norm Signed Relative Error",
            subtitle=f"Mean signed relative error {_percent(grad['mean_signed_relative_error'])}; ±5% lines are diagnostic guidance",
            x_values=steps,
            series=[("Signed relative error", _values(rows, "grad_norm_signed_relative_error", percent=True), "#059669")],
            y_name="(GPU - NPU) / GPU (%)",
            mark_lines=[("Zero baseline", 0.0, "#475569"), ("Upper guidance +5%", 5.0, "#f59e0b"), ("Lower guidance -5%", -5.0, "#f59e0b")],
        ),
    ]
    return save_panel_report(
        path=output_directory / "training_observation.html",
        title="GPU / NPU Training Observation",
        description="Interactive offline evidence. Hover for exact values; zoom, pan, select a window, toggle series, inspect data, or export a chart from its toolbox.",
        sections=[cards, *charts],
    )
