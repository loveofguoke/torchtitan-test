# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Whole-training metric comparison and portable diagnostic charts."""

from __future__ import annotations

import csv
from html import escape
import json
import math
from pathlib import Path
from typing import Any, Iterable

from tests.glm5_2_mindstudio.artifacts import write_json


LOSS_KEY = "loss_metrics/global_avg_loss"
MAX_LOSS_KEY = "loss_metrics/global_max_loss"
GRAD_NORM_KEY = "grad_norm"


def _number(value: Any) -> float:
    if isinstance(value, str):
        return {
            "NaN": math.nan,
            "Infinity": math.inf,
            "-Infinity": -math.inf,
        }[value]
    return float(value)


def read_training_metrics(path: Path) -> dict[int, dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"training metrics not found: {path}")
    records: dict[int, dict[str, float]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        payload = json.loads(line)
        step = int(payload["step"])
        if step in records:
            raise ValueError(f"duplicate step {step} at {path}:{line_number}")
        values = payload["metrics"]
        missing = [
            key for key in (LOSS_KEY, MAX_LOSS_KEY, GRAD_NORM_KEY)
            if key not in values
        ]
        if missing:
            raise ValueError(f"missing metrics {missing} at {path}:{line_number}")
        records[step] = {
            "loss": _number(values[LOSS_KEY]),
            "global_max_loss": _number(values[MAX_LOSS_KEY]),
            "grad_norm": _number(values[GRAD_NORM_KEY]),
        }
    if not records:
        raise ValueError(f"training metrics are empty: {path}")
    return records


def _first_nonfinite_metrics(path: Path) -> dict[str, int]:
    """Return the first observed NaN/Inf step for every logged metric."""

    first: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        step = int(payload["step"])
        for name, raw_value in payload["metrics"].items():
            try:
                value = _number(raw_value)
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(value):
                first.setdefault(str(name), step)
    return dict(sorted(first.items()))


def _relative_error(reference: float, candidate: float) -> float:
    if not math.isfinite(reference) or not math.isfinite(candidate):
        return math.nan
    return abs(candidate - reference) / max(abs(reference), 1e-12)


def _first_step(
    rows: Iterable[dict[str, Any]],
    predicate,
) -> int | None:
    return next((int(row["step"]) for row in rows if predicate(row)), None)


def _finite_mean(values: Iterable[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def _spike_steps(
    values: list[tuple[int, float]],
    threshold: float | None,
) -> list[int]:
    if threshold is None:
        return []
    result: list[int] = []
    for (_, previous), (step, current) in zip(values, values[1:]):
        if not math.isfinite(previous) or not math.isfinite(current):
            continue
        relative_jump = abs(current - previous) / max(abs(previous), 1e-12)
        if relative_jump > threshold:
            result.append(step)
    return result


def _svg_chart(
    path: Path,
    *,
    title: str,
    y_label: str,
    series: tuple[tuple[str, str, list[tuple[int, float]]], ...],
    reference_lines: tuple[tuple[str, str, float], ...] = (),
    include_zero: bool = False,
    subtitle: str | None = None,
) -> None:
    width, height = 1120, 560
    left, top, right, bottom = 105, 95, 35, 80
    points = [point for _, _, values in series for point in values]
    finite = [(step, value) for step, value in points if math.isfinite(value)]
    if not finite:
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="480">\n'
            '<rect width="100%" height="100%" fill="white"/>\n'
            f'<text x="80" y="40" font-size="20">{title}</text>\n'
            '<text x="80" y="90" font-size="16">'
            'No finite values; inspect summary.json for NaN/Inf steps.'
            '</text>\n</svg>\n',
            encoding="utf-8",
        )
        return
    min_step = min(step for step, _ in finite)
    max_step = max(step for step, _ in finite)
    min_value = min(value for _, value in finite)
    max_value = max(value for _, value in finite)
    if reference_lines:
        min_value = min(min_value, *(value for _, _, value in reference_lines))
        max_value = max(max_value, *(value for _, _, value in reference_lines))
    if include_zero:
        min_value = min(min_value, 0.0)
        max_value = max(max_value, 0.0)
    raw_span = max_value - min_value
    padding = max(raw_span * 0.05, abs(max_value) * 0.01, 1e-12)
    if include_zero and min_value == 0:
        display_min = 0.0
    else:
        display_min = min_value - padding
    display_max = max_value + padding
    step_span = max(max_step - min_step, 1)
    value_span = max(display_max - display_min, 1e-12)

    def point(step: int, value: float) -> tuple[float, float]:
        x = left + (step - min_step) / step_span * (width - left - right)
        y = top + (display_max - value) / value_span * (height - top - bottom)
        return x, y

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" font-size="24" '
        f'text-anchor="middle" font-weight="700">{escape(title)}</text>',
    ]
    if subtitle:
        lines.append(
            f'<text x="{width / 2}" y="55" font-size="13" '
            f'text-anchor="middle" fill="#4b5563">{escape(subtitle)}</text>'
        )
    plot_bottom = height - bottom
    x_tick_count = min(8, max_step - min_step + 1)
    y_tick_count = 6
    for index in range(y_tick_count + 1):
        value = display_min + value_span * index / y_tick_count
        y = plot_bottom - (plot_bottom - top) * index / y_tick_count
        lines.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" '
                f'y2="{y:.2f}" stroke="#d1d5db" stroke-width="1"/>',
                f'<text x="{left-12}" y="{y+4:.2f}" font-size="12" '
                f'text-anchor="end">{value:.6g}</text>',
            ]
        )
    for index in range(x_tick_count + 1):
        step = round(min_step + step_span * index / x_tick_count)
        x, _ = point(step, display_min)
        lines.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" '
                f'y2="{plot_bottom}" stroke="#eef0f3" stroke-width="1"/>',
                f'<text x="{x:.2f}" y="{plot_bottom+24}" font-size="12" '
                f'text-anchor="middle">{step}</text>',
            ]
        )
    lines.extend(
        [
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" '
        'stroke="#333"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
        f'y2="{height-bottom}" stroke="#333"/>',
            f'<text x="{(left + width - right) / 2}" y="{height-22}" '
            'font-size="14" text-anchor="middle">Training step</text>',
            f'<text x="25" y="{(top + plot_bottom) / 2}" font-size="14" '
            f'text-anchor="middle" transform="rotate(-90 25 '
            f'{(top + plot_bottom) / 2})">{escape(y_label)}</text>',
        ]
    )
    legend_x = left
    for name, color, values in series:
        segments: list[list[str]] = [[]]
        for step, value in values:
            if math.isfinite(value):
                x, y = point(step, value)
                segments[-1].append(f"{x:.2f},{y:.2f}")
            elif segments[-1]:
                segments.append([])
        for segment in segments:
            if segment:
                lines.append(
                    f'<polyline points="{" ".join(segment)}" fill="none" '
                    f'stroke="{color}" stroke-width="2"/>'
                )
        lines.extend(
            [
                f'<line x1="{legend_x}" y1="75" x2="{legend_x+24}" y2="75" '
                f'stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x+30}" y="79" font-size="12">'
                f'{escape(name)}</text>',
            ]
        )
        legend_x += 170
    for name, color, value in reference_lines:
        _, y = point(min_step, value)
        lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" '
            f'y2="{y:.2f}" stroke="{color}" stroke-width="2" '
            'stroke-dasharray="8 6"/>'
        )
        lines.extend(
            [
                f'<line x1="{legend_x}" y1="75" x2="{legend_x+24}" y2="75" '
                f'stroke="{color}" stroke-width="2" stroke-dasharray="8 6"/>',
                f'<text x="{legend_x+30}" y="79" font-size="12">'
                f'{escape(name)}</text>',
            ]
        )
        legend_x += 190
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_training_metrics(
    *,
    reference_path: Path,
    candidate_path: Path,
    output_directory: Path,
    loss_relative_threshold: float = 0.01,
    grad_norm_relative_threshold: float | None = None,
    spike_relative_threshold: float | None = None,
) -> Path:
    reference = read_training_metrics(reference_path)
    candidate = read_training_metrics(candidate_path)
    if set(reference) != set(candidate):
        missing_reference = sorted(set(candidate) - set(reference))
        missing_candidate = sorted(set(reference) - set(candidate))
        raise ValueError(
            "training metric steps differ: "
            f"missing_reference={missing_reference}, "
            f"missing_candidate={missing_candidate}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for step in sorted(reference):
        ref = reference[step]
        cand = candidate[step]
        rows.append(
            {
                "step": step,
                "reference_loss": ref["loss"],
                "candidate_loss": cand["loss"],
                "loss_relative_error": _relative_error(
                    ref["loss"], cand["loss"]
                ),
                "reference_grad_norm": ref["grad_norm"],
                "candidate_grad_norm": cand["grad_norm"],
                "grad_norm_relative_error": _relative_error(
                    ref["grad_norm"], cand["grad_norm"]
                ),
            }
        )
    csv_path = output_directory / "training_metrics_compare.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    reference_loss = [(step, reference[step]["loss"]) for step in sorted(reference)]
    candidate_loss = [(step, candidate[step]["loss"]) for step in sorted(candidate)]
    reference_grad = [
        (step, reference[step]["grad_norm"]) for step in sorted(reference)
    ]
    candidate_grad = [
        (step, candidate[step]["grad_norm"]) for step in sorted(candidate)
    ]
    _svg_chart(
        output_directory / "loss.svg",
        title="Training Loss",
        y_label="Loss",
        series=(
            ("GPU reference", "#2563eb", reference_loss),
            ("NPU candidate", "#dc2626", candidate_loss),
        ),
        include_zero=True,
    )
    _svg_chart(
        output_directory / "grad_norm.svg",
        title="Gradient Norm",
        y_label="L2 norm",
        series=(
            ("GPU reference", "#2563eb", reference_grad),
            ("NPU candidate", "#dc2626", candidate_grad),
        ),
        include_zero=True,
    )
    loss_error = [
        (row["step"], row["loss_relative_error"] * 100.0) for row in rows
    ]
    grad_error = [
        (row["step"], row["grad_norm_relative_error"] * 100.0)
        for row in rows
    ]
    loss_mean = _finite_mean(value for _, value in loss_error)
    first_loss_difference = _first_step(
        rows,
        lambda row: math.isfinite(row["loss_relative_error"])
        and row["loss_relative_error"] > loss_relative_threshold,
    )
    _svg_chart(
        output_directory / "loss_relative_error.svg",
        title="Loss Relative Error",
        y_label="Relative error (%)",
        series=(("Loss error", "#7c3aed", loss_error),),
        reference_lines=(
            ("Zero-error baseline", "#dc2626", 0.0),
            (
                f"Guidance {loss_relative_threshold * 100:g}%",
                "#f59e0b",
                loss_relative_threshold * 100.0,
            ),
        ),
        include_zero=True,
        subtitle=(
            f"Mean error: {loss_mean:.6g}%; first step above guidance: "
            f"{first_loss_difference if first_loss_difference is not None else 'none'}"
            if loss_mean is not None
            else "No finite Loss relative errors"
        ),
    )
    grad_reference_lines = [("Zero-error baseline", "#dc2626", 0.0)]
    if grad_norm_relative_threshold is not None:
        grad_reference_lines.append(
            (
                f"Configured {grad_norm_relative_threshold * 100:g}%",
                "#f59e0b",
                grad_norm_relative_threshold * 100.0,
            )
        )
    grad_mean = _finite_mean(value for _, value in grad_error)
    _svg_chart(
        output_directory / "grad_norm_relative_error.svg",
        title="Gradient Norm Relative Error",
        y_label="Relative error (%)",
        series=(("Grad Norm error", "#059669", grad_error),),
        reference_lines=tuple(grad_reference_lines),
        include_zero=True,
        subtitle=(
            f"Mean error: {grad_mean:.6g}%; no universal acceptance threshold"
            if grad_mean is not None
            else "No finite Grad Norm relative errors"
        ),
    )
    first_observed_step = min(reference)
    if _first_nonfinite_metrics(candidate_path):
        symptom = "candidate-nan-or-inf"
    elif first_loss_difference == first_observed_step:
        symptom = "first-step-loss-difference"
    elif first_loss_difference is not None:
        symptom = "later-window-loss-difference"
    else:
        symptom = "no-loss-difference-observed"
    summary = {
        "schema": "torchtitan.glm5_2.mindstudio_training_observation",
        "schema_version": 1,
        "reference": str(reference_path.resolve()),
        "candidate": str(candidate_path.resolve()),
        "step_count": len(rows),
        "observation": {
            "first_step": first_observed_step,
            "last_step": max(reference),
            "diagnostic_symptom": symptom,
            "reference_first_nonfinite_metrics": _first_nonfinite_metrics(
                reference_path
            ),
            "candidate_first_nonfinite_metrics": _first_nonfinite_metrics(
                candidate_path
            ),
        },
        "loss": {
            "guidance_relative_threshold": loss_relative_threshold,
            "first_step_above_threshold": first_loss_difference,
            "mean_relative_error": _finite_mean(
                row["loss_relative_error"] for row in rows
            ),
            "reference_nonfinite_step": _first_step(
                rows, lambda row: not math.isfinite(row["reference_loss"])
            ),
            "candidate_nonfinite_step": _first_step(
                rows, lambda row: not math.isfinite(row["candidate_loss"])
            ),
            "reference_spike_steps": _spike_steps(
                reference_loss, spike_relative_threshold
            ),
            "candidate_spike_steps": _spike_steps(
                candidate_loss, spike_relative_threshold
            ),
        },
        "grad_norm": {
            "diagnostic_relative_threshold": grad_norm_relative_threshold,
            "first_step_above_threshold": (
                _first_step(
                    rows,
                    lambda row: math.isfinite(row["grad_norm_relative_error"])
                    and row["grad_norm_relative_error"]
                    > grad_norm_relative_threshold,
                )
                if grad_norm_relative_threshold is not None
                else None
            ),
            "mean_relative_error": _finite_mean(
                row["grad_norm_relative_error"] for row in rows
            ),
            "reference_nonfinite_step": _first_step(
                rows, lambda row: not math.isfinite(row["reference_grad_norm"])
            ),
            "candidate_nonfinite_step": _first_step(
                rows, lambda row: not math.isfinite(row["candidate_grad_norm"])
            ),
            "reference_spike_steps": _spike_steps(
                reference_grad, spike_relative_threshold
            ),
            "candidate_spike_steps": _spike_steps(
                candidate_grad, spike_relative_threshold
            ),
        },
        "notes": [
            "The 1% loss value follows the official practice guide's symptom "
            "classification; it is not a universal delivery verdict.",
            "Grad Norm and spike thresholds are diagnostic settings and are "
            "evaluated only when explicitly provided.",
            "The diagnostic symptom selects the next investigation branch; "
            "it is not a delivery PASS/FAIL verdict.",
        ],
    }
    summary_path = output_directory / "summary.json"
    write_json(summary_path, summary)
    return summary_path
