# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Plot four-panel GPU/NPU alignment figures for distributed long runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ddp_long_v2 import _mean_curve
from .export_distributed_alignment_results import (
    LONG_SCENARIO_SUFFIX,
    LONG_TOPOLOGIES,
    _aligned_series,
    _load_artifacts,
    _scenario,
)


GPU_COLOR = "#374151"
NPU_COLOR = "#2563EB"
GUIDE_COLOR = "#DC2626"
GRID_COLOR = "#D1D5DB"
TEXT_COLOR = "#111827"
MUTED_COLOR = "#4B5563"


@dataclass(frozen=True)
class AlignmentData:
    topology: str
    steps: tuple[int, ...]
    gpu_loss: tuple[float, ...]
    npu_loss: tuple[float, ...]
    gpu_grad: tuple[float, ...]
    npu_grad: tuple[float, ...]
    contract: Mapping[str, Any]
    source_paths: tuple[Path, ...]


@dataclass(frozen=True)
class PlotSeries:
    label: str
    color: str
    steps: tuple[int, ...]
    values: tuple[float, ...]
    width: float = 2.4


@dataclass(frozen=True)
class Guide:
    value: float
    label: str | None = None
    css_class: str = "guide"


def _load_alignment(artifact_root: Path, topology: str) -> AlignmentData:
    scenario_root = artifact_root / _scenario(topology, LONG_SCENARIO_SUFFIX)
    gpu, npu = _load_artifacts(scenario_root, topology)
    readers = (*gpu, *npu)
    steps, loss_curves = _aligned_series(readers, "loss")
    grad_steps, grad_curves = _aligned_series(readers, "grad_norm")
    if grad_steps != steps:
        raise ValueError(f"{topology}: loss and global grad norm steps do not match")
    if len(steps) < 2:
        raise ValueError(f"{topology}: at least two observations are required")
    values = (*loss_curves, *grad_curves)
    if not all(math.isfinite(value) for curve in values for value in curve):
        raise ValueError(f"{topology}: loss or global grad norm contains NaN/Inf")
    return AlignmentData(
        topology=topology,
        steps=steps,
        gpu_loss=_mean_curve(loss_curves[:2]),
        npu_loss=_mean_curve(loss_curves[2:]),
        gpu_grad=_mean_curve(grad_curves[:2]),
        npu_grad=_mean_curve(grad_curves[2:]),
        contract=gpu[0].training_contract,
        source_paths=tuple(reader.path for reader in readers),
    )


def _moving_average(
    steps: Sequence[int],
    values: Sequence[float],
    window: int,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if window < 1:
        raise ValueError("smoothing window must be positive")
    width = min(window, len(values))
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)
    averages = tuple(
        (prefix[end] - prefix[end - width]) / width
        for end in range(width, len(values) + 1)
    )
    return tuple(steps[width - 1 :]), averages


def _cumulative_mean(values: Sequence[float]) -> tuple[float, ...]:
    total = 0.0
    result = []
    for index, value in enumerate(values, start=1):
        total += value
        result.append(total / index)
    return tuple(result)


def _relative_diagnostics(
    data: AlignmentData,
    warmup_steps: int,
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...]]:
    retained = tuple(
        index for index, step in enumerate(data.steps) if step > warmup_steps
    )
    if not retained:
        raise ValueError(
            f"{data.topology}: no observations remain after {warmup_steps}-step warmup"
        )
    steps = tuple(data.steps[index] for index in retained)
    loss_error = tuple(
        abs(data.gpu_loss[index] - data.npu_loss[index])
        / max(abs(data.gpu_loss[index]), 1e-12)
        * 100.0
        for index in retained
    )
    grad_error = tuple(
        (data.gpu_grad[index] - data.npu_grad[index])
        / max(abs(data.gpu_grad[index]), 1e-12)
        * 100.0
        for index in retained
    )
    return steps, _cumulative_mean(loss_error), _cumulative_mean(grad_error)


def _extent(
    series: Sequence[PlotSeries],
    guides: Sequence[Guide],
    *,
    floor_zero: bool = False,
    symmetric: bool = False,
) -> tuple[float, float]:
    values = [value for item in series for value in item.values]
    values.extend(guide.value for guide in guides)
    if floor_zero:
        values.append(0.0)
    low = min(values)
    high = max(values)
    if symmetric:
        bound = max(abs(low), abs(high), 1e-9) * 1.08
        return -bound, bound
    if floor_zero:
        low = 0.0
    span = high - low
    padding = max(span * 0.07, abs(high) * 0.015, 1e-6)
    return low if floor_zero else low - padding, high + padding


def _format_tick(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1000.0:
        return f"{value:.0f}"
    if magnitude >= 10.0:
        return f"{value:.1f}"
    if magnitude >= 1.0:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _path(
    steps: Sequence[int],
    values: Sequence[float],
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    x_extent: tuple[float, float],
    y_extent: tuple[float, float],
) -> str:
    x_low, x_high = x_extent
    y_low, y_high = y_extent
    points = []
    for index, (step, value) in enumerate(zip(steps, values)):
        x = left + (step - x_low) * width / (x_high - x_low)
        y = top + (y_high - value) * height / (y_high - y_low)
        points.append(f"{'M' if index == 0 else 'L'}{x:.2f},{y:.2f}")
    return " ".join(points)


def _project_y(
    value: float,
    *,
    top: float,
    height: float,
    y_extent: tuple[float, float],
) -> float:
    low, high = y_extent
    return top + (high - value) * height / (high - low)


def _panel(
    panel_id: str,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[PlotSeries],
    guides: Sequence[Guide] = (),
    warmup_end: int | None = None,
    floor_zero: bool = False,
    symmetric: bool = False,
    endpoint_percent: bool = False,
    signed_endpoint: bool = False,
    x_domain: tuple[int, int] | None = None,
) -> list[str]:
    plot_left = x + 105
    plot_top = y + 58
    plot_width = width - 135
    plot_height = height - 128
    x_extent = (
        (float(x_domain[0]), float(x_domain[1]))
        if x_domain is not None
        else (
            float(min(item.steps[0] for item in series)),
            float(max(item.steps[-1] for item in series)),
        )
    )
    y_extent = _extent(
        series,
        guides,
        floor_zero=floor_zero,
        symmetric=symmetric,
    )
    lines = [
        f'<g id="{escape(panel_id)}">',
        (
            f'<text x="{x + width / 2:.1f}" y="{y + 25:.1f}" '
            f'class="panel-title" text-anchor="middle">{escape(title)}</text>'
        ),
    ]
    for index in range(6):
        fraction = index / 5
        grid_x = plot_left + fraction * plot_width
        step = round(x_extent[0] + fraction * (x_extent[1] - x_extent[0]))
        lines.extend(
            [
                (
                    f'<line x1="{grid_x:.1f}" y1="{plot_top:.1f}" '
                    f'x2="{grid_x:.1f}" y2="{plot_top + plot_height:.1f}" '
                    'class="grid"/>'
                ),
                (
                    f'<text x="{grid_x:.1f}" y="{plot_top + plot_height + 26:.1f}" '
                    f'class="tick" text-anchor="middle">{step}</text>'
                ),
            ]
        )
        grid_y = plot_top + (1.0 - fraction) * plot_height
        value = y_extent[0] + fraction * (y_extent[1] - y_extent[0])
        lines.extend(
            [
                (
                    f'<line x1="{plot_left:.1f}" y1="{grid_y:.1f}" '
                    f'x2="{plot_left + plot_width:.1f}" y2="{grid_y:.1f}" '
                    'class="grid"/>'
                ),
                (
                    f'<text x="{plot_left - 12:.1f}" y="{grid_y + 5:.1f}" '
                    f'class="tick" text-anchor="end">{_format_tick(value)}</text>'
                ),
            ]
        )

    if warmup_end is not None and warmup_end > x_extent[0]:
        warmup_x = plot_left + (
            min(float(warmup_end), x_extent[1]) - x_extent[0]
        ) * plot_width / (x_extent[1] - x_extent[0])
        lines.extend(
            [
                (
                    f'<rect x="{plot_left:.1f}" y="{plot_top:.1f}" '
                    f'width="{max(0.0, warmup_x - plot_left):.1f}" '
                    f'height="{plot_height:.1f}" class="warmup"/>'
                ),
                (
                    f'<text x="{(plot_left + warmup_x) / 2:.1f}" '
                    f'y="{plot_top + 18:.1f}" class="warmup-label" '
                    'text-anchor="middle">warmup</text>'
                ),
            ]
        )

    for guide in guides:
        guide_y = _project_y(
            guide.value,
            top=plot_top,
            height=plot_height,
            y_extent=y_extent,
        )
        lines.append(
            f'<line x1="{plot_left:.1f}" y1="{guide_y:.1f}" '
            f'x2="{plot_left + plot_width:.1f}" y2="{guide_y:.1f}" '
            f'class="{escape(guide.css_class)}"/>'
        )

    lines.append(f'<g clip-path="url(#clip-{escape(panel_id)})">')
    for item in series:
        path = _path(
            item.steps,
            item.values,
            left=plot_left,
            top=plot_top,
            width=plot_width,
            height=plot_height,
            x_extent=x_extent,
            y_extent=y_extent,
        )
        lines.append(
            f'<path d="{path}" fill="none" stroke="{item.color}" '
            f'stroke-width="{item.width:.1f}" stroke-linejoin="round"/>'
        )
    lines.append("</g>")

    lines.extend(
        [
            (
                f'<rect x="{plot_left:.1f}" y="{plot_top:.1f}" '
                f'width="{plot_width:.1f}" height="{plot_height:.1f}" '
                'class="plot-frame"/>'
            ),
            (
                f'<text x="{plot_left + plot_width / 2:.1f}" '
                f'y="{y + height - 13:.1f}" class="axis-title" '
                f'text-anchor="middle">{escape(x_label)}</text>'
            ),
            (
                f'<text x="{x + 23:.1f}" y="{plot_top + plot_height / 2:.1f}" '
                'class="axis-title" text-anchor="middle" '
                f'transform="rotate(-90 {x + 23:.1f} '
                f'{plot_top + plot_height / 2:.1f})">{escape(y_label)}</text>'
            ),
        ]
    )

    legend_entries = [(item.color, item.label, False) for item in series]
    legend_entries.extend(
        (GUIDE_COLOR, guide.label, guide.css_class == "guide")
        for guide in guides
        if guide.label is not None
    )
    legend_x = plot_left + plot_width - 330
    legend_y = plot_top + 21
    for index, (color, label, dashed) in enumerate(legend_entries):
        item_y = legend_y + index * 27
        dash = ' stroke-dasharray="10 7"' if dashed else ""
        lines.extend(
            [
                (
                    f'<line x1="{legend_x:.1f}" y1="{item_y:.1f}" '
                    f'x2="{legend_x + 38:.1f}" y2="{item_y:.1f}" '
                    f'stroke="{color}" stroke-width="3"{dash}/>'
                ),
                (
                    f'<text x="{legend_x + 50:.1f}" y="{item_y + 5:.1f}" '
                    f'class="legend">{escape(label)}</text>'
                ),
            ]
        )

    if endpoint_percent:
        item = series[-1]
        value = item.values[-1]
        value_y = _project_y(
            value,
            top=plot_top,
            height=plot_height,
            y_extent=y_extent,
        )
        y_offset = -11 if value >= 0.0 else 22
        label = f"{value:+.4f}%" if signed_endpoint else f"{value:.4f}%"
        lines.append(
            f'<text x="{plot_left + plot_width - 10:.1f}" '
            f'y="{value_y + y_offset:.1f}" text-anchor="end" '
            f'fill="{item.color}" class="endpoint">{label}</text>'
        )
    lines.append("</g>")
    return lines


def _contract_summary(data: AlignmentData) -> tuple[str, str]:
    training = data.contract.get("training", {})
    token_plan = data.contract.get("token_plan", {})
    if not isinstance(training, Mapping):
        training = {}
    if not isinstance(token_plan, Mapping):
        token_plan = {}
    batch = training.get("global_batch_size", token_plan.get("global_batch_size", "?"))
    sequence = training.get("sequence_length", token_plan.get("sequence_length", "?"))
    seed = training.get("seed", "?")
    dtype = str(training.get("mixed_precision_param", "BF16")).upper()
    if dtype == "BFLOAT16":
        dtype = "BF16"
    subtitle = (
        f"Same setup: global batch {batch} | sequence length {sequence} | "
        f"seed {seed} | GPU/NPU repeats averaged"
    )
    return subtitle, dtype


def _render_figure(
    data: AlignmentData,
    *,
    smoothing_window: int,
    warmup_steps: int,
) -> tuple[str, dict[str, float]]:
    smooth_steps, gpu_loss = _moving_average(
        data.steps, data.gpu_loss, smoothing_window
    )
    _, npu_loss = _moving_average(data.steps, data.npu_loss, smoothing_window)
    grad_steps, gpu_grad = _moving_average(
        data.steps, data.gpu_grad, smoothing_window
    )
    _, npu_grad = _moving_average(data.steps, data.npu_grad, smoothing_window)
    error_steps, loss_error, grad_error = _relative_diagnostics(data, warmup_steps)
    subtitle, dtype = _contract_summary(data)
    width = 2000
    height = 1320
    panel_width = 900
    panel_height = 505
    left_x = 70
    right_x = 1030
    top_y = 145
    bottom_y = 720
    clip_rectangles = (
        ("loss", left_x, top_y),
        ("grad", right_x, top_y),
        ("loss-error", left_x, bottom_y),
        ("grad-error", right_x, bottom_y),
    )
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            'aria-labelledby="figure-title figure-description">'
        ),
        (
            f'<title id="figure-title">GPU vs NPU {escape(data.topology.upper())} '
            'distributed precision alignment</title>'
        ),
        (
            '<desc id="figure-description">Loss, global grad norm, and running '
            'relative-error trajectories for matched GPU and NPU long runs.</desc>'
        ),
        "<defs>",
    ]
    for panel_id, panel_x, panel_y in clip_rectangles:
        lines.append(
            f'<clipPath id="clip-{panel_id}"><rect x="{panel_x + 105}" '
            f'y="{panel_y + 58}" width="{panel_width - 135}" '
            f'height="{panel_height - 128}"/></clipPath>'
        )
    lines.extend(
        [
            "</defs>",
            "<style>",
            (
                "text { font-family: DejaVu Sans, Arial, sans-serif; "
                f"fill: {TEXT_COLOR}; }}"
            ),
            ".figure-title { font-size: 34px; font-weight: 700; }",
            f".subtitle {{ font-size: 18px; fill: {MUTED_COLOR}; }}",
            ".panel-title { font-size: 24px; font-weight: 500; }",
            f".tick {{ font-size: 17px; fill: {MUTED_COLOR}; }}",
            f".axis-title {{ font-size: 19px; fill: {MUTED_COLOR}; }}",
            ".legend { font-size: 17px; }",
            ".endpoint { font-size: 18px; font-weight: 700; }",
            f".grid {{ stroke: {GRID_COLOR}; stroke-width: 1; opacity: 0.65; }}",
            ".plot-frame { fill: none; stroke: #9CA3AF; stroke-width: 1.2; }",
            ".warmup { fill: #9CA3AF; opacity: 0.13; }",
            ".warmup-label { fill: #6B7280; font-size: 16px; }",
            (
                f".guide {{ stroke: {GUIDE_COLOR}; stroke-width: 2; "
                "stroke-dasharray: 10 7; }}"
            ),
            ".baseline { stroke: #6B7280; stroke-width: 1.4; }",
            "</style>",
            f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
            (
                f'<text x="{width / 2}" y="52" class="figure-title" '
                'text-anchor="middle">GPU vs NPU distributed precision alignment '
                f'— {escape(data.topology.upper())} ({len(data.steps)} steps, '
                f'{escape(dtype)})</text>'
            ),
            (
                f'<text x="{width / 2}" y="91" class="subtitle" '
                f'text-anchor="middle">{escape(subtitle)}</text>'
            ),
        ]
    )
    lines.extend(
        _panel(
            "loss",
            x=left_x,
            y=top_y,
            width=panel_width,
            height=panel_height,
            title=f"Loss trajectory ({smoothing_window}-step moving average)",
            x_label="Training step",
            y_label="loss",
            series=(
                PlotSeries("GPU reference", GPU_COLOR, smooth_steps, gpu_loss, 3.0),
                PlotSeries("NPU candidate", NPU_COLOR, smooth_steps, npu_loss, 2.6),
            ),
            warmup_end=warmup_steps,
            x_domain=(data.steps[0], data.steps[-1]),
        )
    )
    lines.extend(
        _panel(
            "grad",
            x=right_x,
            y=top_y,
            width=panel_width,
            height=panel_height,
            title=(
                "Global grad norm trajectory "
                f"({smoothing_window}-step moving average)"
            ),
            x_label="Training step",
            y_label="global grad norm",
            series=(
                PlotSeries("GPU reference", GPU_COLOR, grad_steps, gpu_grad, 3.0),
                PlotSeries("NPU candidate", NPU_COLOR, grad_steps, npu_grad, 2.6),
            ),
            warmup_end=warmup_steps,
            x_domain=(data.steps[0], data.steps[-1]),
        )
    )
    lines.extend(
        _panel(
            "loss-error",
            x=left_x,
            y=bottom_y,
            width=panel_width,
            height=panel_height,
            title=(
                "Running loss mean absolute relative error "
                f"(after {warmup_steps}-step warmup)"
            ),
            x_label="Training step",
            y_label="Running mean absolute relative error (%)",
            series=(
                PlotSeries("NPU vs GPU", NPU_COLOR, error_steps, loss_error, 3.2),
            ),
            guides=(Guide(1.0, "1% diagnostic guide (non-gating)"),),
            floor_zero=True,
            endpoint_percent=True,
        )
    )
    lines.extend(
        _panel(
            "grad-error",
            x=right_x,
            y=bottom_y,
            width=panel_width,
            height=panel_height,
            title=(
                "Running grad norm signed mean relative error "
                f"(after {warmup_steps}-step warmup)"
            ),
            x_label="Training step",
            y_label="(GPU - NPU) / |GPU| (%)",
            series=(
                PlotSeries("NPU vs GPU", NPU_COLOR, error_steps, grad_error, 3.2),
            ),
            guides=(
                Guide(5.0, "+/-5% diagnostic guides (non-gating)"),
                Guide(-5.0),
                Guide(0.0, css_class="baseline"),
            ),
            symmetric=True,
            endpoint_percent=True,
            signed_endpoint=True,
        )
    )
    lines.extend(
        [
            (
                f'<text x="{width / 2}" y="1280" class="subtitle" '
                'text-anchor="middle">Lower panels use every retained raw step; '
                'under V2 they are diagnostic and do not independently decide '
                'PASS/FAIL.</text>'
            ),
            "</svg>",
        ]
    )
    diagnostics = {
        "final_running_loss_mare_percent": loss_error[-1],
        "final_running_grad_signed_mean_percent": grad_error[-1],
    }
    return "\n".join(lines) + "\n", diagnostics


def plot_distributed_gpu_npu_alignment(
    artifact_root: str | Path,
    output_root: str | Path,
    *,
    topologies: Sequence[str] = LONG_TOPOLOGIES,
    smoothing_window: int = 31,
    warmup_steps: int = 100,
) -> tuple[tuple[Path, ...], Path]:
    """Write one four-panel GPU/NPU SVG per distributed topology."""

    if not topologies:
        raise ValueError("at least one topology is required")
    if smoothing_window < 1:
        raise ValueError("smoothing window must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup steps must be nonnegative")
    artifact_root = Path(artifact_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    paths = []
    records = []
    for topology in topologies:
        data = _load_alignment(artifact_root, topology)
        svg, diagnostics = _render_figure(
            data,
            smoothing_window=smoothing_window,
            warmup_steps=warmup_steps,
        )
        path = output_root / (
            f"gpu-vs-npu-{topology}-{len(data.steps)}steps.svg"
        )
        path.write_text(svg, encoding="utf-8")
        paths.append(path)
        records.append(
            {
                "topology": topology,
                "steps": len(data.steps),
                "plot": str(path),
                "sources": [str(source) for source in data.source_paths],
                **diagnostics,
            }
        )
    manifest_path = output_root / "gpu_npu_alignment_plots.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_root": str(artifact_root),
                "smoothing_window": smoothing_window,
                "warmup_steps": warmup_steps,
                "plots": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return tuple(paths), manifest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot four-panel GPU/NPU alignment for long distributed runs."
    )
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--topology",
        action="append",
        choices=LONG_TOPOLOGIES,
        help="Plot only this topology; repeat for multiple topologies.",
    )
    parser.add_argument("--smoothing-window", type=int, default=31)
    parser.add_argument("--warmup-steps", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths, manifest = plot_distributed_gpu_npu_alignment(
        args.artifact_root,
        args.output_root,
        topologies=tuple(args.topology or LONG_TOPOLOGIES),
        smoothing_window=args.smoothing_window,
        warmup_steps=args.warmup_steps,
    )
    print(f"Generated {len(paths)} GPU/NPU alignment figures:")
    for path in paths:
        print(path)
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
