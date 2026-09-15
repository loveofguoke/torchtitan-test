# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Render GPU/NPU loss-step curves from existing precision artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
import json
import math
from pathlib import Path
from typing import Sequence

from .artifacts import PrecisionArtifactReader
from .distributed_long_convergence_v2 import (
    DEFAULT_TOPOLOGIES,
    _scenario,
)


GPU_COLOR = "#0072B2"
NPU_COLOR = "#D55E00"
GRID_COLOR = "#D9DEE7"
TEXT_COLOR = "#20242B"


@dataclass(frozen=True)
class LossCurves:
    topology: str
    scenario: str
    steps: tuple[int, ...]
    gpu_runs: tuple[tuple[float, ...], ...]
    npu_runs: tuple[tuple[float, ...], ...]
    source_paths: tuple[Path, ...]


def _read_loss_curves(artifact_root: Path, topology: str) -> LossCurves:
    scenario = _scenario(topology)
    scenario_root = artifact_root / scenario
    paths = tuple(
        scenario_root / name
        for name in (
            "reference-r1",
            "reference-r2",
            "candidate-r1",
            "candidate-r2",
        )
    )
    readers = tuple(PrecisionArtifactReader(path) for path in paths)
    series = tuple(reader.loss_series() for reader in readers)
    steps = tuple(series[0])
    if any(tuple(item) != steps for item in series[1:]):
        raise ValueError(f"artifact step sequences do not match for {topology}")
    if len(steps) < 2:
        raise ValueError(f"at least two loss observations are required for {topology}")
    runs = tuple(tuple(item[step] for step in steps) for item in series)
    if not all(math.isfinite(value) for run in runs for value in run):
        raise ValueError(f"loss contains a non-finite value for {topology}")
    return LossCurves(
        topology=topology,
        scenario=scenario,
        steps=steps,
        gpu_runs=runs[:2],
        npu_runs=runs[2:],
        source_paths=paths,
    )


def _mean_curve(curves: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not curves:
        raise ValueError("at least one loss curve is required")
    width = len(curves[0])
    if any(len(curve) != width for curve in curves):
        raise ValueError("loss curves have different lengths")
    return tuple(sum(values) / len(values) for values in zip(*curves))


def _moving_average(
    steps: Sequence[int],
    values: Sequence[float],
    window_size: int,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if window_size < 1:
        raise ValueError("smoothing window must be positive")
    if len(steps) != len(values) or not values:
        raise ValueError("steps and loss values must have the same nonzero length")
    width = min(window_size, len(values))
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)
    smoothed = tuple(
        (prefix[end] - prefix[end - width]) / width
        for end in range(width, len(values) + 1)
    )
    return tuple(steps[width - 1 :]), smoothed


def _extent(values: Sequence[float]) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    if low == high:
        padding = max(abs(low) * 0.05, 1.0)
    else:
        padding = (high - low) * 0.06
    return low - padding, high + padding


def _format_loss(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1000.0 or (0.0 < magnitude < 0.001):
        return f"{value:.2e}"
    if magnitude >= 10.0:
        return f"{value:.2f}"
    return f"{value:.4f}"


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

    def project(step: int, value: float) -> tuple[float, float]:
        x = left + (step - x_low) * width / (x_high - x_low)
        y = top + (y_high - value) * height / (y_high - y_low)
        return x, y

    points = (project(step, value) for step, value in zip(steps, values))
    return " ".join(
        f"{'M' if index == 0 else 'L'}{x:.2f},{y:.2f}"
        for index, (x, y) in enumerate(points)
    )


def _panel(
    curves: LossCurves,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    smoothing_window: int,
    show_raw: bool,
    show_legend: bool,
) -> list[str]:
    plot_left = x + 72
    plot_top = y + 48
    plot_width = width - 96
    plot_height = height - 112
    gpu_mean = _mean_curve(curves.gpu_runs)
    npu_mean = _mean_curve(curves.npu_runs)
    smooth_steps, gpu_smooth = _moving_average(
        curves.steps, gpu_mean, smoothing_window
    )
    _, npu_smooth = _moving_average(curves.steps, npu_mean, smoothing_window)
    visible_values = (
        tuple(value for run in curves.gpu_runs + curves.npu_runs for value in run)
        if show_raw
        else gpu_smooth + npu_smooth
    )
    x_extent = (float(curves.steps[0]), float(curves.steps[-1]))
    y_extent = _extent(visible_values)
    lines = [
        f'<g class="panel" data-topology="{escape(curves.topology)}">',
        (
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" '
            f'height="{height:.1f}" rx="8" fill="#FFFFFF" stroke="#C8CED8"/>'
        ),
        (
            f'<text x="{x + 20:.1f}" y="{y + 29:.1f}" class="panel-title">'
            f'{escape(curves.topology.upper())}</text>'
        ),
    ]

    for index in range(5):
        fraction = index / 4
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
                    f'<text x="{grid_x:.1f}" y="{plot_top + plot_height + 22:.1f}" '
                    f'class="tick" text-anchor="middle">{step}</text>'
                ),
            ]
        )
        grid_y = plot_top + (1.0 - fraction) * plot_height
        loss = y_extent[0] + fraction * (y_extent[1] - y_extent[0])
        lines.extend(
            [
                (
                    f'<line x1="{plot_left:.1f}" y1="{grid_y:.1f}" '
                    f'x2="{plot_left + plot_width:.1f}" y2="{grid_y:.1f}" '
                    'class="grid"/>'
                ),
                (
                    f'<text x="{plot_left - 10:.1f}" y="{grid_y + 4:.1f}" '
                    f'class="tick" text-anchor="end">{_format_loss(loss)}</text>'
                ),
            ]
        )

    lines.append(
        f'<g clip-path="url(#clip-{escape(curves.topology)})">'
    )
    if show_raw:
        for run in curves.gpu_runs:
            path = _path(
                curves.steps,
                run,
                left=plot_left,
                top=plot_top,
                width=plot_width,
                height=plot_height,
                x_extent=x_extent,
                y_extent=y_extent,
            )
            lines.append(f'<path d="{path}" class="gpu raw"/>')
        for run in curves.npu_runs:
            path = _path(
                curves.steps,
                run,
                left=plot_left,
                top=plot_top,
                width=plot_width,
                height=plot_height,
                x_extent=x_extent,
                y_extent=y_extent,
            )
            lines.append(f'<path d="{path}" class="npu raw"/>')

    for values, css_class in ((gpu_smooth, "gpu smooth"), (npu_smooth, "npu smooth")):
        path = _path(
            smooth_steps,
            values,
            left=plot_left,
            top=plot_top,
            width=plot_width,
            height=plot_height,
            x_extent=x_extent,
            y_extent=y_extent,
        )
        lines.append(f'<path d="{path}" class="{css_class}"/>')
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
                f'y="{y + height - 16:.1f}" class="axis-title" '
                'text-anchor="middle">Optimizer step</text>'
            ),
            (
                f'<text x="{x + 18:.1f}" y="{plot_top + plot_height / 2:.1f}" '
                'class="axis-title" text-anchor="middle" '
                f'transform="rotate(-90 {x + 18:.1f} '
                f'{plot_top + plot_height / 2:.1f})">Training loss</text>'
            ),
        ]
    )
    if show_legend:
        legend_y = y + 24
        legend_x = x + width - 510
        labels = (
            ("gpu raw", "GPU raw repeats", 0),
            ("npu raw", "NPU raw repeats", 122),
            ("gpu smooth", f"GPU {smoothing_window}-step MA", 250),
            ("npu smooth", f"NPU {smoothing_window}-step MA", 380),
        )
        for css_class, label, offset in labels:
            start = legend_x + offset
            lines.extend(
                [
                    (
                        f'<line x1="{start:.1f}" y1="{legend_y:.1f}" '
                        f'x2="{start + 20:.1f}" y2="{legend_y:.1f}" '
                        f'class="{css_class}"/>'
                    ),
                    (
                        f'<text x="{start + 25:.1f}" y="{legend_y + 4:.1f}" '
                        f'class="legend">{escape(label)}</text>'
                    ),
                ]
            )
    lines.append("</g>")
    return lines


def _document(
    curves: Sequence[LossCurves],
    *,
    smoothing_window: int,
    overview: bool,
) -> str:
    if overview:
        columns = 2
        panel_width = 670
        panel_height = 300
        gap = 20
        margin = 20
        header = 70
        rows = math.ceil(len(curves) / columns)
        width = margin * 2 + columns * panel_width + (columns - 1) * gap
        height = header + rows * panel_height + max(0, rows - 1) * gap + margin
        title = "GPU vs NPU loss-step overview"
    else:
        panel_width = 1060
        panel_height = 610
        margin = 20
        header = 0
        width = 1100
        height = 650
        title = f"{curves[0].topology.upper()} GPU vs NPU loss-step curves"

    clip_paths = []
    for index, curve in enumerate(curves):
        if overview:
            panel_x = margin + (index % 2) * (panel_width + gap)
            panel_y = header + (index // 2) * (panel_height + gap)
        else:
            panel_x = margin
            panel_y = margin
        clip_paths.append(
            (
                f'<clipPath id="clip-{escape(curve.topology)}">'
                f'<rect x="{panel_x + 72:.1f}" y="{panel_y + 48:.1f}" '
                f'width="{panel_width - 96:.1f}" '
                f'height="{panel_height - 112:.1f}"/></clipPath>'
            )
        )

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            'aria-labelledby="chart-title chart-description">'
        ),
        f'<title id="chart-title">{escape(title)}</title>',
        (
            '<desc id="chart-description">Training loss by optimizer step for '
            'GPU reference and NPU candidate precision artifacts.</desc>'
        ),
        "<defs>",
        *clip_paths,
        "</defs>",
        "<style>",
        (
            "text { font-family: Inter, system-ui, -apple-system, sans-serif; "
            f"fill: {TEXT_COLOR}; }}"
        ),
        ".panel-title { font-size: 16px; font-weight: 700; }",
        ".tick { font-size: 11px; }",
        ".axis-title { font-size: 12px; font-weight: 600; }",
        ".legend { font-size: 10px; }",
        f".grid {{ stroke: {GRID_COLOR}; stroke-width: 1; }}",
        ".plot-frame { fill: none; stroke: #858D99; stroke-width: 1; }",
        ".raw, .smooth { fill: none; stroke-linejoin: round; }",
        f".gpu {{ stroke: {GPU_COLOR}; }}",
        f".npu {{ stroke: {NPU_COLOR}; }}",
        ".raw { stroke-width: 1; opacity: 0.28; }",
        ".smooth { stroke-width: 2.5; opacity: 0.95; }",
        ".npu.smooth { stroke-dasharray: 8 5; }",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#F5F7FA"/>',
    ]
    if overview:
        lines.extend(
            [
                (
                    f'<text x="{margin:.1f}" y="30" '
                    'style="font-size:20px;font-weight:700">'
                    'GPU vs NPU loss-step overview</text>'
                ),
                (
                    f'<line x1="{width - 480}" y1="26" x2="{width - 450}" '
                    'y2="26" class="gpu smooth"/>'
                ),
                (
                    f'<text x="{width - 442}" y="30" class="axis-title">'
                    f'GPU {smoothing_window}-step moving average</text>'
                ),
                (
                    f'<line x1="{width - 240}" y1="26" x2="{width - 210}" '
                    'y2="26" class="npu smooth"/>'
                ),
                (
                    f'<text x="{width - 202}" y="30" class="axis-title">'
                    f'NPU {smoothing_window}-step moving average</text>'
                ),
                (
                    f'<text x="{margin:.1f}" y="52" class="tick">'
                    'Per-topology y-axis is scaled to its observed loss range.</text>'
                ),
            ]
        )

    for index, curve in enumerate(curves):
        if overview:
            panel_x = margin + (index % 2) * (panel_width + gap)
            panel_y = header + (index // 2) * (panel_height + gap)
        else:
            panel_x = margin
            panel_y = margin
        lines.extend(
            _panel(
                curve,
                x=panel_x,
                y=panel_y,
                width=panel_width,
                height=panel_height,
                smoothing_window=smoothing_window,
                show_raw=not overview,
                show_legend=not overview,
            )
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def plot_topology_matrix(
    artifact_root: str | Path,
    output_root: str | Path,
    *,
    topologies: Sequence[str] = DEFAULT_TOPOLOGIES,
    smoothing_window: int = 100,
) -> tuple[Path, tuple[Path, ...], Path]:
    """Write one detailed SVG per topology plus a smoothed overview SVG."""

    if not topologies:
        raise ValueError("at least one topology is required")
    if smoothing_window < 1:
        raise ValueError("smoothing window must be positive")
    artifact_root = Path(artifact_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    curves = tuple(
        _read_loss_curves(artifact_root, topology) for topology in topologies
    )

    detail_paths = []
    for curve in curves:
        path = output_root / f"loss_step_{curve.topology}.svg"
        path.write_text(
            _document((curve,), smoothing_window=smoothing_window, overview=False),
            encoding="utf-8",
        )
        detail_paths.append(path)
    overview_path = output_root / "loss_step_overview.svg"
    overview_path.write_text(
        _document(curves, smoothing_window=smoothing_window, overview=True),
        encoding="utf-8",
    )

    manifest_path = output_root / "loss_step_plots.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_root": str(artifact_root),
                "smoothing_window": smoothing_window,
                "overview": str(overview_path),
                "topologies": [
                    {
                        "topology": curve.topology,
                        "scenario": curve.scenario,
                        "steps": len(curve.steps),
                        "sources": [str(path) for path in curve.source_paths],
                        "plot": str(detail_path),
                    }
                    for curve, detail_path in zip(curves, detail_paths)
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return overview_path, tuple(detail_paths), manifest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot GPU/NPU loss-step curves from formal precision artifacts."
    )
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--topology",
        action="append",
        choices=DEFAULT_TOPOLOGIES,
        help="Plot only the selected topology; repeat for multiple topologies.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=100,
        help="Moving-average window shown as the emphasized curve (default: 100).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    overview, details, manifest = plot_topology_matrix(
        args.artifact_root,
        args.output_root,
        topologies=tuple(args.topology or DEFAULT_TOPOLOGIES),
        smoothing_window=args.smoothing_window,
    )
    print(f"Overview: {overview}")
    print(f"Topology plots: {len(details)}")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
