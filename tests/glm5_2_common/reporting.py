# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Shared offline report components backed by Panel and Apache ECharts."""

from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence


class ReportingDependencyError(RuntimeError):
    """Raised when the optional offline-report stack is unavailable."""


def _stack() -> tuple[Any, Any, Any]:
    try:
        import panel as pn
        from bokeh.resources import INLINE
        from pyecharts import options as opts
    except ImportError as error:
        raise ReportingDependencyError(
            "Interactive reports require Panel and pyecharts; run "
            "`python -m pip install -r requirements-reporting.txt`."
        ) from error
    pn.extension("echarts")
    return pn, INLINE, opts


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def echarts_line(
    *,
    title: str,
    subtitle: str,
    x_values: Sequence[int | float | str],
    series: Sequence[tuple[str, Sequence[float | None], str]],
    y_name: str,
    mark_lines: Sequence[tuple[str, float, str]] = (),
    mark_areas: Sequence[tuple[str, int | float, int | float, str]] = (),
    height: int = 430,
) -> Any:
    """Build one pyecharts Line and expose it through Panel's ECharts pane."""

    pn, _, opts = _stack()
    from pyecharts.charts import Line

    chart = Line(init_opts=opts.InitOpts(width="100%", height=f"{height}px"))
    chart.add_xaxis(list(x_values))
    for index, (name, values, color) in enumerate(series):
        line_opts = opts.LineStyleOpts(width=2, color=color)
        markline_opts = None
        markarea_opts = None
        if index == 0 and mark_lines:
            markline_opts = opts.MarkLineOpts(
                symbol=["none", "none"],
                label_opts=opts.LabelOpts(position="insideEndTop"),
                data=[
                    opts.MarkLineItem(name=label, y=value, linestyle_opts=opts.LineStyleOpts(color=line_color))
                    for label, value, line_color in mark_lines
                ],
            )
        if index == 0 and mark_areas:
            markarea_opts = opts.MarkAreaOpts(
                is_silent=True,
                data=[
                    opts.MarkAreaItem(
                        name=label,
                        x=(start, end),
                        itemstyle_opts=opts.ItemStyleOpts(color=color, opacity=0.09),
                    )
                    for label, start, end, color in mark_areas
                ],
            )
        chart.add_yaxis(
            name,
            list(values),
            is_symbol_show=False,
            is_connect_nones=False,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=line_opts,
            markline_opts=markline_opts,
            markarea_opts=markarea_opts,
        )
    chart.set_global_opts(
        title_opts=opts.TitleOpts(title=title, subtitle=subtitle, pos_left="2%"),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        legend_opts=opts.LegendOpts(pos_top="12%"),
        toolbox_opts=opts.ToolboxOpts(
            is_show=True,
            feature={
                "dataZoom": {"yAxisIndex": "none"},
                "restore": {},
                "saveAsImage": {},
                "dataView": {"readOnly": True},
            },
        ),
        datazoom_opts=[
            opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
            opts.DataZoomOpts(type_="slider", range_start=0, range_end=100),
        ],
        xaxis_opts=opts.AxisOpts(name="Training step", type_="category", boundary_gap=False),
        yaxis_opts=opts.AxisOpts(name=y_name, is_scale=True, splitline_opts=opts.SplitLineOpts(is_show=True)),
    )
    option = json.loads(chart.dump_options())
    return pn.pane.ECharts(option, height=height, sizing_mode="stretch_width")


def metric_cards(items: Iterable[tuple[str, str, str]]) -> Any:
    """Return responsive, project-owned metric cards."""

    pn, _, _ = _stack()
    cards = []
    for label, value, detail in items:
        cards.append(
            pn.pane.HTML(
                '<section class="metric-card">'
                f'<div class="metric-label">{html.escape(label)}</div>'
                f'<div class="metric-value">{html.escape(value)}</div>'
                f'<div class="metric-detail">{html.escape(detail)}</div>'
                "</section>",
                sizing_mode="stretch_width",
            )
        )
    return pn.FlexBox(*cards, sizing_mode="stretch_width")


def save_panel_report(
    *,
    path: Path,
    title: str,
    sections: Sequence[Any],
    description: str = "",
) -> Path:
    """Save a self-contained, offline Panel report."""

    pn, inline, _ = _stack()
    css = """
    :root { --report-blue:#2563eb; --report-ink:#172033; --report-muted:#64748b; }
    body { background:#f6f8fb; color:var(--report-ink); }
    .report-shell { max-width:1500px; margin:0 auto; padding:22px; }
    .report-title { font-size:30px; font-weight:750; margin:0 0 5px; }
    .report-description { color:var(--report-muted); margin:0 0 18px; }
    .metric-card { background:#fff; border:1px solid #dce2ea; border-radius:10px;
      padding:14px 16px; min-width:210px; box-shadow:0 2px 9px #1720330d; }
    .metric-label { color:var(--report-muted); font-size:12px; text-transform:uppercase; }
    .metric-value { font-size:24px; font-weight:750; margin:4px 0; }
    .metric-detail { color:var(--report-muted); font-size:12px; }
    .bk-panel-models-layout-Card { margin:12px 0; }
    """
    header = pn.pane.HTML(
        '<header><h1 class="report-title">' + html.escape(title) + "</h1>"
        '<p class="report-description">' + html.escape(description) + "</p></header>",
        sizing_mode="stretch_width",
    )
    body = pn.Column(header, *sections, css_classes=["report-shell"], sizing_mode="stretch_width")
    path.parent.mkdir(parents=True, exist_ok=True)
    body.save(path, resources=inline, embed=True, title=title, max_states=1, css=[css])
    document = path.read_text(encoding="utf-8")
    document = re.sub(
        r'<link[^>]+href="https://cdn\.holoviz\.org/[^>]+>',
        "",
        document,
    )
    path.write_text(document, encoding="utf-8")
    return path
