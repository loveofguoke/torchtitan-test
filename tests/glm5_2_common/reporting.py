# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Shared offline report components backed by Panel and Apache ECharts."""

from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence


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
    mark_points: Sequence[tuple[str, int | float, float, str]] = (),
    height: int = 760,
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
        markpoint_opts = None
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
        if index == 0 and mark_points:
            markpoint_opts = opts.MarkPointOpts(
                symbol_size=64,
                label_opts=opts.LabelOpts(font_size=13),
                data=[
                    opts.MarkPointItem(
                        name=label,
                        coord=[x_value, y_value],
                        value=f"{y_value:.4g}",
                        itemstyle_opts=opts.ItemStyleOpts(color=point_color),
                    )
                    for label, x_value, y_value, point_color in mark_points
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
            markpoint_opts=markpoint_opts,
        )
    chart.set_global_opts(
        title_opts=opts.TitleOpts(
            title=title,
            subtitle=subtitle,
            pos_left="3%",
            item_gap=14,
            title_textstyle_opts=opts.TextStyleOpts(font_size=24),
            subtitle_textstyle_opts=opts.TextStyleOpts(font_size=15),
        ),
        tooltip_opts=opts.TooltipOpts(
            trigger="axis", axis_pointer_type="cross", textstyle_opts=opts.TextStyleOpts(font_size=15)
        ),
        legend_opts=opts.LegendOpts(
            pos_top="18%",
            item_gap=24,
            textstyle_opts=opts.TextStyleOpts(font_size=15),
        ),
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
        xaxis_opts=opts.AxisOpts(
            name="训练步 / Step",
            type_="category",
            boundary_gap=False,
            name_gap=38,
            name_textstyle_opts=opts.TextStyleOpts(font_size=15),
            axislabel_opts=opts.LabelOpts(font_size=13, margin=16),
        ),
        yaxis_opts=opts.AxisOpts(
            name=y_name,
            is_scale=True,
            name_gap=52,
            name_textstyle_opts=opts.TextStyleOpts(font_size=15),
            axislabel_opts=opts.LabelOpts(font_size=13, margin=16),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
    )
    option = json.loads(chart.dump_options())
    option["grid"] = {
        "left": "10%",
        "right": "5%",
        "top": "29%",
        "bottom": "17%",
        "containLabel": True,
    }
    chart_pane = pn.pane.ECharts(
        option,
        height=height,
        sizing_mode="stretch_width",
    )
    return pn.Column(
        chart_pane,
        margin=(30, 0, 64, 0),
        sizing_mode="stretch_width",
    )


def summary_table(
    *,
    columns: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> Any:
    """Render a compact summary table without external table dependencies."""

    pn, _, _ = _stack()
    head = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(value)}</td>" for value in row)
        + "</tr>"
        for row in rows
    )
    return pn.pane.HTML(
        '<div class="summary-table-wrap"><table class="summary-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>",
        sizing_mode="stretch_width",
    )


def section_heading(title: str, description: str) -> Any:
    """Render a visible bilingual report section boundary."""

    pn, _, _ = _stack()
    return pn.pane.HTML(
        '<section class="report-section"><h2>'
        + html.escape(title)
        + "</h2><p>"
        + html.escape(description)
        + "</p></section>",
        sizing_mode="stretch_width",
    )


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
    .report-shell { max-width:1900px; margin:0 auto; padding:36px 44px; }
    .report-title { font-size:40px; font-weight:750; margin:0 0 10px; }
    .report-description { color:var(--report-muted); font-size:18px; line-height:1.6; margin:0 0 26px; }
    .report-section { margin:36px 0 14px; border-left:6px solid var(--report-blue);
      background:#edf4ff; padding:16px 20px; border-radius:8px; }
    .report-section h2 { font-size:28px; margin:0 0 6px; }
    .report-section p { color:var(--report-muted); font-size:17px; line-height:1.55; margin:0; }
    .summary-table-wrap { overflow-x:auto; margin:8px 0 22px; }
    .summary-table { width:100%; border-collapse:collapse; background:#fff;
      border:1px solid #ccd5e2; border-radius:10px; overflow:hidden;
      box-shadow:0 2px 9px #1720330d; font-size:18px; }
    .summary-table th { background:#eaf1fb; color:#24324a; font-weight:700;
      padding:14px 16px; text-align:left; }
    .summary-table td { border-top:1px solid #dce2ea; padding:13px 14px;
      font-variant-numeric:tabular-nums; line-height:1.45; }
    .summary-table tbody tr:hover { background:#f4f8fd; }
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
