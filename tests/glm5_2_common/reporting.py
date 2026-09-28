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
    pn.extension("echarts", "tabulator")
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
    x_name: str = "训练步 / Step",
    mark_lines: Sequence[tuple[str, float, str]] = (),
    mark_areas: Sequence[tuple[str, int | float, int | float, str]] = (),
    mark_points: Sequence[tuple[str, int | float, float, str]] = (),
    height: int = 760,
) -> Any:
    """Build one pyecharts Line and expose it through Panel's ECharts pane."""

    pn, _, opts = _stack()
    from pyecharts.charts import Line

    chart = Line(init_opts=opts.InitOpts(width="100%", height=f"{height}px"))
    x_categories = [str(value) for value in x_values]
    chart.add_xaxis(x_categories)
    is_short_window = len(x_values) <= 20
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
                label_opts=opts.LabelOpts(is_show=False),
                data=[
                    opts.MarkAreaItem(
                        name=label,
                        x=(str(start), str(end)),
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
                        coord=[str(x_value), y_value],
                        value=f"{y_value:.4g}",
                        itemstyle_opts=opts.ItemStyleOpts(color=point_color),
                    )
                    for label, x_value, y_value, point_color in mark_points
                ],
            )
        chart.add_yaxis(
            name,
            list(values),
            is_symbol_show=is_short_window,
            symbol_size=8 if is_short_window else 4,
            is_clip=not is_short_window,
            is_connect_nones=False,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=line_opts,
            markline_opts=markline_opts,
            markarea_opts=markarea_opts,
            markpoint_opts=markpoint_opts,
        )
    chart.set_global_opts(
        tooltip_opts=opts.TooltipOpts(
            trigger="axis", axis_pointer_type="cross", textstyle_opts=opts.TextStyleOpts(font_size=15)
        ),
        legend_opts=opts.LegendOpts(
            pos_top="3%",
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
            name=x_name,
            type_="category",
            boundary_gap=False,
            name_gap=38,
            name_textstyle_opts=opts.TextStyleOpts(font_size=15),
            axislabel_opts=opts.LabelOpts(
                font_size=13,
                margin=16,
                interval=0 if is_short_window else None,
            ),
            axistick_opts=opts.AxisTickOpts(
                is_align_with_label=True,
            ),
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
        "left": "7%",
        "right": "8%",
        "top": "13%",
        "bottom": "17%",
        "containLabel": True,
    }
    area_legend = ""
    if mark_areas:
        area_items = "".join(
            '<span style="display:inline-flex;align-items:center;gap:8px;'
            'font-size:17px;color:#334155;margin:0 24px 8px 0">'
            f'<span style="display:inline-block;width:24px;height:14px;'
            f'border:2px solid {html.escape(color)};background:{html.escape(color)}22;'
            'border-radius:3px"></span>'
            + html.escape(label)
            + "</span>"
            for label, _, _, color in mark_areas
        )
        area_legend = (
            '<div style="display:flex;flex-wrap:wrap;margin:18px 0 0">'
            + area_items
            + "</div>"
        )
    chart_heading = pn.pane.HTML(
        '<div class="chart-heading" style="margin:0 0 20px;padding:0 8px">'
        '<h3 style="color:#172033;font-size:27px;line-height:1.35;'
        'font-weight:700;margin:0 0 10px">'
        + html.escape(title)
        + '</h3><p style="color:#526178;font-size:18px;line-height:1.65;'
        'margin:0">'
        + html.escape(subtitle)
        + "</p>"
        + area_legend
        + "</div>",
        sizing_mode="stretch_width",
    )
    chart_pane = pn.pane.ECharts(
        option,
        height=height,
        sizing_mode="stretch_width",
    )
    return pn.Column(
        chart_heading,
        chart_pane,
        margin=(42, 0, 78, 0),
        sizing_mode="stretch_width",
    )


def echarts_heatmap(
    *,
    title: str,
    subtitle: str,
    x_values: Sequence[int | float | str],
    y_values: Sequence[int | float | str],
    values: Sequence[tuple[int, int, float]],
    value_name: str,
    height: int = 680,
) -> Any:
    """Build an interactive rank-by-step heatmap for diagnostic evidence."""

    pn, _, opts = _stack()
    from pyecharts.charts import HeatMap

    maximum = max((value for _, _, value in values), default=1.0)
    chart = HeatMap(init_opts=opts.InitOpts(width="100%", height=f"{height}px"))
    chart.add_xaxis([str(value) for value in x_values])
    chart.add_yaxis(
        value_name,
        [str(value) for value in y_values],
        [[x, y, value] for x, y, value in values],
        label_opts=opts.LabelOpts(is_show=False),
    )
    chart.set_global_opts(
        tooltip_opts=opts.TooltipOpts(trigger="item"),
        toolbox_opts=opts.ToolboxOpts(
            is_show=True,
            feature={"restore": {}, "saveAsImage": {}, "dataView": {"readOnly": True}},
        ),
        visualmap_opts=opts.VisualMapOpts(
            min_=0,
            max_=maximum,
            pos_right="2%",
            pos_top="middle",
            is_calculable=True,
        ),
        xaxis_opts=opts.AxisOpts(
            name="训练步 / Step",
            name_gap=36,
            axislabel_opts=opts.LabelOpts(font_size=13),
        ),
        yaxis_opts=opts.AxisOpts(
            name="Rank",
            name_gap=42,
            axislabel_opts=opts.LabelOpts(font_size=13),
        ),
        datazoom_opts=[opts.DataZoomOpts(type_="inside", range_start=0, range_end=100)],
    )
    option = json.loads(chart.dump_options())
    option["grid"] = {
        "left": "7%",
        "right": "14%",
        "top": "8%",
        "bottom": "13%",
        "containLabel": True,
    }
    heading = pn.pane.HTML(
        '<div style="margin:0 0 20px;padding:0 8px">'
        '<h3 style="color:#172033;font-size:27px;line-height:1.35;'
        'font-weight:700;margin:0 0 10px">'
        + html.escape(title)
        + '</h3><p style="color:#526178;font-size:18px;line-height:1.65;margin:0">'
        + html.escape(subtitle)
        + "</p></div>",
        sizing_mode="stretch_width",
    )
    return pn.Column(
        heading,
        pn.pane.ECharts(option, height=height, sizing_mode="stretch_width"),
        margin=(42, 0, 78, 0),
        sizing_mode="stretch_width",
    )


def summary_table(
    *,
    columns: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> Any:
    """Render a compact summary table without external table dependencies."""

    pn, _, _ = _stack()
    header_style = (
        "background:#eaf1fb;color:#24324a;font-size:20px;font-weight:700;"
        "padding:18px 20px;text-align:left;border:1px solid #bcc8d8"
    )
    cell_style = (
        "background:#fff;color:#172033;font-size:19px;line-height:1.55;"
        "padding:17px 20px;text-align:left;border:1px solid #ccd5e2;"
        "font-variant-numeric:tabular-nums"
    )
    head = "".join(
        f'<th style="{header_style}">{html.escape(column)}</th>'
        for column in columns
    )
    body = "".join(
        "<tr>"
        + "".join(
            f'<td style="{cell_style}">{html.escape(value)}</td>'
            for value in row
        )
        + "</tr>"
        for row in rows
    )
    return pn.pane.HTML(
        '<div class="summary-table-wrap" style="width:100%;overflow-x:auto;'
        'margin:20px 0 42px"><table class="summary-table" style="width:100%;'
        'border-collapse:collapse;background:#fff;border:1px solid #bcc8d8;'
        'box-shadow:0 3px 12px #17203314">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>",
        sizing_mode="stretch_width",
    )


def interactive_table(
    *,
    title: str,
    description: str,
    rows: Sequence[dict[str, Any]],
    columns: Sequence[str],
    page_size: int = 30,
    pagination: bool = True,
) -> Any:
    """Render a filterable, sortable, offline table."""

    pn, _, _ = _stack()
    import pandas as pd

    frame = pd.DataFrame(
        [{column: row.get(column) for column in columns} for row in rows],
        columns=list(columns),
    )
    options: dict[str, Any] = {
        "show_index": False,
        "header_filters": True,
        "sizing_mode": "stretch_width",
        "height": 760 if pagination else min(6000, max(760, 36 * (len(frame) + 2))),
        "layout": "fit_data_stretch",
    }
    if pagination:
        options.update(pagination="local", page_size=page_size)
    heading = pn.pane.HTML(
        '<div class="table-heading" style="margin:0 0 18px;padding:0">'
        '<h3 style="color:#172033;font-size:25px;line-height:1.35;'
        'font-weight:700;margin:0 0 8px">'
        + html.escape(title)
        + '</h3><p style="color:#526178;font-size:17px;line-height:1.6;'
        'margin:0">'
        + html.escape(description)
        + "</p></div>",
        sizing_mode="stretch_width",
    )
    table = pn.widgets.Tabulator(
        frame,
        theme="simple",
        stylesheets=[
            """
            :host .tabulator { border:1px solid #bcc8d8; border-radius:9px;
              overflow:hidden; box-shadow:0 3px 12px #17203314;
              font-size:16px; background:#fff; }
            :host .tabulator-header { background:#eaf1fb; color:#24324a;
              font-weight:700; border-bottom:1px solid #bcc8d8; }
            :host .tabulator-col { background:#eaf1fb !important;
              border-right:1px solid #ccd5e2 !important; }
            :host .tabulator-cell { padding:11px 13px;
              border-right:1px solid #e2e8f0; }
            :host .tabulator-row { border-bottom:1px solid #e2e8f0; }
            :host .tabulator-row:hover { background:#f4f8fd !important; }
            """
        ],
        **options,
    )
    return pn.Column(
        heading,
        table,
        sizing_mode="stretch_width",
        margin=(30, 0, 64, 0),
    )


def section_heading(title: str, description: str) -> Any:
    """Render a visible bilingual report section boundary."""

    pn, _, _ = _stack()
    return pn.pane.HTML(
        '<section class="report-section" style="margin:58px 0 28px;'
        'border-left:7px solid #2563eb;background:#edf4ff;padding:20px 26px;'
        'border-radius:9px"><h2 style="color:#172033;font-size:32px;'
        'line-height:1.3;margin:0 0 9px">'
        + html.escape(title)
        + '</h2><p style="color:#526178;font-size:19px;line-height:1.65;'
        'margin:0">'
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
    html, body { margin:0; padding:0; }
    .report-shell { width:calc(100% - 96px) !important; max-width:1400px;
      box-sizing:border-box; margin:36px auto 72px !important;
      padding:48px 64px 72px !important; background:#fff;
      border:1px solid #dce3ec; border-radius:14px;
      box-shadow:0 6px 24px #17203312; }
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
    .tabulator { font-variant-numeric:tabular-nums; }
    .bk-panel-models-layout-Card { margin:12px 0; }
    @media (max-width:900px) {
      .report-shell { width:calc(100% - 28px) !important;
        margin:14px auto 36px !important; padding:28px 22px 44px !important; }
    }
    """
    header = pn.pane.HTML(
        '<header style="margin:0 0 48px"><h1 class="report-title" '
        'style="color:#172033;font-size:42px;line-height:1.3;font-weight:750;'
        'margin:0 0 18px">'
        + html.escape(title)
        + '</h1><p class="report-description" style="color:#526178;'
        'font-size:20px;line-height:1.7;margin:0">'
        + html.escape(description)
        + "</p></header>",
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
