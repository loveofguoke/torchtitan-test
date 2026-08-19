"""Portable analysis and HTML rendering for profiler experiments."""

from __future__ import annotations

import csv
import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for record in records:
        step = int(record["step"])
        for name, value in record.get("metrics", {}).items():
            series[name].append((step, float(value)))

    summaries: dict[str, dict[str, float | int | None]] = {}
    for name, points in series.items():
        values = [value for _, value in points]
        summaries[name] = {
            "count": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "p90": _percentile(values, 0.90),
            "p99": _percentile(values, 0.99),
            "min": min(values),
            "max": max(values),
            "last": values[-1],
        }
    return {
        "steps": sorted({int(record["step"]) for record in records}),
        "series": {name: points for name, points in series.items()},
        "summary": summaries,
    }


def file_inventory(directory: Path) -> dict[str, Any]:
    files = [path for path in directory.rglob("*") if path.is_file()]
    by_suffix: dict[str, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "bytes": 0}
    )
    entries = []
    for path in files:
        size = path.stat().st_size
        suffix = path.suffix.lower() or "[no extension]"
        by_suffix[suffix]["count"] += 1
        by_suffix[suffix]["bytes"] += size
        entries.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "bytes": size,
            }
        )
    entries.sort(key=lambda item: item["bytes"], reverse=True)
    return {
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in entries),
        "by_suffix": dict(sorted(by_suffix.items())),
        "largest_files": entries[:20],
    }


def _normalized(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def _duration_scale(header: str) -> float:
    normalized = header.lower()
    if "(ns)" in normalized or normalized.endswith("ns"):
        return 0.001
    if "(ms)" in normalized or normalized.endswith("ms"):
        return 1000.0
    if "(s)" in normalized or normalized.endswith("seconds"):
        return 1_000_000.0
    return 1.0


def _find_column(headers: list[str], candidates: Iterable[str]) -> str | None:
    normalized_headers = {_normalized(header): header for header in headers}
    for candidate in candidates:
        if candidate in normalized_headers:
            return normalized_headers[candidate]
    return None


def _duration_category(name: str) -> str:
    normalized = name.lower().replace("_", "")
    communication_markers = (
        "hccl",
        "allreduce",
        "allgather",
        "alltoall",
        "reducescatter",
        "broadcast",
        "sendrecv",
    )
    memory_markers = ("memcpy", "memset", "dma", "memorycopy")
    if any(marker in normalized for marker in communication_markers):
        return "communication"
    if any(marker in normalized for marker in memory_markers):
        return "memory"
    return "compute_or_runtime"


def extract_top_csv_entries(
    profiler_directory: Path,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Aggregate recognized kernel/operator/API duration CSVs defensively."""

    tables = []
    for path in profiler_directory.rglob("*.csv"):
        try:
            with path.open(encoding="utf-8-sig", errors="replace", newline="") as stream:
                reader = csv.DictReader(stream)
                headers = reader.fieldnames or []
                name_column = _find_column(
                    headers,
                    ("name", "opname", "operatorname", "apiname", "taskname"),
                )
                duration_column = next(
                    (header for header in headers if "duration" in header.lower()),
                    None,
                )
                if name_column is None or duration_column is None:
                    continue
                scale = _duration_scale(duration_column)
                totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
                for row in reader:
                    name = (row.get(name_column) or "").strip()
                    raw_duration = (row.get(duration_column) or "").strip()
                    if not name or not raw_duration:
                        continue
                    try:
                        duration_us = float(raw_duration) * scale
                    except ValueError:
                        continue
                    totals[name][0] += duration_us
                    totals[name][1] += 1
        except OSError:
            continue
        if not totals:
            continue
        rows = [
            {"name": name, "total_us": values[0], "calls": int(values[1])}
            for name, values in totals.items()
        ]
        rows.sort(key=lambda row: row["total_us"], reverse=True)
        category_totals: dict[str, float] = defaultdict(float)
        for row in rows:
            category_totals[_duration_category(row["name"])] += row["total_us"]
        tables.append(
            {
                "source": path.relative_to(profiler_directory).as_posix(),
                "duration_column": duration_column,
                "category_totals_us": dict(category_totals),
                "rows": rows[:limit],
            }
        )
    tables.sort(
        key=lambda table: sum(row["total_us"] for row in table["rows"]),
        reverse=True,
    )
    return tables[:8]


def build_analysis(run_directory: Path) -> dict[str, Any]:
    metrics = summarize_metrics(read_metrics(run_directory / "metrics.jsonl"))
    profiler_directory = run_directory / "trainer_output" / "profiling" / "traces"
    if not profiler_directory.is_dir():
        profiler_directory = run_directory / "trainer_output" / "profiling"
    inventory = (
        file_inventory(profiler_directory)
        if profiler_directory.is_dir()
        else {"file_count": 0, "total_bytes": 0, "by_suffix": {}, "largest_files": []}
    )
    advisor_path = run_directory / "advisor.json"
    advisor = (
        json.loads(advisor_path.read_text(encoding="utf-8"))
        if advisor_path.is_file()
        else None
    )
    return {
        "metrics": metrics,
        "profiler_directory": str(profiler_directory),
        "profiler_inventory": inventory,
        "top_csv_tables": (
            extract_top_csv_entries(profiler_directory)
            if profiler_directory.is_dir()
            else []
        ),
        "advisor": advisor,
    }


def _format_number(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if not isinstance(value, float):
        return html.escape(str(value))
    magnitude = abs(value)
    if magnitude and (magnitude < 1e-3 or magnitude >= 1e5):
        return f"{value:.4e}"
    return f"{value:.6g}"


def _human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"


def _find_metric(
    summary: dict[str, Any],
    fragments: tuple[str, ...],
) -> tuple[str, dict[str, Any]] | None:
    for name, values in summary.items():
        normalized = name.lower()
        if all(fragment in normalized for fragment in fragments):
            return name, values
    return None


def _line_chart(
    points: list[tuple[int, float]],
    *,
    label: str,
    color: str,
    profile_start: int,
    profile_end: int,
) -> str:
    if not points:
        return '<div class="empty">No samples</div>'
    width, height = 780, 220
    padding_left, padding_right, padding_top, padding_bottom = 64, 20, 20, 38
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_max += 1
    if y_min == y_max:
        padding = abs(y_min) * 0.05 or 1.0
        y_min -= padding
        y_max += padding

    def x_position(value: float) -> float:
        return padding_left + (value - x_min) / (x_max - x_min) * (
            width - padding_left - padding_right
        )

    def y_position(value: float) -> float:
        return padding_top + (y_max - value) / (y_max - y_min) * (
            height - padding_top - padding_bottom
        )

    polyline = " ".join(
        f"{x_position(step):.2f},{y_position(value):.2f}" for step, value in points
    )
    shade_start = max(profile_start, x_min)
    shade_end = min(profile_end, x_max)
    shade = ""
    if shade_start <= shade_end:
        start_x = x_position(shade_start)
        end_x = x_position(shade_end)
        shade = (
            f'<rect x="{start_x:.2f}" y="{padding_top}" '
            f'width="{max(end_x - start_x, 2):.2f}" '
            f'height="{height - padding_top - padding_bottom}" '
            'fill="#38bdf8" opacity="0.12"/>'
        )
    return f"""
    <svg class="chart" viewBox="0 0 {width} {height}" role="img"
         aria-label="{html.escape(label)} by training step">
      {shade}
      <line x1="{padding_left}" y1="{height-padding_bottom}" x2="{width-padding_right}"
            y2="{height-padding_bottom}" class="axis"/>
      <line x1="{padding_left}" y1="{padding_top}" x2="{padding_left}"
            y2="{height-padding_bottom}" class="axis"/>
      <polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.5"/>
      <text x="{padding_left}" y="{height-10}" class="tick">step {x_min}</text>
      <text x="{width-padding_right}" y="{height-10}" text-anchor="end" class="tick">step {x_max}</text>
      <text x="{padding_left-8}" y="{padding_top+5}" text-anchor="end" class="tick">{_format_number(y_max)}</text>
      <text x="{padding_left-8}" y="{height-padding_bottom}" text-anchor="end" class="tick">{_format_number(y_min)}</text>
    </svg>
    """


def _bar_chart(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<div class="empty">No recognized duration table</div>'
    maximum = max(row["total_us"] for row in rows) or 1.0
    result = ['<div class="bars">']
    for row in rows[:12]:
        width = row["total_us"] / maximum * 100
        result.append(
            '<div class="bar-row">'
            f'<div class="bar-name" title="{html.escape(row["name"])}">'
            f'{html.escape(row["name"])}</div>'
            '<div class="bar-track">'
            f'<div class="bar-fill" style="width:{width:.2f}%"></div></div>'
            f'<div class="bar-value">{row["total_us"] / 1000:.3f} ms '
            f'({row["calls"]} calls)</div></div>'
        )
    result.append("</div>")
    return "".join(result)


def _composition_chart(table: dict[str, Any]) -> str:
    totals = table.get("category_totals_us", {})
    total = sum(totals.values())
    if total <= 0:
        return ""
    colors = {
        "compute_or_runtime": "#2563eb",
        "communication": "#f97316",
        "memory": "#10b981",
    }
    labels = {
        "compute_or_runtime": "Compute / runtime",
        "communication": "Communication",
        "memory": "Memory movement",
    }
    segments = []
    legend = []
    for category in ("compute_or_runtime", "communication", "memory"):
        value = totals.get(category, 0.0)
        fraction = value / total * 100
        segments.append(
            f'<div title="{labels[category]} {fraction:.2f}%" '
            f'style="width:{fraction:.5f}%;background:{colors[category]}"></div>'
        )
        legend.append(
            f'<span><i style="background:{colors[category]}"></i>'
            f'{labels[category]}: {fraction:.2f}%</span>'
        )
    return (
        '<div class="composition">' + "".join(segments) + "</div>"
        '<div class="legend">' + "".join(legend) + "</div>"
        '<p class="subtle">Heuristic classification by kernel/operator name. '
        "Overlapping communication and compute require msprof-analyze for "
        "authoritative distributed analysis.</p>"
    )


def render_html_report(
    *,
    manifest: dict[str, Any],
    analysis: dict[str, Any],
    output_path: Path,
) -> None:
    config = manifest["config"]
    metric_data = analysis["metrics"]
    summary = metric_data["summary"]
    series = metric_data["series"]
    # Trainer calls profiler.step() after finishing each one-based training
    # step, while the scheduler itself is zero based. Recording therefore
    # starts on the following training step.
    profile_start = config["skip_steps"] + config["warmup_steps"] + 1
    profile_end = profile_start + config["active_steps"] - 1

    cards = []
    card_specs = (
        (("end_to_end",), "Median step time", "median", "s"),
        (("throughput",), "Mean throughput", "mean", " tok/s"),
        (("tflops",), "Mean TFLOPS", "mean", ""),
        (("mfu",), "Mean MFU", "mean", "%"),
        (("max_active",), "Peak active memory", "max", " GiB"),
    )
    for fragments, label, statistic, suffix in card_specs:
        found = _find_metric(summary, fragments)
        if found:
            _, values = found
            cards.append(
                f'<div class="card"><span>{html.escape(label)}</span>'
                f'<strong>{_format_number(values[statistic])}{suffix}</strong></div>'
            )

    charts = []
    chart_specs = (
        (("end_to_end",), "Step time", "#2563eb"),
        (("throughput",), "Throughput", "#059669"),
        (("tflops",), "TFLOPS", "#7c3aed"),
        (("max_active",), "Active memory", "#d97706"),
    )
    for fragments, label, color in chart_specs:
        found = _find_metric(series, fragments)
        if found:
            name, points = found
            charts.append(
                f'<section><h2>{html.escape(label)}</h2><p class="subtle">'
                f'{html.escape(name)}; blue region is the active profiling window.</p>'
                + _line_chart(
                    points,
                    label=label,
                    color=color,
                    profile_start=profile_start,
                    profile_end=profile_end,
                )
                + "</section>"
            )

    configuration_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in config.items()
        if key not in {"environment", "extra_args"}
    )
    profiler_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th>"
        f"<td>{html.escape(str(value))}</td></tr>"
        for key, value in manifest.get("profiler_environment", {}).items()
    )
    inventory = analysis["profiler_inventory"]
    suffix_rows = "".join(
        f"<tr><td>{html.escape(suffix)}</td><td>{values['count']}</td>"
        f"<td>{_human_bytes(values['bytes'])}</td></tr>"
        for suffix, values in inventory["by_suffix"].items()
    ) or '<tr><td colspan="3">No profiler files discovered</td></tr>'

    top_tables = analysis["top_csv_tables"]
    composition_table = next(
        (
            table
            for table in top_tables
            if "kernel" in table["source"].lower()
        ),
        top_tables[0] if top_tables else None,
    )
    composition_section = (
        '<section><h2>Time composition probe</h2><p class="subtle">Source: '
        f'{html.escape(composition_table["source"])}</p>'
        f'{_composition_chart(composition_table)}</section>'
        if composition_table is not None
        else ""
    )
    top_sections = "".join(
        f'<section><h2>Top duration entries</h2><p class="subtle">Source: '
        f'{html.escape(table["source"])}</p>{_bar_chart(table["rows"])}</section>'
        for table in top_tables[:3]
    )
    if not top_sections:
        top_sections = (
            '<section><h2>Operator and kernel detail</h2><div class="callout">'
            "No recognized duration CSV is available yet. For offline captures, run "
            "<code>--analyze --offline-parse</code>; for deeper diagnosis, run "
            "<code>--analyze --advisor</code>.</div></section>"
        )

    advisor = analysis.get("advisor")
    advisor_section = ""
    if advisor is not None:
        advisor_section = (
            '<section><h2>msprof-analyze advisor</h2>'
            f'<p>Return code: <strong>{advisor.get("return_code")}</strong></p>'
            '<details><summary>Command and output</summary><pre>'
            f'{html.escape(" ".join(advisor.get("command", [])))}\n\n'
            f'{html.escape(advisor.get("stdout", ""))}\n'
            f'{html.escape(advisor.get("stderr", ""))}</pre></details></section>'
        )

    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(manifest['run_name'])} performance report</title>
<style>
:root{{--bg:#f5f7fb;--panel:#fff;--text:#172033;--muted:#657089;--line:#dfe5ef;--accent:#2563eb}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:28px}} h1{{font-size:27px;margin:0 0 6px}} h2{{font-size:18px;margin:0 0 4px}}
.subtle{{color:var(--muted);margin:0 0 14px}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:22px 0}}
.card,section{{background:var(--panel);border:1px solid var(--line);border-radius:12px;box-shadow:0 3px 14px #1720330a}}
.card{{padding:16px}} .card span{{display:block;color:var(--muted)}} .card strong{{font-size:22px}}
section{{padding:20px;margin:16px 0;overflow:auto}} table{{border-collapse:collapse;width:100%}} th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left}}
th{{background:#f8fafc;position:sticky;top:0}} code{{background:#edf2ff;padding:2px 5px;border-radius:4px}} .chart{{width:100%;min-width:600px}}
.axis{{stroke:#9aa6ba;stroke-width:1}} .tick{{fill:#657089;font-size:11px}} .empty,.callout{{padding:18px;background:#f8fafc;border-radius:8px;color:var(--muted)}}
.bars{{display:grid;gap:8px}} .bar-row{{display:grid;grid-template-columns:minmax(180px,2fr) minmax(220px,4fr) 150px;gap:10px;align-items:center}}
.bar-name{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}} .bar-track{{height:12px;background:#e8edf6;border-radius:8px;overflow:hidden}} .bar-fill{{height:100%;background:linear-gradient(90deg,#2563eb,#38bdf8)}} .bar-value{{text-align:right;color:var(--muted);font-variant-numeric:tabular-nums}}
.composition{{height:28px;border-radius:7px;overflow:hidden;display:flex;background:#e8edf6}} .composition>div{{min-width:0}} .legend{{display:flex;flex-wrap:wrap;gap:16px;margin:12px 0}} .legend i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}}
@media(max-width:700px){{main{{padding:14px}}.bar-row{{grid-template-columns:1fr}}.bar-value{{text-align:left}}}}
</style></head><body><main>
<h1>GLM-5.2 performance probe</h1>
<p class="subtle">{html.escape(manifest['run_name'])} · {html.escape(manifest['device'])} · {html.escape(manifest['topology'])} · {html.escape(manifest['preset'])}</p>
<div class="cards">{''.join(cards) or '<div class="card"><span>Training metrics</span><strong>Unavailable</strong></div>'}</div>
{''.join(charts)}
{composition_section}
{top_sections}
{advisor_section}
<section><h2>Profiler data inventory</h2><p class="subtle">{inventory['file_count']} files, {_human_bytes(inventory['total_bytes'])}; raw directory: {html.escape(analysis['profiler_directory'])}</p>
<table><thead><tr><th>Type</th><th>Files</th><th>Size</th></tr></thead><tbody>{suffix_rows}</tbody></table></section>
<section><h2>Effective Ascend profiler controls</h2><table><tbody>{profiler_rows or '<tr><td>No NPU-specific controls</td></tr>'}</tbody></table></section>
<section><h2>Experiment configuration</h2><table><tbody>{configuration_rows}</tbody></table></section>
</main></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
