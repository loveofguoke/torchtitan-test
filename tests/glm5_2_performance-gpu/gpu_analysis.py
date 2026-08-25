"""Parse Torch Profiler Chrome traces and render compact GPU reports."""

from __future__ import annotations

from collections import defaultdict
import html
import json
import math
from pathlib import Path
import re
from statistics import mean, median
from typing import Any, Iterable


def read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
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
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    by_name: dict[str, list[float]] = defaultdict(list)
    steps: list[int] = []
    for record in records:
        steps.append(int(record.get("step", 0)))
        for name, value in record.get("metrics", {}).items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                by_name[name].append(float(value))
    return {
        "steps": sorted(set(steps)),
        "metrics": {
            name: {
                "count": len(values),
                "min": min(values),
                "median": median(values),
                "mean": mean(values),
                "p90": _percentile(values, 0.90),
                "max": max(values),
                "last": values[-1],
            }
            for name, values in sorted(by_name.items())
        },
    }


def _event_category(event: dict[str, Any]) -> str:
    name = str(event.get("name", "")).lower()
    category = str(event.get("cat", "")).lower()
    combined = f"{category} {name}"
    communication_markers = (
        "nccl",
        "allreduce",
        "all_reduce",
        "allgather",
        "all_gather",
        "reduce_scatter",
        "broadcast",
        "sendrecv",
        "c10d",
    )
    if any(marker in combined for marker in communication_markers):
        return "communication"
    if "kernel" in category or category in {"gpu", "cuda_kernel"}:
        return "cuda_kernel"
    if "cuda_runtime" in category or "cuda_driver" in category:
        return "cuda_runtime"
    if "cpu_op" in category or category == "operator":
        return "cpu_operator"
    if "memory" in category or "memory" in name:
        return "memory"
    return "other"


def _rank_from_path(path: Path) -> int | None:
    match = re.search(r"rank[_-]?(\d+)", path.name)
    return int(match.group(1)) if match else None


def parse_trace(path: Path, *, top_limit: int = 30) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("traceEvents", payload) if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        raise ValueError(f"traceEvents is not a list in {path}")

    categories: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"count": 0, "total_us": 0.0, "max_us": 0.0}
    )
    names: dict[tuple[str, str], dict[str, float | int]] = defaultdict(
        lambda: {"count": 0, "total_us": 0.0, "max_us": 0.0}
    )
    complete_events = 0
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        duration = event.get("dur")
        if not isinstance(duration, (int, float)) or duration < 0:
            continue
        complete_events += 1
        duration_us = float(duration)
        category = _event_category(event)
        name = str(event.get("name", "<unnamed>"))
        for bucket in (categories[category], names[(category, name)]):
            bucket["count"] = int(bucket["count"]) + 1
            bucket["total_us"] = float(bucket["total_us"]) + duration_us
            bucket["max_us"] = max(float(bucket["max_us"]), duration_us)

    top_events = sorted(
        (
            {"category": category, "name": name, **values}
            for (category, name), values in names.items()
        ),
        key=lambda row: float(row["total_us"]),
        reverse=True,
    )[:top_limit]
    return {
        "path": str(path),
        "rank": _rank_from_path(path),
        "complete_events": complete_events,
        "categories": dict(sorted(categories.items())),
        "top_events": top_events,
    }


def inspect_traces(trace_directory: Path) -> dict[str, Any]:
    paths = sorted(trace_directory.rglob("rank*_trace.json"))
    traces = []
    errors = []
    for path in paths:
        try:
            traces.append(parse_trace(path))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append({"path": str(path), "error": repr(error)})
    return {
        "directory": str(trace_directory),
        "trace_count": len(paths),
        "parsed_count": len(traces),
        "traces": traces,
        "errors": errors,
    }


def build_analysis(run_directory: Path) -> dict[str, Any]:
    trace_directory = run_directory / "trainer_output" / "profiling" / "traces"
    metrics = summarize_metrics(read_metrics(run_directory / "metrics.jsonl"))
    traces = inspect_traces(trace_directory)
    return {"metrics": metrics, "profiler": traces}


def _number(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):,.{digits}f}"


def render_html_report(
    *, manifest: dict[str, Any], analysis: dict[str, Any], output_path: Path
) -> None:
    metrics = analysis["metrics"]["metrics"]
    metric_rows = "".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{_number(values['median'])}</td>"
        f"<td>{_number(values['p90'])}</td>"
        f"<td>{_number(values['max'])}</td>"
        "</tr>"
        for name, values in metrics.items()
    ) or '<tr><td colspan="4">No metrics were recorded.</td></tr>'

    trace_sections = []
    for trace in analysis["profiler"]["traces"]:
        category_rows = "".join(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{values['count']}</td>"
            f"<td>{_number(float(values['total_us']) / 1000.0)}</td>"
            f"<td>{_number(float(values['max_us']) / 1000.0)}</td>"
            "</tr>"
            for name, values in trace["categories"].items()
        )
        event_rows = "".join(
            "<tr>"
            f"<td>{html.escape(str(row['category']))}</td>"
            f"<td>{html.escape(str(row['name']))}</td>"
            f"<td>{row['count']}</td>"
            f"<td>{_number(float(row['total_us']) / 1000.0)}</td>"
            "</tr>"
            for row in trace["top_events"]
        )
        trace_sections.append(
            f"<h2>Rank {trace['rank']} trace</h2>"
            f"<p><code>{html.escape(trace['path'])}</code>; "
            f"{trace['complete_events']} complete events.</p>"
            "<h3>Category totals</h3><table><thead><tr>"
            "<th>Category</th><th>Count</th><th>Total ms</th><th>Max ms</th>"
            f"</tr></thead><tbody>{category_rows}</tbody></table>"
            "<h3>Top events by accumulated duration</h3><table><thead><tr>"
            "<th>Category</th><th>Name</th><th>Count</th><th>Total ms</th>"
            f"</tr></thead><tbody>{event_rows}</tbody></table>"
        )
    if not trace_sections:
        trace_sections.append("<h2>Profiler traces</h2><p>No CUDA trace was found.</p>")

    title = html.escape(str(manifest.get("run_name", "GPU performance report")))
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;line-height:1.45;color:#172033}}
table{{border-collapse:collapse;width:100%;margin:1rem 0 2rem}}
th,td{{border:1px solid #ccd3df;padding:.45rem;text-align:left}}
th{{background:#eef2f8}} code{{overflow-wrap:anywhere}}
</style></head><body><h1>{title}</h1>
<p>Topology: <strong>{html.escape(str(manifest.get('topology')))}</strong>; CUDA devices:
<strong>{html.escape(str(manifest.get('visible_devices')))}</strong>.</p>
<p>Open each <code>rank*_trace.json</code> in Chrome tracing or Perfetto for the
full timeline. Category totals overlap and must not be summed as wall time.</p>
<h2>Training metrics</h2><table><thead><tr><th>Metric</th><th>Median</th>
<th>P90</th><th>Max</th></tr></thead><tbody>{metric_rows}</tbody></table>
{''.join(trace_sections)}</body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
