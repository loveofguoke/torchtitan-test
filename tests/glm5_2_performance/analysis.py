"""Portable analysis and HTML rendering for profiler experiments."""

from __future__ import annotations

import csv
import html
import json
import math
import re
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


def summarize_profile_phases(
    records: list[dict[str, Any]],
    *,
    config: dict[str, Any] | None,
    profiler_environment: dict[str, str] | None,
) -> dict[str, Any]:
    """Separate steady state, collection overhead, and synchronous parse tax."""
    if not records or not config:
        return {"phase_order": [], "phases": {}, "comparison": {}}

    skip_steps = int(config["skip_steps"])
    warmup_steps = int(config["warmup_steps"])
    active_steps = int(config["active_steps"])
    active_start = skip_steps + warmup_steps + 1
    active_end = active_start + active_steps - 1
    parse_mode = (profiler_environment or {}).get(
        "TORCHTITAN_NPU_PROFILER_PARSE_MODE", "unknown"
    )
    available_steps = {int(record["step"]) for record in records}
    final_step = max(available_steps)

    phase_steps: dict[str, list[int]] = {
        "startup": [step for step in (1,) if step in available_steps],
        "baseline": [
            step
            for step in range(2, min(skip_steps, final_step) + 1)
            if step in available_steps
        ],
        "profile_warmup": [
            step
            for step in range(skip_steps + 1, min(active_start, final_step + 1))
            if step in available_steps
        ],
        "profile_active": [
            step
            for step in range(active_start, min(active_end, final_step) + 1)
            if step in available_steps
        ],
    }
    boundary_step = active_end + 1
    if parse_mode == "sync" and boundary_step in available_steps:
        phase_steps["sync_parse_boundary"] = [boundary_step]
        post_start = boundary_step + 1
    else:
        post_start = boundary_step
    phase_steps["post_profile"] = [
        step
        for step in range(post_start, final_step + 1)
        if step in available_steps
    ]
    phase_steps = {name: steps for name, steps in phase_steps.items() if steps}

    records_by_step = {int(record["step"]): record for record in records}
    phases: dict[str, Any] = {}
    for name, steps in phase_steps.items():
        phase_summary = summarize_metrics([records_by_step[step] for step in steps])
        phases[name] = {
            "steps": steps,
            "summary": phase_summary["summary"],
        }

    step_metric = "time_metrics/end_to_end(s)"
    throughput_metric = "throughput(tps)"

    def statistic(phase: str, metric: str, name: str) -> float | None:
        value = phases.get(phase, {}).get("summary", {}).get(metric, {}).get(name)
        return float(value) if value is not None else None

    baseline_step = statistic("baseline", step_metric, "median")
    active_step = statistic("profile_active", step_metric, "median")
    post_step = statistic("post_profile", step_metric, "median")
    boundary_time = statistic("sync_parse_boundary", step_metric, "median")
    baseline_throughput = statistic("baseline", throughput_metric, "median")
    active_throughput = statistic("profile_active", throughput_metric, "median")

    def percent_change(value: float | None, reference: float | None) -> float | None:
        if value is None or reference in {None, 0.0}:
            return None
        return (value / reference - 1.0) * 100.0

    parse_reference = post_step if post_step is not None else baseline_step
    comparison = {
        "parse_mode": parse_mode,
        "baseline_median_step_seconds": baseline_step,
        "active_median_step_seconds": active_step,
        "active_step_overhead_percent": percent_change(active_step, baseline_step),
        "post_profile_median_step_seconds": post_step,
        "baseline_median_throughput_tps": baseline_throughput,
        "active_median_throughput_tps": active_throughput,
        "active_throughput_change_percent": percent_change(
            active_throughput, baseline_throughput
        ),
        "parse_boundary_step": (
            boundary_step if "sync_parse_boundary" in phases else None
        ),
        "parse_boundary_step_seconds": boundary_time,
        "parse_stall_excess_seconds": (
            boundary_time - parse_reference
            if boundary_time is not None and parse_reference is not None
            else None
        ),
    }
    return {
        "schedule": {
            "skip_steps": skip_steps,
            "warmup_steps": warmup_steps,
            "active_steps": active_steps,
            "active_start": active_start,
            "active_end": active_end,
        },
        "phase_order": list(phase_steps),
        "phases": phases,
        "comparison": comparison,
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
    if "aicpu" in normalized:
        return "aicpu"
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


def extract_kernel_shape_entries(
    profiler_directory: Path,
    *,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Aggregate kernel duration by operator and concrete input signature."""
    tables = []
    for path in profiler_directory.rglob("kernel_details.csv"):
        groups: dict[tuple[str, str, str], dict[str, float]] = defaultdict(
            lambda: {"total_us": 0.0, "calls": 0.0, "max_us": 0.0}
        )
        try:
            with path.open(encoding="utf-8-sig", errors="replace", newline="") as stream:
                reader = csv.DictReader(stream)
                headers = reader.fieldnames or []
                name_column = _find_column(headers, ("name", "opname"))
                duration_column = next(
                    (header for header in headers if "duration" in header.lower()),
                    None,
                )
                if (
                    name_column is None
                    or duration_column is None
                    or "Input Shapes" not in headers
                    or "Input Data Types" not in headers
                ):
                    continue
                scale = _duration_scale(duration_column)
                for row in reader:
                    name = (row.get(name_column) or "").strip()
                    try:
                        duration_us = float(row.get(duration_column) or "") * scale
                    except ValueError:
                        continue
                    if not name:
                        continue
                    shapes = (row.get("Input Shapes") or "-").strip() or "-"
                    dtypes = (row.get("Input Data Types") or "-").strip() or "-"
                    group = groups[(name, shapes, dtypes)]
                    group["total_us"] += duration_us
                    group["calls"] += 1
                    group["max_us"] = max(group["max_us"], duration_us)
        except OSError:
            continue
        rows = [
            {
                "name": name,
                "input_shapes": shapes,
                "input_dtypes": dtypes,
                "total_us": values["total_us"],
                "calls": int(values["calls"]),
                "mean_us": values["total_us"] / values["calls"],
                "max_us": values["max_us"],
            }
            for (name, shapes, dtypes), values in groups.items()
        ]
        rows.sort(key=lambda row: row["total_us"], reverse=True)
        tables.append(
            {
                "source": path.relative_to(profiler_directory).as_posix(),
                "rows": rows[:limit],
            }
        )
    return tables


def extract_l2_cache_entries(
    profiler_directory: Path,
    *,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Summarize available L2 hit/victim samples by operator name."""
    entries = []
    for path in profiler_directory.rglob("l2_cache.csv"):
        groups: dict[str, dict[str, list[float] | int]] = defaultdict(
            lambda: {"samples": 0, "hit_rates": [], "victim_rates": []}
        )
        try:
            with path.open(encoding="utf-8-sig", errors="replace", newline="") as stream:
                reader = csv.DictReader(stream)
                for row in reader:
                    name = (row.get("Op Name") or row.get("Name") or "").strip()
                    if not name:
                        continue
                    group = groups[name]
                    group["samples"] = int(group["samples"]) + 1
                    for column, target in (
                        ("Hit Rate", "hit_rates"),
                        ("Victim Rate", "victim_rates"),
                    ):
                        try:
                            value = float(row.get(column) or "")
                        except ValueError:
                            continue
                        values = group[target]
                        assert isinstance(values, list)
                        values.append(value)
        except OSError:
            continue
        rows = []
        for name, values in groups.items():
            hit_rates = values["hit_rates"]
            victim_rates = values["victim_rates"]
            assert isinstance(hit_rates, list) and isinstance(victim_rates, list)
            rows.append(
                {
                    "name": name,
                    "samples": int(values["samples"]),
                    "mean_hit_rate": (
                        statistics.fmean(hit_rates) if hit_rates else None
                    ),
                    "mean_victim_rate": (
                        statistics.fmean(victim_rates) if victim_rates else None
                    ),
                }
            )
        rows.sort(key=lambda row: row["samples"], reverse=True)
        entries.append(
            {
                "source": path.relative_to(profiler_directory).as_posix(),
                "rows": rows[:limit],
            }
        )
    return entries


def inspect_profiler_deliverables(profiler_directory: Path) -> dict[str, Any]:
    """Classify official Ascend outputs and portable GPU traces."""

    if not profiler_directory.is_dir():
        return {"ascend_profiles": [], "gpu_traces": [], "insight_ready": False}

    ascend_profiles = []
    for profile_root in sorted(profiler_directory.rglob("*_ascend_pt")):
        if not profile_root.is_dir():
            continue
        output = profile_root / "ASCEND_PROFILER_OUTPUT"
        text_files = {
            name: (output / name).is_file()
            for name in (
                "api_statistic.csv",
                "communication.json",
                "communication_matrix.json",
                "data_preprocess.csv",
                "hccs.csv",
                "kernel_details.csv",
                "l2_cache.csv",
                "memory_record.csv",
                "nic.csv",
                "npu_module_mem.csv",
                "operator_details.csv",
                "operator_memory.csv",
                "op_statistic.csv",
                "pcie.csv",
                "roce.csv",
                "soc_pmu.csv",
                "step_trace_time.csv",
                "task_time.csv",
                "trace_view.json",
            )
        }
        database_files = sorted(
            path.name
            for pattern in ("analysis.db", "ascend_pytorch_profiler*.db")
            for path in output.glob(pattern)
            if path.is_file()
        )
        profiler_info_files = sorted(
            path.name for path in profile_root.glob("profiler_info*.json")
        )
        profiler_metadata_files = sorted(
            path.name for path in profile_root.glob("profiler_metadata*.json")
        )
        parsed_text = any(text_files.values())
        parsed_database = bool(database_files)
        ascend_profiles.append(
            {
                "root": str(profile_root),
                "relative_root": profile_root.relative_to(
                    profiler_directory
                ).as_posix(),
                "raw_ready": bool(profiler_info_files),
                "parsed_text": parsed_text,
                "parsed_database": parsed_database,
                "insight_ready": parsed_text or parsed_database,
                "text_files": text_files,
                "database_files": database_files,
                "profiler_info_files": profiler_info_files,
                "profiler_metadata_files": profiler_metadata_files,
            }
        )

    gpu_traces = sorted(
        str(path)
        for path in profiler_directory.rglob("*.json")
        if "trace" in path.name.lower()
        and not any(parent.name.endswith("_ascend_pt") for parent in path.parents)
    )
    return {
        "ascend_profiles": ascend_profiles,
        "gpu_traces": gpu_traces,
        "insight_ready": bool(gpu_traces)
        or any(profile["insight_ready"] for profile in ascend_profiles),
    }


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    return (
        json.loads(path.read_text(encoding="utf-8"))
        if path.is_file()
        else None
    )


def _compiler_diagnostics(runtime_log: Path) -> dict[str, Any]:
    """Summarize compiler, fallback, and profiler-parser runtime messages."""

    categories = {
        "graph_breaks": ("graph break", "graph_break"),
        "recompiles": ("recompiling function", "recompile reason"),
        "backend_failures": (
            "backendcompilerfailed",
            "torch._dynamo.exc",
            "inductorerror",
        ),
        "npu_cpu_fallbacks": ("will fall back to run on the cpu",),
        "aicpu_fallbacks": ("kernel is running on aicpu",),
    }
    result: dict[str, Any] = {
        "compile_markers": 0,
        "cann_parse_seconds": None,
        "all_parse_seconds": None,
        **{name: 0 for name in categories},
        "examples": [],
    }
    if not runtime_log.is_file():
        return result
    for line in runtime_log.read_text(encoding="utf-8", errors="replace").splitlines():
        normalized = line.lower()
        duration_match = re.search(
            r"total time of (\d+):(\d+):(\d+(?:\.\d+)?)", normalized
        )
        if duration_match:
            hours, minutes, seconds = duration_match.groups()
            duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
            if "cann profiling data parsed" in normalized:
                result["cann_parse_seconds"] = duration
            elif "all profiling data parsed" in normalized:
                result["all_parse_seconds"] = duration
        if "compiling each" in normalized and "torch.compile" in normalized:
            result["compile_markers"] += 1
        for name, patterns in categories.items():
            if not any(pattern in normalized for pattern in patterns):
                continue
            result[name] += 1
            if len(result["examples"]) < 12:
                result["examples"].append(line[:1000])
            break
    return result


def build_analysis(
    run_directory: Path,
    *,
    metrics_path: Path | None = None,
    config: dict[str, Any] | None = None,
    profiler_environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    records = read_metrics(metrics_path or run_directory / "metrics.jsonl")
    metrics = summarize_metrics(records)
    profiler_directory = run_directory / "trainer_output" / "profiling" / "traces"
    if not profiler_directory.is_dir():
        profiler_directory = run_directory / "trainer_output" / "profiling"
    inventory = (
        file_inventory(profiler_directory)
        if profiler_directory.is_dir()
        else {"file_count": 0, "total_bytes": 0, "by_suffix": {}, "largest_files": []}
    )
    deliverables = inspect_profiler_deliverables(profiler_directory)
    return {
        "metrics": metrics,
        "profile_phases": summarize_profile_phases(
            records,
            config=config,
            profiler_environment=profiler_environment,
        ),
        "profiler_directory": str(profiler_directory),
        "profiler_inventory": inventory,
        "top_csv_tables": (
            extract_top_csv_entries(profiler_directory)
            if profiler_directory.is_dir()
            else []
        ),
        "kernel_shape_tables": (
            extract_kernel_shape_entries(profiler_directory)
            if profiler_directory.is_dir()
            else []
        ),
        "l2_cache_tables": (
            extract_l2_cache_entries(profiler_directory)
            if profiler_directory.is_dir()
            else []
        ),
        "deliverables": deliverables,
        "compiler_diagnostics": _compiler_diagnostics(
            run_directory / "runtime.log"
        ),
        "diagnostics": {
            name: _read_optional_json(run_directory / f"{name}.json")
            for name in ("advisor", "cluster", "compare")
        },
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
        "aicpu": "#dc2626",
    }
    labels = {
        "compute_or_runtime": "Compute / runtime",
        "communication": "Communication",
        "memory": "Memory movement",
        "aicpu": "AiCPU fallback",
    }
    segments = []
    legend = []
    for category in ("compute_or_runtime", "communication", "memory", "aicpu"):
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
    step_time_series = _find_metric(series, ("end_to_end",))
    profile_window_seconds = None
    if step_time_series is not None:
        profile_window_seconds = sum(
            value
            for step, value in step_time_series[1]
            if profile_start <= step <= profile_end
        )
        cards.append(
            '<div class="card"><span>Profiled step time</span>'
            f'<strong>{_format_number(profile_window_seconds)} s</strong></div>'
        )

    compiler = analysis.get("compiler_diagnostics", {})
    phase_analysis = analysis.get("profile_phases", {})
    phase_comparison = phase_analysis.get("comparison", {})
    active_overhead = phase_comparison.get("active_step_overhead_percent")
    if active_overhead is not None:
        cards.append(
            '<div class="card"><span>Active collection overhead</span>'
            f'<strong>{_format_number(active_overhead)}%</strong></div>'
        )
    parse_stall = phase_comparison.get("parse_stall_excess_seconds")
    if parse_stall is not None:
        cards.append(
            '<div class="card"><span>Sync parse stall</span>'
            f'<strong>{_format_number(parse_stall)} s</strong></div>'
        )
    if compiler.get("all_parse_seconds") is not None:
        cards.append(
            '<div class="card"><span>Profiler parser wall time</span>'
            f'<strong>{_format_number(compiler["all_parse_seconds"])} s</strong></div>'
        )
    compile_enabled = any(
        "--compile." in str(value) for value in config.get("extra_args", [])
    )
    if compile_enabled:
        cards.extend(
            [
                '<div class="card"><span>Graph breaks</span>'
                f'<strong>{compiler.get("graph_breaks", 0)}</strong></div>',
                '<div class="card"><span>Recompile messages</span>'
                f'<strong>{compiler.get("recompiles", 0)}</strong></div>',
                '<div class="card"><span>Backend failures</span>'
                f'<strong>{compiler.get("backend_failures", 0)}</strong></div>',
            ]
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
    compiler_examples = "\n".join(compiler.get("examples", []))
    compiler_section = ""
    if compile_enabled:
        compiler_section = f"""
<section><h2>torch.compile health</h2>
<p class="subtle">Counts are diagnostic log lines, not unique FX graphs. Use
<code>--compiler-diagnostics</code> to enable graph-break and recompile logs.</p>
<table><tbody>
<tr><th>Regional compile markers</th><td>{compiler.get('compile_markers', 0)}</td></tr>
<tr><th>Graph-break messages</th><td>{compiler.get('graph_breaks', 0)}</td></tr>
<tr><th>Recompile messages</th><td>{compiler.get('recompiles', 0)}</td></tr>
<tr><th>Backend-failure messages</th><td>{compiler.get('backend_failures', 0)}</td></tr>
</tbody></table>
<details><summary>Representative compiler log lines</summary><pre>{html.escape(compiler_examples or 'No compiler diagnostic line was recorded.')}</pre></details>
</section>"""
    runtime_section = f"""
<section><h2>NPU runtime diagnostics</h2>
<p class="subtle">Counts are warning lines, not operator invocation counts. Parser
time is extracted from the Ascend profiler completion messages.</p>
<table><tbody>
<tr><th>NPU-to-CPU fallback warnings</th><td>{compiler.get('npu_cpu_fallbacks', 0)}</td></tr>
<tr><th>AiCPU fallback warnings</th><td>{compiler.get('aicpu_fallbacks', 0)}</td></tr>
<tr><th>CANN parse time</th><td>{_format_number(compiler.get('cann_parse_seconds'))} s</td></tr>
<tr><th>Total profiler parse time</th><td>{_format_number(compiler.get('all_parse_seconds'))} s</td></tr>
</tbody></table>
<details><summary>Representative diagnostic lines</summary><pre>{html.escape(compiler_examples or 'No runtime diagnostic line was recorded.')}</pre></details>
</section>"""

    baseline_step = phase_comparison.get("baseline_median_step_seconds")
    phase_rows = ""
    for phase_name in phase_analysis.get("phase_order", []):
        phase = phase_analysis["phases"][phase_name]
        steps = phase["steps"]
        step_range = str(steps[0]) if len(steps) == 1 else f"{steps[0]}–{steps[-1]}"
        phase_summary = phase["summary"]
        step_time = phase_summary.get("time_metrics/end_to_end(s)", {}).get(
            "median"
        )
        throughput = phase_summary.get("throughput(tps)", {}).get("median")
        overhead = (
            (step_time / baseline_step - 1.0) * 100.0
            if step_time is not None and baseline_step not in {None, 0.0}
            else None
        )
        overhead_text = "-" if overhead is None else f"{_format_number(overhead)}%"
        phase_rows += (
            f"<tr><td>{html.escape(phase_name)}</td><td>{step_range}</td>"
            f"<td>{_format_number(step_time)}</td>"
            f"<td>{_format_number(throughput)}</td>"
            f"<td>{overhead_text}</td></tr>"
        )
    phase_section = (
        '<section><h2>Profiler phase overhead</h2>'
        f'<p class="subtle">Parse mode: {html.escape(str(phase_comparison.get("parse_mode", "unknown")))}. '
        'The baseline excludes startup, profiler warmup, active collection, and '
        'the synchronous parser boundary.</p>'
        '<table><thead><tr><th>Phase</th><th>Steps</th><th>Median step (s)</th>'
        '<th>Median throughput (tok/s)</th><th>Step overhead vs baseline</th>'
        f'</tr></thead><tbody>{phase_rows}</tbody></table></section>'
        if phase_rows
        else ""
    )
    training_arguments = " ".join(str(value) for value in config.get("extra_args", []))
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

    kernel_shape_sections = ""
    for table in analysis.get("kernel_shape_tables", [])[:2]:
        rows = "".join(
            "<tr>"
            f'<td><code>{html.escape(row["name"])}</code></td>'
            f'<td><code>{html.escape(row["input_shapes"])}</code></td>'
            f'<td><code>{html.escape(row["input_dtypes"])}</code></td>'
            f'<td>{row["calls"]}</td>'
            f'<td>{_format_number(row["total_us"] / 1000.0)}</td>'
            f'<td>{_format_number(row["mean_us"])}</td>'
            f'<td>{_format_number(row["max_us"])}</td>'
            "</tr>"
            for row in table["rows"][:20]
        )
        kernel_shape_sections += (
            '<section><h2>Top kernel shape signatures</h2>'
            f'<p class="subtle">Source: {html.escape(table["source"])}. '
            "Grouped by kernel name, input shapes, and input dtypes.</p>"
            '<table><thead><tr><th>Kernel</th><th>Input shapes</th>'
            '<th>Dtypes</th><th>Calls</th><th>Total (ms)</th>'
            '<th>Mean (us)</th><th>Max (us)</th></tr></thead>'
            f"<tbody>{rows}</tbody></table></section>"
        )

    l2_sections = ""
    for table in analysis.get("l2_cache_tables", [])[:2]:
        rows = "".join(
            "<tr>"
            f'<td><code>{html.escape(row["name"])}</code></td>'
            f'<td>{row["samples"]}</td>'
            f'<td>{_format_number(row["mean_hit_rate"])}</td>'
            f'<td>{_format_number(row["mean_victim_rate"])}</td>'
            "</tr>"
            for row in table["rows"][:20]
        )
        l2_sections += (
            '<section><h2>L2 cache samples</h2>'
            f'<p class="subtle">Source: {html.escape(table["source"])}. '
            "N/A samples are excluded from each mean.</p>"
            '<table><thead><tr><th>Operator</th><th>Samples</th>'
            '<th>Mean hit rate</th><th>Mean victim rate</th></tr></thead>'
            f"<tbody>{rows}</tbody></table></section>"
        )

    diagnostics = analysis.get("diagnostics", {})
    diagnostic_sections = ""
    for name in ("advisor", "cluster", "compare"):
        diagnostic = diagnostics.get(name)
        if diagnostic is None:
            continue
        diagnostic_sections += (
            f'<section><h2>msprof-analyze {name}</h2>'
            f'<p>Return code: <strong>{diagnostic.get("return_code")}</strong></p>'
            '<details><summary>Command and output</summary><pre>'
            f'{html.escape(" ".join(diagnostic.get("command", [])))}\n\n'
            f'{html.escape(diagnostic.get("stdout", ""))}\n'
            f'{html.escape(diagnostic.get("stderr", ""))}</pre></details></section>'
        )

    deliverables = analysis.get(
        "deliverables",
        {"ascend_profiles": [], "gpu_traces": [], "insight_ready": False},
    )

    def badge(ready: bool, ready_text: str, missing_text: str) -> str:
        css_class = "ready" if ready else "pending"
        text = ready_text if ready else missing_text
        return f'<span class="status {css_class}">{html.escape(text)}</span>'

    ascend_rows = ""
    for profile in deliverables["ascend_profiles"]:
        files = [
            name for name, present in profile["text_files"].items() if present
        ] + profile["database_files"] + profile["profiler_metadata_files"]
        file_details = ", ".join(files) or "No parsed files"
        ascend_rows += (
            f'<tr><td><code>{html.escape(profile["relative_root"])}</code>'
            f'<br><span class="subtle">{html.escape(file_details)}</span></td>'
            f'<td>{badge(profile["raw_ready"], "captured", "metadata missing")}</td>'
            f'<td>{badge(profile["parsed_text"], "text ready", "text pending")}</td>'
            f'<td>{badge(profile["parsed_database"], "DB ready", "DB pending")}</td>'
            f'<td>{badge(profile["insight_ready"], "ready", "parse required")}</td></tr>'
        )
    if not ascend_rows:
        ascend_rows = '<tr><td colspan="5">No Ascend profile root discovered</td></tr>'

    gpu_trace_items = "".join(
        f"<li><code>{html.escape(path)}</code></li>"
        for path in deliverables["gpu_traces"][:20]
    ) or "<li>No GPU trace discovered</li>"

    has_capture = inventory["file_count"] > 0
    has_parsed_output = bool(deliverables["gpu_traces"]) or deliverables[
        "insight_ready"
    ]
    pipeline = (
        badge(has_capture, "1 Capture complete", "1 Capture missing")
        + badge(has_parsed_output, "2 Parse complete", "2 Parse pending")
        + badge(
            (diagnostics.get("advisor") or {}).get("return_code") == 0,
            "3 Advisor complete",
            "3 Advisor optional",
        )
        + badge(
            (diagnostics.get("cluster") or {}).get("return_code") == 0,
            "4 Cluster complete",
            "4 Cluster optional",
        )
        + badge(
            (diagnostics.get("compare") or {}).get("return_code") == 0,
            "5 Compare complete",
            "5 Compare optional",
        )
        + badge(
            deliverables["insight_ready"],
            "6 Insight ready",
            "6 Insight pending",
        )
    )

    preflight = manifest.get("preflight", {})
    preflight_rows = ""
    byte_fields = {
        "storage_free_bytes_before_capture",
        "storage_free_bytes_after_capture",
        "storage_total_bytes",
        "run_output_bytes",
        "recommended_free_bytes_for_same_capture",
    }
    for key, value in preflight.items():
        if key == "warnings":
            continue
        rendered = _human_bytes(value) if key in byte_fields else str(value)
        preflight_rows += (
            f"<tr><th>{html.escape(key)}</th><td>{html.escape(rendered)}</td></tr>"
        )
    warning_items = "".join(
        f"<li>{html.escape(warning)}</li>"
        for warning in preflight.get("warnings", [])
    ) or "<li>No preflight warning was recorded.</li>"
    profile_duration_warning = ""
    if profile_window_seconds is not None and profile_window_seconds > 300:
        profile_duration_warning = (
            '<div class="callout warning">The active profiling window exceeded '
            "the official five-minute recommendation. Reduce active steps.</div>"
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
.pipeline{{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0}} .status{{display:inline-flex;align-items:center;padding:5px 9px;border-radius:999px;font-size:12px;font-weight:650}}
.status.ready{{background:#dcfce7;color:#166534}} .status.pending{{background:#f1f5f9;color:#64748b}} pre{{white-space:pre-wrap;word-break:break-word}}
.callout.warning{{background:#fff7ed;color:#9a3412;border:1px solid #fed7aa}}
@media(max-width:700px){{main{{padding:14px}}.bar-row{{grid-template-columns:1fr}}.bar-value{{text-align:left}}}}
</style></head><body><main>
<h1>GLM-5.2 performance probe</h1>
<p class="subtle">{html.escape(manifest['run_name'])} · {html.escape(manifest['device'])} · {html.escape(manifest['topology'])} · {html.escape(manifest['preset'])}</p>
<div class="pipeline">{pipeline}</div>
<div class="cards">{''.join(cards) or '<div class="card"><span>Training metrics</span><strong>Unavailable</strong></div>'}</div>
{profile_duration_warning}
{phase_section}
{''.join(charts)}
{runtime_section}
{compiler_section}
{composition_section}
{top_sections}
{kernel_shape_sections}
{l2_sections}
{diagnostic_sections}
<section><h2>Official Ascend deliverables</h2><p class="subtle">Each row is one <code>*_ascend_pt</code> profile root. Text/DB readiness is checked against the official PyTorch Profiler output layout.</p>
<table><thead><tr><th>Profile root and parsed files</th><th>Raw</th><th>Text</th><th>Database</th><th>MindStudio Insight</th></tr></thead><tbody>{ascend_rows}</tbody></table></section>
<section><h2>Open in MindStudio Insight</h2><div class="callout">Copy the complete <code>*_ascend_pt</code> directory to a local disk, then import that profile root in MindStudio Insight. Do not select only <code>ASCEND_PROFILER_OUTPUT</code>. For cluster analysis, import the generated <code>cluster_analysis_output</code> directory. This HTML is an experiment index and summary; it does not replace the official Timeline, Communication, Operator, or Memory views.</div>
<details><summary>GPU trace inputs available for comparison</summary><ul>{gpu_trace_items}</ul></details></section>
<section><h2>Profiler data inventory</h2><p class="subtle">{inventory['file_count']} files, {_human_bytes(inventory['total_bytes'])}; raw directory: {html.escape(analysis['profiler_directory'])}</p>
<table><thead><tr><th>Type</th><th>Files</th><th>Size</th></tr></thead><tbody>{suffix_rows}</tbody></table></section>
<section><h2>Profiling preflight</h2><ul>{warning_items}</ul><table><tbody>{preflight_rows or '<tr><td>No preflight metadata</td></tr>'}</tbody></table></section>
<section><h2>Effective Ascend profiler controls</h2><table><tbody>{profiler_rows or '<tr><td>No NPU-specific controls</td></tr>'}</tbody></table></section>
<section><h2>Experiment configuration</h2><table><tbody>{configuration_rows}</tbody></table></section>
<section><h2>Additional training arguments</h2><pre>{html.escape(training_arguments or 'None')}</pre></section>
</main></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
