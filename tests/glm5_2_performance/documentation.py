"""Human-readable documentation for performance exploration evidence."""

from __future__ import annotations

import json
from pathlib import Path
import shlex
from typing import Any

from .config import performance_topologies


def card_scope(topology: str) -> str:
    """Return the stable directory name for a topology's world size."""

    return f"{performance_topologies()[topology].world_size}-card"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_history(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _number(value: object, digits: int = 2) -> str:
    return "-" if value is None else f"{float(value):,.{digits}f}"


def _run_identity(directory: Path) -> dict[str, Any]:
    manifest = _read_json(directory / "manifest.json")
    history = _read_history(directory / "command_history.jsonl")
    first_history = history[0] if history else {}
    topology = str(manifest.get("topology") or first_history.get("topology") or "unknown")
    config = manifest.get("config")
    if not isinstance(config, dict):
        config = {}
    failure = _read_json(directory / "failure.json")
    analysis = _read_json(directory / "analysis.json")
    artifacts = _read_json(directory / "artifacts.json")
    if failure:
        status = "failed"
    elif manifest and analysis:
        status = "completed"
    elif manifest:
        status = "captured"
    else:
        status = "command-only"
    profiler_enabled = config.get("profiler_enabled")
    if profiler_enabled is None:
        profiler_enabled = not any(
            "--profiler-off" in str(record.get("shell_command", ""))
            for record in history
        )
    return {
        "run_name": str(manifest.get("run_name") or directory.name),
        "topology": topology,
        "config": config,
        "manifest": manifest,
        "history": history,
        "failure": failure,
        "analysis": analysis,
        "artifacts": artifacts,
        "status": status,
        "mode": "profiler-active" if profiler_enabled is not False else "profiler-off",
        "timestamp": str(first_history.get("timestamp") or ""),
    }


def _metrics(identity: dict[str, Any]) -> dict[str, Any]:
    analysis = identity["analysis"]
    comparison = analysis.get("profile_phases", {}).get("comparison", {})
    steady = (
        analysis.get("profile_phases", {})
        .get("phases", {})
        .get("steady", {})
        .get("summary", {})
    )
    step = steady.get("time_metrics/end_to_end(s)", {})
    memory = steady.get("memory/max_active(GiB)", {})
    return {
        "step_ms": (
            float(comparison["baseline_median_step_seconds"]) * 1000
            if comparison.get("baseline_median_step_seconds") is not None
            else None
        ),
        "step_p90_ms": (
            float(step["p90"]) * 1000 if step.get("p90") is not None else None
        ),
        "device_tps": comparison.get("baseline_median_throughput_tps"),
        "job_tps": comparison.get("baseline_job_throughput_tps"),
        "memory_gib": memory.get("max"),
    }


def _command(value: object) -> str | None:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return shlex.join(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def write_run_readme(directory: Path) -> Path:
    """Render one self-contained process/result document beside run JSON."""

    identity = _run_identity(directory)
    manifest = identity["manifest"]
    config = identity["config"]
    analysis = identity["analysis"]
    artifacts = identity["artifacts"]
    metrics = _metrics(identity)
    topology = identity["topology"]
    world_size = (
        performance_topologies()[topology].world_size
        if topology in performance_topologies()
        else None
    )
    lines = [
        f"# {identity['run_name']}",
        "",
        "## experiment",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| status | {identity['status']} |",
        f"| topology | {topology} |",
        f"| world size | {world_size or '-'} |",
        f"| mode | {identity['mode']} |",
        f"| device | {manifest.get('device') or (identity['history'][0].get('device') if identity['history'] else '-')} |",
        f"| preset | {manifest.get('preset') or config.get('preset') or '-'} |",
        f"| steps | {config.get('steps', '-')} |",
        f"| local / global batch | {config.get('local_batch_size', '-')} / {config.get('global_batch_size', '-')} |",
        f"| sequence length | {config.get('sequence_length', '-')} |",
        f"| parameter / reduction dtype | {config.get('mixed_precision_param', '-')} / {config.get('mixed_precision_reduce', '-')} |",
        f"| replicate | {config.get('replicate', '-')} |",
        f"| visible devices | {', '.join(manifest.get('visible_devices') or []) or '-'} |",
        f"| started | {identity['timestamp'] or '-'} |",
        "",
        "## process",
        "",
    ]
    process_index = 1
    for record in identity["history"]:
        command = _command(record.get("shell_command") or record.get("argv"))
        if not command:
            continue
        lines.extend(
            [
                f"{process_index}. Run the `{record.get('action', 'driver')}` driver:",
                "",
                "   ```bash",
                f"   {command}",
                "   ```",
                "",
            ]
        )
        process_index += 1
    training_command = _command(manifest.get("command"))
    if training_command:
        lines.extend(
            [
                f"{process_index}. The workflow generated and executed this training command:",
                "",
                "   ```bash",
                f"   {training_command}",
                "   ```",
                "",
            ]
        )
        process_index += 1
    tool_rows = []
    for path in sorted((directory / "tool_commands").glob("*.json")):
        record = _read_json(path)
        command = _command(record.get("command"))
        tool_rows.append((path, record.get("return_code"), command))
    if tool_rows:
        lines.extend(
            [
                f"{process_index}. Parse and analyze the capture with the official tools:",
                "",
            ]
        )
        for path, return_code, command in tool_rows:
            lines.extend(
                [
                    f"   - `{path.stem}` (return code `{return_code}`)",
                    "",
                    "     ```bash",
                    f"     {command or '-'}",
                    "     ```",
                    "",
                ]
            )
        process_index += 1
    lines.extend(
        [
            f"{process_index}. Produce compact evidence (`manifest.json`, `metrics.jsonl`, "
            "`analysis.json`, tool status files) and the HTML report when analysis completed.",
            "",
            "## results",
            "",
            "| metric | value |",
            "| --- | ---: |",
            f"| median step | {_number(metrics['step_ms'])} ms |",
            f"| p90 step | {_number(metrics['step_p90_ms'])} ms |",
            f"| throughput / device | {_number(metrics['device_tps'])} tok/s |",
            f"| throughput / job | {_number(metrics['job_tps'])} tok/s |",
            f"| peak active hbm | {_number(metrics['memory_gib'], 3)} GiB |",
            "",
        ]
    )
    communication = analysis.get("communication_summary", {}).get("rows", [])
    if communication:
        lines.extend(
            [
                "### collectives",
                "",
                "| rank | kind | calls/step | payload mb/step | transit ms/step | effective gb/s |",
                "| ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in communication:
            lines.append(
                f"| {row.get('rank', '-')} | {row.get('kind', '-')} | "
                f"{_number(row.get('calls_per_step'))} | "
                f"{_number(row.get('payload_mb_per_step'))} | "
                f"{_number(row.get('transit_ms_per_step'))} | "
                f"{_number(row.get('effective_bandwidth_gbps'))} |"
            )
        lines.append("")
    if identity["failure"]:
        lines.extend(
            [
                "### failure",
                "",
                f"`{identity['failure'].get('error', 'unknown failure')}`",
                "",
                "See [failure.json](failure.json) and "
                "[failed_runtime.log](failed_runtime.log) for the complete evidence.",
                "",
            ]
        )
    lines.extend(
        [
            "## outputs",
            "",
            "| output | purpose |",
            "| --- | --- |",
        ]
    )
    local_outputs = (
        ("command_history.jsonl", "exact driver history and environment"),
        ("manifest.json", "resolved configuration and generated training command"),
        ("metrics.jsonl", "raw per-step metrics"),
        ("analysis.json", "structured derived analysis"),
        ("artifacts.json", "raw run, compact artifact, report, and MindStudio paths"),
        ("failure.json", "failure command and exception"),
        ("failed_runtime.log", "failed training stdout/stderr"),
    )
    for name, purpose in local_outputs:
        if (directory / name).is_file():
            lines.append(f"| [{name}]({name}) | {purpose} |")
    if tool_rows:
        lines.append("| [tool_commands](tool_commands/) | official analysis commands and return codes |")
    lines.append("")
    for label, key in (
        ("raw run", "raw_run_directory"),
        ("compact artifact", "artifact_directory"),
        ("html report", "report_path"),
        ("MindStudio input", "mindstudio_input"),
        ("TensorBoard logdir", "tensorboard_root"),
        ("flame graph stacks/SVG root", "flamegraph_root"),
        ("torch.compile visualization", "graph_visualization_root"),
    ):
        if artifacts.get(key):
            lines.append(f"- {label}: `{artifacts[key]}`")
    lines.extend(
        [
            "",
            "Profiler-off results are throughput evidence. Profiler-active results are "
            "used for attribution only and must not be compared as authoritative throughput.",
            "",
        ]
    )
    output = directory / "readme.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def run_identity(directory: Path) -> dict[str, Any]:
    """Expose normalized run metadata to exploration index generators."""

    value = _run_identity(directory)
    value["metrics"] = _metrics(value)
    return value
