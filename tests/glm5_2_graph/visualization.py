"""Capture and summarize official ``torch.compile`` diagnostics.

Each rank writes Dynamo ``TORCH_TRACE`` events and Inductor debug output inside
its run directory. ``tlparse`` turns the trace into an interactive report with
graph breaks, guards, recompiles, generated FX graphs, and source frames. These
are compiler diagnostics, distinct from NPU execution timelines collected by
the performance profiler.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from html import escape
from typing import Any


GRAPH_DIAGNOSTICS_ENV = "GLM5_GRAPH_CAPTURE_DIAGNOSTICS"
RUN_DIRECTORY_ENV = "GLM5_EXPERIMENT_RUN_DIRECTORY"


def configure_graph_diagnostics() -> Path | None:
    """Route compiler traces into the current experiment run directory."""

    if os.environ.get(GRAPH_DIAGNOSTICS_ENV, "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None
    run_value = os.environ.get(RUN_DIRECTORY_ENV)
    if not run_value:
        raise RuntimeError(
            f"{RUN_DIRECTORY_ENV} is required when {GRAPH_DIAGNOSTICS_ENV} is enabled"
        )
    root = Path(run_value).resolve() / "graph_visualization"
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
    rank_root = root / f"rank_{rank}"
    trace_root = rank_root / "torch_trace"
    compile_root = rank_root / "inductor"
    trace_root.mkdir(parents=True, exist_ok=True)
    compile_root.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_TRACE"] = str(trace_root)
    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    os.environ["TORCH_COMPILE_DEBUG_DIR"] = str(compile_root)
    return root


def _relative_files(paths: list[Path], root: Path) -> list[str]:
    return [path.relative_to(root).as_posix() for path in sorted(paths)]


def _tlparse_output(root: Path, trace_path: Path) -> Path:
    digest = hashlib.sha256(str(trace_path).encode("utf-8")).hexdigest()[:8]
    return root / "tlparse" / f"{trace_path.stem}-{digest}"


def generate_tlparse_reports(run_directory: Path) -> list[dict[str, Any]]:
    """Generate one tlparse HTML report per structured trace when installed."""

    root = run_directory / "graph_visualization"
    traces = sorted(root.rglob("torch_trace/*.log")) if root.is_dir() else []
    executable = shutil.which("tlparse")
    results: list[dict[str, Any]] = []
    for trace_path in traces:
        output = _tlparse_output(root, trace_path)
        index = output / "index.html"
        command = [
            executable or "tlparse",
            str(trace_path),
            "-o",
            str(output),
        ]
        if (
            index.is_file()
            and index.stat().st_mtime_ns >= trace_path.stat().st_mtime_ns
        ):
            results.append(
                {
                    "trace": str(trace_path),
                    "output": str(output),
                    "index": str(index),
                    "status": "ready",
                    "command": command,
                }
            )
            continue
        if executable is None:
            results.append(
                {
                    "trace": str(trace_path),
                    "output": str(output),
                    "index": None,
                    "status": "tool_missing",
                    "command": command,
                }
            )
            continue
        if output.exists():
            if output.is_symlink():
                raise ValueError(
                    f"refusing to replace symlinked tlparse output: {output}"
                )
            shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        (output / "tlparse.log").write_text(completed.stdout, encoding="utf-8")
        results.append(
            {
                "trace": str(trace_path),
                "output": str(output),
                "index": str(index) if index.is_file() else None,
                "status": (
                    "ready"
                    if completed.returncode == 0 and index.is_file()
                    else "failed"
                ),
                "return_code": completed.returncode,
                "command": command,
            }
        )
    if results:
        manifest = root / "tlparse_manifest.json"
        manifest.write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return results


def inspect_graph_visualizations(run_directory: Path) -> dict[str, Any]:
    """Inventory structured compile traces, tlparse HTML, FX, IR, and code."""

    root = run_directory / "graph_visualization"
    if not root.is_dir():
        return {
            "ready": False,
            "root": str(root),
            "structured_traces": [],
            "tlparse_reports": [],
            "fx_graphs": [],
            "inductor_ir": [],
            "generated_code": [],
            "tlparse_available": shutil.which("tlparse") is not None,
        }
    traces = sorted(root.rglob("torch_trace/*.log"))
    tlparse_indexes = (
        sorted((root / "tlparse").rglob("index.html"))
        if (root / "tlparse").is_dir()
        else []
    )
    fx_graphs = sorted(root.rglob("fx_graph*.py"))
    inductor_ir = sorted(
        path
        for pattern in ("ir_pre_fusion.txt", "ir_post_fusion.txt")
        for path in root.rglob(pattern)
    )
    generated_code = sorted(root.rglob("output_code.py"))
    return {
        "ready": bool(traces or tlparse_indexes or fx_graphs),
        "root": str(root),
        "structured_traces": _relative_files(traces, root),
        "tlparse_reports": _relative_files(tlparse_indexes, root),
        "fx_graphs": _relative_files(fx_graphs, root),
        "inductor_ir": _relative_files(inductor_ir, root),
        "generated_code": _relative_files(generated_code, root),
        "tlparse_available": shutil.which("tlparse") is not None,
    }


def _trace_events(trace: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        trace.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            value["_line"] = line_number
            events.append(value)
    return events


def _event_text(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False, sort_keys=True)


def analyze_graph_compilation(run_directory: Path) -> dict[str, Any]:
    """Summarize official structured traces without replacing tlparse."""

    inventory = inspect_graph_visualizations(run_directory)
    root = Path(inventory["root"])
    ranks: dict[str, dict[str, Any]] = {}
    totals = {"events": 0, "graph_breaks": 0, "recompiles": 0}
    for relative in inventory["structured_traces"]:
        trace = root / relative
        rank = next((part for part in trace.parts if part.startswith("rank_")), "rank_unknown")
        record = ranks.setdefault(
            rank,
            {"traces": [], "events": 0, "graph_breaks": [], "recompiles": []},
        )
        events = _trace_events(trace)
        record["traces"].append(relative)
        record["events"] += len(events)
        totals["events"] += len(events)
        for event in events:
            text = _event_text(event)
            lowered = text.lower()
            item = {
                "trace": relative,
                "line": event["_line"],
                "compile_id": event.get("compile_id"),
                "frame_id": event.get("frame_id"),
                "payload": event,
            }
            if "graph_break" in lowered or "graph break" in lowered:
                record["graph_breaks"].append(item)
                totals["graph_breaks"] += 1
            if "recompile" in lowered or "guard_failure" in lowered:
                record["recompiles"].append(item)
                totals["recompiles"] += 1
    return {
        "schema_version": 1,
        "run_directory": str(run_directory.resolve()),
        "inventory": inventory,
        "totals": totals,
        "ranks": ranks,
        "mapping_semantics": (
            "Module/source to FX to Inductor IR to generated kernel is generally "
            "many-to-many because decomposition, fusion, elimination and fallback apply."
        ),
    }


def generate_graph_compilation_report(run_directory: Path) -> dict[str, Any]:
    """Write a portable JSON/HTML index beside official compiler artifacts."""

    generate_tlparse_reports(run_directory)
    report = analyze_graph_compilation(run_directory)
    root = Path(report["inventory"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "compilation_report.json"
    html_path = root / "compilation_report.html"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = []
    details = []
    for rank, item in sorted(report["ranks"].items()):
        rows.append(
            "<tr>"
            f"<td>{escape(rank)}</td><td>{item['events']}</td>"
            f"<td>{len(item['graph_breaks'])}</td>"
            f"<td>{len(item['recompiles'])}</td>"
            "</tr>"
        )
        for kind in ("graph_breaks", "recompiles"):
            for event in item[kind]:
                details.append(
                    f"<details><summary>{escape(rank)} · {escape(kind)} · "
                    f"{escape(event['trace'])}:{event['line']}</summary><pre>"
                    f"{escape(json.dumps(event['payload'], ensure_ascii=False, indent=2))}"
                    "</pre></details>"
                )
    inventory = report["inventory"]
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>torch.compile report</title>"
        "<style>body{font:14px system-ui;margin:2rem;max-width:1100px}"
        "table{border-collapse:collapse}td,th{padding:.45rem .8rem;border-bottom:1px solid #bbb}"
        ".flow{display:flex;flex-wrap:wrap;gap:.5rem;margin:1rem 0}.flow span{padding:.5rem;"
        "border:1px solid #aaa;border-radius:.3rem}</style>"
        "<h1>torch.compile compilation report</h1>"
        "<div class='flow'><span>TorchTitan module</span><span>→ Dynamo/FX</span>"
        "<span>→ AOTAutograd</span><span>→ Inductor IR</span>"
        "<span>→ TorchNPU lowering</span><span>→ Triton/DVM kernel</span>"
        "<span>→ NPU runtime</span></div>"
        f"<p>FX graphs: {len(inventory['fx_graphs'])}; IR files: "
        f"{len(inventory['inductor_ir'])}; generated code: "
        f"{len(inventory['generated_code'])}; tlparse reports: "
        f"{len(inventory['tlparse_reports'])}.</p>"
        "<table><thead><tr><th>Rank</th><th>Trace events</th>"
        "<th>Graph-break records</th><th>Recompile records</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table><h2>Break and recompile evidence</h2>"
        + ("".join(details) if details else "<p>No matching structured events.</p>")
        + "<p>Open the tlparse index for the complete source stack, guards, FX "
        "graphs and generated code. Mapping is many-to-many.</p>",
        encoding="utf-8",
    )
    report["json"] = str(json_path)
    report["html"] = str(html_path)
    return report


__all__ = [
    "GRAPH_DIAGNOSTICS_ENV",
    "RUN_DIRECTORY_ENV",
    "configure_graph_diagnostics",
    "generate_tlparse_reports",
    "analyze_graph_compilation",
    "generate_graph_compilation_report",
    "inspect_graph_visualizations",
]
