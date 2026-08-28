"""Capture and summarize official torch.compile visualization artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
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


__all__ = [
    "GRAPH_DIAGNOSTICS_ENV",
    "RUN_DIRECTORY_ENV",
    "configure_graph_diagnostics",
    "generate_tlparse_reports",
    "inspect_graph_visualizations",
]
