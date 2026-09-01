"""Discover and render portable performance visualization artifacts.

Official tools remain the source of truth: TensorBoard for scalar curves,
MindStudio Insight/Perfetto-compatible traces for interactive timelines, and
torch_npu stack/memory exports for folded stacks and allocation history. This
module discovers those outputs, renders portable SVG/HTML where possible, and
records commands for richer GUI inspection without copying raw multi-GB traces
into reports.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


def _relative_paths(paths: list[Path], root: Path) -> list[str]:
    return [path.relative_to(root).as_posix() for path in sorted(paths)]


def _matching_relative_files(
    root: Path, patterns: tuple[str, ...]
) -> list[str]:
    if not root.is_dir():
        return []
    files = {
        path.relative_to(root).as_posix()
        for pattern in patterns
        for path in root.rglob(pattern)
        if path.is_file()
    }
    return sorted(files)


def mindstudio_insight_handoff(
    profile_root: Path,
    *,
    analysis_root: Path | None = None,
    portable_base: Path | None = None,
) -> dict[str, Any]:
    """Describe portable import roots and issue-driven Insight views."""

    profile = profile_root.resolve()
    analysis = analysis_root.resolve() if analysis_root is not None else None
    import_targets = [str(profile)]
    if analysis is not None and analysis.is_dir():
        import_targets.extend(
            str(path)
            for path in sorted(
                path
                for path in analysis.rglob("cluster_analysis_output")
                if path.is_dir()
            )
        )
    portable = portable_base.resolve() if portable_base is not None else None
    portable_import_targets = []
    for index, target in enumerate(import_targets):
        target_path = Path(target)
        relative_path = None
        if portable is not None:
            try:
                relative_path = target_path.relative_to(portable).as_posix()
            except ValueError:
                relative_path = None
        portable_import_targets.append(
            {
                "kind": "profile" if index == 0 else "cluster",
                "server_path": str(target_path),
                "relative_path": relative_path,
            }
        )
    profile_inventory = {
        "databases": _matching_relative_files(profile, ("*.db",)),
        "timelines": _matching_relative_files(
            profile, ("trace_view.json", "msprof_*.json", "*.json.gz")
        ),
        "tables": _matching_relative_files(profile, ("*.csv", "*.xlsx")),
        "reports": _matching_relative_files(profile, ("*.html",)),
    }
    all_evidence = [
        *profile_inventory["databases"],
        *profile_inventory["timelines"],
        *profile_inventory["tables"],
        *profile_inventory["reports"],
    ]

    def view_evidence(*tokens: str, database_fallback: bool = True) -> dict[str, Any]:
        matches = [
            path
            for path in all_evidence
            if any(token in path.lower() for token in tokens)
        ]
        if matches:
            return {"evidence_status": "ready", "evidence": matches[:20]}
        if database_fallback and profile_inventory["databases"]:
            return {
                "evidence_status": "inspect_database",
                "evidence": profile_inventory["databases"][:20],
            }
        return {"evidence_status": "missing", "evidence": []}

    summary_evidence = (
        {"evidence_status": "ready", "evidence": all_evidence[:20]}
        if all_evidence
        else {"evidence_status": "missing", "evidence": []}
    )
    return {
        "schema": "torchtitan.glm5_2.mindstudio_performance_handoff",
        "schema_version": 2,
        "profile_root": str(profile),
        "analysis_root": str(analysis) if analysis is not None else None,
        "import_targets": import_targets,
        "portable_base": str(portable) if portable is not None else None,
        "portable_import_targets": portable_import_targets,
        "profile_inventory": profile_inventory,
        "views": [
            {
                "name": "Summary",
                "use_when": "start every system-level investigation",
                "question": (
                    "Is time dominated by computing, communication, or free "
                    "device intervals, and are ranks or stages imbalanced?"
                ),
                **summary_evidence,
            },
            {
                "name": "Communication",
                "use_when": "multi-rank communication is exposed or imbalanced",
                "question": (
                    "Which rank, domain, collective, or link has abnormal wait, "
                    "payload, bandwidth, or non-overlapped time?"
                ),
                **view_evidence("communication", "hccs", "pcie", "roce"),
            },
            {
                "name": "Timeline",
                "use_when": "the summary points to a concrete time interval",
                "question": (
                    "Which Python, CANN, Runtime, stream, kernel, or collective "
                    "caused it, and did Host launch readiness delay it?"
                ),
                **view_evidence("trace_view", "msprof_"),
            },
            {
                "name": "Operator",
                "use_when": "one operator family dominates computing time",
                "question": (
                    "Are count, duration, shape, utilization, or fallback "
                    "behavior abnormal?"
                ),
                **view_evidence("operator", "op_summary", "kernel"),
            },
            {
                "name": "Memory",
                "use_when": "HBM pressure, allocation churn, or retention is suspected",
                "question": (
                    "Which operator or lifetime interval owns active, reserved, "
                    "or repeatedly allocated device memory?"
                ),
                **view_evidence("memory", "mem_"),
            },
            {
                "name": "RL",
                "use_when": "the input contains supported RL MSTX ranges",
                "question": (
                    "Where are bubbles or imbalance across rollout, inference, "
                    "reward, and training stages?"
                ),
                **view_evidence(
                    "rollout", "reward", "verl", "reinforce", database_fallback=False
                ),
            },
        ],
        "notes": [
            "Import a whole profile or cluster output root, not a rank-0 file.",
            "RL view requires supported RL framework data and MSTX ranges.",
            "A tool stage completing is not a performance PASS verdict.",
        ],
    }


def inspect_tensorboard(run_directory: Path) -> dict[str, Any]:
    """Describe scalar TensorBoard event files emitted by TorchTitan."""

    event_files = sorted(
        path
        for path in run_directory.rglob("events.out.tfevents.*")
        if path.is_file()
    )
    roots = sorted({path.parent for path in event_files})
    preferred_root = (
        run_directory / "trainer_output" / "tensorboard"
        if (run_directory / "trainer_output" / "tensorboard").is_dir()
        else (roots[0] if len(roots) == 1 else run_directory)
    )
    return {
        "ready": bool(event_files),
        "root": str(preferred_root),
        "relative_root": (
            preferred_root.relative_to(run_directory).as_posix()
            if preferred_root.is_relative_to(run_directory)
            else str(preferred_root)
        ),
        "event_files": _relative_paths(event_files, run_directory),
        "event_count": len(event_files),
        "total_bytes": sum(path.stat().st_size for path in event_files),
        "command": [
            "tensorboard",
            "--logdir",
            str(preferred_root),
            "--port",
            "6006",
            "--bind_all",
        ],
    }


def inspect_stack_visualizations(
    profiler_directory: Path,
) -> dict[str, Any]:
    """Describe official HTML and portable folded-stack flame graphs."""

    if not profiler_directory.is_dir():
        return {
            "stack_files": [],
            "flamegraphs": [],
            "flamegraph_tool": None,
            "mindstudio_flamegraphs": [],
            "mindstudio_flamegraph_logs": [],
            "mindstudio_flamegraph_tool": None,
        }
    stack_files = sorted(
        path
        for path in profiler_directory.rglob("*")
        if path.is_file()
        and "stack" in path.name.lower()
        and path.suffix.lower() in {".log", ".stacks"}
    )
    flamegraphs = sorted(
        path
        for path in profiler_directory.rglob("*.svg")
        if path.is_file() and "flame" in path.name.lower()
    )
    mindstudio_flamegraphs = sorted(
        path
        for path in profiler_directory.rglob("flamegraph.html")
        if path.is_file() and "mindstudio_flamegraphs" in path.parts
    )
    mindstudio_logs = sorted(
        path
        for path in profiler_directory.rglob("mindstudio_flamegraph.log")
        if path.is_file()
    )
    return {
        "stack_files": _relative_paths(stack_files, profiler_directory),
        "flamegraphs": _relative_paths(flamegraphs, profiler_directory),
        "flamegraph_tool": find_flamegraph_script(),
        "mindstudio_flamegraphs": _relative_paths(
            mindstudio_flamegraphs, profiler_directory
        ),
        "mindstudio_flamegraph_logs": _relative_paths(
            mindstudio_logs, profiler_directory
        ),
        "mindstudio_flamegraph_tool": find_mindstudio_flamegraph_script(),
    }


def inspect_memory_visualizations(
    profiler_directory: Path,
) -> dict[str, Any]:
    """Describe official Ascend memory timeline exports."""

    if not profiler_directory.is_dir():
        return {"ready": False, "html": [], "series": [], "raw": []}
    candidates = sorted(
        path
        for path in profiler_directory.rglob("*memory_timeline*")
        if path.is_file()
    )
    html_files = [path for path in candidates if path.suffix.lower() == ".html"]
    raw_files = [path for path in candidates if path.name.endswith("raw.json.gz")]
    series_files = [
        path
        for path in candidates
        if path.name.endswith((".json", ".json.gz")) and path not in raw_files
    ]
    return {
        "ready": bool(html_files),
        "html": _relative_paths(html_files, profiler_directory),
        "series": _relative_paths(series_files, profiler_directory),
        "raw": _relative_paths(raw_files, profiler_directory),
    }


def inspect_analysis_outputs(run_directory: Path) -> dict[str, Any]:
    """Inventory msprof-analyze status files and generated deliverables."""

    patterns = (
        "advisor.json",
        "cluster.json",
        "compare.json",
        "cluster_time_summary.json",
        "free_analysis.json",
        "communication_bottleneck_rank_*.json",
    )
    status_paths = sorted(
        {path for pattern in patterns for path in run_directory.glob(pattern)}
    )
    entries = []
    for status_path in status_paths:
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        command = [str(value) for value in status.get("command", [])]
        output_root = None
        if "-o" in command:
            index = command.index("-o") + 1
            if index < len(command):
                output_root = Path(command[index])
                if not output_root.is_absolute():
                    output_root = run_directory / output_root
        files = (
            sorted(path for path in output_root.rglob("*") if path.is_file())
            if output_root is not None and output_root.is_dir()
            else []
        )
        entries.append(
            {
                "name": status_path.stem,
                "status_file": status_path.relative_to(run_directory).as_posix(),
                "return_code": status.get("return_code"),
                "command": command,
                "output_root": str(output_root) if output_root is not None else None,
                "files": _relative_paths(files, run_directory),
                "file_count": len(files),
                "total_bytes": sum(path.stat().st_size for path in files),
            }
        )
    return {"ready": bool(entries), "entries": entries}


def find_flamegraph_script() -> str | None:
    """Resolve Brendan Gregg's flamegraph.pl without downloading tools."""

    configured = os.environ.get("TORCHTITAN_FLAMEGRAPH_PL")
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"TORCHTITAN_FLAMEGRAPH_PL does not exist: {path}"
            )
        return str(path)
    executable = shutil.which("flamegraph.pl")
    return str(Path(executable).resolve()) if executable else None


def find_mindstudio_flamegraph_script() -> str | None:
    """Resolve the official MindStudio Insight flame graph generator."""

    configured = os.environ.get("TORCHTITAN_MSINSIGHT_FLAMEGRAPH")
    if not configured:
        return None
    path = Path(configured).expanduser().resolve()
    if path.is_dir():
        path = path / "flamegraph.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"TORCHTITAN_MSINSIGHT_FLAMEGRAPH does not exist: {path}"
        )
    return str(path)


def render_mindstudio_flamegraphs(profiler_directory: Path) -> list[Path]:
    """Render official interactive HTML from each Ascend profiler database."""

    script = find_mindstudio_flamegraph_script()
    if script is None or not profiler_directory.is_dir():
        return []
    databases = sorted(
        path
        for path in profiler_directory.rglob("ascend_pytorch_profiler_*.db")
        if path.is_file()
    )
    generated: list[Path] = []
    output_parent = profiler_directory / "mindstudio_flamegraphs"
    for database in databases:
        relative_database = database.relative_to(profiler_directory)
        digest = hashlib.sha256(
            relative_database.as_posix().encode("utf-8")
        ).hexdigest()[:8]
        stable_name = f"{database.stem}-{digest}"
        output_directory = output_parent / stable_name
        output_path = output_directory / "flamegraph.html"
        if (
            output_path.is_file()
            and output_path.stat().st_mtime_ns >= database.stat().st_mtime_ns
        ):
            generated.append(output_path)
            continue
        if output_directory.is_symlink():
            raise RuntimeError(
                "refusing to replace symlinked MindStudio flame graph output: "
                f"{output_directory}"
            )
        if output_directory.exists():
            shutil.rmtree(output_directory)
        output_directory.mkdir(parents=True)
        result = subprocess.run(
            [
                sys.executable,
                script,
                str(database),
                "--output",
                str(output_directory),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (output_directory / "mindstudio_flamegraph.log").write_text(
            result.stdout, encoding="utf-8"
        )
        if result.returncode == 0 and output_path.is_file():
            generated.append(output_path)
    return generated


def render_flamegraphs(profiler_directory: Path) -> list[Path]:
    """Render every folded stack file when flamegraph.pl is available."""

    inventory = inspect_stack_visualizations(profiler_directory)
    script = inventory["flamegraph_tool"]
    if script is None:
        return []
    perl = shutil.which("perl")
    if perl is None:
        return []
    generated: list[Path] = []
    for relative_path in inventory["stack_files"]:
        stack_path = profiler_directory / relative_path
        output_path = stack_path.with_name(stack_path.stem + "_flamegraph.svg")
        if (
            output_path.is_file()
            and output_path.stat().st_mtime_ns >= stack_path.stat().st_mtime_ns
        ):
            generated.append(output_path)
            continue
        with output_path.open("w", encoding="utf-8") as output:
            result = subprocess.run(
                [
                    perl,
                    script,
                    "--title",
                    stack_path.stem.replace("_", " "),
                    "--countname",
                    "us.",
                    str(stack_path),
                ],
                stdout=output,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        if result.returncode:
            output_path.unlink(missing_ok=True)
            continue
        generated.append(output_path)
    return generated


__all__ = [
    "find_flamegraph_script",
    "find_mindstudio_flamegraph_script",
    "inspect_analysis_outputs",
    "inspect_memory_visualizations",
    "inspect_stack_visualizations",
    "inspect_tensorboard",
    "mindstudio_insight_handoff",
    "render_flamegraphs",
    "render_mindstudio_flamegraphs",
]
