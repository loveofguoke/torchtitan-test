#!/usr/bin/env python3
"""Organize historical performance evidence and regenerate readable indexes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tests.glm5_2_performance.config import performance_topologies  # noqa: E402
from tests.glm5_2_performance.documentation import (  # noqa: E402
    card_scope,
    run_identity,
    write_run_readme,
)


SCOPES = ("1-card", "2-card", "4-card", "8-card", "16-card")


def _root() -> Path:
    return Path(__file__).resolve().parents[4]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _topology(path: Path) -> str | None:
    manifest = _read_json(path / "manifest.json")
    if manifest.get("topology") in performance_topologies():
        return str(manifest["topology"])
    history = path / "command_history.jsonl"
    if history.is_file():
        for line in history.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except (TypeError, ValueError):
                continue
            if record.get("topology") in performance_topologies():
                return str(record["topology"])
    names = sorted(performance_topologies(), key=len, reverse=True)
    for name in names:
        if re.search(rf"(?:^|-)({re.escape(name)})(?:-|$)", path.name):
            return name
    return None


def _move(source: Path, destination: Path, moves: dict[str, str]) -> None:
    if source.resolve() == destination.resolve():
        return
    if destination.exists():
        raise FileExistsError(f"migration target already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    old = str(source.resolve())
    source.rename(destination)
    moves[old] = str(destination.resolve())


def _migrate_directories(base: Path, moves: dict[str, str]) -> None:
    if not base.is_dir():
        return
    for path in sorted(base.iterdir()):
        if not path.is_dir() or path.name in SCOPES:
            continue
        topology = _topology(path)
        if topology is None:
            continue
        _move(path, base / card_scope(topology) / topology / path.name, moves)


def _migrate_html_reports(base: Path, moves: dict[str, str]) -> None:
    if not base.is_dir():
        return
    old_eight = base / "eight_card" / "npu-eight-card-topology-comparison.html"
    if old_eight.is_file():
        _move(old_eight, base / "8-card" / "comparison.html", moves)
    for path in sorted(base.glob("*.html")):
        if path.name == "npu-multi-card-topology-comparison.html":
            _move(path, base / "4-card" / "comparison.html", moves)
            continue
        topology = _topology(path.with_suffix(""))
        if topology is None:
            _move(path, base / "suites" / path.name, moves)
            continue
        _move(
            path,
            base / card_scope(topology) / topology / path.name,
            moves,
        )
    old_eight_parent = base / "eight_card"
    if old_eight_parent.is_dir() and not any(old_eight_parent.iterdir()):
        old_eight_parent.rmdir()


def _migrate_exploration_reports(reports: Path, moves: dict[str, str]) -> None:
    mappings = {
        reports / "NPU_EXPLORATION_REPORT.md": reports / "summary.md",
        reports / "MULTI_CARD_TOPOLOGY_REPORT.md": reports / "4-card" / "comparison.md",
        reports / "FAILED_ATTEMPTS.md": reports / "failures.md",
        reports / "LEGACY_ANALYZE_ALIASES.md": reports / "reference" / "legacy-aliases.md",
        reports / "eight_card" / "README.md": reports / "8-card" / "index.md",
        reports / "eight_card" / "COMMANDS.md": reports / "8-card" / "commands.md",
        reports / "eight_card" / "EIGHT_CARD_ANALYSIS.md": reports / "8-card" / "summary.md",
        reports / "eight_card" / "EIGHT_CARD_TOPOLOGY_REPORT.md": reports / "8-card" / "comparison.md",
        reports / "eight_card" / "ddp8" / "DDP8_ANALYSIS.md": reports / "8-card" / "ddp8" / "analysis.md",
        reports / "eight_card" / "fsdp8" / "FSDP8_ANALYSIS.md": reports / "8-card" / "fsdp8" / "analysis.md",
    }
    for source, destination in mappings.items():
        if source.exists():
            _move(source, destination, moves)
    old = reports / "eight_card"
    if old.is_dir():
        for path in sorted(old.rglob("*"), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        if old.exists() and not any(old.iterdir()):
            old.rmdir()


def _replace(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _replace(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace(item, replacements) for item in value]
    if isinstance(value, str):
        for old, new in sorted(replacements.items(), key=lambda item: -len(item[0])):
            value = value.replace(old, new)
        return value
    return value


def _update_current_paths(root: Path, moves: dict[str, str]) -> None:
    compact = root / "tests" / "glm5_2_performance" / "explorations" / "runs"
    candidates = []
    for base in (compact, root / "performance_artifacts"):
        if base.is_dir():
            candidates.extend(base.rglob("manifest.json"))
            candidates.extend(base.rglob("analysis.json"))
            candidates.extend(base.rglob("artifacts.json"))
    if compact.is_dir():
        candidates.extend(compact.rglob("tool_commands/*.json"))
    for path in candidates:
        value = _read_json(path)
        updated = _replace(value, moves)
        if (
            path.parent.name == "tool_commands"
            and updated != value
            and "command" in value
        ):
            updated["original_command"] = value.get(
                "original_command", value["command"]
            )
        if updated != value:
            path.write_text(
                json.dumps(updated, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )


def _legacy_path_mappings(root: Path) -> dict[str, str]:
    """Map old flat paths to their current scoped locations after migration."""

    mappings: dict[str, str] = {}
    for base in (
        root / "performance_runs",
        root / "performance_artifacts",
        root / "performance_reports",
    ):
        for scope in SCOPES:
            scope_path = base / scope
            if not scope_path.is_dir():
                continue
            for path in scope_path.glob("*/*"):
                if path.is_dir() or path.suffix == ".html":
                    mappings[str((base / path.name).resolve())] = str(path.resolve())
    return mappings


def migrate(root: Path) -> dict[str, str]:
    moves: dict[str, str] = {}
    exploration = root / "tests" / "glm5_2_performance" / "explorations"
    _migrate_directories(exploration / "runs", moves)
    _migrate_directories(root / "performance_runs", moves)
    _migrate_directories(root / "performance_artifacts", moves)
    _migrate_html_reports(root / "performance_reports", moves)
    _migrate_exploration_reports(exploration / "reports", moves)
    old_index = exploration / "README.md"
    if old_index.is_file():
        _move(old_index, exploration / "index.md", moves)
    replacements = _legacy_path_mappings(root)
    replacements.update(moves)
    _update_current_paths(root, replacements)
    return moves


def _link(path: Path, target: Path) -> str:
    return Path(os.path.relpath(target, path.parent)).as_posix()


def _number(value: object, digits: int = 2) -> str:
    return "-" if value is None else f"{float(value):,.{digits}f}"


def _run_directories(runs_root: Path, scope: str, topology: str) -> list[Path]:
    base = runs_root / scope / topology
    if not base.is_dir():
        return []
    return sorted(
        (path for path in base.iterdir() if path.is_dir()),
        key=lambda path: (run_identity(path)["timestamp"], path.name),
    )


def _write_topology_document(
    reports_root: Path,
    runs_root: Path,
    scope: str,
    topology: str,
    directories: list[Path],
) -> Path:
    topology_config = performance_topologies()[topology]
    output = reports_root / scope / topology / "experiment.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    identities = [run_identity(path) for path in directories]
    lines = [
        f"# {topology} experiments",
        "",
        "## topology",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| world size | {topology_config.world_size} |",
        f"| dp replicate / shard | {topology_config.dp_replicate} / {topology_config.dp_shard} |",
        f"| tp / cp / pp / ep | {topology_config.tp} / {topology_config.cp} / {topology_config.pp} / {topology_config.ep} |",
        f"| recorded runs | {len(directories)} |",
        "",
        "## run results",
        "",
        "| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for directory, identity in zip(directories, identities):
        metrics = identity["metrics"]
        config = identity["config"]
        readme = directory / "readme.md"
        lines.append(
            f"| [{identity['run_name']}]({_link(output, readme)}) | "
            f"{identity['status']} | {identity['mode']} | "
            f"{identity['manifest'].get('preset') or config.get('preset') or '-'} | "
            f"{_number(metrics['step_ms'])} ms | "
            f"{_number(metrics['device_tps'])} | {_number(metrics['job_tps'])} | "
            f"{_number(metrics['memory_gib'], 3)} GiB |"
        )
    lines.extend(["", "## experiment sequence", ""])
    for index, (directory, identity) in enumerate(zip(directories, identities), 1):
        history = identity["history"]
        lines.extend(
            [
                f"### {index}. {identity['run_name']}",
                "",
                f"- Status: `{identity['status']}`",
                f"- Started: `{identity['timestamp'] or '-'}`",
                f"- Full process and outputs: [run readme]({_link(output, directory / 'readme.md')})",
            ]
        )
        artifacts = identity["artifacts"]
        if artifacts.get("report_path"):
            lines.append(f"- HTML report: `{artifacts['report_path']}`")
        for record in history:
            if record.get("shell_command"):
                lines.extend(
                    [
                        "",
                        "```bash",
                        str(record["shell_command"]),
                        "```",
                    ]
                )
        lines.append("")
    completed = [
        identity
        for identity in identities
        if identity["status"] == "completed"
        and identity["metrics"].get("job_tps") is not None
    ]
    if completed:
        best = max(completed, key=lambda item: float(item["metrics"]["job_tps"]))
        lines.extend(
            [
                "## current summary",
                "",
                f"The highest recorded job throughput is "
                f"{_number(best['metrics']['job_tps'])} tok/s from "
                f"`{best['run_name']}`. Profiler-off runs are throughput evidence; "
                "profiler-active runs are attribution evidence only.",
                "",
            ]
        )
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _write_scope_index(
    reports_root: Path,
    runs_root: Path,
    scope: str,
    topologies: list[str],
) -> Path:
    output = reports_root / scope / "index.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {scope} performance experiments",
        "",
        "| topology | runs | best profiler-off tok/s/job | experiment document |",
        "| --- | ---: | ---: | --- |",
    ]
    comparisons = []
    for topology in topologies:
        directories = _run_directories(runs_root, scope, topology)
        identities = [run_identity(path) for path in directories]
        baseline = [
            identity
            for identity in identities
            if identity["mode"] == "profiler-off"
            and identity["status"] == "completed"
            and identity["metrics"].get("job_tps") is not None
        ]
        best = max(
            (float(identity["metrics"]["job_tps"]) for identity in baseline),
            default=None,
        )
        experiment = reports_root / scope / topology / "experiment.md"
        lines.append(
            f"| {topology} | {len(directories)} | {_number(best)} | "
            f"[experiment]({_link(output, experiment)}) |"
        )
        if best is not None:
            comparisons.append((topology, best))
    lines.extend(["", "## comparison", ""])
    if comparisons:
        comparisons.sort(key=lambda item: item[1], reverse=True)
        leader, leader_tps = comparisons[0]
        lines.append(
            f"`{leader}` currently leads the profiler-off screening at "
            f"{_number(leader_tps)} tok/s/job. This is a run index and screening "
            "comparison; validity constraints remain in each run and analysis document."
        )
    else:
        lines.append(
            "No profiler-off authority run is available in this scope; use the "
            "topology documents for profiler attribution results."
        )
    lines.extend(["", "- [scope summary](summary.md)"])
    if (reports_root / scope / "comparison.md").is_file():
        lines.append("- [scope comparison](comparison.md)")
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    summary = reports_root / scope / "summary.md"
    if not summary.exists():
        summary.write_text("\n".join(lines), encoding="utf-8")
    return output


def generate_documents(root: Path) -> None:
    exploration = root / "tests" / "glm5_2_performance" / "explorations"
    runs_root = exploration / "runs"
    reports_root = exploration / "reports"
    scope_topologies: dict[str, list[str]] = {}
    for scope_path in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        scope = scope_path.name
        if scope not in SCOPES:
            continue
        topologies = sorted(path.name for path in scope_path.iterdir() if path.is_dir())
        scope_topologies[scope] = topologies
        for topology in topologies:
            directories = _run_directories(runs_root, scope, topology)
            for directory in directories:
                write_run_readme(directory)
            _write_topology_document(
                reports_root, runs_root, scope, topology, directories
            )
        _write_scope_index(reports_root, runs_root, scope, topologies)
    index = reports_root / "index.md"
    lines = [
        "# npu performance exploration reports",
        "",
        "The hierarchy is rank count → topology → experiment/run evidence.",
        "",
        "| scope | topology count | documents |",
        "| --- | ---: | --- |",
    ]
    for scope, topologies in sorted(
        scope_topologies.items(), key=lambda item: int(item[0].split("-", 1)[0])
    ):
        scope_index = reports_root / scope / "index.md"
        lines.append(
            f"| {scope} | {len(topologies)} | [index]({_link(index, scope_index)}) |"
        )
    lines.extend(
        [
            "",
            "- [overall analysis](summary.md)",
            "- [failed attempts](failures.md)",
            "- [reference notes](reference/)",
            "",
        ]
    )
    index.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--migrate", action="store_true")
    parser.add_argument("--root", type=Path, default=_root())
    args = parser.parse_args()
    root = args.root.resolve()
    moves = migrate(root) if args.migrate else {}
    generate_documents(root)
    print(f"moved {len(moves)} paths")
    print(root / "tests" / "glm5_2_performance" / "explorations" / "reports" / "index.md")


if __name__ == "__main__":
    main()
