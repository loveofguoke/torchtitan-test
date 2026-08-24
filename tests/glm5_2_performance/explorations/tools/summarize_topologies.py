#!/usr/bin/env python3
"""Build a reproducible evidence index for NPU topology explorations."""

from __future__ import annotations

import argparse
from datetime import datetime
import html
import json
from pathlib import Path
import shlex
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tests.glm5_2_performance.config import performance_topologies  # noqa: E402


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: object, digits: int = 2) -> str:
    return "-" if value is None else f"{float(value):,.{digits}f}"


def _latest_runs(root: Path, *, steps: int, replicate: int) -> list[dict]:
    selected: dict[tuple[str, bool], tuple[int, dict]] = {}
    for manifest_path in root.glob("*/manifest.json"):
        analysis_path = manifest_path.with_name("analysis.json")
        if not analysis_path.is_file():
            continue
        manifest = _read_json(manifest_path)
        config = manifest.get("config", {})
        profiler_enabled = config.get("profiler_enabled", True) is not False
        if (
            manifest.get("device") != "npu"
            or manifest.get("preset") != "distributed"
            or int(config.get("steps", -1)) != steps
            or (
                not profiler_enabled
                and int(config.get("replicate", 0)) != replicate
            )
        ):
            continue
        key = (str(manifest["topology"]), profiler_enabled)
        candidate = {
            "manifest_path": manifest_path,
            "manifest": manifest,
            "analysis": _read_json(analysis_path),
        }
        mtime = manifest_path.stat().st_mtime_ns
        if key not in selected or mtime > selected[key][0]:
            selected[key] = (mtime, candidate)
    return [value[1] for value in selected.values()]


def _baseline_row(run: dict) -> dict:
    manifest = run["manifest"]
    analysis = run["analysis"]
    comparison = analysis.get("profile_phases", {}).get("comparison", {})
    memory = (
        analysis.get("metrics", {})
        .get("summary", {})
        .get("memory/max_active(GiB)", {})
        .get("max")
    )
    return {
        "topology": manifest["topology"],
        "step_ms": (
            float(comparison["baseline_median_step_seconds"]) * 1000
            if comparison.get("baseline_median_step_seconds") is not None
            else None
        ),
        "device_tps": comparison.get("baseline_median_throughput_tps"),
        "job_tps": comparison.get("baseline_job_throughput_tps"),
        "memory_gib": memory,
        "directory": run["manifest_path"].parent,
        "run_name": manifest["run_name"],
    }


def _profile_row(run: dict) -> dict:
    manifest = run["manifest"]
    analysis = run["analysis"]
    ranks = analysis.get("distributed_step_trace", {}).get("ranks", [])
    communication = analysis.get("communication_summary", {}).get("rows", [])

    def values(name: str) -> list[float]:
        return [float(row[name]) for row in ranks if row.get(name) is not None]

    exposed = values("median_exposed_communication_ms")
    compute = values("median_compute_ms")
    collectives = []
    for kind in sorted({str(row["kind"]) for row in communication}):
        rows = [row for row in communication if row["kind"] == kind]

        def median(name: str) -> float | None:
            samples = [
                float(row[name]) for row in rows if row.get(name) is not None
            ]
            return statistics.median(samples) if samples else None

        collectives.append(
            {
                "kind": kind,
                "calls": median("calls_per_step"),
                "payload": median("payload_mb_per_step"),
                "transit": median("transit_ms_per_step"),
                "bandwidth": median("effective_bandwidth_gbps"),
            }
        )
    return {
        "topology": manifest["topology"],
        "compute_min": min(compute) if compute else None,
        "compute_max": max(compute) if compute else None,
        "exposed_min": min(exposed) if exposed else None,
        "exposed_max": max(exposed) if exposed else None,
        "collectives": collectives,
        "directory": run["manifest_path"].parent,
        "run_name": manifest["run_name"],
    }


def _markdown(
    baselines: list[dict], profiles: list[dict], *, steps: int, replicate: int
) -> str:
    lines = [
        "# GLM-5.2 NPU multi-card topology report",
        "",
        f"Generated: {datetime.now().astimezone().isoformat()}",
        "",
        "## Scope and comparability",
        "",
        f"This table selects NPU distributed runs with `steps={steps}` and the "
        f"distributed preset; profiler-off screening uses `replicate={replicate}`, "
        "while attribution uses the latest successful replicate. Exact physical "
        "device mappings are retained per run; NPU0 is excluded because it reports "
        "an HCCS lane-drop warning. Profiler-off "
        "runs are throughput evidence; profiler-active runs are attribution only.",
        "",
        "Job throughput is `world_size × rank throughput`, equivalently the "
        "configured global token budget divided by median step time. Peak HBM is "
        "the maximum `memory/max_active(GiB)` sample on the metrics rank.",
        "",
        "## Profiler-off topology screening",
        "",
        "| Topology | Degrees (DP-repl/DP-shard/TP/CP/PP/EP) | Median step (ms) | tok/s/device | tok/s/job | Peak active HBM (GiB) | Evidence |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    topologies = performance_topologies()
    for row in sorted(baselines, key=lambda item: float(item["job_tps"] or 0), reverse=True):
        topology = topologies[row["topology"]]
        degrees = "/".join(
            str(value)
            for value in (
                topology.dp_replicate,
                topology.dp_shard,
                topology.tp,
                topology.cp,
                topology.pp,
                topology.ep,
            )
        )
        lines.append(
            f'| {row["topology"]} | {degrees} | {_number(row["step_ms"])} | '
            f'{_number(row["device_tps"])} | {_number(row["job_tps"])} | '
            f'{_number(row["memory_gib"], 3)} | `{row["directory"]}/analysis.json` |'
        )
    lines.extend(
        [
            "",
            "## Profiler attribution",
            "",
            "| Topology | Compute range (ms) | Exposed communication range (ms) | Collective calls/step | Payload MB/rank/step | Physical transit ms/step | Effective GB/s | Evidence |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in sorted(profiles, key=lambda item: item["topology"]):
        collectives = row["collectives"] or [{"kind": "-"}]
        for index, collective in enumerate(collectives):
            lines.append(
                f'| {row["topology"] if index == 0 else ""} '
                f'`{collective.get("kind", "-")}` | '
                f'{_number(row["compute_min"]) if index == 0 else ""}–'
                f'{_number(row["compute_max"]) if index == 0 else ""} | '
                f'{_number(row["exposed_min"]) if index == 0 else ""}–'
                f'{_number(row["exposed_max"]) if index == 0 else ""} | '
                f'{_number(collective.get("calls"))} | '
                f'{_number(collective.get("payload"))} | '
                f'{_number(collective.get("transit"))} | '
                f'{_number(collective.get("bandwidth"))} | '
                f'`{row["directory"]}/analysis.json` |'
            )
    lines.extend(
        [
            "",
            "## Evidence chain per conclusion",
            "",
            "1. Experiment identity and exact `torchrun` argv: each run's `manifest.json`.",
            "2. Driver invocations and environment: `command_history.jsonl`.",
            "3. Step metrics and HBM: `metrics.jsonl` and `analysis.json`.",
            "4. Rank critical path: parsed `step_trace_time.csv` summarized in `distributed_step_trace`.",
            "5. Collective count/payload/transit: canonical `ASCEND_PROFILER_OUTPUT/communication.json` summarized in `communication_summary`.",
            "6. Official offline tools: `tool_commands/*.json`; MindStudio imports are listed in `artifacts.json`.",
            "",
            "No optimization is accepted from these measurements alone. Each proposed "
            "change needs profiler-off repeats plus a separate profiler attribution run.",
            "",
        ]
    )
    return "\n".join(lines)


def _html(baselines: list[dict], profiles: list[dict]) -> str:
    maximum = max((float(row["job_tps"] or 0) for row in baselines), default=1.0)
    bars = []
    for row in sorted(baselines, key=lambda item: float(item["job_tps"] or 0), reverse=True):
        width = float(row["job_tps"] or 0) / maximum * 100
        bars.append(
            '<div class="row"><strong>' + html.escape(row["topology"]) + '</strong>'
            '<div class="track"><div style="width:' + f"{width:.4f}%" + '"></div></div>'
            '<span>' + _number(row["job_tps"]) + ' tok/s</span></div>'
        )
    profile_rows = ""
    for row in sorted(profiles, key=lambda item: item["topology"]):
        for index, collective in enumerate(row["collectives"] or [{"kind": "-"}]):
            profile_rows += (
                '<tr><td>' + (html.escape(row["topology"]) if index == 0 else "")
                + ' <code>' + html.escape(str(collective.get("kind", "-")))
                + '</code></td><td>'
                + (_number(row["compute_min"]) + '–' + _number(row["compute_max"]) if index == 0 else "")
                + '</td><td>'
                + (_number(row["exposed_min"]) + '–' + _number(row["exposed_max"]) if index == 0 else "")
                + '</td><td>' + _number(collective.get("calls"))
                + '</td><td>' + _number(collective.get("payload"))
                + '</td><td>' + _number(collective.get("transit")) + '</td></tr>'
            )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>NPU topology comparison</title>
<style>body{{font:14px system-ui;margin:32px;color:#172033;max-width:1100px}}h1,h2{{margin-top:28px}}.row{{display:grid;grid-template-columns:120px 1fr 150px;gap:12px;align-items:center;margin:10px 0}}.track{{height:24px;background:#e2e8f0}}.track div{{height:100%;background:#2563eb}}table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border-bottom:1px solid #d7deea;text-align:left}}th{{background:#f4f7fb}}</style></head><body>
<h1>GLM-5.2 NPU topology comparison</h1><p>Profiler-off job throughput; physical NPU1-4; W0 debug model.</p>
<h2>Job throughput</h2>{''.join(bars)}
<h2>Profiler attribution</h2><table><thead><tr><th>Topology</th><th>Compute range (ms)</th><th>Exposed communication range (ms)</th><th>Calls/step</th><th>Payload MB/rank/step</th><th>Transit ms/step</th></tr></thead><tbody>{profile_rows}</tbody></table>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument(
        "--exploration-root",
        type=Path,
        default=Path("tests/glm5_2_performance/explorations/runs"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path(
            "tests/glm5_2_performance/explorations/reports/"
            "MULTI_CARD_TOPOLOGY_REPORT.md"
        ),
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=Path("performance_reports/npu-multi-card-topology-comparison.html"),
    )
    args = parser.parse_args()
    runs = _latest_runs(
        args.exploration_root, steps=args.steps, replicate=args.replicate
    )
    baselines = [
        _baseline_row(run)
        for run in runs
        if run["manifest"].get("config", {}).get("profiler_enabled") is False
    ]
    profiles = [
        _profile_row(run)
        for run in runs
        if run["manifest"].get("config", {}).get("profiler_enabled") is not False
    ]
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.html.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(
        _markdown(baselines, profiles, steps=args.steps, replicate=args.replicate),
        encoding="utf-8",
    )
    args.html.write_text(_html(baselines, profiles), encoding="utf-8")
    history = args.exploration_root.parent / "history" / "report_generation_commands.jsonl"
    history.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "cwd": str(Path.cwd()),
        "argv": [sys.executable, *sys.argv],
        "shell_command": shlex.join([sys.executable, *sys.argv]),
        "markdown": str(args.markdown),
        "html": str(args.html),
    }
    with history.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    print(args.markdown)
    print(args.html)


if __name__ == "__main__":
    main()
