#!/usr/bin/env python3
"""Build a reproducible evidence index for NPU topology explorations."""

from __future__ import annotations

import argparse
from datetime import datetime
import html
import json
import os
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


def _latest_runs(
    root: Path,
    *,
    steps: int,
    replicate: int | None,
    world_size: int | None,
) -> list[dict]:
    selected: dict[tuple[str, bool], tuple[int, dict]] = {}
    for manifest_path in root.rglob("manifest.json"):
        analysis_path = manifest_path.with_name("analysis.json")
        if not analysis_path.is_file():
            continue
        manifest = _read_json(manifest_path)
        config = manifest.get("config", {})
        profiler_enabled = config.get("profiler_enabled", True) is not False
        topology_name = str(manifest["topology"])
        topology = performance_topologies().get(topology_name)
        if (
            manifest.get("device") != "npu"
            or manifest.get("preset") != "distributed"
            or int(config.get("steps", -1)) != steps
            or (
                not profiler_enabled
                and replicate is not None
                and int(config.get("replicate", 0)) != replicate
            )
            or (
                world_size is not None
                and (topology is None or topology.world_size != world_size)
            )
        ):
            continue
        key = (topology_name, profiler_enabled)
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
    steady = (
        analysis.get("profile_phases", {})
        .get("phases", {})
        .get("steady", {})
        .get("summary", {})
    )
    step = steady.get("time_metrics/end_to_end(s)", {})
    memory = steady.get("memory/max_active(GiB)", {}).get("max")
    step_median = step.get("median")
    step_p90 = step.get("p90")
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
        "step_p90_ms": float(step_p90) * 1000 if step_p90 is not None else None,
        "step_max_ms": (
            float(step["max"]) * 1000 if step.get("max") is not None else None
        ),
        "p90_over_median_pct": (
            (float(step_p90) / float(step_median) - 1.0) * 100
            if step_p90 is not None and step_median
            else None
        ),
        "replicate": int(manifest.get("config", {}).get("replicate", 0)),
        "visible_devices": manifest.get("visible_devices", []),
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
    baselines: list[dict],
    profiles: list[dict],
    *,
    steps: int,
    replicate: int | None,
    world_size: int | None,
    output_path: Path,
) -> str:
    replicate_text = "latest successful" if replicate is None else str(replicate)
    world_size_text = "all" if world_size is None else str(world_size)
    ddp8 = next((row for row in baselines if row["topology"] == "ddp8"), None)
    ddp8_drift = _number(ddp8["p90_over_median_pct"]) if ddp8 else "unknown"
    health_scope = (
        " The eight-card runs include NPU0, which currently reports a health "
        "warning; therefore those results are diagnostic comparisons, not "
        "healthy-hardware acceptance evidence."
        if world_size in (None, 8)
        else ""
    )
    lines = [
        (
            f"# GLM-5.2 NPU {world_size}-card topology report"
            if world_size is not None
            else "# GLM-5.2 NPU multi-card topology report"
        ),
        "",
        f"Generated: {datetime.now().astimezone().isoformat()}",
        "",
        "## Scope and comparability",
        "",
        f"This table selects NPU distributed runs with `steps={steps}`, "
        f"`world_size={world_size_text}`, and the distributed preset; profiler-off "
        f"screening uses `{replicate_text}` replicate, "
        "while attribution uses the latest successful replicate. Exact physical "
        f"device mappings are retained per run.{health_scope} Profiler-off "
        "runs are throughput evidence; profiler-active runs are attribution only.",
        "",
        "Job throughput is `world_size × rank throughput`, equivalently the "
        "configured global token budget divided by median step time. Peak HBM is "
        "the maximum `memory/max_active(GiB)` sample on the metrics rank.",
        "",
        "## Validity constraints",
        "",
    ]
    if world_size in (None, 8):
        lines.extend(
            [
                "- The first DDP8 profiler-off attempt (`replicate=1`) failed during "
                "concurrent TSD initialization on physical NPU4/6/7 "
                "(`507033: TsdOpen failed`). Sequential set-device probes then passed; "
                "the table therefore records successful `replicate=2` and retains the "
                "failed run directory as evidence.",
                f"- DDP8 replicate 2 has {ddp8_drift}% p90 drift and disagrees with "
                "the non-active portion of the profiler run. Treat its throughput as "
                "contaminated until an idle-system repeat converges; the rank/collective "
                "attribution remains useful.",
            ]
        )
    lines.extend(
        [
            "- Every topology currently has one authoritative profiler-off replicate "
            "at most. Rankings are screening results and need repeated idle-system "
            "confirmation before acceptance.",
            "",
            "## Profiler-off topology screening",
            "",
            "| Topology | Degrees (DP-repl/DP-shard/TP/CP/PP/EP) | Replicate | Median / p90 / max step (ms) | p90 drift | tok/s/device | tok/s/job | Peak active HBM (GiB) | Evidence |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    topologies = performance_topologies()

    def evidence_link(row: dict) -> str:
        target = row["directory"] / "analysis.json"
        relative = Path(os.path.relpath(target, output_path.parent)).as_posix()
        return f"[analysis.json]({relative})"

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
            f'| {row["topology"]} | {degrees} | {row["replicate"]} | '
            f'{_number(row["step_ms"])} / {_number(row["step_p90_ms"])} / '
            f'{_number(row["step_max_ms"])} | '
            f'{_number(row["p90_over_median_pct"])}% | '
            f'{_number(row["device_tps"])} | {_number(row["job_tps"])} | '
            f'{_number(row["memory_gib"], 3)} | {evidence_link(row)} |'
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
                f'{evidence_link(row)} |'
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


def _html(
    baselines: list[dict], profiles: list[dict], *, world_size: int | None
) -> str:
    maximum = max((float(row["job_tps"] or 0) for row in baselines), default=1.0)
    bars = []
    for row in sorted(baselines, key=lambda item: float(item["job_tps"] or 0), reverse=True):
        width = float(row["job_tps"] or 0) / maximum * 100
        bars.append(
            '<div class="row"><strong>' + html.escape(row["topology"]) + '</strong>'
            '<div class="track"><div style="width:' + f"{width:.4f}%" + '"></div></div>'
            '<span>' + _number(row["job_tps"]) + ' tok/s</span></div>'
        )
    stability_bars = []
    maximum_drift = max(
        (float(row["p90_over_median_pct"] or 0) for row in baselines), default=1.0
    )
    for row in sorted(
        baselines,
        key=lambda item: float(item["p90_over_median_pct"] or 0),
        reverse=True,
    ):
        drift = float(row["p90_over_median_pct"] or 0)
        width = drift / maximum_drift * 100 if maximum_drift else 0
        stability_bars.append(
            '<div class="row"><strong>' + html.escape(row["topology"]) + '</strong>'
            '<div class="track drift"><div style="width:' + f"{width:.4f}%" + '"></div></div>'
            '<span>' + _number(drift) + '%</span></div>'
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
    health_note = (
        " NPU0 health warning makes the 8-card set diagnostic rather than "
        "acceptance evidence."
        if world_size in (None, 8)
        else ""
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>NPU topology comparison</title>
<style>body{{font:14px system-ui;margin:32px;color:#172033;max-width:1100px}}h1,h2{{margin-top:28px}}.row{{display:grid;grid-template-columns:120px 1fr 150px;gap:12px;align-items:center;margin:10px 0}}.track{{height:24px;background:#e2e8f0}}.track div{{height:100%;background:#2563eb}}.track.drift div{{background:#ea580c}}table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border-bottom:1px solid #d7deea;text-align:left}}th{{background:#f4f7fb}}</style></head><body>
<h1>GLM-5.2 NPU topology comparison</h1><p>Profiler-off job throughput; world size {world_size or 'all'}; W0 debug model.{health_note}</p>
<h2>Job throughput</h2>{''.join(bars)}
<h2>Step-time stability: p90 drift above median</h2>{''.join(stability_bars)}
<h2>Profiler attribution</h2><table><thead><tr><th>Topology</th><th>Compute range (ms)</th><th>Exposed communication range (ms)</th><th>Calls/step</th><th>Payload MB/rank/step</th><th>Transit ms/step</th></tr></thead><tbody>{profile_rows}</tbody></table>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument(
        "--latest-replicate",
        action="store_true",
        help="select the latest successful profiler-off replicate per topology",
    )
    parser.add_argument("--world-size", type=int)
    parser.add_argument(
        "--exploration-root",
        type=Path,
        default=Path("tests/glm5_2_performance/explorations/runs"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path(
            "tests/glm5_2_performance/explorations/reports/4-card/comparison.md"
        ),
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=Path("performance_reports/4-card/comparison.html"),
    )
    args = parser.parse_args()
    runs = _latest_runs(
        args.exploration_root,
        steps=args.steps,
        replicate=None if args.latest_replicate else args.replicate,
        world_size=args.world_size,
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
        _markdown(
            baselines,
            profiles,
            steps=args.steps,
            replicate=None if args.latest_replicate else args.replicate,
            world_size=args.world_size,
            output_path=args.markdown,
        ),
        encoding="utf-8",
    )
    args.html.write_text(
        _html(baselines, profiles, world_size=args.world_size), encoding="utf-8"
    )
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
