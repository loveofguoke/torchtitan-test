"""Evidence-backed single-topology performance diagnosis.

The diagnosis is a project-owned index over training metrics and official
profiler outputs. It never replaces msprof-analyze, MindStudio Insight, or
Nsight reports, and heuristic findings are never promoted to pass/fail.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import performance_topologies


SCHEMA = "torchtitan.glm5_2.performance.self_diagnosis.v1"


def _branch(
    name: str,
    status: str,
    summary: str,
    *,
    evidence: list[dict[str, Any]],
    next_actions: list[str],
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "summary": summary,
        "evidence": evidence,
        "next_actions": next_actions,
    }


def _metric(summary: dict[str, Any], fragment: str, statistic: str) -> float | None:
    for name, values in summary.items():
        if fragment in name.lower() and values.get(statistic) is not None:
            return float(values[statistic])
    return None


def _numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def _topology_focus(topology_name: str) -> list[str]:
    topology = performance_topologies()[topology_name]
    focus = ["compute", "host", "memory"]
    if topology.world_size > 1:
        focus.extend(["rank_balance", "communication"])
    if topology.dp_replicate > 1:
        focus.append("gradient_collective")
    if topology.dp_shard > 1:
        focus.extend(["parameter_all_gather", "gradient_reduce_scatter"])
    if topology.tp > 1:
        focus.extend(["tensor_parallel_collectives", "small_collectives"])
    if topology.cp > 1:
        focus.append("context_parallel_exchange")
    if topology.pp > 1:
        focus.extend(["pipeline_bubbles", "stage_balance"])
    if topology.ep > 1:
        focus.extend(["all_to_all", "expert_load_balance"])
    return list(dict.fromkeys(focus))


def build_self_diagnosis(
    *,
    manifest: dict[str, Any],
    analysis: dict[str, Any],
) -> dict[str, Any]:
    """Build a non-verdict diagnosis from already captured evidence."""

    config = manifest["config"]
    topology_name = str(manifest["topology"])
    profiler_enabled = bool(config.get("profiler_enabled", True))
    summary = analysis.get("metrics", {}).get("summary", {})
    distributed = analysis.get("distributed_step_trace", {})
    rank_rows = distributed.get("ranks", [])
    cross_rank = distributed.get("cross_rank", {})
    communication_rows = analysis.get("communication_summary", {}).get("rows", [])
    compiler = analysis.get("compiler_diagnostics", {})

    step_median = _metric(summary, "end_to_end", "median")
    throughput_mean = _metric(summary, "throughput", "mean")
    peak_memory = _metric(summary, "max_active", "max")
    mfu_mean = _metric(summary, "mfu", "mean")
    tflops_mean = _metric(summary, "tflops", "mean")
    baseline_evidence = [
        {"metric": "median_step_seconds", "value": step_median},
        {"metric": "mean_throughput_tps", "value": throughput_mean},
        {"metric": "mean_tflops", "value": tflops_mean},
        {"metric": "mean_mfu_percent", "value": mfu_mean},
        {"metric": "peak_active_memory_gib", "value": peak_memory},
    ]
    baseline_evidence = [row for row in baseline_evidence if row["value"] is not None]
    baseline_status = "observed" if baseline_evidence else "not_available"
    baseline_summary = (
        "Profiler-off steady-state evidence is suitable for baseline decisions."
        if not profiler_enabled and baseline_evidence
        else "Metrics are instrumented and support localization, not final baseline claims."
        if baseline_evidence
        else "No training metric evidence is available."
    )

    rank_evidence: list[dict[str, Any]] = []
    for name, values in cross_rank.items():
        ratio = values.get("max_over_median")
        if ratio is not None:
            rank_evidence.append(
                {
                    "metric": name,
                    "max_over_median": float(ratio),
                    "max_rank": values.get("max_rank"),
                }
            )
    skewed = [row for row in rank_evidence if row["max_over_median"] >= 1.10]
    rank_status = (
        "suspect"
        if skewed
        else "observed"
        if rank_evidence
        else "not_available"
    )

    exposed = _numeric_values(rank_rows, "exposed_communication_percent")
    wait = _numeric_values(communication_rows, "wait_percent")
    communication_evidence: list[dict[str, Any]] = []
    if exposed:
        communication_evidence.append(
            {"metric": "max_exposed_communication_percent", "value": max(exposed)}
        )
    if wait:
        communication_evidence.append(
            {"metric": "max_collective_wait_percent", "value": max(wait)}
        )
    communication_suspect = (exposed and max(exposed) >= 20.0) or (
        wait and max(wait) >= 50.0
    )
    communication_status = (
        "suspect"
        if communication_suspect
        else "observed"
        if communication_evidence
        else "not_available"
    )

    free_values = _numeric_values(rank_rows, "free_percent")
    fallbacks = int(compiler.get("npu_cpu_fallbacks", 0)) + int(
        compiler.get("aicpu_fallbacks", 0)
    )
    host_evidence: list[dict[str, Any]] = [
        {"metric": "runtime_fallback_warning_count", "value": fallbacks}
    ]
    if free_values:
        host_evidence.append(
            {"metric": "max_device_free_percent", "value": max(free_values)}
        )
    host_suspect = fallbacks > 0 or (free_values and max(free_values) >= 20.0)

    memory_ready = bool(analysis.get("memory_visualizations", {}).get("ready"))
    memory_evidence = [
        {"metric": "peak_active_memory_gib", "value": peak_memory},
        {"metric": "memory_timeline_ready", "value": memory_ready},
    ]

    operator_tables = analysis.get("top_csv_tables", [])
    operator_evidence = [
        {
            "metric": "duration_table",
            "source": table.get("source"),
            "top_entries": table.get("rows", [])[:5],
        }
        for table in operator_tables[:3]
    ]
    compute_status = "observed" if operator_evidence else "not_available"

    graph_evidence = [
        {
            "metric": "graph_break_messages",
            "value": int(compiler.get("graph_breaks", 0)),
        },
        {
            "metric": "recompile_messages",
            "value": int(compiler.get("recompiles", 0)),
        },
        {
            "metric": "backend_failure_messages",
            "value": int(compiler.get("backend_failures", 0)),
        },
    ]
    graph_suspect = any(int(row["value"]) > 0 for row in graph_evidence)

    branches = {
        "baseline": _branch(
            "Profiler-off baseline",
            baseline_status,
            baseline_summary,
            evidence=baseline_evidence,
            next_actions=[
                "Run at least three profiler-off repeats after warmup before accepting throughput or step-time changes."
            ] if profiler_enabled else [],
        ),
        "rank_balance": _branch(
            "Cross-rank balance",
            rank_status,
            "At least one rank metric exceeds 1.10x the cross-rank median."
            if skewed
            else "No material rank skew was derived from available StepTrace evidence."
            if rank_evidence
            else "No cross-rank StepTrace evidence is available.",
            evidence=rank_evidence,
            next_actions=[
                "Open the slow-rank and slow-link outputs, then align the slow rank with Timeline compute, communication, and Free intervals."
            ] if skewed else [],
        ),
        "communication": _branch(
            "Communication exposure",
            communication_status,
            "Communication is materially exposed or collective wait is high."
            if communication_suspect
            else "Available communication evidence does not cross the project triage guidance."
            if communication_evidence
            else "Communication evidence is not available for this capture.",
            evidence=communication_evidence,
            next_actions=[
                "Use cluster_time_summary, communication bottleneck, communication matrix, slow-rank, and slow-link outputs to separate transit from wait and readiness skew."
            ] if communication_suspect else [],
        ),
        "host": _branch(
            "Host and device starvation",
            "suspect" if host_suspect else "observed",
            "Fallback warnings or device Free time indicate a possible Host/runtime bottleneck."
            if host_suspect
            else "No fallback warning or material device Free signal was derived.",
            evidence=host_evidence,
            next_actions=[
                "Inspect Host Timeline, DataLoader, Python GC, stream synchronization, affinity, and API statistics before recapturing a deeper profile."
            ] if host_suspect else [],
        ),
        "operator": _branch(
            "Operator and kernel hotspots",
            compute_status,
            "Operator duration tables are indexed; inspect dominant kernels, shapes, dtypes, call counts, and fusion opportunities."
            if operator_evidence
            else "No parsed operator duration table is available.",
            evidence=operator_evidence,
            next_actions=[
                "Run offline parse or Advisor, then use operator, shape, L2, fusion, block-dim, and operator-MFU evidence for the selected hotspot."
            ] if not operator_evidence else [],
        ),
        "memory": _branch(
            "Memory capacity and lifetime",
            "observed"
            if memory_ready or peak_memory is not None
            else "not_available",
            "Interactive memory lifetime evidence is available."
            if memory_ready
            else "Only aggregate peak memory is available; allocation lifetime is not yet localized."
            if peak_memory is not None
            else "No memory evidence is available.",
            evidence=memory_evidence,
            next_actions=[
                "Use the memory preset to classify parameters, gradients, optimizer state, activations, temporary tensors, and reserved-versus-allocated gaps."
            ] if not memory_ready else [],
        ),
        "graph": _branch(
            "Graph and compilation health",
            "suspect" if graph_suspect else "observed",
            "Graph breaks, recompiles, or backend failures were logged."
            if graph_suspect
            else "No graph-break, recompile, or backend-failure message was counted.",
            evidence=graph_evidence,
            next_actions=[
                "Inspect TORCH_TRACE/tlparse and backend IR, then compare fused and unfused operator evidence."
            ] if graph_suspect else [],
        ),
    }
    suspects = [
        name
        for name, branch in branches.items()
        if branch["status"] == "suspect"
    ]
    return {
        "schema": SCHEMA,
        "scope": "single_topology",
        "verdict_policy": "diagnostic_only_no_pass_fail",
        "run_name": manifest["run_name"],
        "device": manifest["device"],
        "topology": topology_name,
        "preset": manifest["preset"],
        "capture_semantics": (
            "instrumented_localization"
            if profiler_enabled
            else "profiler_off_baseline"
        ),
        "topology_focus": _topology_focus(topology_name),
        "triage_guidance": {
            "cross_rank_max_over_median": 1.10,
            "exposed_communication_percent": 20.0,
            "collective_wait_percent": 50.0,
            "device_free_percent": 20.0,
            "meaning": "Project triage guidance only; not an acceptance threshold.",
        },
        "suspect_branches": suspects,
        "branches": branches,
        "topn_routing": {
            "communication": "Top1 communication bottleneck",
            "operator": "Top2 operator or fusion bottleneck",
            "host": "Top3 Host Bound",
            "rank_balance": "Top4 cluster fluctuation or slow rank/link",
            "graph": "Top7 version or graph regression",
            "memory": "Capacity/lifetime branch before operator recapture",
        },
    }


def write_self_diagnosis(run_directory: Path, diagnosis: dict[str, Any]) -> dict[str, str]:
    """Write the run-owned diagnosis bundle and return its paths."""

    output = run_directory / "diagnosis" / "self"
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "diagnosis.json"
    markdown_path = output / "README.md"
    json_path.write_text(
        json.dumps(diagnosis, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Single-topology performance diagnosis",
        "",
        "This is a project-owned diagnostic index. It does not replace official tool output and does not issue pass/fail verdicts.",
        "",
        f"- Run: `{diagnosis['run_name']}`",
        f"- Device: `{diagnosis['device']}`",
        f"- Topology: `{diagnosis['topology']}`",
        f"- Preset: `{diagnosis['preset']}`",
        f"- Suspect branches: `{', '.join(diagnosis['suspect_branches']) or 'none derived'}`",
        "",
        "## Branches",
        "",
        "| Branch | Status | Summary |",
        "|---|---|---|",
    ]
    for key, branch in diagnosis["branches"].items():
        lines.append(
            f"| `{key}` | `{branch['status']}` | {branch['summary']} |"
        )
    lines.extend(
        [
            "",
            "See `diagnosis.json` for evidence rows, topology focus, triage guidance, and next actions.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(json_path), "readme": str(markdown_path)}
