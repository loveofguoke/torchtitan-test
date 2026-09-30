"""Aggregate profiler-off runs and compare compatible performance groups."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

from tests.glm5_2_common.cli import print_output_path, reset_output_generation
from tests.glm5_2_performance.analysis import read_metrics


STEP_TIME = "time_metrics/end_to_end(s)"
THROUGHPUT = "throughput(tps)"


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _metric_name(metrics: dict[str, Any], markers: tuple[str, ...]) -> str | None:
    for name in metrics:
        normalized = name.lower()
        if all(marker in normalized for marker in markers):
            return name
    return None


def _overview(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "experiment.json"
    if not path.is_file():
        raise FileNotFoundError(f"experiment overview is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _normalized_contract(overview: dict[str, Any]) -> dict[str, Any]:
    raw = overview.get("configuration") or overview.get("contract")
    if not isinstance(raw, dict):
        raise ValueError("experiment overview has no configuration or contract")
    topology = raw.get("topology", overview.get("topology"))
    topology_name = (
        topology.get("name") or topology.get("slug")
        if isinstance(topology, dict)
        else topology
    )
    topology_degrees: dict[str, Any]
    if isinstance(topology, dict):
        aliases = {
            "world_size": "world_size",
            "dp_replicate": "data_parallel_replicate_degree",
            "dp_shard": "data_parallel_shard_degree",
            "tp": "tensor_parallel_degree",
            "pp": "pipeline_parallel_degree",
            "cp": "context_parallel_degree",
            "ep": "expert_parallel_degree",
        }
        topology_degrees = {
            short: topology.get(short, topology.get(long))
            for short, long in aliases.items()
            if topology.get(short, topology.get(long)) is not None
        }
    else:
        try:
            from tests.glm5_2_performance.config import performance_topologies

            selected = performance_topologies()[str(topology_name)]
            topology_degrees = {
                "world_size": selected.world_size,
                "dp_replicate": selected.dp_replicate,
                "dp_shard": selected.dp_shard,
                "tp": selected.tp,
                "pp": selected.pp,
                "cp": selected.cp,
                "ep": selected.ep,
            }
        except KeyError:
            topology_degrees = {}
    return {
        "module": raw.get("module"),
        "model_config": raw.get("model_config", raw.get("config")),
        "topology": topology_name,
        "topology_degrees": topology_degrees,
        "steps": raw.get("steps"),
        "local_batch_size": raw.get("local_batch_size"),
        "global_batch_size": raw.get("global_batch_size"),
        "sequence_length": raw.get("sequence_length"),
        "seed": raw.get("seed"),
        "training_dtype": raw.get("training_dtype", "float32"),
        "mixed_precision_param": raw.get("mixed_precision_param", "bfloat16"),
        "mixed_precision_reduce": raw.get("mixed_precision_reduce", "float32"),
        "profiler_enabled": raw.get("profiler_enabled", True),
    }


def _run_summary(run_dir: Path, *, skip_steps: int) -> dict[str, Any]:
    overview = _overview(run_dir)
    contract = _normalized_contract(overview)
    if contract["profiler_enabled"] is not False:
        raise ValueError(
            "performance authority requires a profiler-off run: " f"{run_dir}"
        )
    records = [
        record
        for record in read_metrics(run_dir / "metrics.jsonl")
        if int(record["step"]) > skip_steps
    ]
    if not records:
        raise ValueError(
            f"no metrics remain after skipping {skip_steps} steps: {run_dir}"
        )
    series: dict[str, list[float]] = {}
    for record in records:
        for name, value in record.get("metrics", {}).items():
            series.setdefault(name, []).append(float(value))
    step_time_name = STEP_TIME if STEP_TIME in series else _metric_name(
        series, ("end_to_end",)
    )
    throughput_name = THROUGHPUT if THROUGHPUT in series else _metric_name(
        series, ("throughput",)
    )
    if step_time_name is None or throughput_name is None:
        raise ValueError(f"step time or throughput metric is missing: {run_dir}")
    step_times = series[step_time_name]
    throughputs = series[throughput_name]

    def aggregate(markers: tuple[str, ...], operation: str) -> float | None:
        name = _metric_name(series, markers)
        if name is None:
            return None
        values = series[name]
        return max(values) if operation == "max" else statistics.fmean(values)

    return {
        "run": str(run_dir.resolve()),
        "contract": contract,
        "measurement": {
            "skip_steps": skip_steps,
            "measured_steps": len(records),
            "first_step": min(int(record["step"]) for record in records),
            "last_step": max(int(record["step"]) for record in records),
            "step_time_median_s": statistics.median(step_times),
            "step_time_p90_s": _percentile(step_times, 0.90),
            "step_time_p95_s": _percentile(step_times, 0.95),
            "throughput_median_tps": statistics.median(throughputs),
            "throughput_mean_tps": statistics.fmean(throughputs),
            "tflops_mean": aggregate(("tflops",), "mean"),
            "mfu_mean_percent": aggregate(("mfu",), "mean"),
            "peak_active_memory_gib": aggregate(("max_active",), "max"),
        },
    }


def _contract_without_steps(contract: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in contract.items() if key != "steps"}


def _validate_group(runs: list[dict[str, Any]], label: str) -> dict[str, Any]:
    expected = _contract_without_steps(runs[0]["contract"])
    for run in runs[1:]:
        actual = _contract_without_steps(run["contract"])
        if actual != expected:
            raise ValueError(
                f"{label} repeat contract mismatch:\nexpected={expected}\nactual={actual}"
            )
    return expected


def _aggregate_group(
    label: str, run_dirs: Iterable[Path], *, skip_steps: int
) -> dict[str, Any]:
    runs = [_run_summary(path.resolve(), skip_steps=skip_steps) for path in run_dirs]
    if not runs:
        raise ValueError(f"{label} must contain at least one run")
    contract = _validate_group(runs, label)
    metric_names = tuple(runs[0]["measurement"])
    aggregate: dict[str, Any] = {}
    for name in metric_names:
        values = [run["measurement"].get(name) for run in runs]
        numeric = [float(value) for value in values if isinstance(value, (int, float))]
        if not numeric:
            continue
        mean = statistics.fmean(numeric)
        aggregate[name] = {
            "median": statistics.median(numeric),
            "mean": mean,
            "min": min(numeric),
            "max": max(numeric),
            "cv_percent": (
                statistics.pstdev(numeric) / abs(mean) * 100
                if len(numeric) > 1 and mean
                else 0.0
            ),
        }
    return {
        "label": label,
        "contract": contract,
        "repeat_count": len(runs),
        "measurement_policy": {
            "profiler_enabled": False,
            "skip_steps": skip_steps,
            "recommended_min_repeats": 3,
            "repeat_count_sufficient": len(runs) >= 3,
            "note": "MLPerf-style measurement discipline only; this is not an MLPerf result.",
        },
        "runs": runs,
        "aggregate": aggregate,
    }


def _compare(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if reference["contract"] != candidate["contract"]:
        raise ValueError(
            "reference and candidate contracts differ; refusing performance comparison:\n"
            f"reference={reference['contract']}\ncandidate={candidate['contract']}"
        )
    metrics: dict[str, Any] = {}
    for name, reference_value in reference["aggregate"].items():
        candidate_value = candidate["aggregate"].get(name)
        if not candidate_value:
            continue
        baseline = reference_value["median"]
        current = candidate_value["median"]
        metrics[name] = {
            "reference_median": baseline,
            "candidate_median": current,
            "candidate_vs_reference_percent": (
                (current / baseline - 1) * 100 if baseline else None
            ),
        }
    return {
        "contract_match": True,
        "metrics": metrics,
        "interpretation": (
            "Lower is better for step time and memory; higher is better for "
            "throughput, TFLOPS, and MFU. Percent changes are evidence, not "
            "automatic acceptance verdicts."
        ),
    }


def _format(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _render(output: Path, payload: dict[str, Any]) -> None:
    groups = [payload["reference"]]
    if payload.get("candidate"):
        groups.append(payload["candidate"])
    group_rows = "".join(
        "<tr>"
        f"<td>{html.escape(group['label'])}</td>"
        f"<td>{group['repeat_count']}</td>"
        f"<td>{'yes' if group['measurement_policy']['repeat_count_sufficient'] else 'no'}</td>"
        f"<td><code>{html.escape(group['contract']['topology'])}</code></td>"
        "</tr>"
        for group in groups
    )
    metric_names = sorted(
        {name for group in groups for name in group["aggregate"]}
    )
    metric_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(name)}</code></td>"
        + "".join(
            f"<td>{_format(group['aggregate'].get(name, {}).get('median'))}</td>"
            f"<td>{_format(group['aggregate'].get(name, {}).get('cv_percent'))}%</td>"
            for group in groups
        )
        + (
            f"<td>{_format(payload['comparison']['metrics'].get(name, {}).get('candidate_vs_reference_percent'))}%</td>"
            if payload.get("comparison")
            else ""
        )
        + "</tr>"
        for name in metric_names
    )
    headers = "".join(
        f"<th>{html.escape(group['label'])} median</th><th>{html.escape(group['label'])} CV</th>"
        for group in groups
    )
    if payload.get("comparison"):
        headers += "<th>Candidate vs reference</th>"
    run_rows = "".join(
        "<tr>"
        f"<td>{html.escape(group['label'])}</td>"
        f"<td><code>{html.escape(run['run'])}</code></td>"
        f"<td>{run['measurement']['first_step']}-{run['measurement']['last_step']}</td>"
        f"<td>{_format(run['measurement']['step_time_median_s'])}</td>"
        f"<td>{_format(run['measurement']['throughput_median_tps'])}</td>"
        "</tr>"
        for group in groups
        for run in group["runs"]
    )
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GLM5.2 profiler-off performance evidence</title><style>
body{{margin:0;background:#f5f7fb;color:#172033;font:15px/1.6 system-ui}}
main{{max-width:1280px;margin:auto;padding:36px 32px}}section{{background:#fff;border:1px solid #dfe5ef;border-radius:12px;padding:22px;margin:20px 0;overflow:auto}}
table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #e5eaf2;padding:10px;text-align:left;vertical-align:top}}th{{background:#f7f9fc}}code{{white-space:pre-wrap;word-break:break-all}}.note{{color:#516074}}
</style></head><body><main><h1>Profiler-off 重复实验与性能对比</h1>
<p class="note">只比较通过实验契约校验的正常训练。重复数不足三次会明确展示，但不会伪造 PASS/FAIL。这里借鉴 MLPerf 的测量纪律，不声称是 MLPerf 结果。</p>
<section><h2>实验组</h2><table><thead><tr><th>Group</th><th>Repeats</th><th>>=3</th><th>Topology</th></tr></thead><tbody>{group_rows}</tbody></table></section>
<section><h2>聚合指标</h2><table><thead><tr><th>Metric</th>{headers}</tr></thead><tbody>{metric_rows}</tbody></table><p class="note">Step time/显存越低越好；吞吐、TFLOPS、MFU 越高越好。CV 用于判断重复稳定性。</p></section>
<section><h2>逐次运行</h2><table><thead><tr><th>Group</th><th>Run</th><th>Measured steps</th><th>Median step time (s)</th><th>Median throughput (tps)</th></tr></thead><tbody>{run_rows}</tbody></table></section>
</main></body></html>"""
    (output / "comparison.html").write_text(document, encoding="utf-8")


def build_comparison(
    *,
    reference_runs: list[Path],
    reference_label: str,
    candidate_runs: list[Path] | None,
    candidate_label: str,
    skip_steps: int,
    output: Path,
    force: bool = False,
) -> dict[str, Any]:
    reference = _aggregate_group(reference_label, reference_runs, skip_steps=skip_steps)
    candidate = (
        _aggregate_group(candidate_label, candidate_runs, skip_steps=skip_steps)
        if candidate_runs
        else None
    )
    payload: dict[str, Any] = {
        "schema": "torchtitan.glm5_2.performance.comparison.v1",
        "reference": reference,
        "candidate": candidate,
        "comparison": _compare(reference, candidate) if candidate else None,
    }
    existing_path = output / "comparison.json"
    if existing_path.is_file():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        if existing == payload and not force:
            if not (output / "comparison.html").is_file():
                _render(output, payload)
            return payload
        if not force:
            raise FileExistsError(
                "comparison output belongs to different inputs; use another "
                f"--output or --force: {output}"
            )
    if force and output.exists():
        reset_output_generation((output,), label="performance comparison")
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _render(output, payload)
    lines = [
        "# Profiler-off performance evidence",
        "",
        f"- reference: `{reference_label}` ({reference['repeat_count']} repeats)",
        f"- candidate: `{candidate_label}` ({candidate['repeat_count']} repeats)"
        if candidate
        else "- candidate: not configured",
        f"- skipped warmup steps: `{skip_steps}`",
        "",
        "Open `comparison.html` for the complete self-contained report and "
        "`comparison.json` for machine-readable evidence.",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def run_cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", action="append", type=Path, required=True)
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--candidate-run", action="append", type=Path)
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--skip-steps", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.skip_steps < 0:
        parser.error("--skip-steps must be non-negative")
    build_comparison(
        reference_runs=args.reference_run,
        reference_label=args.reference_label,
        candidate_runs=args.candidate_run,
        candidate_label=args.candidate_label,
        skip_steps=args.skip_steps,
        output=args.output.resolve(),
        force=args.force,
    )
    for name in ("comparison.html", "comparison.json", "README.md"):
        print_output_path("Performance comparison output", args.output.resolve() / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
