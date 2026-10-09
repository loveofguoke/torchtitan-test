"""Aggregate profiler-off runs and compare compatible performance groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any, Iterable

from tests.glm5_2_common.cli import (
    display_repository_path,
    print_output_path,
    reset_output_generation,
)
from tests.glm5_2_common.reporting import (
    echarts_line,
    interactive_table,
    save_panel_report,
    section_heading,
    summary_table,
)
from tests.glm5_2_performance.analysis import read_metrics
from tests.glm5_2_performance.benchmark_metrics import (
    bootstrap_median_ci,
    metric_statistics,
    numerical_validity,
    percentile,
)


STEP_TIME = "time_metrics/end_to_end(s)"
THROUGHPUT = "throughput(tps)"
ABLATION_FACTORS = (
    "graph_mode",
    "compile_components",
    "training_dtype",
    "mixed_precision_param",
    "mixed_precision_reduce",
    "extra_args",
)
LOWER_IS_BETTER = {
    "step_time_median_s",
    "step_time_p90_s",
    "step_time_p95_s",
    "step_time_p99_s",
    "step_time_cv_percent",
    "step_time_mad_s",
    "step_time_iqr_s",
    "peak_active_memory_gib",
}
DIAGNOSTIC_ONLY = {"step_time_drift_percent_per_100_steps"}
MEASUREMENT_LEVEL_MIN_REPEATS = {
    "exploratory": 1,
    "development": 3,
    "formal": 5,
    "release": 5,
}
MEASUREMENT_LEVEL_MIN_STEPS = {
    "exploratory": 1,
    "development": 20,
    "formal": 100,
    "release": 100,
}
MEASUREMENT_METADATA = {
    "skip_steps",
    "measured_steps",
    "first_step",
    "last_step",
}


def _percentile(values: list[float], fraction: float) -> float | None:
    return percentile(values, fraction)


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
    extra_args = [
        argument
        for argument in raw.get("extra_args", [])
        if not str(argument).startswith("--compile.")
    ]
    fixture = overview.get("fixture", raw.get("fixture"))
    fixture = fixture if isinstance(fixture, dict) else {}
    token_plan = fixture.get("token_plan")
    token_plan = token_plan if isinstance(token_plan, dict) else {}
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
        "workload": raw.get("workload", "representative"),
        "workload_args": raw.get("workload_args", []),
        "fixture_checkpoint_sha256": fixture.get("checkpoint_sha256"),
        "fixture_token_plan_sha256": token_plan.get("sha256"),
        "training_dtype": raw.get("training_dtype", "float32"),
        "mixed_precision_param": raw.get("mixed_precision_param", "bfloat16"),
        "mixed_precision_reduce": raw.get("mixed_precision_reduce", "float32"),
        "graph_mode": raw.get("graph_mode", "eager"),
        "compile_components": raw.get("compile_components", ["model"]),
        "deterministic": raw.get("deterministic", False),
        "telemetry_interval_seconds": raw.get("telemetry_interval_seconds"),
        # GraphFeatureConfig generates --compile.* arguments from graph_mode
        # and compile_components. Raw compile arguments are rejected by the
        # performance CLI, so removing generated duplicates cannot hide a
        # user-controlled contract difference.
        "extra_args": extra_args,
        "profiler_enabled": raw.get("profiler_enabled", True),
    }


def _execution_context(overview: dict[str, Any]) -> dict[str, Any]:
    raw = overview.get("configuration") or overview.get("contract") or {}
    return {
        "device": overview.get("device", raw.get("device")),
        "device_selection": overview.get("device_selection"),
        "codegen_backend": raw.get("npu_codegen"),
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
    step_statistics = metric_statistics(step_times)
    throughput_statistics = metric_statistics(throughputs)
    validity = numerical_validity(records)

    def aggregate(markers: tuple[str, ...], operation: str) -> float | None:
        name = _metric_name(series, markers)
        if name is None:
            return None
        values = series[name]
        return max(values) if operation == "max" else statistics.fmean(values)

    return {
        "run": str(run_dir.resolve()),
        "contract": contract,
        "execution_context": _execution_context(overview),
        "measurement": {
            "skip_steps": skip_steps,
            "measured_steps": len(records),
            "first_step": min(int(record["step"]) for record in records),
            "last_step": max(int(record["step"]) for record in records),
            "step_time_median_s": statistics.median(step_times),
            "step_time_p90_s": _percentile(step_times, 0.90),
            "step_time_p95_s": _percentile(step_times, 0.95),
            "step_time_p99_s": _percentile(step_times, 0.99),
            "step_time_cv_percent": step_statistics["cv_percent"],
            "step_time_mad_s": step_statistics["mad"],
            "step_time_iqr_s": step_statistics["iqr"],
            "step_time_drift_percent_per_100_steps": step_statistics[
                "drift_percent_per_100_steps"
            ],
            "throughput_median_tps": statistics.median(throughputs),
            "throughput_mean_tps": statistics.fmean(throughputs),
            "throughput_p95_tps": throughput_statistics["p95"],
            "tflops_mean": aggregate(("tflops",), "mean"),
            "mfu_mean_percent": aggregate(("mfu",), "mean"),
            "peak_active_memory_gib": aggregate(("max_active",), "max"),
            "validity": validity,
        },
        "series": {
            "step": [int(record["step"]) for record in records],
            "step_time_s": step_times,
            "throughput_tps": throughputs,
            "tflops": series.get(_metric_name(series, ("tflops",)) or "", []),
            "mfu_percent": series.get(_metric_name(series, ("mfu",)) or "", []),
            "active_memory_gib": series.get(
                _metric_name(series, ("max_active",)) or "", []
            ),
        },
    }


def _contract_without_steps(contract: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in contract.items() if key != "steps"}


def _validate_group(runs: list[dict[str, Any]], label: str) -> dict[str, Any]:
    expected = _contract_without_steps(runs[0]["contract"])
    expected_context = runs[0]["execution_context"]
    for run in runs[1:]:
        actual = _contract_without_steps(run["contract"])
        if actual != expected:
            raise ValueError(
                f"{label} repeat contract mismatch:\nexpected={expected}\nactual={actual}"
            )
        if run["execution_context"] != expected_context:
            raise ValueError(
                f"{label} repeat execution context mismatch:\n"
                f"expected={expected_context}\n"
                f"actual={run['execution_context']}"
            )
    return expected


def _aggregate_group(
    label: str, run_dirs: Iterable[Path], *, skip_steps: int
) -> dict[str, Any]:
    runs = [_run_summary(path.resolve(), skip_steps=skip_steps) for path in run_dirs]
    if not runs:
        raise ValueError(f"{label} must contain at least one run")
    expected_steps = runs[0]["series"]["step"]
    for run in runs[1:]:
        if run["series"]["step"] != expected_steps:
            raise ValueError(
                "profiler-off runs have different measured step windows; "
                "repeat comparison requires identical per-step evidence"
            )
    contract = _validate_group(runs, label)
    metric_names = tuple(
        name
        for name in runs[0]["measurement"]
        if name != "validity" and name not in MEASUREMENT_METADATA
    )
    aggregate: dict[str, Any] = {}
    for name in metric_names:
        values = [run["measurement"].get(name) for run in runs]
        numeric = [float(value) for value in values if isinstance(value, (int, float))]
        if not numeric:
            continue
        aggregate[name] = {
            **metric_statistics(numeric),
            "bootstrap_median_95ci": bootstrap_median_ci(numeric),
        }
    invalid_runs = [
        run["run"]
        for run in runs
        if run["measurement"]["validity"]["status"] == "invalid"
    ]
    unavailable_runs = [
        run["run"]
        for run in runs
        if run["measurement"]["validity"]["status"] == "not_available"
    ]
    if invalid_runs:
        validity = "invalid"
    elif unavailable_runs:
        validity = "not_available"
    else:
        validity = "valid"
    repeat_count = len(runs)
    tier = (
        "formal"
        if repeat_count >= 5
        else "development"
        if repeat_count >= 3
        else "insufficient"
    )
    return {
        "label": label,
        "contract": contract,
        "execution_context": runs[0]["execution_context"],
        "repeat_count": repeat_count,
        "measurement_policy": {
            "profiler_enabled": False,
            "skip_steps": skip_steps,
            "recommended_min_repeats": 3,
            "formal_min_repeats": 5,
            "repeat_count_sufficient": repeat_count >= 3,
            "formal_repeat_count_sufficient": repeat_count >= 5,
            "tier": tier,
            "validity": validity,
            "invalid_runs": invalid_runs,
            "validity_not_available_runs": unavailable_runs,
            "note": "MLPerf-style measurement discipline only; this is not an MLPerf result.",
        },
        "runs": runs,
        "aggregate": aggregate,
    }


def _contract_differences(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    keys = sorted(set(reference) | set(candidate))
    return {
        key: {"reference": reference.get(key), "candidate": candidate.get(key)}
        for key in keys
        if reference.get(key) != candidate.get(key)
    }


def _compare(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    ablation_factor: str | None = None,
) -> dict[str, Any]:
    differences = _contract_differences(
        reference["contract"], candidate["contract"]
    )
    if ablation_factor is None and differences:
        raise ValueError(
            "reference and candidate contracts differ; refusing performance comparison:\n"
            f"reference={reference['contract']}\ncandidate={candidate['contract']}"
        )
    if ablation_factor is not None:
        if ablation_factor not in ABLATION_FACTORS:
            raise ValueError(f"unsupported ablation factor: {ablation_factor}")
        if set(differences) != {ablation_factor}:
            raise ValueError(
                "ablation must change exactly its declared factor; "
                f"factor={ablation_factor!r}, differences={differences}"
            )
        if reference["execution_context"] != candidate["execution_context"]:
            raise ValueError(
                "ablation requires the same device and physical-device selection; "
                f"reference={reference['execution_context']}, "
                f"candidate={candidate['execution_context']}"
            )
    if (
        reference["runs"][0]["series"]["step"]
        != candidate["runs"][0]["series"]["step"]
    ):
        raise ValueError(
            "reference and candidate measured step windows differ; refusing "
            "per-step performance comparison"
        )
    metrics: dict[str, Any] = {}
    for name, reference_value in reference["aggregate"].items():
        candidate_value = candidate["aggregate"].get(name)
        if not candidate_value:
            continue
        baseline = reference_value["median"]
        current = candidate_value["median"]
        reference_ci = reference_value.get("bootstrap_median_95ci")
        candidate_ci = candidate_value.get("bootstrap_median_95ci")
        ci_relation = "not_available"
        verdict = "inconclusive"
        if reference_ci is not None and candidate_ci is not None:
            if candidate_ci[1] < reference_ci[0]:
                ci_relation = "candidate_lower"
                verdict = (
                    "improvement" if name in LOWER_IS_BETTER else "regression"
                )
            elif candidate_ci[0] > reference_ci[1]:
                ci_relation = "candidate_higher"
                verdict = "regression" if name in LOWER_IS_BETTER else "improvement"
            else:
                ci_relation = "overlap"
        if name in DIAGNOSTIC_ONLY:
            verdict = "diagnostic_only"
        metrics[name] = {
            "reference_median": baseline,
            "candidate_median": current,
            "candidate_vs_reference_percent": (
                (current / baseline - 1) * 100 if baseline else None
            ),
            "optimization_improvement_percent": (
                None
                if name in DIAGNOSTIC_ONLY
                else ((baseline - current) / baseline * 100)
                if baseline and name in LOWER_IS_BETTER
                else ((current - baseline) / baseline * 100)
                if baseline
                else None
            ),
            "better_direction": (
                "diagnostic"
                if name in DIAGNOSTIC_ONLY
                else "lower"
                if name in LOWER_IS_BETTER
                else "higher"
            ),
            "repeat_ranges_overlap": not (
                reference_value["max"] < candidate_value["min"]
                or candidate_value["max"] < reference_value["min"]
            ),
            "bootstrap_95ci_relation": ci_relation,
            "verdict": verdict,
        }
    headline_verdicts = {
        metrics[name]["verdict"]
        for name in ("step_time_median_s", "throughput_median_tps")
        if name in metrics
    }
    if headline_verdicts == {"improvement"}:
        overall_verdict = "improvement"
    elif headline_verdicts == {"regression"}:
        overall_verdict = "regression"
    else:
        overall_verdict = "inconclusive"
    return {
        "contract_match": True,
        "comparison_kind": "ablation" if ablation_factor else "peer",
        "ablation_factor": ablation_factor,
        "contract_differences": differences,
        "metrics": metrics,
        "overall_verdict": overall_verdict,
        "verdict_policy": (
            "Headline step-time and throughput bootstrap 95% intervals must "
            "both be separated and agree; overlap or disagreement is inconclusive."
        ),
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
    group_table = summary_table(
        columns=("Group", "Repeats", ">=3", "Topology", "Warmup skipped"),
        rows=tuple(
            (
                group["label"],
                str(group["repeat_count"]),
                "yes"
                if group["measurement_policy"]["repeat_count_sufficient"]
                else "no",
                str(group["contract"]["topology"]),
                str(group["measurement_policy"]["skip_steps"]),
            )
            for group in groups
        ),
    )
    metric_names = sorted(
        {name for group in groups for name in group["aggregate"]}
    )
    aggregate_columns = ["Metric"]
    for group in groups:
        aggregate_columns.extend(
            (
                f"{group['label']} median",
                f"{group['label']} CV",
                f"{group['label']} bootstrap 95% CI",
            )
        )
    if payload.get("comparison"):
        aggregate_columns.append("Candidate vs reference")
        aggregate_columns.append("Bootstrap verdict")
        if payload["comparison"].get("comparison_kind") == "ablation":
            aggregate_columns.append("Optimization improvement")
    aggregate_rows = []
    for name in metric_names:
        row = [name]
        for group in groups:
            values = group["aggregate"].get(name, {})
            row.extend(
                (
                    _format(values.get("median")),
                    f"{_format(values.get('cv_percent'))}%",
                    _format(values.get("bootstrap_median_95ci")),
                )
            )
        if payload.get("comparison"):
            relative_change = payload["comparison"]["metrics"].get(
                name, {}
            ).get("candidate_vs_reference_percent")
            row.append(
                f"{_format(relative_change)}%"
            )
            row.append(
                payload["comparison"]["metrics"].get(name, {}).get(
                    "verdict", "-"
                )
            )
            if payload["comparison"].get("comparison_kind") == "ablation":
                improvement = payload["comparison"]["metrics"].get(
                    name, {}
                ).get("optimization_improvement_percent")
                row.append(f"{_format(improvement)}%")
        aggregate_rows.append(tuple(row))
    aggregate_table = summary_table(
        columns=tuple(aggregate_columns), rows=tuple(aggregate_rows)
    )
    run_rows = [
        {
            "Group": group["label"],
            "Repeat": index,
            "Measured steps": (
                f"{run['measurement']['first_step']}-"
                f"{run['measurement']['last_step']}"
            ),
            "Median step time (s)": run["measurement"]["step_time_median_s"],
            "P90 step time (s)": run["measurement"]["step_time_p90_s"],
            "P95 step time (s)": run["measurement"]["step_time_p95_s"],
            "P99 step time (s)": run["measurement"]["step_time_p99_s"],
            "Step CV (%)": run["measurement"]["step_time_cv_percent"],
            "Step MAD (s)": run["measurement"]["step_time_mad_s"],
            "Step IQR (s)": run["measurement"]["step_time_iqr_s"],
            "Drift (%/100 steps)": run["measurement"][
                "step_time_drift_percent_per_100_steps"
            ],
            "Median throughput (tps)": run["measurement"][
                "throughput_median_tps"
            ],
            "Validity": run["measurement"]["validity"]["status"],
            "Path": display_repository_path(Path(run["run"])),
        }
        for group in groups
        for index, run in enumerate(group["runs"], start=1)
    ]
    sections: list[Any] = [
        section_heading(
            "证据来源与职责 / Evidence Provenance",
            (
                "训练与性能原始证据由 TorchTitan 指标、Ascend PyTorch "
                "Profiler、msProf 和 msprof-analyze 等官方工具产生；本报告只负责"
                "实验合同校验、重复聚合、派生统计与索引，不重新实现 profiler。"
            ),
        ),
        summary_table(
            columns=("Layer", "Owner", "Responsibility"),
            rows=(
                (
                    "Capture and diagnosis",
                    "Official Ascend ms tools",
                    "Timeline, operator, communication, memory, advisor evidence",
                ),
                (
                    "Experiment harness",
                    "torchtitan-test",
                    "Orchestration, identity, lifecycle, contract validation",
                ),
                (
                    "This report",
                    "torchtitan-test",
                    "Profiler-off repeat aggregation and derived effect sizes",
                ),
            ),
        ),
        section_heading(
            "实验合同与重复数 / Contract and Repeats",
            (
                "仅聚合 profiler-off 正常训练。当前测量等级："
                f"{payload['measurement_level']}；少于该等级要求的重复数时拒绝生成。"
            ),
        ),
        group_table,
        section_heading(
            "聚合指标 / Aggregate Metrics",
            "Step time 和显存越低越好；吞吐、TFLOPS、MFU 越高越好；CV 衡量重复稳定性。",
        ),
        aggregate_table,
    ]
    if payload.get("comparison"):
        sections.extend(
            (
                section_heading(
                    "置信区间结论 / Confidence-interval Verdict",
                    payload["comparison"]["verdict_policy"],
                ),
                summary_table(
                    columns=("Overall verdict", "Measurement level"),
                    rows=(
                        (
                            payload["comparison"]["overall_verdict"],
                            payload["measurement_level"],
                        ),
                    ),
                ),
            )
        )
    chart_specs = (
        ("step_time_s", "逐 Step 耗时 / Step Time", "秒 / Seconds"),
        ("throughput_tps", "逐 Step 吞吐 / Throughput", "Tokens/s"),
        ("tflops", "逐 Step TFLOPS", "TFLOPS"),
        ("mfu_percent", "逐 Step MFU", "MFU (%)"),
        ("active_memory_gib", "逐 Step 活跃显存 / Active Memory", "GiB"),
    )
    colors = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2")
    metric_charts = []
    for metric, title, y_name in chart_specs:
        chart_series = []
        x_values = None
        color_index = 0
        for group in groups:
            for repeat, run in enumerate(group["runs"], start=1):
                values = run["series"].get(metric, [])
                if not values:
                    continue
                x_values = run["series"]["step"]
                chart_series.append(
                    (
                        f"{group['label']} r{repeat}",
                        values,
                        colors[color_index % len(colors)],
                    )
                )
                color_index += 1
        if chart_series and x_values:
            metric_charts.append(
                echarts_line(
                    title=title,
                    subtitle="叠加全部重复运行；悬停读取点值，缩放检查长尾与漂移。",
                    x_values=x_values,
                    series=chart_series,
                    y_name=y_name,
                )
            )
    sections.extend(
        (
            section_heading(
                "逐 Step 重复性 / Per-step Repeatability",
                "不同 repeat 叠加后可直接观察启动偏差、稳态漂移和孤立长尾。",
            ),
            *metric_charts,
        )
    )
    if payload.get("comparison"):
        changes = payload["comparison"]["metrics"]
        if payload["comparison"].get("comparison_kind") == "ablation":
            difference = payload["comparison"]["contract_differences"]
            factor = payload["comparison"]["ablation_factor"]
            sections.extend(
                (
                    section_heading(
                        "消融合同 / Ablation Contract",
                        "框架已验证两组实验只改变声明的一个配置字段。",
                    ),
                    summary_table(
                        columns=("Factor", "Baseline", "Variant"),
                        rows=(
                            (
                                str(factor),
                                json.dumps(
                                    difference[factor]["reference"],
                                    ensure_ascii=False,
                                ),
                                json.dumps(
                                    difference[factor]["candidate"],
                                    ensure_ascii=False,
                                ),
                            ),
                        ),
                    ),
                )
            )
        sections.extend(
            (
                section_heading(
                    "候选相对基准变化 / Candidate vs Reference",
                    "正值表示候选数值更大；Step time/显存与吞吐/MFU 的好坏方向不同，不自动形成 PASS/FAIL。",
                ),
                echarts_line(
                    title="指标相对变化 / Relative Change",
                    subtitle="候选中位数相对 reference 中位数的百分比变化。",
                    x_values=list(changes),
                    series=[
                        (
                            "Candidate vs reference",
                            [
                                values.get("candidate_vs_reference_percent")
                                for values in changes.values()
                            ],
                            "#7c3aed",
                        )
                    ],
                    y_name="变化 / Change (%)",
                    x_name="指标 / Metric",
                    mark_lines=(("Zero baseline", 0.0, "#475569"),),
                    height=680,
                ),
            )
        )
    sections.extend(
        (
            section_heading(
                "逐次运行明细 / Run Inventory",
                "表格支持筛选和排序；路径用于回溯原始 metrics、日志与实验合同。",
            ),
            interactive_table(
                title="Profiler-off repeats",
                description="每一行是一份独立正常训练证据。",
                rows=run_rows,
                columns=(
                    "Group",
                    "Repeat",
                    "Measured steps",
                    "Median step time (s)",
                    "P90 step time (s)",
                    "P95 step time (s)",
                    "Median throughput (tps)",
                    "Path",
                ),
                pagination=False,
            ),
        )
    )
    save_panel_report(
        path=output / "comparison.html",
        title="Profiler-off 重复实验与性能对比",
        description="交互式离线证据：重复数不足三次不会伪造结论；采用 MLPerf 风格测量纪律，但不是 MLPerf 结果。",
        sections=sections,
    )


def build_comparison(
    *,
    reference_runs: list[Path],
    reference_label: str,
    candidate_runs: list[Path] | None,
    candidate_label: str,
    skip_steps: int,
    output: Path,
    ablation_factor: str | None = None,
    measurement_level: str = "exploratory",
    force: bool = False,
) -> dict[str, Any]:
    if measurement_level not in MEASUREMENT_LEVEL_MIN_REPEATS:
        raise ValueError(f"unsupported measurement level: {measurement_level}")
    minimum_repeats = MEASUREMENT_LEVEL_MIN_REPEATS[measurement_level]
    minimum_steps = MEASUREMENT_LEVEL_MIN_STEPS[measurement_level]
    if len(reference_runs) < minimum_repeats or (
        candidate_runs is not None and len(candidate_runs) < minimum_repeats
    ):
        raise ValueError(
            f"{measurement_level} measurement requires at least "
            f"{minimum_repeats} independent repeats per configured group"
        )
    if ablation_factor is not None:
        if not candidate_runs:
            raise ValueError("performance ablation requires candidate runs")
        if len(reference_runs) < 3 or len(candidate_runs) < 3:
            raise ValueError(
                "performance ablation requires at least three independent "
                "repeats on each side"
            )
    reference = _aggregate_group(reference_label, reference_runs, skip_steps=skip_steps)
    candidate = (
        _aggregate_group(candidate_label, candidate_runs, skip_steps=skip_steps)
        if candidate_runs
        else None
    )
    groups = [reference, *([candidate] if candidate is not None else [])]
    short_runs = [
        run["run"]
        for group in groups
        for run in group["runs"]
        if run["measurement"]["measured_steps"] < minimum_steps
    ]
    if short_runs:
        raise ValueError(
            f"{measurement_level} measurement requires at least {minimum_steps} "
            f"measured steady steps per run; short runs: {short_runs}"
        )
    if measurement_level in {"formal", "release"}:
        invalid_groups = [
            group["label"]
            for group in groups
            if group["measurement_policy"]["validity"] != "valid"
        ]
        if invalid_groups:
            raise ValueError(
                f"{measurement_level} measurement requires explicit finite "
                f"loss/gradient evidence; unavailable or invalid groups: {invalid_groups}"
            )
    if measurement_level == "release" and candidate is None:
        raise ValueError("release measurement requires reference and candidate groups")
    payload: dict[str, Any] = {
        "schema": "torchtitan.glm5_2.performance.comparison.v1",
        "measurement_level": measurement_level,
        "measurement_level_contract": {
            "minimum_repeats_per_group": minimum_repeats,
            "minimum_measured_steps_per_run": minimum_steps,
            "requires_numerical_validity": measurement_level
            in {"formal", "release"},
            "requires_candidate": measurement_level == "release",
        },
        "evidence_provenance": {
            "capture_and_diagnosis": "official Ascend ms tools",
            "harness": "torchtitan-test orchestration and contract validation",
            "report": "derived profiler-off aggregation; not a profiler",
        },
        "reference": reference,
        "candidate": candidate,
        "comparison": (
            _compare(
                reference,
                candidate,
                ablation_factor=ablation_factor,
            )
            if candidate
            else None
        ),
    }
    if (
        measurement_level == "release"
        and payload["comparison"]["overall_verdict"] == "inconclusive"
    ):
        raise ValueError(
            "release measurement is inconclusive: headline bootstrap 95% "
            "intervals overlap or step-time and throughput disagree"
        )
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
        f"- measurement level: `{measurement_level}`",
        (
            f"- ablation factor: `{ablation_factor}`"
            if ablation_factor
            else "- comparison kind: strict peer comparison"
        ),
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
    parser.add_argument(
        "--ablation-factor",
        choices=ABLATION_FACTORS,
        help=(
            "declare the one configuration field intentionally changed "
            "between reference and candidate"
        ),
    )
    parser.add_argument("--skip-steps", type=int, default=10)
    parser.add_argument(
        "--measurement-level",
        choices=tuple(MEASUREMENT_LEVEL_MIN_REPEATS),
        default="exploratory",
        help=(
            "enforce repeat and validity gates: exploratory=1, development=3, "
            "formal=5 finite runs, release=formal plus a conclusive candidate "
            "comparison; "
            "development requires 20 and formal/release 100 measured steady steps"
        ),
    )
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
        ablation_factor=args.ablation_factor,
        measurement_level=args.measurement_level,
        force=args.force,
    )
    for name in ("comparison.html", "comparison.json", "README.md"):
        print_output_path("Performance comparison output", args.output.resolve() / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
