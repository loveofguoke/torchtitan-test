# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Project-owned analysis and visualization for official Monitor V2 CSV files."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from tests.glm5_2_common.reporting import (
    echarts_heatmap,
    echarts_line,
    interactive_table,
    save_panel_report,
    section_heading,
    summary_table,
)

from .artifacts import write_json


IDENTITY_COLUMNS = (
    "rank",
    "monitor",
    "vpp_stage",
    "step",
    "module_name",
    "scope",
    "micro_step",
)
METRIC_COLUMNS = ("min", "max", "mean", "norm", "nans")
NEAR_ZERO_EPSILON = 1e-8


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _rank(path: Path) -> int:
    for part in path.parts:
        if part.startswith("rank_"):
            return int(part.removeprefix("rank_"))
    raise ValueError(f"Monitor CSV is not below rank_<id>: {path}")


def _monitor_kind(path: Path) -> str:
    name = path.stem
    marker = "_step"
    return name.split(marker, 1)[0] if marker in name else name


def read_monitor_rows(official_root: Path, role: str) -> list[dict[str, Any]]:
    """Read official Monitor V2 CSV files without changing their semantics."""

    rows: list[dict[str, Any]] = []
    for path in sorted(official_root.glob("rank_*/**/*.csv")):
        rank = _rank(path.relative_to(official_root))
        monitor = _monitor_kind(path)
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for order, raw in enumerate(csv.DictReader(stream)):
                row: dict[str, Any] = {
                    "role": role,
                    "rank": rank,
                    "monitor": monitor,
                    "source_file": path.relative_to(official_root).as_posix(),
                    "row_order": order,
                }
                row.update({str(key): value for key, value in raw.items()})
                rows.append(row)
    return rows


def _identity(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(column, "")) for column in IDENTITY_COLUMNS)


def align_monitor_rows(
    reference_rows: Iterable[dict[str, Any]],
    candidate_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Align GPU/NPU rows by rank, step, monitored object, and monitor scope."""

    reference = {_identity(row): row for row in reference_rows}
    candidate = {_identity(row): row for row in candidate_rows}
    aligned: list[dict[str, Any]] = []
    for identity in sorted(set(reference) | set(candidate)):
        golden = reference.get(identity)
        target = candidate.get(identity)
        row = dict(zip(IDENTITY_COLUMNS, identity, strict=True))
        row["match_status"] = (
            "matched" if golden is not None and target is not None
            else "reference_only" if golden is not None
            else "candidate_only"
        )
        row["reference_source"] = golden.get("source_file", "") if golden else ""
        row["candidate_source"] = target.get("source_file", "") if target else ""
        row["layer_order"] = (
            target.get("row_order", golden.get("row_order", 0))
            if target else golden.get("row_order", 0)
        )
        for metric in METRIC_COLUMNS:
            reference_value = _float(golden.get(metric)) if golden else None
            candidate_value = _float(target.get(metric)) if target else None
            row[f"reference_{metric}"] = reference_value
            row[f"candidate_{metric}"] = candidate_value
            if reference_value is not None and candidate_value is not None:
                signed = candidate_value - reference_value
                row[f"signed_{metric}_difference"] = signed
                row[f"absolute_{metric}_difference"] = abs(signed)
                row[f"relative_{metric}_error"] = abs(signed) / max(
                    abs(reference_value), 1e-12
                )
                scale = max(
                    abs(reference_value),
                    abs(candidate_value),
                    NEAR_ZERO_EPSILON,
                )
                row[f"scaled_{metric}_error"] = abs(signed) / scale
                row[f"reference_{metric}_near_zero"] = (
                    abs(reference_value) <= NEAR_ZERO_EPSILON
                )
            else:
                row[f"signed_{metric}_difference"] = None
                row[f"absolute_{metric}_difference"] = None
                row[f"relative_{metric}_error"] = None
                row[f"scaled_{metric}_error"] = None
                row[f"reference_{metric}_near_zero"] = None
        reference_norm = row.get("reference_norm")
        candidate_norm = row.get("candidate_norm")
        if reference_norm is not None and candidate_norm is not None:
            reference_zero = abs(float(reference_norm)) <= NEAR_ZERO_EPSILON
            candidate_zero = abs(float(candidate_norm)) <= NEAR_ZERO_EPSILON
            row["norm_zero_status"] = (
                "both_near_zero"
                if reference_zero and candidate_zero
                else "gpu_near_zero_npu_nonzero"
                if reference_zero
                else "npu_near_zero_gpu_nonzero"
                if candidate_zero
                else "comparable_scale"
            )
        else:
            row["norm_zero_status"] = "unmatched"
        aligned.append(row)
    return aligned


def _write_aligned_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else list(IDENTITY_COLUMNS)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _top_anomalies(rows: list[dict[str, Any]], limit: int = 50) -> list[dict[str, Any]]:
    comparable = [
        row for row in rows
        if row.get("scaled_norm_error") is not None
    ]
    return sorted(
        comparable,
        key=lambda row: (
            float(row["scaled_norm_error"]),
            float(row.get("absolute_norm_difference") or 0.0),
        ),
        reverse=True,
    )[:limit]


def _layer_charts(
    rows: list[dict[str, Any]],
    *,
    focus_step: str | None,
) -> list[Any]:
    if focus_step is None:
        return []
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("step")) == focus_step and row.get("monitor") == "weight_grad":
            groups[(str(row["rank"]), str(row["vpp_stage"]), str(row["scope"]))].append(row)
    pairs: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for (rank, vpp_stage, scope), group in groups.items():
        pairs[(rank, vpp_stage)][scope] = group
    ranked = sorted(
        pairs.items(),
        key=lambda item: max(
            (
                float(row.get("scaled_norm_error") or 0.0)
                for group in item[1].values()
                for row in group
            ),
            default=0.0,
        ),
        reverse=True,
    )
    charts: list[Any] = []
    for (rank, vpp_stage), scopes in ranked[:8]:
        ordered_scopes = [
            (scope, scopes[scope])
            for scope in ("unreduced", "reduced")
            if scope in scopes
        ]
        for scope, group in ordered_scopes:
            ordered = sorted(group, key=lambda row: int(row.get("layer_order", 0)))
            charts.append(echarts_line(
                title=(
                    f"异常 Step {focus_step}：Rank {rank} / VPP {vpp_stage} / {scope} "
                    "逐层梯度 Norm"
                ),
                subtitle=(
                    "横轴保持官方 CSV 的反向层顺序；对比 GPU 标杆与 NPU 调试侧，"
                    "用于观察异常是否集中在 embedding、输出层或某一连续层段。"
                ),
                x_values=[str(row["module_name"]) for row in ordered],
                series=(
                    (
                        "GPU / Reference",
                        [row.get("reference_norm") for row in ordered],
                        "#2563eb",
                    ),
                    (
                        "NPU / Candidate",
                        [row.get("candidate_norm") for row in ordered],
                        "#ea580c",
                    ),
                ),
                x_name="反向层/参数顺序 / Backward parameter order",
                y_name="梯度 Norm / Gradient norm",
                height=760,
            ))
        for scope, group in ordered_scopes:
            ordered = sorted(group, key=lambda row: int(row.get("layer_order", 0)))
            charts.append(echarts_line(
                title=(
                    f"异常 Step {focus_step}：Rank {rank} / VPP {vpp_stage} / {scope} "
                    "逐层梯度统计"
                ),
                subtitle=(
                    "Monitor 不保存完整梯度 Tensor；这里展示官方 CSV 中的 "
                    "min/mean/max 统计，用于观察分布范围和整体偏移。"
                ),
                x_values=[str(row["module_name"]) for row in ordered],
                series=tuple(
                    (
                        f"{role} {metric}",
                        [row.get(f"{prefix}_{metric}") for row in ordered],
                        color,
                    )
                    for role, prefix, colors in (
                        ("GPU", "reference", ("#1d4ed8", "#2563eb", "#60a5fa")),
                        ("NPU", "candidate", ("#c2410c", "#ea580c", "#fb923c")),
                    )
                    for metric, color in zip(("min", "mean", "max"), colors, strict=True)
                ),
                x_name="反向层/参数顺序 / Backward parameter order",
                y_name="梯度统计值 / Gradient statistic",
                height=760,
            ))
    return charts


def _overall_scope_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            row.get("monitor") == "weight_grad"
            and row.get("match_status") == "matched"
            and row.get("scope") in {"unreduced", "reduced"}
            and row.get("reference_norm") is not None
            and row.get("candidate_norm") is not None
        ):
            groups[(int(row["step"]), str(row["scope"]))].append(row)
    result: list[dict[str, Any]] = []
    for (step, scope), group in sorted(groups.items()):
        reference = math.sqrt(sum(float(row["reference_norm"]) ** 2 for row in group))
        candidate = math.sqrt(sum(float(row["candidate_norm"]) ** 2 for row in group))
        absolute = abs(candidate - reference)
        result.append({
            "step": step,
            "scope": scope,
            "gpu_aggregate_norm": reference,
            "npu_aggregate_norm": candidate,
            "absolute_difference": absolute,
            "scaled_error": absolute / max(reference, candidate, NEAR_ZERO_EPSILON),
            "parameter_rows": len(group),
        })
    return result


def _overall_scope_charts(rows: list[dict[str, Any]]) -> list[Any]:
    aggregate = _overall_scope_rows(rows)
    charts: list[Any] = []
    for scope, label in (
        ("unreduced", "Reduce 前 / Unreduced"),
        ("reduced", "Reduce 后 / Reduced"),
    ):
        selected = [row for row in aggregate if row["scope"] == scope]
        if not selected:
            continue
        charts.append(echarts_line(
            title=f"{label}：完整监控窗口梯度 Norm 汇总",
            subtitle=(
                "每个 step 对已匹配参数行做 sqrt(sum(parameter_norm^2))。"
                "该值用于观察趋势和尖刺；分片/复制参数可能使它不同于训练日志的全局 Grad Norm。"
            ),
            x_values=[str(row["step"]) for row in selected],
            series=(
                ("GPU / Reference", [row["gpu_aggregate_norm"] for row in selected], "#2563eb"),
                ("NPU / Candidate", [row["npu_aggregate_norm"] for row in selected], "#ea580c"),
            ),
            x_name="训练 Step / Training step",
            y_name="汇总梯度 Norm / Aggregated gradient norm",
            height=620,
        ))
    return charts


def _rank_step_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("monitor") == "weight_grad" and row.get("scaled_norm_error") is not None:
            groups[(str(row["rank"]), str(row["step"]))].append(row)
    summary: list[dict[str, Any]] = []
    for (rank, step), group in groups.items():
        worst = max(group, key=lambda row: float(row["scaled_norm_error"]))
        stable_scale_errors = sorted(
            float(row["scaled_norm_error"])
            for row in group
            if row.get("norm_zero_status") == "comparable_scale"
        )
        p95_index = max(0, math.ceil(0.95 * len(stable_scale_errors)) - 1)
        p95_error = (
            stable_scale_errors[p95_index] if stable_scale_errors else None
        )
        near_zero_mismatches = sum(
            row.get("norm_zero_status")
            in {"gpu_near_zero_npu_nonzero", "npu_near_zero_gpu_nonzero"}
            for row in group
        )
        paired_identity = (
            str(worst.get("monitor", "")),
            str(worst.get("vpp_stage", "")),
            str(worst.get("module_name", "")),
            str(worst.get("micro_step", "")),
        )
        paired = {
            str(row.get("scope")): row
            for row in group
            if (
                str(row.get("monitor", "")),
                str(row.get("vpp_stage", "")),
                str(row.get("module_name", "")),
                str(row.get("micro_step", "")),
            ) == paired_identity
        }
        unreduced_row = paired.get("unreduced")
        reduced_row = paired.get("reduced")
        unreduced = (
            float(unreduced_row["scaled_norm_error"])
            if unreduced_row is not None else None
        )
        reduced = (
            float(reduced_row["scaled_norm_error"])
            if reduced_row is not None else None
        )
        if unreduced is not None and reduced is not None:
            if reduced > max(unreduced * 2.0, 0.05):
                localization = "同参数 reduce 后误差明显放大：检查通信、规约后处理和裁剪"
            elif unreduced > max(reduced * 2.0, 0.05):
                localization = "同参数 reduce 前误差更大、规约后收敛：检查反向累积和分片"
            else:
                localization = "同参数 reduce 前后均有相近差异：优先检查上游反向计算"
        elif unreduced is not None:
            localization = "仅采集到 reduce 前梯度"
        elif reduced is not None:
            localization = "仅采集到 reduce 后梯度"
        else:
            localization = "scope 不含 reduced/unreduced"
        summary.append(
            {
                "rank": rank,
                "step": step,
                "max_scaled_norm_error": float(worst["scaled_norm_error"]),
                "p95_scaled_norm_error_excluding_near_zero": p95_error,
                "near_zero_mismatch_count": near_zero_mismatches,
                "max_absolute_norm_difference": float(
                    worst.get("absolute_norm_difference") or 0.0
                ),
                "worst_reference_norm": worst.get("reference_norm"),
                "worst_candidate_norm": worst.get("candidate_norm"),
                "worst_zero_status": worst.get("norm_zero_status"),
                "reference_norm_near_zero": bool(
                    worst.get("reference_norm_near_zero")
                ),
                "worst_parameter": worst["module_name"],
                "worst_scope": worst["scope"],
                "unreduced_max_error": unreduced,
                "reduced_max_error": reduced,
                "localization_hint": localization,
            }
        )
    return sorted(
        summary,
        key=lambda row: (
            float(row["max_scaled_norm_error"]),
            float(row["max_absolute_norm_difference"]),
        ),
        reverse=True,
    )


def _rank_step_heatmap(rows: list[dict[str, Any]]) -> Any | None:
    summary = _rank_step_summary(rows)
    if not summary:
        return None
    steps = sorted({str(row["step"]) for row in summary}, key=lambda value: int(value))
    ranks = sorted({str(row["rank"]) for row in summary}, key=lambda value: int(value))
    x_index = {value: index for index, value in enumerate(steps)}
    y_index = {value: index for index, value in enumerate(ranks)}
    values = [
        (
            x_index[str(row["step"])],
            y_index[str(row["rank"])],
            100.0 * float(
                row["p95_scaled_norm_error_excluding_near_zero"] or 0.0
            ),
        )
        for row in summary
    ]
    return echarts_heatmap(
        title="Rank × Step 梯度 Norm P95 尺度化误差",
        subtitle=(
            "每个格子统计该 rank、step 下非近零参考项的 P95 尺度化误差；"
            "近零不匹配单独计数，避免所有格子被少数除零项顶到 100%。"
        ),
        x_values=steps,
        y_values=ranks,
        values=values,
        value_name="P95 尺度化误差 / %",
    )


def write_monitor_analysis(
    *,
    reference_official: Path,
    candidate_official: Path,
    output_directory: Path,
    monitor_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a self-contained project analysis from official Monitor V2 CSVs."""

    reference_rows = read_monitor_rows(reference_official, "reference")
    candidate_rows = read_monitor_rows(candidate_official, "candidate")
    aligned = align_monitor_rows(reference_rows, candidate_rows)
    anomalies = _top_anomalies(aligned)
    rank_step_summary = _rank_step_summary(aligned)
    overall_scope_rows = _overall_scope_rows(aligned)
    reduced_scope_rows = [
        row for row in overall_scope_rows if row["scope"] == "reduced"
    ]
    focus_step = (
        str(max(
            reduced_scope_rows,
            key=lambda row: (row["scaled_error"], row["absolute_difference"]),
        )["step"])
        if reduced_scope_rows
        else str(anomalies[0]["step"])
        if anomalies
        else None
    )
    focus_rows = [row for row in aligned if str(row.get("step")) == focus_step]
    focus_aggregate = [
        row for row in overall_scope_rows if str(row["step"]) == focus_step
    ]
    matched = sum(row["match_status"] == "matched" for row in aligned)
    reference_only = sum(row["match_status"] == "reference_only" for row in aligned)
    candidate_only = sum(row["match_status"] == "candidate_only" for row in aligned)
    nonfinite_rows = sum(
        float(row.get("reference_nans") or 0) > 0
        or float(row.get("candidate_nans") or 0) > 0
        for row in aligned
    )
    summary = {
        "schema": "torchtitan.glm5_2.monitor_analysis",
        "schema_version": 1,
        "policy": "project_derived_diagnostic_without_official_verdict",
        "verdict": "diagnostic-only",
        "status_counts": {},
        "reference_rows": len(reference_rows),
        "candidate_rows": len(candidate_rows),
        "matched_rows": matched,
        "reference_only_rows": reference_only,
        "candidate_only_rows": candidate_only,
        "rows_with_reported_nans": nonfinite_rows,
        "focus_step": focus_step,
        "focus_scope_aggregate": focus_aggregate,
        "top_anomalies": anomalies,
        "rank_step_summary": rank_step_summary,
        "monitor_config": monitor_config or {},
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_aligned_csv(output_directory / "aligned_metrics.csv", aligned)
    _write_aligned_csv(output_directory / "anomaly_summary.csv", anomalies)
    _write_aligned_csv(output_directory / "rank_step_summary.csv", rank_step_summary)
    write_json(output_directory / "analysis.json", summary)

    heatmap = _rank_step_heatmap(aligned)
    sections: list[Any] = [
        section_heading(
            "Monitor 覆盖与匹配 / Coverage and alignment",
            "CSV 是 msProbe Monitor V2 官方输出；本页面的跨端对齐、误差和图表由项目生成，"
            "只用于定位，不构成官方 PASS/FAIL 判定。",
        ),
        summary_table(
            columns=("项目 / Item", "数值 / Value", "含义 / Interpretation"),
            rows=(
                ("GPU rows", str(len(reference_rows)), "GPU 官方 CSV 行数"),
                ("NPU rows", str(len(candidate_rows)), "NPU 官方 CSV 行数"),
                ("Matched", str(matched), "rank/step/module/scope 均匹配"),
                ("GPU only", str(reference_only), "仅 GPU 出现，需检查结构或采集覆盖"),
                ("NPU only", str(candidate_only), "仅 NPU 出现，需检查结构或采集覆盖"),
                ("NaN rows", str(nonfinite_rows), "两端 nans 指标非零的匹配行"),
            ),
        ),
        section_heading(
            "Monitor 采集配置 / Capture configuration",
            "这是本次 TrainerMonitorV2 的实际配置。weight_grad 同时产生 "
            "unreduced（reduce 前）与 reduced（reduce 后、optimizer.step 前）统计。",
        ),
        summary_table(
            columns=("配置项 / Option", "值 / Value"),
            rows=tuple(
                (str(key), json.dumps(value, ensure_ascii=False))
                for key, value in (monitor_config or {}).items()
            ) or (("配置", "未记录"),),
        ),
        section_heading(
            "Reduce 前后总体梯度 / Overall unreduced and reduced gradients",
            "先分别查看完整监控窗口的 reduce 前与 reduce 后梯度汇总趋势，再进入异常 step、rank 和参数。",
        ),
        *_overall_scope_charts(aligned),
        interactive_table(
            rows=overall_scope_rows,
            columns=(
                "step", "scope", "gpu_aggregate_norm", "npu_aggregate_norm",
                "absolute_difference", "scaled_error", "parameter_rows",
            ),
            pagination=False,
        ),
        section_heading(
            "Step 与 Rank 定界 / Step and rank localization",
            "先确认异常是否只发生在单个 step/rank，还是跨 step/rank 持续传播。",
        ),
        *( [heatmap] if heatmap is not None else [] ),
        interactive_table(
            rows=rank_step_summary,
            columns=(
                "step", "rank", "max_scaled_norm_error",
                "p95_scaled_norm_error_excluding_near_zero",
                "near_zero_mismatch_count",
                "max_absolute_norm_difference", "reference_norm_near_zero",
                "worst_reference_norm", "worst_candidate_norm",
                "worst_zero_status",
                "worst_parameter", "worst_scope", "unreduced_max_error",
                "reduced_max_error", "localization_hint",
            ),
            pagination=False,
        ),
        section_heading(
            f"异常 Step {focus_step or '-'} 证据 / Focus-step evidence",
            "该 step 由 reduced 汇总曲线的最大尺度化差异自动选出。先看 reduce 前后汇总，"
            "再按 rank、scope、参数和近零类型筛选明细。",
        ),
        summary_table(
            columns=("Step", "Scope", "GPU 汇总 Norm", "NPU 汇总 Norm", "绝对差", "尺度化误差"),
            rows=tuple(
                (
                    str(row["step"]), str(row["scope"]),
                    f"{float(row['gpu_aggregate_norm']):.8g}",
                    f"{float(row['npu_aggregate_norm']):.8g}",
                    f"{float(row['absolute_difference']):.8g}",
                    f"{100.0 * float(row['scaled_error']):.4f}%",
                )
                for row in focus_aggregate
            ) or (("-", "-", "-", "-", "-", "-"),),
        ),
        interactive_table(
            rows=focus_rows,
            columns=(
                "rank", "step", "scope", "module_name", "reference_norm",
                "candidate_norm", "absolute_norm_difference", "scaled_norm_error",
                "norm_zero_status", "reference_min", "candidate_min",
                "reference_mean", "candidate_mean", "reference_max", "candidate_max",
            ),
            pagination=False,
        ),
        section_heading(
            "自动提取异常 / Automatically extracted anomalies",
            "按尺度化梯度 norm 误差排序；GPU norm 接近零时同时查看绝对误差，"
            "避免把除以近零值产生的巨大百分比误判为同等严重的问题。",
        ),
        interactive_table(
            rows=anomalies,
            columns=(
                "step", "rank", "scope", "module_name", "reference_norm",
                "candidate_norm", "signed_norm_difference",
                "absolute_norm_difference", "scaled_norm_error",
                "reference_norm_near_zero", "relative_norm_error",
            ),
            pagination=False,
        ),
        section_heading(
            "异常 Step 逐层梯度 / Per-layer gradients at the anomalous step",
            "按官方 CSV 行顺序还原反向层顺序，分别查看 reduce 前后以及各 rank。",
        ),
        *_layer_charts(aligned, focus_step=focus_step),
        section_heading(
            "完整对齐明细 / Full aligned Monitor table",
            "完整 GPU/NPU CSV 对齐结果已连续内嵌，不分页；可在列头按 rank、step、"
            "scope、数值大小等任意列筛选和排序；"
            "aligned_metrics.csv 同时保留便于脚本处理。",
        ),
        interactive_table(
            rows=aligned,
            columns=(
                "rank", "monitor", "step", "vpp_stage", "scope", "micro_step",
                "module_name", "match_status", "reference_norm", "candidate_norm",
                "signed_norm_difference", "absolute_norm_difference",
                "scaled_norm_error", "reference_norm_near_zero",
                "norm_zero_status", "relative_norm_error", "reference_mean",
                "candidate_mean", "reference_min", "candidate_min", "reference_max",
                "candidate_max", "reference_nans", "candidate_nans",
            ),
            pagination=False,
        ),
    ]
    report = save_panel_report(
        path=output_directory / "monitor_report.html",
        title="GPU / NPU Monitor 梯度诊断",
        description=(
            "基于 msProbe Monitor V2 官方 CSV 的项目派生交互分析。悬停查看数值，缩放选择层段，"
            "并结合 aligned_metrics.csv 回溯所有原始匹配行。"
        ),
        sections=sections,
    )
    summary["report"] = report.name
    summary["analysis_files"] = [
        "aligned_metrics.csv",
        "anomaly_summary.csv",
        "rank_step_summary.csv",
        "analysis.json",
        report.name,
    ]
    write_json(output_directory / "analysis.json", summary)
    return summary
