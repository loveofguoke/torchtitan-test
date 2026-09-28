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
            else:
                row[f"signed_{metric}_difference"] = None
                row[f"absolute_{metric}_difference"] = None
                row[f"relative_{metric}_error"] = None
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
        if row.get("relative_norm_error") is not None
    ]
    return sorted(
        comparable,
        key=lambda row: float(row["relative_norm_error"]),
        reverse=True,
    )[:limit]


def _layer_charts(rows: list[dict[str, Any]], anomalies: list[dict[str, Any]]) -> list[Any]:
    if not anomalies:
        return []
    focus_step = str(anomalies[0]["step"])
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("step")) == focus_step and row.get("monitor") == "weight_grad":
            groups[(str(row["rank"]), str(row["vpp_stage"]), str(row["scope"]))].append(row)
    ranked = sorted(
        groups.items(),
        key=lambda item: max(
            (float(row.get("relative_norm_error") or 0.0) for row in item[1]),
            default=0.0,
        ),
        reverse=True,
    )
    charts: list[Any] = []
    for (rank, vpp_stage, scope), group in ranked[:16]:
        ordered = sorted(group, key=lambda row: int(row.get("layer_order", 0)))
        charts.append(
            echarts_line(
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
            )
        )
    return charts


def _rank_step_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("monitor") == "weight_grad" and row.get("relative_norm_error") is not None:
            groups[(str(row["rank"]), str(row["step"]))].append(row)
    summary: list[dict[str, Any]] = []
    for (rank, step), group in groups.items():
        worst = max(group, key=lambda row: float(row["relative_norm_error"]))
        scopes = {
            str(scope): max(
                (float(row["relative_norm_error"]) for row in group if row.get("scope") == scope),
                default=None,
            )
            for scope in {row.get("scope") for row in group}
        }
        unreduced = scopes.get("unreduced")
        reduced = scopes.get("reduced")
        if unreduced is not None and reduced is not None:
            localization = (
                "reduce 前已异常：优先检查反向计算、累积和参数层"
                if unreduced >= reduced * 0.5
                else "reduce 后明显放大：优先检查通信、规约后处理和裁剪"
            )
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
                "max_relative_norm_error": float(worst["relative_norm_error"]),
                "worst_parameter": worst["module_name"],
                "worst_scope": worst["scope"],
                "unreduced_max_error": unreduced,
                "reduced_max_error": reduced,
                "localization_hint": localization,
            }
        )
    return sorted(
        summary,
        key=lambda row: float(row["max_relative_norm_error"]),
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
            100.0 * float(row["max_relative_norm_error"]),
        )
        for row in summary
    ]
    return echarts_heatmap(
        title="Rank × Step 最大梯度 Norm 相对误差",
        subtitle=(
            "每个格子取该 rank、step 下所有参数与 reduced/unreduced scope 的最大相对误差；"
            "用于先发现局部尖刺，再进入下方逐层曲线。颜色是定位信号，不是通过阈值。"
        ),
        x_values=steps,
        y_values=ranks,
        values=values,
        value_name="最大相对误差 / %",
    )


def write_monitor_analysis(
    *,
    reference_official: Path,
    candidate_official: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Create a self-contained project analysis from official Monitor V2 CSVs."""

    reference_rows = read_monitor_rows(reference_official, "reference")
    candidate_rows = read_monitor_rows(candidate_official, "candidate")
    aligned = align_monitor_rows(reference_rows, candidate_rows)
    anomalies = _top_anomalies(aligned)
    rank_step_summary = _rank_step_summary(aligned)
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
        "focus_step": anomalies[0]["step"] if anomalies else None,
        "top_anomalies": anomalies,
        "rank_step_summary": rank_step_summary,
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
            "Step 与 Rank 定界 / Step and rank localization",
            "先确认异常是否只发生在单个 step/rank，还是跨 step/rank 持续传播。",
        ),
        *( [heatmap] if heatmap is not None else [] ),
        summary_table(
            columns=(
                "Step", "Rank", "最大误差", "最异常参数", "Scope", "Reduce 前后判断",
            ),
            rows=tuple(
                (
                    str(row["step"]),
                    str(row["rank"]),
                    f"{100.0 * float(row['max_relative_norm_error']):.4f}%",
                    str(row["worst_parameter"]),
                    str(row["worst_scope"]),
                    str(row["localization_hint"]),
                )
                for row in rank_step_summary[:50]
            ) or (("-", "-", "-", "没有可比较数据", "-", "-"),),
        ),
        section_heading(
            "自动提取异常 / Automatically extracted anomalies",
            "按梯度 norm 相对误差排序，用于选择异常 step、rank、scope 和参数。",
        ),
        summary_table(
            columns=(
                "Step", "Rank", "Scope", "Parameter", "GPU norm", "NPU norm", "Relative error",
            ),
            rows=tuple(
                (
                    str(row["step"]),
                    str(row["rank"]),
                    str(row["scope"]),
                    str(row["module_name"]),
                    f"{float(row['reference_norm']):.6g}",
                    f"{float(row['candidate_norm']):.6g}",
                    f"{100.0 * float(row['relative_norm_error']):.4f}%",
                )
                for row in anomalies
            ) or (("-", "-", "-", "没有可比较 norm", "-", "-", "-"),),
        ),
        section_heading(
            "异常 Step 逐层梯度 / Per-layer gradients at the anomalous step",
            "按官方 CSV 行顺序还原反向层顺序，分别查看 reduce 前后以及各 rank。",
        ),
        *_layer_charts(aligned, anomalies),
        section_heading(
            "完整对齐明细 / Full aligned Monitor table",
            "完整 GPU/NPU CSV 对齐结果已内嵌，可按任意列筛选、排序和分页；"
            "aligned_metrics.csv 同时保留便于脚本处理。",
        ),
        interactive_table(
            rows=aligned,
            columns=(
                "rank", "monitor", "step", "vpp_stage", "scope", "micro_step",
                "module_name", "match_status", "reference_norm", "candidate_norm",
                "signed_norm_difference", "relative_norm_error", "reference_mean",
                "candidate_mean", "reference_min", "candidate_min", "reference_max",
                "candidate_max", "reference_nans", "candidate_nans",
            ),
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
