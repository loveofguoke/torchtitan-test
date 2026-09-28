# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Interactive whole-training observation report."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

from tests.glm5_2_common.reporting import (
    echarts_line,
    save_panel_report,
    section_heading,
    summary_table,
)


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4%}"


def _number(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6g}"


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
    return sum(values) / len(values) if values else None


def _values(rows: Sequence[dict[str, Any]], key: str, *, percent: bool = False) -> list[float | None]:
    scale = 100.0 if percent else 1.0
    return [float(row[key]) * scale if math.isfinite(float(row[key])) else None for row in rows]


def write_observation_report(
    *,
    output_directory: Path,
    rows: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> Path:
    loss = summary["loss"]
    grad = summary["grad_norm"]
    early = loss["early_window"]
    first_exceeded = loss["first_step_above_threshold"]
    nonfinite = summary["observation"]["nonfinite_analysis"]
    steps = [int(row["step"]) for row in rows]
    early_rows = [
        row
        for row in rows
        if early["first_step"] <= row["step"] <= early["last_step"]
    ]
    early_steps = [int(row["step"]) for row in early_rows]
    areas: list[tuple[str, int, int, str]] = []
    if early["first_step"] is not None:
        areas.append(
            (
                "首 Steps 检查窗口 / First-step inspection window",
                early["first_step"],
                early["last_step"],
                "#2563eb",
            )
        )
    if first_exceeded is not None:
        areas.append(
            (
                "首次超限后窗口 / After first guidance exceedance",
                first_exceeded,
                steps[-1],
                "#f59e0b",
            )
        )
    loss_threshold_pct = float(loss["guidance_relative_threshold"]) * 100.0
    grad_threshold = grad["diagnostic_relative_threshold"]
    overview = summary_table(
        columns=("Metric", "Value", "Interpretation"),
        rows=(
            (
                "Compared step window",
                f"{steps[0]}-{steps[-1]} ({summary['step_count']} steps)",
                "The common GPU/NPU observation interval.",
            ),
            (
                "Loss mean relative error",
                _percent(loss["mean_relative_error"]),
                f"Compared with {loss_threshold_pct:g}% diagnostic guidance.",
            ),
            (
                "First Loss guidance exceedance",
                "none" if first_exceeded is None else f"step {first_exceeded}",
                "A localization trigger, not an automatic delivery verdict.",
            ),
            (
                "Grad Norm mean relative error",
                _percent(grad["mean_relative_error"]),
                "Use together with the median and largest-error steps below.",
            ),
            (
                "GPU NaN/Inf metrics",
                str(nonfinite["reference"]["metric_count"]),
                "Number of monitored metrics containing a non-finite value.",
            ),
            (
                "NPU NaN/Inf metrics",
                str(nonfinite["candidate"]["metric_count"]),
                "Number of monitored metrics containing a non-finite value.",
            ),
        ),
    )
    nonfinite_rows = []
    for endpoint, label in (("reference", "GPU 标杆 / Reference"), ("candidate", "NPU 调试 / Candidate")):
        result = nonfinite[endpoint]
        details = "; ".join(
            f"{item['metric']}: {item['kind']} at step {item['first_step']}"
            for item in result["metrics"]
        ) or "未发现 / None observed"
        nonfinite_rows.append(
            (
                label,
                str(result["metric_count"]),
                "-" if result["first_step"] is None else str(result["first_step"]),
                details,
            )
        )
    nonfinite_table = summary_table(
        columns=("端点 / Endpoint", "异常指标数 / Count", "首次 Step", "明细 / Details"),
        rows=tuple(nonfinite_rows),
    )
    grad_largest = grad.get("largest_relative_error_steps", [])
    grad_anomaly_table = summary_table(
        columns=(
            "Step",
            "GPU Grad Norm",
            "NPU Grad Norm",
            "相对误差 / Relative error",
            "定位提示 / Diagnostic note",
        ),
        rows=tuple(
            (
                str(item["step"]),
                _number(item["reference"]),
                _number(item["candidate"]),
                _percent(item["relative_error"]),
                "检查原始值是否接近 0，并对照相邻 step 的 Loss、梯度和更新。",
            )
            for item in grad_largest
        ),
    )
    max_grad_step = grad.get("max_relative_error_step")
    max_grad_error = grad.get("max_relative_error")
    max_grad_points = (
        (("最大相对误差", max_grad_step, float(max_grad_error) * 100.0, "#dc2626"),)
        if max_grad_step is not None and max_grad_error is not None
        else ()
    )
    loss_charts = [
        echarts_line(
            title="训练 Loss 全程对比 / Training Loss",
            subtitle=f"平均相对误差 {_percent(loss['mean_relative_error'])}; 首次超过指导线: {first_exceeded if first_exceeded is not None else '无'}",
            x_values=steps,
            series=[
                ("GPU reference", _values(rows, "reference_loss"), "#2563eb"),
                ("NPU candidate", _values(rows, "candidate_loss"), "#dc2626"),
            ],
            y_name="Loss / 损失",
            mark_areas=areas,
        ),
        echarts_line(
            title="首 Steps Loss 对比 / First Steps Loss",
            subtitle=f"仅显示 step {early['first_step']}..{early['last_step']}，不混入后续训练窗口",
            x_values=early_steps,
            series=[
                ("GPU 标杆 / Reference", _values(early_rows, "reference_loss"), "#2563eb"),
                ("NPU 调试 / Candidate", _values(early_rows, "candidate_loss"), "#dc2626"),
            ],
            y_name="Loss / 损失",
        ),
        echarts_line(
            title="首 Steps Loss 相对误差 / First Steps Relative Error",
            subtitle=f"窗口 {early['first_step']}..{early['last_step']}: 均值 {_percent(early['mean_relative_error'])}, 最大值 {_percent(early['max_relative_error'])}",
            x_values=early_steps,
            series=[("Loss 相对误差", _values(early_rows, "loss_relative_error", percent=True), "#7c3aed")],
            y_name="相对误差 / Relative error (%)",
            mark_lines=[("Zero baseline", 0.0, "#475569"), (f"Guidance {loss_threshold_pct:g}%", loss_threshold_pct, "#f59e0b")],
        ),
        echarts_line(
            title="全程 Loss 相对误差 / Whole-training Relative Error",
            subtitle=f"首次超限后均值 {_percent(loss['post_first_threshold_window']['mean_relative_error']) if first_exceeded is not None else 'N/A'}",
            x_values=steps,
            series=[("Loss relative error", _values(rows, "loss_relative_error", percent=True), "#7c3aed")],
            y_name="相对误差 / Relative error (%)",
            mark_lines=[("Zero baseline", 0.0, "#475569"), (f"Guidance {loss_threshold_pct:g}%", loss_threshold_pct, "#f59e0b")],
            mark_areas=areas,
        ),
        echarts_line(
            title="Loss 有符号差值 / Signed Difference",
            subtitle=f"平均有符号差值 {_number(_mean(rows, 'loss_signed_difference'))}; NPU - GPU",
            x_values=steps,
            series=[("Signed difference", _values(rows, "loss_signed_difference"), "#7c3aed")],
            y_name="NPU - GPU / 差值",
            mark_lines=[("Zero baseline", 0.0, "#475569")],
        ),
    ]
    grad_charts = [
        echarts_line(
            title="梯度范数对比 / Gradient Norm",
            subtitle=f"平均相对误差 {_percent(grad['mean_relative_error'])}; 中位数 {_percent(grad.get('median_relative_error'))}",
            x_values=steps,
            series=[
                ("GPU reference", _values(rows, "reference_grad_norm"), "#2563eb"),
                ("NPU candidate", _values(rows, "candidate_grad_norm"), "#dc2626"),
            ],
            y_name="L2 Norm / 范数",
        ),
        echarts_line(
            title="梯度范数相对误差 / Grad Norm Relative Error",
            subtitle=f"最大值 {_percent(max_grad_error)}，位于 step {max_grad_step}; 极值是定位信号，不由均值掩盖",
            x_values=steps,
            series=[("Grad Norm relative error", _values(rows, "grad_norm_relative_error", percent=True), "#059669")],
            y_name="相对误差 / Relative error (%)",
            mark_lines=[("Zero baseline", 0.0, "#475569")] + ([("Configured guidance", float(grad_threshold) * 100.0, "#f59e0b")] if grad_threshold is not None else []),
            mark_points=max_grad_points,
        ),
        echarts_line(
            title="梯度范数有符号差值 / Grad Norm Signed Difference",
            subtitle=f"平均有符号差值 {_number(_mean(rows, 'grad_norm_signed_difference'))}; 零线用于观察持续偏斜",
            x_values=steps,
            series=[("Signed difference", _values(rows, "grad_norm_signed_difference"), "#059669")],
            y_name="NPU - GPU / 差值",
            mark_lines=[("Zero baseline", 0.0, "#475569")],
        ),
        echarts_line(
            title="梯度范数有符号相对误差 / Signed Relative Error",
            subtitle=f"平均有符号相对误差 {_percent(grad['mean_signed_relative_error'])}; ±5% 仅为诊断指导线",
            x_values=steps,
            series=[("Signed relative error", _values(rows, "grad_norm_signed_relative_error", percent=True), "#059669")],
            y_name="(GPU - NPU) / GPU (%)",
            mark_lines=[("Zero baseline", 0.0, "#475569"), ("Upper guidance +5%", 5.0, "#f59e0b"), ("Lower guidance -5%", -5.0, "#f59e0b")],
        ),
    ]
    return save_panel_report(
        path=output_directory / "training_observation.html",
        title="GPU / NPU 训练观察（Training Observation）",
        description="交互式离线证据：悬停查看精确值，框选或滚轮缩放，拖动平移，切换曲线，并可从工具箱查看数据或导出图片。",
        sections=[
            section_heading("总体摘要 / Overview", "先看现象分类，再进入分支定位；表中阈值是诊断指导，不自动等同于交付结论。"),
            overview,
            section_heading("NaN / Inf 与溢出检查", "标准流程首先检查两端所有已记录数值指标是否出现非有限值；即使未发现，也明确记录为零。"),
            nonfinite_table,
            section_heading("Loss 对齐分析", "依次查看全程曲线、真正截取的首 Steps 窗口、全程相对误差和有符号差值。"),
            *loss_charts,
            section_heading("Grad Norm 对齐与异常点", "均值可能掩盖孤立极值。下表列出相对误差最大的 step；异常点需结合原始范数、相邻 Loss 和参数更新继续定位。"),
            grad_anomaly_table,
            *grad_charts,
        ],
    )
