"""Interactive training-metric reports for performance experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tests.glm5_2_common.reporting import (
    echarts_line,
    interactive_table,
    save_panel_report,
    section_heading,
    summary_table,
)


def _metric(
    values: dict[str, Any], fragment: str
) -> tuple[str, Any] | None:
    return next(
        (
            (name, metric_values)
            for name, metric_values in values.items()
            if fragment in name.lower()
        ),
        None,
    )


def _number(value: Any, suffix: str = "") -> str:
    return "N/A" if value is None else f"{float(value):.6g}{suffix}"


def write_training_metrics_report(
    *,
    manifest: dict[str, Any],
    analysis: dict[str, Any],
    output_path: Path,
) -> Path:
    """Write one self-contained interactive baseline/localization report."""

    config = manifest["config"]
    metrics = analysis.get("metrics", {})
    summary = metrics.get("summary", {})
    series = metrics.get("series", {})
    steps = list(metrics.get("steps", []))
    profiler_enabled = bool(config.get("profiler_enabled", True))
    skip_steps = int(config.get("skip_steps", 0))
    warmup_area = (
        (
            (
                "启动与暖机 / Startup and warmup",
                steps[0],
                min(skip_steps, steps[-1]),
                "#f59e0b",
            ),
        )
        if steps and skip_steps >= steps[0]
        else ()
    )
    summary_specs = (
        ("end_to_end", "Median step time", "median", " s", "越低越好；正式结论看暖机后的稳态窗口。"),
        ("throughput", "Mean throughput", "mean", " tok/s", "越高越好；确认是每设备还是整个作业口径。"),
        ("tflops", "Mean TFLOPS", "mean", "", "有效计算吞吐，需结合 MFU 和算子结构解释。"),
        ("mfu", "Mean MFU", "mean", "%", "模型 FLOPs 与设备峰值的比值，不等于设备利用率。"),
        (
            "max_active",
            "Peak active memory",
            "max",
            " GiB",
            "用于容量判断；碎片仍需 Memory Timeline。",
        ),
    )
    summary_rows = []
    for fragment, label, statistic, suffix, note in summary_specs:
        found = _metric(summary, fragment)
        if found:
            summary_rows.append(
                (label, _number(found[1].get(statistic), suffix), note)
            )
    phase_rows = []
    phases = analysis.get("profile_phases", {})
    for name in phases.get("phase_order", []):
        phase = phases["phases"][name]
        phase_steps = phase.get("steps", [])
        phase_summary = phase.get("summary", {})
        step_time = _metric(phase_summary, "end_to_end")
        throughput = _metric(phase_summary, "throughput")
        phase_rows.append(
            (
                name,
                "-" if not phase_steps else f"{phase_steps[0]}-{phase_steps[-1]}",
                _number(step_time[1].get("median") if step_time else None, " s"),
                _number(throughput[1].get("median") if throughput else None, " tok/s"),
                "性能基线" if name == "steady" and not profiler_enabled else "诊断阶段",
            )
        )
    charts = []
    chart_specs = (
        ("end_to_end", "训练 Step 耗时 / Step Time", "秒 / Seconds", "#2563eb"),
        ("throughput", "训练吞吐 / Throughput", "Tokens/s", "#059669"),
        ("tflops", "有效计算吞吐 / TFLOPS", "TFLOPS", "#7c3aed"),
        ("mfu", "模型 FLOPs 利用率 / MFU", "MFU (%)", "#dc2626"),
        ("max_active", "活跃显存 / Active Memory", "GiB", "#d97706"),
    )
    for fragment, title, y_name, color in chart_specs:
        found = _metric(series, fragment)
        if not found:
            continue
        points = found[1]
        charts.append(
            echarts_line(
                title=title,
                subtitle=(
                    f"{found[0]}；悬停查看精确值，框选或滚轮缩放。"
                    + (
                        "橙色区域为启动与暖机，不作为稳态性能结论。"
                        if warmup_area
                        else ""
                    )
                ),
                x_values=[int(step) for step, _ in points],
                series=[(found[0], [float(value) for _, value in points], color)],
                y_name=y_name,
                mark_areas=warmup_area,
            )
        )
    diagnosis = analysis.get("self_diagnosis", {})
    diagnosis_rows = [
        {
            "Branch": key,
            "Status": branch.get("status"),
            "Summary": branch.get("summary"),
            "Evidence": json.dumps(branch.get("evidence", []), ensure_ascii=False),
            "Next actions": "; ".join(branch.get("next_actions", [])),
        }
        for key, branch in diagnosis.get("branches", {}).items()
    ]
    tool_rows = [
        {
            "Type": item.get("type", "output"),
            "Role": item.get("role", "evidence"),
            "Path": item.get("path", ""),
            "How to inspect": item.get("inspect", ""),
        }
        for item in analysis.get("tool_outputs", [])
    ]
    sections: list[Any] = [
        section_heading(
            "执行合同 / Execution Contract",
            "执行模式和代码生成后端属于实验身份；不同合同不会复用同一目录。",
        ),
        summary_table(
            columns=("Field", "Value", "Meaning"),
            rows=(
                ("Graph mode", config.get("graph_mode", "eager"), "eager、Inductor 或 NPU Graphs"),
                ("Compile components", ", ".join(config.get("compile_components", ("model",))), "进入图编译的训练组件"),
                ("NPU codegen", config.get("npu_codegen") or "installed default", "DVM 或 Ascend Triton 代码生成后端"),
                ("Compiler diagnostics", str(bool(config.get("compiler_diagnostics", False))), "graph break/recompile/dynamic shape 日志"),
            ),
        ),
        section_heading(
            "总体摘要 / Overview",
            "Profiler-off 用于性能数值结论；Profiler-active 只用于定位原因。",
        ),
        summary_table(
            columns=("Metric", "Value", "Interpretation"),
            rows=tuple(summary_rows),
        ),
    ]
    if phase_rows:
        sections.extend(
            (
                section_heading(
                    "训练阶段 / Measurement Windows",
                    "明确区分启动、暖机、稳态和采集阶段，避免把编译或 Profiler 开销算进基线。",
                ),
                summary_table(
                    columns=(
                        "Phase",
                        "Steps",
                        "Median step",
                        "Median throughput",
                        "Use",
                    ),
                    rows=tuple(phase_rows),
                ),
            )
        )
    sections.extend(
        (
            section_heading(
                "逐 Step 交互图 / Interactive Step Metrics",
                "所有图均支持悬停、缩放、平移、图例切换、数据查看和图片导出。",
            ),
            *charts,
        )
    )
    if diagnosis_rows:
        sections.extend(
            (
                section_heading(
                    "单拓扑诊断 / Self Diagnosis",
                    "诊断状态用于选择下一步工具，不是自动 PASS/FAIL。",
                ),
                interactive_table(
                    title="诊断分支",
                    description="按状态、分支或证据筛选；详细证据保留原始 JSON。",
                    rows=diagnosis_rows,
                    columns=("Branch", "Status", "Summary", "Evidence", "Next actions"),
                    pagination=False,
                ),
            )
        )
    if tool_rows:
        sections.extend(
            (
                section_heading(
                    "官方工具产物 / Native Tool Outputs",
                    "项目报告负责索引和解释；时间线、数据库与原生报告仍由对应官方工具打开。",
                ),
                interactive_table(
                    title="原生证据清单 / Native evidence inventory",
                    description="路径保留到原始采集、聚合统计和日志，便于从结论回溯证据。",
                    rows=tool_rows,
                    columns=("Type", "Role", "Path", "How to inspect"),
                    pagination=False,
                ),
            )
        )
    mode = "Profiler-active 归因" if profiler_enabled else "Profiler-off 性能基线"
    return save_panel_report(
        path=output_path,
        title=f"{manifest['topology']} 性能报告 / Performance Report",
        description=(
            f"{mode}；模型 {config.get('model_config', config.get('config', '-'))}，"
            f"共 {config.get('steps', len(steps))} steps。报告完全离线，可直接下载一个 HTML 阅读。"
        ),
        sections=sections,
    )
