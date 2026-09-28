# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Human-readable index around official MindStudio result files."""

from __future__ import annotations

import csv
import html
import json
import os
from pathlib import Path
from typing import Any, Sequence

from .artifacts import write_json


def _relative_link(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _embedded_csv_table(path: Path) -> str:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    header = "".join(f"<th>{html.escape(name)}</th>" for name in fieldnames)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row.get(name, '')))}</td>"
            for name in fieldnames
        )
        + "</tr>"
        for row in rows
    )
    return (
        '<div class="table-scroll"><table class="metrics"><thead><tr>'
        + header
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table></div>"
    )


def _embedded_text_file(path: Path) -> str:
    return f"<pre>{html.escape(path.read_text(encoding='utf-8'))}</pre>"


def _embedded_svg(path: Path) -> str:
    payload = path.read_text(encoding="utf-8").strip()
    if payload.startswith("<?xml"):
        payload = payload.split("?>", maxsplit=1)[-1].lstrip()
    if not payload.startswith("<svg"):
        return f"<pre>{html.escape(payload)}</pre>"
    return f'<div class="chart">{payload}</div>'


def _embedded_html_document(path: Path) -> str:
    payload = html.escape(path.read_text(encoding="utf-8"), quote=True)
    return (
        '<iframe title="交互式训练观察 / Interactive training observation" '
        'style="display:block;width:100%;height:10000px;border:0;overflow:hidden" '
        'scrolling="no" onload="this.style.height=this.contentDocument.'
        "documentElement.scrollHeight+'px'\" "
        f'srcdoc="{payload}"></iframe>'
    )


def _copy_ready_grad_anomaly_summary(entry: dict[str, Any]) -> str:
    grad = entry["grad_norm"]
    anomalies = grad.get("prominent_anomalies") or []
    median_error = grad.get("median_relative_error")
    median_text = "N/A" if median_error is None else f"{median_error:.2%}"
    if not anomalies:
        return (
            f"拓扑 {entry['topology']}：未自动识别出显著孤立的 Grad Norm 异常点；"
            f"全窗口相对误差中位数={median_text}。"
        )

    lines = [
        f"拓扑 {entry['topology']}：自动识别出 {len(anomalies)} 个显著 Grad Norm 异常点；"
        f"全窗口相对误差中位数={median_text}。"
    ]
    for anomaly in anomalies:
        ratio = anomaly.get("candidate_to_reference_ratio")
        ratio_text = "N/A" if ratio is None else f"{ratio:.4g}x"
        lines.append(
            f"step {anomaly['step']}：GPU={anomaly['reference']:.6g}，"
            f"NPU={anomaly['candidate']:.6g}，"
            f"绝对差={anomaly['absolute_difference']:.6g}，"
            f"NPU/GPU={ratio_text}，"
            f"Grad Norm 相对误差={anomaly['relative_error']:.2%}，"
            f"同 step Loss 相对误差={anomaly['loss_relative_error']:.2%}。"
        )
        neighbors = []
        for label, key in (("前一", "previous"), ("后一", "next")):
            adjacent = anomaly.get(key)
            if adjacent is None:
                continue
            neighbors.append(
                f"{label} step {adjacent['step']}={adjacent['relative_error']:.2%}"
            )
        if neighbors:
            lines.append("相邻 Grad Norm 相对误差：" + "；".join(neighbors) + "。")
    return "\n".join(lines)


def _supplemental_entries(
    repository_root: Path,
    patterns: Sequence[str],
) -> list[dict[str, str]]:
    """Resolve configured evidence globs into stable report links."""

    entries: list[dict[str, str]] = []
    for pattern in patterns:
        matches = sorted(repository_root.glob(pattern))
        report_files: list[Path] = []
        for match in matches:
            if match.is_file():
                report_files.append(match)
            elif match.is_dir():
                report_files.extend(sorted(match.rglob("*.html")))
        if not report_files:
            entries.append(
                {
                    "pattern": pattern,
                    "status": "missing",
                    "path": "",
                }
            )
            continue
        for report_file in dict.fromkeys(report_files):
            entries.append(
                {
                    "pattern": pattern,
                    "status": "available",
                    "path": report_file.resolve()
                    .relative_to(repository_root.resolve())
                    .as_posix(),
                }
            )
    return entries


def _official_diagnostic_entries(
    repository_root: Path,
    report_directory: Path,
) -> list[dict[str, str]]:
    """Discover official derived diagnostics produced beside the main verdict."""

    patterns = (
        (
            "precision_precheck",
            "Official API precision pre-check comparison",
            "*/precision_precheck/compare-r*/precheck_report.html",
        ),
        (
            "graph_visualization_guide",
            "Official hierarchical graph visualization guide",
            "*/graph-visualize-r*/README.md",
        ),
        (
            "graph_visualization_index",
            "Official hierarchical graph visualization index",
            "*/graph-visualize-r*/index.json",
        ),
        (
            "training_monitor",
            "Project-derived TrainerMonitorV2 CSV analysis",
            "*/monitor_analysis/monitor_report.html",
        ),
    )
    entries: list[dict[str, str]] = []
    repository = repository_root.resolve()
    for kind, label, pattern in patterns:
        for path in sorted(report_directory.glob(pattern)):
            if not path.is_file():
                continue
            resolved = path.resolve()
            entries.append(
                {
                    "kind": kind,
                    "label": label,
                    "scope": path.relative_to(report_directory).as_posix(),
                    "path": resolved.relative_to(repository).as_posix(),
                }
            )
    return entries


def _observation_html_document(
    *,
    title: str,
    html_rows: Sequence[str],
    evidence_sections: Sequence[str] = (),
) -> str:
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:42px auto;max-width:1900px;"
        "color:#172033;font-size:18px;line-height:1.6}"
        "h1{font-size:40px;line-height:1.3;margin:0 0 18px}"
        "h2{font-size:30px;line-height:1.35;margin:52px 0 22px}"
        "p{margin:12px 0 20px}"
        "table{border-collapse:collapse;width:100%;margin:24px 0 44px}"
        "th,td{border:1px solid #ccd4e0;padding:14px 16px;text-align:left}"
        "th{background:#eef3fa;position:sticky;top:0}"
        "code{background:#f4f6f8;padding:2px 4px}"
        "details{margin:20px 0;border:1px solid #ccd4e0;padding:12px}"
        "summary{cursor:pointer;font-weight:700}"
        "pre{white-space:pre-wrap;word-break:break-word;background:#f6f8fa;"
        "padding:12px;max-height:520px;overflow:auto}"
        ".chart{overflow:auto;margin:12px 0 28px}.chart svg{max-width:100%;height:auto}"
        ".table-scroll{max-height:640px;overflow:auto}.metrics{font-size:12px}"
        "</style></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        "<p>流程 / Workflow: <code>training observation</code>.</p>"
        "<p>本报告比较未注入调试工具的 GPU 标杆与 NPU 调试侧训练指标，用于现象分类，"
        "不声明通用的交付 PASS/FAIL 结论。</p>"
        "<h2>训练观察 / Training observations</h2><table><thead><tr>"
        "<th>拓扑 / Topology</th><th>观察现象 / Symptom</th><th>窗口 / Window</th>"
        "<th>Loss 平均相对误差</th>"
        "<th>Loss 首次超过指导线</th>"
        "<th>首次超限后平均误差</th>"
        "<th>后续超限比例</th><th>首 Steps Loss 窗口</th>"
        "<th>Grad Norm 平均有符号误差</th><th>NaN/Inf</th>"
        "<th>报告 / Report</th>"
        "</tr></thead><tbody>"
        + "".join(html_rows)
        + "</tbody></table>"
        + "".join(evidence_sections)
        + "</body></html>"
    )


def _write_observation_report_index(
    *,
    report_directory: Path,
    experiment_name: str,
    rows: Sequence[dict[str, Any]],
) -> Path:
    """Write a training-observation report without an msProbe verdict."""

    observations: list[dict[str, Any]] = []
    evidence_names = (
        "summary.json",
        "training_metrics_compare.csv",
        "training_observation.html",
        "loss.svg",
        "grad_norm.svg",
        "loss_relative_error.svg",
        "grad_norm_relative_error.svg",
        "loss_signed_difference.svg",
        "grad_norm_signed_difference.svg",
        "early_loss.svg",
        "early_loss_relative_error.svg",
        "grad_norm_signed_relative_error.svg",
    )
    for row in rows:
        official_summary = Path(row["official_result"])
        summary = json.loads(official_summary.read_text(encoding="utf-8"))
        observation = summary.get("training_observation")
        if not isinstance(observation, dict):
            observation_path = official_summary.parent / "summary.json"
            observation = json.loads(observation_path.read_text(encoding="utf-8"))
        output_directory = official_summary.parent
        evidence = [
            path
            for name in evidence_names
            if (path := output_directory / name).is_file()
        ]
        details = observation["observation"]
        observations.append(
            {
                "topology": str(row["topology"]),
                "diagnostic_symptom": details["diagnostic_symptom"],
                "first_step": details["first_step"],
                "last_step": details["last_step"],
                "step_count": observation["step_count"],
                "loss": observation["loss"],
                "grad_norm": observation["grad_norm"],
                "reference_first_nonfinite_metrics": details[
                    "reference_first_nonfinite_metrics"
                ],
                "candidate_first_nonfinite_metrics": details[
                    "candidate_first_nonfinite_metrics"
                ],
                "nonfinite_analysis": details.get("nonfinite_analysis"),
                "topology_report": str(Path(row["download_report"]).resolve()),
                "evidence": [str(path.resolve()) for path in evidence],
                "runtime_log": str(Path(row["runtime_log"]).resolve()),
            }
        )

    write_json(
        report_directory / "report.json",
        {
            "schema": "torchtitan.glm5_2.mindstudio_report",
            "schema_version": 2,
            "experiment": experiment_name,
            "workflow": "observation",
            "training_observations": observations,
            "delivery_verdict": None,
            "meaning": (
                "Baseline classifies whole-training symptoms. It does not "
                "define a universal delivery pass/fail verdict."
            ),
        },
    )

    markdown_lines = [
        f"# {experiment_name}",
        "",
        "Workflow: `training observation`",
        "",
        "This report compares uninstrumented GPU reference and NPU candidate "
        "training metrics. It classifies the observed symptom and does not "
        "define a universal delivery PASS/FAIL verdict.",
        "",
        "## Training observations",
        "",
        "| Topology | Observed symptom | Window | Mean Loss relative error | "
        "First Loss step above guidance | Mean error after first exceedance | "
        "Subsequent exceedance rate | Early Loss window | "
        "Mean signed Grad Norm error | NaN/Inf | Runtime log |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    html_rows: list[str] = []
    for entry in observations:
        runtime_log = Path(entry["runtime_log"])
        runtime_link = _relative_link(runtime_log, report_directory)
        loss = entry["loss"]
        mean_error = loss.get("mean_relative_error")
        mean_error_text = "N/A" if mean_error is None else f"{mean_error:.2%}"
        first_above = loss.get("first_step_above_threshold")
        first_above_text = "none" if first_above is None else str(first_above)
        post_window = loss.get("post_first_threshold_window", {})
        post_mean = post_window.get("mean_relative_error")
        post_mean_text = "N/A" if post_mean is None else f"{post_mean:.2%}"
        post_fraction = post_window.get("fraction_above_threshold")
        post_fraction_text = (
            "N/A" if post_fraction is None else f"{post_fraction:.2%}"
        )
        early_window = loss.get("early_window", {})
        early_mean = early_window.get("mean_relative_error")
        early_max = early_window.get("max_relative_error")
        early_fraction = early_window.get("fraction_above_threshold")
        early_window_text = (
            "N/A"
            if early_mean is None or early_max is None
            else (
                f"{early_window.get('first_step')}..{early_window.get('last_step')}: "
                f"mean {early_mean:.2%}, max {early_max:.2%}, "
                f"above {early_fraction:.2%}"
            )
        )
        grad_signed_mean = entry["grad_norm"].get("mean_signed_relative_error")
        grad_signed_mean_text = (
            "N/A" if grad_signed_mean is None else f"{grad_signed_mean:.2%}"
        )
        candidate_nonfinite = entry["candidate_first_nonfinite_metrics"]
        reference_nonfinite = entry["reference_first_nonfinite_metrics"]
        nonfinite = entry.get("nonfinite_analysis")
        no_nonfinite = (
            isinstance(nonfinite, dict)
            and nonfinite.get("comparison") == "neither-endpoint-observed"
        ) or (
            not isinstance(nonfinite, dict)
            and not candidate_nonfinite
            and not reference_nonfinite
        )
        nonfinite_text = (
            "Neither endpoint observed NaN/Inf"
            if no_nonfinite
            else "NaN/Inf observed; inspect endpoint details"
        )
        endpoint_nonfinite_text: dict[str, str] = {}
        for role, fallback in (
            ("candidate", candidate_nonfinite),
            ("reference", reference_nonfinite),
        ):
            endpoint = (
                nonfinite.get(role)
                if isinstance(nonfinite, dict)
                else None
            )
            metrics = endpoint.get("metrics") if isinstance(endpoint, dict) else None
            if isinstance(endpoint, dict) and endpoint.get("status") == "none-observed":
                endpoint_nonfinite_text[role] = "none observed"
            elif isinstance(metrics, list) and metrics:
                endpoint_nonfinite_text[role] = "; ".join(
                    f"{item['metric']}: {item['kind']} at step {item['first_step']}"
                    for item in metrics
                )
            elif fallback:
                endpoint_nonfinite_text[role] = "; ".join(
                    f"{name}: non-finite at step {step}"
                    for name, step in sorted(fallback.items())
                )
            else:
                endpoint_nonfinite_text[role] = "none observed"
        window = f"{entry['first_step']}..{entry['last_step']}"
        topology_report = Path(entry["topology_report"])
        topology_report.parent.mkdir(parents=True, exist_ok=True)
        legacy_topology_report = report_directory / f"{entry['topology']}.html"
        if legacy_topology_report.is_file():
            legacy_topology_report.unlink()
        topology_report_link = _relative_link(topology_report, report_directory)
        markdown_lines.append(
            f"| {entry['topology']} | {entry['diagnostic_symptom']} | "
            f"{window} ({entry['step_count']} steps) | {mean_error_text} | "
            f"{first_above_text} | {post_mean_text} | {post_fraction_text} | "
            f"{early_window_text} | {grad_signed_mean_text} | {nonfinite_text} | "
            f"[{runtime_log.name}]({runtime_link}) |"
        )
        html_rows.append(
            "<tr>"
            f"<td>{html.escape(entry['topology'])}</td>"
            f"<td>{html.escape(entry['diagnostic_symptom'])}</td>"
            f"<td>{html.escape(window)} ({entry['step_count']} steps)</td>"
            f"<td>{html.escape(mean_error_text)}</td>"
            f"<td>{html.escape(first_above_text)}</td>"
            f"<td>{html.escape(post_mean_text)}</td>"
            f"<td>{html.escape(post_fraction_text)}</td>"
            f"<td>{html.escape(early_window_text)}</td>"
            f"<td>{html.escape(grad_signed_mean_text)}</td>"
            f"<td>{html.escape(nonfinite_text)}</td>"
            f'<td><a href="{html.escape(topology_report_link)}">open report</a></td>'
            "</tr>"
        )
        markdown_lines.extend(
            (
                "",
                f"### {entry['topology']} evidence",
                "",
                f"- [Open self-contained topology report]({topology_report_link})",
            )
        )
        evidence_by_name: dict[str, Path] = {}
        for raw_path in entry["evidence"]:
            path = Path(raw_path)
            link = _relative_link(path, report_directory)
            markdown_lines.append(f"- [{path.name}]({link})")
            evidence_by_name[path.name] = path
        markdown_lines.extend(
            (
                f"- NaN/Inf: {nonfinite_text}",
                "- Candidate NaN/Inf details: "
                f"{endpoint_nonfinite_text['candidate']}",
                "- Reference NaN/Inf details: "
                f"{endpoint_nonfinite_text['reference']}",
            )
        )
        static_chart_sections = "".join(
            f"<h4>{html.escape(name)}</h4>{_embedded_svg(evidence_by_name[name])}"
            for name in (
                "loss.svg",
                "grad_norm.svg",
                "early_loss.svg",
                "early_loss_relative_error.svg",
                "loss_relative_error.svg",
                "grad_norm_relative_error.svg",
                "loss_signed_difference.svg",
                "grad_norm_signed_difference.svg",
                "grad_norm_signed_relative_error.svg",
            )
            if name in evidence_by_name
        )
        interactive_charts = (
            _embedded_html_document(evidence_by_name["training_observation.html"])
            if "training_observation.html" in evidence_by_name
            else "<p>Interactive training charts are unavailable.</p>"
        )
        chart_sections = (
            interactive_charts
            + "<details><summary>Static SVG fallbacks</summary>"
            + static_chart_sections
            + "</details>"
        )
        metrics_table = (
            _embedded_csv_table(evidence_by_name["training_metrics_compare.csv"])
            if "training_metrics_compare.csv" in evidence_by_name
            else "<p>Training metric table is unavailable.</p>"
        )
        summary_payload = (
            _embedded_text_file(evidence_by_name["summary.json"])
            if "summary.json" in evidence_by_name
            else "<p>Observation summary is unavailable.</p>"
        )
        runtime_payload = _embedded_text_file(runtime_log)
        anomaly_summary = _copy_ready_grad_anomaly_summary(entry)
        anomaly_section = (
            '<section class="automatic-anomaly-summary">'
            "<h2>自动提取异常事实 / Copy-ready anomaly facts</h2>"
            "<p>以下内容由逐 step 指标自动生成，可直接复制用于后续定位；"
            "它是事实摘要，不是根因结论。</p>"
            f"<pre>{html.escape(anomaly_summary)}</pre></section>"
        )
        evidence_section = (
            anomaly_section
            + chart_sections
            + "<section class=\"supporting-evidence\">"
            + "<h2>补充证据 / Supporting evidence</h2>"
            + f"<p><strong>NaN/Inf:</strong> {html.escape(nonfinite_text)}<br>"
            + "Candidate details: "
            + html.escape(endpoint_nonfinite_text["candidate"])
            + "<br>Reference details: "
            + html.escape(endpoint_nonfinite_text["reference"])
            + "</p>"
            + "<details><summary>Per-step metric comparison</summary>"
            + metrics_table
            + "</details>"
            + "<details><summary>Observation summary JSON</summary>"
            + summary_payload
            + "</details>"
            + "<details><summary>Runtime log</summary>"
            + runtime_payload
            + "</details></section>"
        )
        topology_row = html_rows[-1].replace(
            f'<a href="{html.escape(topology_report_link)}">open report</a>',
            "self-contained below",
        )
        topology_report.write_text(
            _observation_html_document(
                title=f"训练观察：{experiment_name} / {entry['topology']}",
                html_rows=(topology_row,),
                evidence_sections=(evidence_section,),
            ),
            encoding="utf-8",
        )

    markdown_path = report_directory / "README.md"
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    html_path = report_directory / f"{experiment_name}.html"
    html_path.write_text(
        _observation_html_document(
            title=experiment_name,
            html_rows=html_rows,
        ),
        encoding="utf-8",
    )
    return html_path


def write_report_index(
    *,
    repository_root: Path,
    report_directory: Path,
    experiment_name: str,
    workflow: str,
    rows: Sequence[dict[str, Any]],
    supplemental_report_patterns: Sequence[str],
) -> Path:
    report_directory.mkdir(parents=True, exist_ok=True)
    if workflow == "observation":
        return _write_observation_report_index(
            report_directory=report_directory,
            experiment_name=experiment_name,
            rows=rows,
        )
    supplemental_reports = _supplemental_entries(
        repository_root,
        supplemental_report_patterns,
    )
    official_diagnostics = _official_diagnostic_entries(
        repository_root,
        report_directory,
    )
    write_json(
        report_directory / "report.json",
        {
            "schema": "torchtitan.glm5_2.mindstudio_report",
            "schema_version": 2,
            "experiment": experiment_name,
            "workflow": workflow,
            "official_results": list(rows),
            "official_diagnostics": official_diagnostics,
            "supplemental_reports": supplemental_reports,
        },
    )

    is_monitor = workflow == "monitor"
    workflow_label = (
        "Project-derived Monitor analysis"
        if is_monitor
        else f"Official workflow: `{workflow}`"
    )
    evidence_description = (
        "Monitor V2 only owns the endpoint CSV captures. The links below point "
        "to project-derived alignment and visualization, which is diagnostic "
        "evidence without an official PASS/FAIL verdict."
        if is_monitor
        else "The verdicts below are copied or aggregated from official msProbe "
        "outputs. Tool-stage completion and numerical PASS are reported separately."
    )
    table_title = "Monitor diagnostic analyses" if is_monitor else "Official verdicts"
    verdict_column = "Analysis status" if is_monitor else "Official verdict"
    result_column = "Analysis report" if is_monitor else "Official result"
    diagnostic_title = "Related diagnostic evidence" if is_monitor else "Official diagnostics"

    markdown_lines = [
        f"# {experiment_name}",
        "",
        workflow_label,
        "",
        evidence_description,
        "",
        f"## {table_title}",
        "",
        f"| Topology | {verdict_column} | Result counts | {result_column} | Runtime log |",
        "|---|---|---|---|---|",
    ]
    html_rows: list[str] = []
    for row in rows:
        topology = str(row["topology"])
        verdict = str(row["verdict"])
        counts = ", ".join(
            f"{key}={value}"
            for key, value in sorted(row.get("status_counts", {}).items())
        ) or "not parsed"
        official = Path(row["official_result"])
        runtime_log = Path(row["runtime_log"])
        if is_monitor and row.get("download_report"):
            official_link = _relative_link(
                Path(row["download_report"]), report_directory
            )
        else:
            official_link = _relative_link(official, report_directory)
        runtime_link = _relative_link(runtime_log, report_directory)
        markdown_lines.append(
            f"| {topology} | {verdict} | {counts} | "
            f"[{official.name}]({official_link}) | "
            f"[{runtime_log.name}]({runtime_link}) |"
        )
        html_rows.append(
            "<tr>"
            f"<td>{html.escape(topology)}</td>"
            f"<td class=\"verdict {html.escape(verdict)}\">"
            f"{html.escape(verdict)}</td>"
            f"<td>{html.escape(counts)}</td>"
            f"<td><a href=\"{html.escape(official_link)}\">"
            f"{'analysis report' if is_monitor else 'official output'}</a></td>"
            f"<td><a href=\"{html.escape(runtime_link)}\">runtime log</a></td>"
            "</tr>"
        )
    markdown_lines.extend(("", f"## {diagnostic_title}", ""))
    if official_diagnostics:
        for entry in official_diagnostics:
            diagnostic_path = repository_root / entry["path"]
            link = _relative_link(diagnostic_path, report_directory)
            markdown_lines.append(
                f"- {entry['label']}: [{entry['scope']}]({link})"
            )
    else:
        markdown_lines.append("- No related diagnostic output has been generated.")
    markdown_lines.extend(("", "## Supplemental long-run evidence", ""))
    if supplemental_reports:
        for entry in supplemental_reports:
            if entry["status"] == "available":
                report_path = repository_root / entry["path"]
                link = _relative_link(report_path, report_directory)
                markdown_lines.append(
                    f"- [{report_path.name}]({link}) "
                    f"(configured by `{entry['pattern']}`)"
                )
            else:
                markdown_lines.append(
                    f"- Not synchronized: `{entry['pattern']}`"
                )
    else:
        markdown_lines.append("- None configured.")
    markdown_path = report_directory / "README.md"
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    supplemental_items: list[str] = []
    for entry in supplemental_reports:
        if entry["status"] == "available":
            report_path = repository_root / entry["path"]
            link = _relative_link(report_path, report_directory)
            supplemental_items.append(
                f'<li><a href="{html.escape(link)}">'
                f"{html.escape(report_path.name)}</a> "
                f"(configured by <code>{html.escape(entry['pattern'])}</code>)</li>"
            )
        else:
            supplemental_items.append(
                "<li>Not synchronized: "
                f"<code>{html.escape(entry['pattern'])}</code></li>"
            )
    supplemental_html = "".join(supplemental_items) or "<li>None configured.</li>"
    diagnostic_items: list[str] = []
    for entry in official_diagnostics:
        diagnostic_path = repository_root / entry["path"]
        link = _relative_link(diagnostic_path, report_directory)
        diagnostic_items.append(
            f'<li>{html.escape(entry["label"])}: '
            f'<a href="{html.escape(link)}">'
            f'{html.escape(entry["scope"])}</a></li>'
        )
    diagnostic_html = (
        "".join(diagnostic_items)
        or "<li>No related diagnostic output has been generated.</li>"
    )
    html_path = report_directory / f"{experiment_name}.html"
    html_path.write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(experiment_name)}</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:32px;color:#172033}"
        "table{border-collapse:collapse;width:100%;margin-top:20px}"
        "th,td{border:1px solid #ccd4e0;padding:8px;text-align:left}"
        "th{position:sticky;top:0;background:#eef3fa}"
        ".verdict{font-weight:700}.pass{color:#137333}.warning,.unparsed{color:#9a6700}"
        ".error,.failed{color:#b3261e}code{background:#f4f6f8;padding:2px 4px}"
        "</style></head><body>"
        f"<h1>{html.escape(experiment_name)}</h1>"
        f"<p>{html.escape(workflow_label)}</p>"
        f"<p>{html.escape(evidence_description)}</p>"
        f"<h2>{html.escape(table_title)}</h2><table><thead><tr><th>Topology</th>"
        f"<th>{html.escape(verdict_column)}</th>"
        "<th>Result counts</th>"
        f"<th>{html.escape(result_column)}</th><th>Runtime log</th></tr></thead><tbody>"
        + "".join(html_rows)
        + f"</tbody></table><h2>{html.escape(diagnostic_title)}</h2><ul>"
        + diagnostic_html
        + "</ul><h2>Supplemental long-run evidence</h2><ul>"
        + supplemental_html
        + "</ul></body></html>",
        encoding="utf-8",
    )
    return html_path
