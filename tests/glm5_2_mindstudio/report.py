# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Human-readable index around official MindStudio result files."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any, Sequence

from .artifacts import write_json


def _relative_link(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


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
            "Official TrainerMonitorV2 capture index",
            "*/official_compare/monitor_index.json",
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


def _write_baseline_report_index(
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
        "loss.svg",
        "grad_norm.svg",
        "relative_error.svg",
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
            "workflow": "baseline",
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
        "Workflow: `baseline training observation`",
        "",
        "This report compares uninstrumented GPU reference and NPU candidate "
        "training metrics. It classifies the observed symptom and does not "
        "define a universal delivery PASS/FAIL verdict.",
        "",
        "## Training observations",
        "",
        "| Topology | Observed symptom | Window | Mean Loss relative error | "
        "First Loss step above guidance | Runtime log |",
        "|---|---|---|---|---|---|",
    ]
    html_rows: list[str] = []
    evidence_sections: list[str] = []
    for entry in observations:
        runtime_log = Path(entry["runtime_log"])
        runtime_link = _relative_link(runtime_log, report_directory)
        loss = entry["loss"]
        mean_error = loss.get("mean_relative_error")
        mean_error_text = "N/A" if mean_error is None else f"{mean_error:.6g}"
        first_above = loss.get("first_step_above_threshold")
        first_above_text = "none" if first_above is None else str(first_above)
        window = f"{entry['first_step']}..{entry['last_step']}"
        markdown_lines.append(
            f"| {entry['topology']} | {entry['diagnostic_symptom']} | "
            f"{window} ({entry['step_count']} steps) | {mean_error_text} | "
            f"{first_above_text} | [{runtime_log.name}]({runtime_link}) |"
        )
        html_rows.append(
            "<tr>"
            f"<td>{html.escape(entry['topology'])}</td>"
            f"<td>{html.escape(entry['diagnostic_symptom'])}</td>"
            f"<td>{html.escape(window)} ({entry['step_count']} steps)</td>"
            f"<td>{html.escape(mean_error_text)}</td>"
            f"<td>{html.escape(first_above_text)}</td>"
            f'<td><a href="{html.escape(runtime_link)}">runtime log</a></td>'
            "</tr>"
        )
        markdown_lines.extend(("", f"### {entry['topology']} evidence", ""))
        links: list[str] = []
        for raw_path in entry["evidence"]:
            path = Path(raw_path)
            link = _relative_link(path, report_directory)
            markdown_lines.append(f"- [{path.name}]({link})")
            links.append(
                f'<li><a href="{html.escape(link)}">'
                f"{html.escape(path.name)}</a></li>"
            )
        candidate_nonfinite = entry["candidate_first_nonfinite_metrics"]
        reference_nonfinite = entry["reference_first_nonfinite_metrics"]
        markdown_lines.extend(
            (
                f"- Candidate first NaN/Inf metrics: "
                f"`{json.dumps(candidate_nonfinite, sort_keys=True)}`",
                f"- Reference first NaN/Inf metrics: "
                f"`{json.dumps(reference_nonfinite, sort_keys=True)}`",
            )
        )
        evidence_sections.append(
            f"<h3>{html.escape(entry['topology'])} evidence</h3><ul>"
            + "".join(links)
            + "</ul>"
            + "<p>Candidate first NaN/Inf metrics: <code>"
            + html.escape(json.dumps(candidate_nonfinite, sort_keys=True))
            + "</code><br>Reference first NaN/Inf metrics: <code>"
            + html.escape(json.dumps(reference_nonfinite, sort_keys=True))
            + "</code></p>"
        )

    markdown_path = report_directory / "README.md"
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    html_path = report_directory / f"{experiment_name}.html"
    html_path.write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(experiment_name)}</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:32px;color:#172033}"
        "table{border-collapse:collapse;width:100%;margin-top:20px}"
        "th,td{border:1px solid #ccd4e0;padding:8px;text-align:left}"
        "th{background:#eef3fa}code{background:#f4f6f8;padding:2px 4px}"
        "</style></head><body>"
        f"<h1>{html.escape(experiment_name)}</h1>"
        "<p>Workflow: <code>baseline training observation</code>.</p>"
        "<p>This report compares uninstrumented GPU reference and NPU candidate "
        "training metrics. It classifies the observed symptom and does not "
        "define a universal delivery PASS/FAIL verdict.</p>"
        "<h2>Training observations</h2><table><thead><tr>"
        "<th>Topology</th><th>Observed symptom</th><th>Window</th>"
        "<th>Mean Loss relative error</th>"
        "<th>First Loss step above guidance</th><th>Runtime log</th>"
        "</tr></thead><tbody>"
        + "".join(html_rows)
        + "</tbody></table>"
        + "".join(evidence_sections)
        + "</body></html>",
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
    if workflow == "baseline":
        return _write_baseline_report_index(
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

    markdown_lines = [
        f"# {experiment_name}",
        "",
        f"Official workflow: `{workflow}`",
        "",
        "The verdicts below are copied or aggregated from official msProbe "
        "outputs. Tool-stage completion and numerical PASS are reported "
        "separately.",
        "",
        "## Official verdicts",
        "",
        "| Topology | Official verdict | Result counts | Official result | Runtime log |",
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
            f"<td><a href=\"{html.escape(official_link)}\">official output</a></td>"
            f"<td><a href=\"{html.escape(runtime_link)}\">runtime log</a></td>"
            "</tr>"
        )
    markdown_lines.extend(("", "## Official diagnostics", ""))
    if official_diagnostics:
        for entry in official_diagnostics:
            diagnostic_path = repository_root / entry["path"]
            link = _relative_link(diagnostic_path, report_directory)
            markdown_lines.append(
                f"- {entry['label']}: [{entry['scope']}]({link})"
            )
    else:
        markdown_lines.append("- No derived official diagnostic has been generated.")
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
        or "<li>No derived official diagnostic has been generated.</li>"
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
        f"<p>Official workflow: <code>{html.escape(workflow)}</code>.</p>"
        "<p>Official msProbe files own module/API/compile verdicts. The project "
        "page only indexes and aggregates them; it does not recompute thresholds.</p>"
        "<h2>Official verdicts</h2><table><thead><tr><th>Topology</th>"
        "<th>Official verdict</th>"
        "<th>Result counts</th>"
        "<th>Official result</th><th>Runtime log</th></tr></thead><tbody>"
        + "".join(html_rows)
        + "</tbody></table><h2>Official diagnostics</h2><ul>"
        + diagnostic_html
        + "</ul><h2>Supplemental long-run evidence</h2><ul>"
        + supplemental_html
        + "</ul></body></html>",
        encoding="utf-8",
    )
    return html_path
