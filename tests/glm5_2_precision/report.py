# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Self-contained HTML reporting for formal training precision experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import html
import json
import math
from pathlib import Path
from typing import Any, Sequence

from .artifacts import PrecisionArtifactReader
from .standards import (
    CriterionResult,
    PairEvaluation,
    evaluate_exact_pair,
    evaluate_migration_pair,
    exact_series_match,
)


@dataclass(frozen=True)
class CandidateEvaluation:
    artifact: PrecisionArtifactReader
    pair: PairEvaluation
    reproducible_reference_loss: bool
    reproducible_reference_grad_norm: bool
    reproducible_loss: bool
    reproducible_grad_norm: bool

    @property
    def passed(self) -> bool:
        return (
            self.pair.passed
            and self.reproducible_reference_loss
            and self.reproducible_reference_grad_norm
            and self.reproducible_loss
            and self.reproducible_grad_norm
        )


def _escape(value: Any) -> str:
    return html.escape(str(value))


def _format_float(value: float) -> str:
    return f"{value:.9g}"


def _line_chart(
    *,
    title: str,
    steps: Sequence[int],
    reference: Sequence[float],
    candidate: Sequence[float],
    reference_label: str,
    candidate_label: str,
) -> str:
    width, height = 900, 300
    left, right, top, bottom = 70, 20, 35, 45
    values = [*reference, *candidate]
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        padding = max(abs(minimum) * 0.05, 1e-12)
        minimum -= padding
        maximum += padding
    min_step, max_step = min(steps), max(steps)
    if min_step == max_step:
        max_step += 1

    def x(step: int) -> float:
        return left + (step - min_step) / (max_step - min_step) * (width - left - right)

    def y(value: float) -> float:
        return top + (maximum - value) / (maximum - minimum) * (height - top - bottom)

    def points(series: Sequence[float]) -> str:
        return " ".join(
            f"{x(step):.2f},{y(value):.2f}" for step, value in zip(steps, series)
        )

    grid: list[str] = []
    for index in range(5):
        fraction = index / 4
        grid_y = top + fraction * (height - top - bottom)
        value = maximum - fraction * (maximum - minimum)
        grid.append(
            f'<line x1="{left}" y1="{grid_y:.2f}" x2="{width-right}" '
            f'y2="{grid_y:.2f}" class="grid"/>'
            f'<text x="{left-8}" y="{grid_y+4:.2f}" text-anchor="end">'
            f'{_escape(_format_float(value))}</text>'
        )
    return f"""
    <section class="chart-card">
      <h3>{_escape(title)}</h3>
      <div class="legend"><span class="reference-line"></span>{_escape(reference_label)}
      <span class="candidate-line"></span>{_escape(candidate_label)}</div>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(title)}">
        {''.join(grid)}
        <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>
        <polyline points="{points(reference)}" class="series reference"/>
        <polyline points="{points(candidate)}" class="series candidate"/>
        <text x="{left}" y="{height-15}">step {min_step}</text>
        <text x="{width-right}" y="{height-15}" text-anchor="end">step {max(steps)}</text>
      </svg>
    </section>"""


def _histogram(errors: Sequence[float], title: str) -> str:
    width, height = 900, 260
    left, right, top, bottom = 65, 20, 35, 45
    num_bins = min(40, max(8, round(math.sqrt(len(errors)))))
    maximum = max(errors)
    if maximum == 0.0:
        counts = [len(errors)] + [0] * (num_bins - 1)
    else:
        counts = [0] * num_bins
        for error in errors:
            index = min(int(error / maximum * num_bins), num_bins - 1)
            counts[index] += 1
    max_count = max(counts)
    plot_width = width - left - right
    plot_height = height - top - bottom
    bars: list[str] = []
    for index, count in enumerate(counts):
        bar_width = plot_width / num_bins
        bar_height = count / max_count * plot_height if max_count else 0
        bars.append(
            f'<rect x="{left + index * bar_width + 1:.2f}" '
            f'y="{top + plot_height - bar_height:.2f}" '
            f'width="{max(bar_width - 2, 1):.2f}" height="{bar_height:.2f}"/>'
        )
    return f"""
    <section class="chart-card">
      <h3>{_escape(title)}</h3>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(title)}">
        <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>
        <g class="bars">{''.join(bars)}</g>
        <text x="{left}" y="{height-15}">0</text>
        <text x="{width-right}" y="{height-15}" text-anchor="end">{_escape(_format_float(maximum))}</text>
        <text x="{width/2}" y="{height-3}" text-anchor="middle">absolute error</text>
      </svg>
    </section>"""


def _metrics_table(evaluation: PairEvaluation) -> str:
    rows: list[str] = []
    for series in (evaluation.loss, evaluation.grad_norm):
        metrics = series.metrics
        quantiles = ", ".join(
            f"P{probability * 100:g}={_format_float(value)}"
            for probability, value in metrics.quantiles.items()
        )
        rows.append(
            "<tr>"
            f"<td>{_escape(series.name)}</td>"
            f"<td>{metrics.count}</td>"
            f"<td>{_format_float(metrics.normal)}</td>"
            f"<td>{_format_float(metrics.absolute)}</td>"
            f"<td>{metrics.relative_normal:.6%}</td>"
            f"<td>{metrics.relative_absolute:.6%}</td>"
            f"<td>{_format_float(metrics.max_absolute)}</td>"
            f"<td>{_escape(quantiles)}</td>"
            "</tr>"
        )
    return f"""
    <table><thead><tr><th>Series</th><th>Steps</th><th>Normal</th>
    <th>Mean absolute</th><th>Mean relative</th><th>Mean absolute relative</th>
    <th>Max absolute</th><th>Absolute-error distribution</th></tr></thead>
    <tbody>{''.join(rows)}</tbody></table>"""


def _criteria_table(
    criteria: Sequence[CriterionResult],
    *,
    reproducible_reference_loss: bool,
    reproducible_reference_grad_norm: bool,
    reproducible_loss: bool,
    reproducible_grad_norm: bool,
) -> str:
    rows = [
        "<tr>"
        f'<td><span class="status {"pass" if item.passed else "fail"}">'
        f'{"PASS" if item.passed else "FAIL"}</span></td>'
        f"<td>{_escape(item.name)}</td><td>{_escape(item.observed)}</td>"
        f"<td>{_escape(item.required)}</td><td>{_escape(item.explanation)}</td>"
        "</tr>"
        for item in criteria
    ]
    for name, passed in (
        ("Reference repeat loss zero error", reproducible_reference_loss),
        (
            "Reference repeat global grad norm zero error",
            reproducible_reference_grad_norm,
        ),
        ("Candidate repeat loss zero error", reproducible_loss),
        ("Candidate repeat global grad norm zero error", reproducible_grad_norm),
    ):
        rows.append(
            "<tr>"
            f'<td><span class="status {"pass" if passed else "fail"}">'
            f'{"PASS" if passed else "FAIL"}</span></td>'
            f"<td>{_escape(name)}</td>"
            f"<td>{'identical' if passed else 'different'}</td>"
            "<td>bitwise-identical float series</td>"
            "<td>Repeated runs verify deterministic reproducibility.</td>"
            "</tr>"
        )
    return f"""
    <table><thead><tr><th>Result</th><th>Criterion</th><th>Observed</th>
    <th>Required</th><th>Meaning</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"""


def _config_table(reference: PrecisionArtifactReader, candidate: PrecisionArtifactReader) -> str:
    rows: list[str] = []
    keys = sorted(set(reference.training_contract) | set(candidate.training_contract))
    for key in keys:
        left = reference.training_contract.get(key)
        right = candidate.training_contract.get(key)
        rows.append(
            f"<tr><td>{_escape(key)}</td>"
            f"<td><pre>{_escape(json.dumps(left, sort_keys=True, indent=2))}</pre></td>"
            f"<td><pre>{_escape(json.dumps(right, sort_keys=True, indent=2))}</pre></td></tr>"
        )
    return f"""
    <table><thead><tr><th>Training contract</th><th>Reference</th><th>Candidate</th>
    </tr></thead><tbody>{''.join(rows)}</tbody></table>"""


def _metadata_table(
    reference: PrecisionArtifactReader,
    candidate: PrecisionArtifactReader,
) -> str:
    rows: list[str] = []
    keys = sorted(set(reference.metadata) | set(candidate.metadata))
    for key in keys:
        left = json.dumps(reference.metadata.get(key), sort_keys=True, indent=2)
        right = json.dumps(candidate.metadata.get(key), sort_keys=True, indent=2)
        rows.append(
            f"<tr><td>{_escape(key)}</td>"
            f"<td><pre>{_escape(left)}</pre></td>"
            f"<td><pre>{_escape(right)}</pre></td></tr>"
        )
    return f"""
    <table><thead><tr><th>Runtime metadata</th><th>Reference</th><th>Candidate</th>
    </tr></thead><tbody>{''.join(rows)}</tbody></table>"""


def _validate_contracts(
    reference: PrecisionArtifactReader,
    candidate: PrecisionArtifactReader,
    *,
    experiment_kind: str,
) -> None:
    left = dict(reference.training_contract)
    right = dict(candidate.training_contract)
    left_topology = left.pop("topology")
    right_topology = right.pop("topology")
    if left != right:
        raise ValueError("reference and candidate use different training contracts")
    if experiment_kind == "migration" and left_topology != right_topology:
        raise ValueError("migration artifacts use different parallel topologies")


def _artifact_paths(root: Path, config: Any) -> tuple[list[Path], list[Path]]:
    from .workflow import _artifact_directory

    references = [
        _artifact_directory(root, config, "reference", config.reference, repeat)
        for repeat in range(1, config.reference.repeats + 1)
    ]
    required_repeats = (
        config.standard.migration.required_candidate_repeats
        if config.kind == "migration"
        else config.standard.self_consistency.required_candidate_repeats
    )
    required_reference_repeats = (
        config.standard.migration.required_reference_repeats
        if config.kind == "migration"
        else config.standard.self_consistency.required_reference_repeats
    )
    if config.reference.repeats < required_reference_repeats:
        raise ValueError(
            f"reference defines {config.reference.repeats} repeats, "
            f"but the standard requires {required_reference_repeats}"
        )
    if config.candidate.repeats < required_repeats:
        raise ValueError(
            f"candidate defines {config.candidate.repeats} repeats, "
            f"but the standard requires {required_repeats}"
        )
    candidates = [
        _artifact_directory(root, config, "candidate", config.candidate, repeat)
        for repeat in range(1, config.candidate.repeats + 1)
    ]
    return references, candidates


def compare_and_write_report(root: Path, config: Any) -> Path:
    reference_paths, candidate_paths = _artifact_paths(root, config)
    references = [PrecisionArtifactReader(path) for path in reference_paths]
    reference = references[0]
    candidates = [PrecisionArtifactReader(path) for path in candidate_paths]
    for item in references:
        if item.metadata.get("role") != "reference":
            raise ValueError(f"reference artifact has role {item.metadata.get('role')!r}")
    for item in candidates:
        if item.metadata.get("role") != "candidate":
            raise ValueError(f"candidate artifact has role {item.metadata.get('role')!r}")
    for other_reference in references[1:]:
        _validate_contracts(
            reference, other_reference, experiment_kind=config.kind
        )
    for candidate in candidates:
        _validate_contracts(reference, candidate, experiment_kind=config.kind)
    for repeat, item in enumerate(references, start=1):
        if int(item.metadata.get("repeat", -1)) != repeat:
            raise ValueError(
                f"reference artifact repeat metadata is {item.metadata.get('repeat')!r}, "
                f"expected {repeat}"
            )
    for repeat, item in enumerate(candidates, start=1):
        if int(item.metadata.get("repeat", -1)) != repeat:
            raise ValueError(
                f"candidate artifact repeat metadata is {item.metadata.get('repeat')!r}, "
                f"expected {repeat}"
            )

    reproducible_reference_loss = all(
        exact_series_match(reference.loss_series(), item.loss_series())
        for item in references[1:]
    )
    reproducible_reference_grad_norm = all(
        exact_series_match(reference.grad_norm_series(), item.grad_norm_series())
        for item in references[1:]
    )
    reproducibility_reference = candidates[0]
    reproducible_loss = all(
        exact_series_match(
            reproducibility_reference.loss_series(), candidate.loss_series()
        )
        for candidate in candidates[1:]
    )
    reproducible_grad_norm = all(
        exact_series_match(
            reproducibility_reference.grad_norm_series(),
            candidate.grad_norm_series(),
        )
        for candidate in candidates[1:]
    )

    evaluations: list[CandidateEvaluation] = []
    for candidate in candidates:
        if config.kind == "self_consistency" and not (
            config.standard.self_consistency.randomness_impacted
        ):
            pair = evaluate_exact_pair(
                reference_loss=reference.loss_series(),
                candidate_loss=candidate.loss_series(),
                reference_grad_norm=reference.grad_norm_series(),
                candidate_grad_norm=candidate.grad_norm_series(),
            )
        else:
            migration_standard = (
                config.standard.migration
                if config.kind == "migration"
                else config.standard.self_consistency.migration_fallback
            )
            pair = evaluate_migration_pair(
                reference_loss=reference.loss_series(),
                candidate_loss=candidate.loss_series(),
                reference_grad_norm=reference.grad_norm_series(),
                candidate_grad_norm=candidate.grad_norm_series(),
                standard=migration_standard,
            )
        evaluations.append(
            CandidateEvaluation(
                artifact=candidate,
                pair=pair,
                reproducible_reference_loss=reproducible_reference_loss,
                reproducible_reference_grad_norm=reproducible_reference_grad_norm,
                reproducible_loss=reproducible_loss,
                reproducible_grad_norm=reproducible_grad_norm,
            )
        )

    repeat_criteria = (
        CriterionResult(
            name="Reference repeat loss zero error",
            passed=reproducible_reference_loss,
            observed="identical" if reproducible_reference_loss else "different",
            required="bitwise-identical float series",
            explanation="Repeated reference runs verify deterministic reproducibility.",
        ),
        CriterionResult(
            name="Reference repeat global grad norm zero error",
            passed=reproducible_reference_grad_norm,
            observed=(
                "identical" if reproducible_reference_grad_norm else "different"
            ),
            required="bitwise-identical float series",
            explanation="Repeated reference runs verify deterministic reproducibility.",
        ),
        CriterionResult(
            name="Candidate repeat loss zero error",
            passed=reproducible_loss,
            observed="identical" if reproducible_loss else "different",
            required="bitwise-identical float series",
            explanation="Repeated candidate runs verify deterministic reproducibility.",
        ),
        CriterionResult(
            name="Candidate repeat global grad norm zero error",
            passed=reproducible_grad_norm,
            observed="identical" if reproducible_grad_norm else "different",
            required="bitwise-identical float series",
            explanation="Repeated candidate runs verify deterministic reproducibility.",
        ),
    )

    overall_passed = all(evaluation.passed for evaluation in evaluations)
    sections: list[str] = []
    for run_index, evaluation in enumerate(evaluations, start=1):
        pair = evaluation.pair
        sections.append(
            f"""
            <section id="candidate-{run_index}">
              <h2>Candidate repeat {run_index}</h2>
              {_criteria_table(pair.criteria,
                 reproducible_reference_loss=reproducible_reference_loss,
                 reproducible_reference_grad_norm=reproducible_reference_grad_norm,
                 reproducible_loss=reproducible_loss,
                 reproducible_grad_norm=reproducible_grad_norm)}
              <h3>Error metrics</h3>{_metrics_table(pair)}
              {_line_chart(title="Training loss", steps=pair.loss.steps,
                 reference=pair.loss.reference, candidate=pair.loss.candidate,
                 reference_label=config.reference.name,
                 candidate_label=f"{config.candidate.name} repeat {run_index}")}
              {_line_chart(title="Global grad norm", steps=pair.grad_norm.steps,
                 reference=pair.grad_norm.reference, candidate=pair.grad_norm.candidate,
                 reference_label=config.reference.name,
                 candidate_label=f"{config.candidate.name} repeat {run_index}")}
              {_histogram(pair.loss.metrics.absolute_errors,
                 "Loss absolute-error distribution")}
              {_histogram(pair.grad_norm.metrics.absolute_errors,
                 "Global grad norm absolute-error distribution")}
              <p>Dropped outlier steps: {_escape(pair.dropped_steps or "none")}</p>
            </section>"""
        )

    exploratory = ""
    if config.exploratory_reports:
        links = "".join(
            f'<li><a href="{_escape(path)}">{_escape(path)}</a></li>'
            for path in config.exploratory_reports
        )
        exploratory = (
            "<section><h2>Sampled exploratory traces</h2>"
            "<p>These traces are diagnostic and do not decide the formal result.</p>"
            f"<ul>{links}</ul></section>"
        )

    report_directory = root / config.report_root / config.scenario_name
    if config.report_subdirectory:
        report_directory /= config.report_subdirectory
    report_directory.mkdir(parents=True, exist_ok=True)
    report_path = report_directory / "precision_report.html"
    summary_path = report_directory / "precision_summary.json"
    summary = {
        "scenario_name": config.scenario_name,
        "experiment_kind": config.kind,
        "passed": overall_passed,
        "reference_reproducible_loss": reproducible_reference_loss,
        "reference_reproducible_grad_norm": reproducible_reference_grad_norm,
        "candidate_reproducible_loss": reproducible_loss,
        "candidate_reproducible_grad_norm": reproducible_grad_norm,
        "repeat_criteria": [asdict(item) for item in repeat_criteria],
        "candidate_runs": [
            {
                "repeat": index,
                "passed": evaluation.passed,
                "criteria": [asdict(item) for item in evaluation.pair.criteria],
                "loss_metrics": {
                    key: value
                    for key, value in asdict(evaluation.pair.loss.metrics).items()
                    if key not in {"absolute_errors", "relative_errors"}
                },
                "grad_norm_metrics": {
                    key: value
                    for key, value in asdict(evaluation.pair.grad_norm.metrics).items()
                    if key not in {"absolute_errors", "relative_errors"}
                },
            }
            for index, evaluation in enumerate(evaluations, start=1)
        ],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path.write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(config.scenario_name)} precision report</title>
<style>
:root{{--pass:#17803d;--fail:#b42318;--ink:#17202a;--muted:#64748b;--line:#d8dee9;--panel:#f8fafc;}}
*{{box-sizing:border-box}} body{{font:14px/1.45 system-ui,sans-serif;color:var(--ink);margin:0;background:#fff}}
main{{max-width:1500px;margin:auto;padding:24px}} h1,h2,h3{{scroll-margin-top:58px}}
.hero{{padding:22px;border:1px solid var(--line);background:var(--panel);border-radius:12px}}
.overall{{display:inline-block;padding:6px 13px;border-radius:999px;color:#fff;font-weight:800;background:{'var(--pass)' if overall_passed else 'var(--fail)'}}}
nav{{position:sticky;top:0;z-index:20;background:#fff;border-bottom:1px solid var(--line);padding:10px 24px}}
nav a{{margin-right:18px;color:#155eef;text-decoration:none}} section{{margin:28px 0}}
table{{border-collapse:separate;border-spacing:0;width:100%;border:1px solid var(--line);border-radius:8px;overflow:visible}}
th,td{{padding:9px 10px;border-right:1px solid var(--line);border-bottom:1px solid var(--line);vertical-align:top;text-align:left}}
th:last-child,td:last-child{{border-right:0}} tr:last-child td{{border-bottom:0}}
thead th{{position:sticky;top:41px;z-index:10;background:#eef2f7}} pre{{white-space:pre-wrap;margin:0;font-size:12px}}
.status{{display:inline-block;padding:3px 8px;border-radius:999px;color:#fff;font-weight:700}} .status.pass{{background:var(--pass)}} .status.fail{{background:var(--fail)}}
.chart-card{{border:1px solid var(--line);border-radius:10px;padding:12px;margin:18px 0}} svg{{width:100%;height:auto}}
.grid{{stroke:#e5e7eb;stroke-width:1}} .axis{{stroke:#64748b;stroke-width:1.2}} .series{{fill:none;stroke-width:2}} .series.reference{{stroke:#155eef}} .series.candidate{{stroke:#e34a33}}
.bars rect{{fill:#7c3aed;opacity:.8}} .legend{{color:var(--muted)}} .legend span{{display:inline-block;width:22px;border-top:3px solid;margin:0 7px 3px 18px}} .reference-line{{border-color:#155eef!important}} .candidate-line{{border-color:#e34a33!important}}
</style></head><body>
<nav><a href="#summary">Summary</a><a href="#contract">Training contract</a>{''.join(f'<a href="#candidate-{index}">Candidate {index}</a>' for index in range(1, len(evaluations)+1))}</nav>
<main><section class="hero" id="summary"><h1>Formal GLM 5.2 precision benchmark</h1>
<p><span class="overall">{'PASS' if overall_passed else 'FAIL'}</span></p>
<p><b>Scenario:</b> {_escape(config.scenario_name)}<br><b>Experiment:</b> {_escape(config.kind)}<br>
<b>Reference:</b> {_escape(config.reference.name)} ({_escape(config.reference.device_type)}, {_escape(config.reference.topology.name)})<br>
<b>Candidate:</b> {_escape(config.candidate.name)} ({_escape(config.candidate.device_type)}, {_escape(config.candidate.topology.name)})</p>
<p>Formal decisions use full-precision JSONL loss and global grad norm. Console-rounded values and exploratory traces never decide this result.</p></section>
<section id="contract"><h2>Training contract</h2>{_config_table(reference, candidates[0])}
<h2>Runtime and source metadata</h2>{_metadata_table(reference, candidates[0])}</section>
{''.join(sections)}{exploratory}</main></body></html>""",
        encoding="utf-8",
    )
    return report_path
