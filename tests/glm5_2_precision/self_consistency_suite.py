# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Shared-reference orchestration for distributed self-consistency suites."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import html
import json
from pathlib import Path
from typing import Any, Sequence

from .artifacts import PrecisionArtifactError, PrecisionArtifactReader
from .report import compare_and_write_report
from .workflow import (
    FormalExperimentConfig,
    ParallelTopology,
    _artifact_directory,
    _endpoint_from_process_environment,
    _fixture_endpoint_from_environment,
    _root,
    _run_directory,
    capture_endpoint,
    prepare_fixture,
)


def _artifact_is_complete(path: Path) -> bool:
    """Return whether an artifact exists and passes its checksum contract."""

    try:
        PrecisionArtifactReader(path)
    except (OSError, ValueError, PrecisionArtifactError):
        return False
    return True


def _training_with_precision(config: FormalExperimentConfig, precision: str | None):
    training = config.training
    if precision == "fp32":
        return replace(
            training,
            training_dtype="float32",
            mixed_precision_param="float32",
        )
    if precision == "bf16":
        return replace(
            training,
            training_dtype="float32",
            mixed_precision_param="bfloat16",
        )
    if precision == "full-bf16":
        return replace(
            training,
            training_dtype="bfloat16",
            mixed_precision_param="bfloat16",
        )
    return training


def _suite_config(
    base: FormalExperimentConfig,
    topology: ParallelTopology,
    *,
    precision: str | None,
) -> FormalExperimentConfig:
    reference = _endpoint_from_process_environment(base.reference)
    candidate = _endpoint_from_process_environment(
        replace(base.candidate, topology=topology)
    )
    return replace(
        base,
        reference=reference,
        candidate=candidate,
        training=_training_with_precision(base, precision),
    )


def _candidate_complete(root: Path, config: FormalExperimentConfig) -> bool:
    return all(
        _artifact_directory(
            root,
            config,
            "candidate",
            config.candidate,
            repeat,
        ).is_dir()
        for repeat in range(1, config.candidate.repeats + 1)
    )


def _reference_complete(root: Path, config: FormalExperimentConfig) -> bool:
    return all(
        _artifact_directory(
            root,
            config,
            "reference",
            config.reference,
            repeat,
        ).is_dir()
        for repeat in range(1, config.reference.repeats + 1)
    )


def compare_suite(
    root: Path,
    base: FormalExperimentConfig,
    *,
    topologies: dict[str, ParallelTopology],
    selected: Sequence[str],
    precision: str | None,
    require_all: bool,
) -> Path:
    configs = {
        name: _suite_config(base, topologies[name], precision=precision)
        for name in selected
    }
    first_config = next(iter(configs.values()))
    if not _reference_complete(root, first_config):
        raise FileNotFoundError(
            "shared single-card reference is incomplete; run --capture reference"
        )
    completed = {
        name: config
        for name, config in configs.items()
        if _candidate_complete(root, config)
    }
    missing = [name for name in selected if name not in completed]
    if require_all and missing:
        raise FileNotFoundError(
            f"candidate captures are incomplete for: {', '.join(missing)}"
        )
    if not completed:
        raise FileNotFoundError("no complete distributed candidate captures found")

    suite_directory = root / base.report_root / first_config.storage_name
    rows: list[str] = []
    summaries: dict[str, dict[str, Any]] = {}
    for name, config in completed.items():
        detail_directory = suite_directory / name
        report_path = compare_and_write_report(
            root,
            config,
            report_directory=detail_directory,
        )
        summary = json.loads(
            (detail_directory / "precision_summary.json").read_text(encoding="utf-8")
        )
        summaries[name] = summary
        candidate_runs = summary["candidate_runs"]
        max_loss_absolute = max(
            run["loss_metrics"]["absolute"] for run in candidate_runs
        )
        max_loss_relative = max(
            run["loss_metrics"]["relative_absolute"] for run in candidate_runs
        )
        max_grad_relative = max(
            abs(run["grad_norm_metrics"]["relative_normal"])
            for run in candidate_runs
        )
        passed = bool(summary["passed"])
        rows.append(
            "<tr>"
            f'<td><span class="status {"pass" if passed else "fail"}">'
            f'{"PASS" if passed else "FAIL"}</span></td>'
            f"<td>{html.escape(name)}</td>"
            f"<td>{len(candidate_runs)}</td>"
            f"<td>{max_loss_absolute:.9g}</td>"
            f"<td>{max_loss_relative:.6%}</td>"
            f"<td>{max_grad_relative:.6%}</td>"
            f'<td><a href="{html.escape(name)}/precision_report.html">details</a></td>'
            "</tr>"
        )
    for name in missing:
        rows.append(
            "<tr><td><span class=\"status pending\">NOT RUN</span></td>"
            f"<td>{html.escape(name)}</td><td>-</td><td>-</td><td>-</td>"
            "<td>-</td><td>-</td></tr>"
        )

    completed_passed = all(bool(value["passed"]) for value in summaries.values())
    complete = not missing
    overall_passed = completed_passed and complete
    display_status = (
        "PASS" if overall_passed else "PARTIAL PASS" if completed_passed else "FAIL"
    )
    status_class = "pass" if completed_passed else "fail"
    suite_summary = {
        "scenario_name": first_config.storage_name,
        "passed": overall_passed,
        "completed_candidates_passed": completed_passed,
        "complete": complete,
        "status": display_status,
        "completed_topologies": list(completed),
        "missing_topologies": missing,
        "require_all": require_all,
        "topologies": summaries,
    }
    suite_directory.mkdir(parents=True, exist_ok=True)
    (suite_directory / "suite_summary.json").write_text(
        json.dumps(suite_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = suite_directory / "suite_report.html"
    report.write_text(
        f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GPU distributed self-consistency suite</title><style>
body{{font:14px/1.45 system-ui,sans-serif;max-width:1400px;margin:auto;padding:24px;color:#17202a}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d8dee9;padding:9px;text-align:left}}
th{{background:#eef2f7;position:sticky;top:0}}.status{{display:inline-block;color:#fff;padding:3px 8px;border-radius:999px;font-weight:700}}
.pass{{background:#17803d}}.fail{{background:#b42318}}.pending{{background:#64748b}}a{{color:#155eef}}
</style></head><body><h1>GPU distributed self-consistency suite</h1>
<p><span class="status {status_class}">{display_status}</span></p>
<p>Every completed topology is compared with the same single-card reference and synchronized fixture.
Missing topologies are shown as NOT RUN. A partial suite is not a final PASS;
<code>--require-all</code> makes missing captures an immediate error.</p>
<table><thead><tr><th>Result</th><th>Topology</th><th>Repeats</th>
<th>Worst loss mean absolute</th><th>Worst loss mean absolute relative</th>
<th>Worst |grad norm mean relative|</th><th>Report</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></body></html>""",
        encoding="utf-8",
    )
    return report


def run_suite_cli(
    base: FormalExperimentConfig,
    script_path: str,
    *,
    topologies: dict[str, ParallelTopology],
    candidate_topologies: Sequence[str],
) -> None:
    parser = argparse.ArgumentParser(
        description="Shared-reference distributed self-consistency suite"
    )
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--data", action="store_true")
    actions.add_argument("--capture", choices=("reference", "candidate"))
    actions.add_argument("--capture-all", action="store_true")
    actions.add_argument("--compare", action="store_true")
    actions.add_argument("--list-topologies", action="store_true")
    parser.add_argument("--topology", choices=tuple(candidate_topologies))
    parser.add_argument(
        "--topologies",
        help="comma-separated candidate subset; defaults to the configured suite",
    )
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--precision", choices=("fp32", "bf16", "full-bf16"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    if args.list_topologies:
        for name in candidate_topologies:
            print(f"{name}: {asdict(topologies[name])}")
        return

    selected = (
        tuple(value.strip() for value in args.topologies.split(",") if value.strip())
        if args.topologies
        else tuple(candidate_topologies)
    )
    unknown = sorted(set(selected) - set(candidate_topologies))
    if unknown:
        raise ValueError(f"topologies are not in this suite: {unknown}")
    if args.topology:
        selected = (args.topology,)
    root = _root(script_path)
    first = _suite_config(base, topologies[selected[0]], precision=args.precision)

    if args.data:
        endpoint = _fixture_endpoint_from_environment(first, "cuda")
        print(
            "Prepared fixture:",
            prepare_fixture(root, first, endpoint=endpoint, force=args.force),
        )
        return
    if args.capture == "reference":
        endpoint = first.reference
        repeats = (args.repeat,) if args.repeat else range(1, endpoint.repeats + 1)
        for repeat in repeats:
            print(
                "Captured reference:",
                capture_endpoint(
                    root,
                    first,
                    role="reference",
                    repeat=repeat,
                    force=args.force,
                ),
            )
        return
    if args.capture == "candidate" and len(selected) != 1:
        raise ValueError("--capture candidate requires one --topology")
    if args.capture == "candidate" or args.capture_all:
        for name in selected:
            config = _suite_config(base, topologies[name], precision=args.precision)
            endpoint = config.candidate
            repeats = (
                (args.repeat,) if args.repeat else range(1, endpoint.repeats + 1)
            )
            for repeat in repeats:
                artifact = _artifact_directory(
                    root,
                    config,
                    "candidate",
                    endpoint,
                    repeat,
                )
                artifact_complete = _artifact_is_complete(artifact)
                if args.capture_all and artifact_complete and not args.force:
                    print(f"Skip completed {name} repeat {repeat}: {artifact}")
                    continue
                retry_incomplete = False
                if args.capture_all and not args.force and endpoint.num_nodes == 1:
                    run_directory = _run_directory(
                        root,
                        config,
                        "candidate",
                        endpoint,
                        repeat,
                    )
                    retry_incomplete = run_directory.exists() or artifact.exists()
                    if retry_incomplete:
                        print(
                            f"Retry incomplete {name} repeat {repeat}; "
                            f"replacing stale run output: {run_directory}"
                        )
                path = capture_endpoint(
                    root,
                    config,
                    role="candidate",
                    repeat=repeat,
                    force=args.force or retry_incomplete,
                )
                print(f"Captured {name} repeat {repeat}: {path}")
        return
    print(
        "Self-consistency suite report:",
        compare_suite(
            root,
            base,
            topologies=topologies,
            selected=selected,
            precision=args.precision,
            require_all=args.require_all,
        ),
    )
