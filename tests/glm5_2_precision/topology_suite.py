# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Topology-suite orchestration for formal precision experiments."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, replace
import html
import json
from pathlib import Path
from typing import Any, Sequence

from tests.glm5_2_common.topology import ParallelTopology, select_topologies

from .artifacts import PrecisionArtifactError, PrecisionArtifactReader
from .report import compare_and_write_report, precision_label
from .workflow import (
    FormalExperimentConfig,
    _artifact_directory,
    _endpoint_from_process_environment,
    _fixture_endpoint_from_environment,
    _root,
    capture_endpoint,
    prepare_fixture,
    training_topology_plan,
)


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


def topology_config(
    base: FormalExperimentConfig,
    topology: ParallelTopology,
    *,
    precision: str | None,
    environment_role: str | None = None,
) -> FormalExperimentConfig:
    """Create one topology member while preserving the shared fixture identity."""

    training = _training_with_precision(base, precision)
    fixture_base = replace(base, training=training)
    reference = base.reference
    candidate = base.candidate
    if base.kind == "migration":
        reference = replace(reference, topology=topology)
        candidate = replace(candidate, topology=topology)
    else:
        candidate = replace(candidate, topology=topology)
    if environment_role == "reference":
        reference = _endpoint_from_process_environment(reference)
    elif environment_role == "candidate":
        candidate = _endpoint_from_process_environment(candidate)
    return replace(
        base,
        reference=reference,
        candidate=candidate,
        training=training,
        fixture_name=fixture_base.fixture_storage_name,
    )


def _capture_complete(root: Path, config: FormalExperimentConfig, role: str) -> bool:
    endpoint = config.reference if role == "reference" else config.candidate
    for repeat in range(1, endpoint.repeats + 1):
        path = _artifact_directory(root, config, role, endpoint, repeat)
        try:
            PrecisionArtifactReader(path)
        except (KeyError, OSError, TypeError, ValueError, PrecisionArtifactError):
            return False
    return True


def _suite_directory(root: Path, config: FormalExperimentConfig) -> Path:
    suite_config = replace(config, topology_subdirectory=True)
    return root / config.report_root / suite_config.storage_name


def compare_topology_suite(
    root: Path,
    base: FormalExperimentConfig,
    *,
    topologies: dict[str, ParallelTopology],
    selected: Sequence[str],
    precision: str | None,
    require_all: bool,
    config_transform: Callable[
        [FormalExperimentConfig], FormalExperimentConfig
    ]
    | None = None,
) -> Path:
    """Compare every complete topology and write one suite index."""

    configs = {
        name: topology_config(base, topologies[name], precision=precision)
        for name in selected
    }
    if config_transform is not None:
        configs = {
            name: config_transform(config) for name, config in configs.items()
        }
    completed: dict[str, FormalExperimentConfig] = {}
    missing: dict[str, list[str]] = {}
    for name, config in configs.items():
        missing_roles = [
            role
            for role in ("reference", "candidate")
            if not _capture_complete(root, config, role)
        ]
        if missing_roles:
            missing[name] = missing_roles
        else:
            completed[name] = config
    if require_all and missing:
        detail = ", ".join(
            f"{name} ({'/'.join(roles)})" for name, roles in missing.items()
        )
        raise FileNotFoundError(f"topology captures are incomplete: {detail}")
    if not completed:
        raise FileNotFoundError(
            "no topology has complete reference and candidate captures"
        )

    first = next(iter(configs.values()))
    suite_directory = _suite_directory(root, first)
    rows: list[str] = []
    summaries: dict[str, dict[str, Any]] = {}
    for name, config in completed.items():
        detail_directory = suite_directory / name
        report_path = compare_and_write_report(
            root, config, report_directory=detail_directory
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
            f"<td>{html.escape(name)}</td><td>{len(candidate_runs)}</td>"
            f"<td>{max_loss_absolute:.9g}</td><td>{max_loss_relative:.6%}</td>"
            f"<td>{max_grad_relative:.6%}</td>"
            f'<td><a href="{html.escape(name)}/{html.escape(report_path.name)}">'
            "details</a></td></tr>"
        )
    for name, roles in missing.items():
        rows.append(
            '<tr><td><span class="status pending">NOT RUN</span></td>'
            f"<td>{html.escape(name)}</td><td>-</td><td>-</td><td>-</td>"
            f"<td>-</td><td>Missing {html.escape(', '.join(roles))}</td></tr>"
        )

    completed_passed = all(bool(summary["passed"]) for summary in summaries.values())
    complete = not missing
    overall_passed = completed_passed and complete
    status = (
        "PASS"
        if overall_passed
        else "PARTIAL PASS" if completed_passed else "FAIL"
    )
    status_class = "pass" if completed_passed else "fail"
    suite_summary = {
        "fixture": first.fixture_storage_name,
        "kind": first.kind,
        "passed": overall_passed,
        "completed_topologies_passed": completed_passed,
        "complete": complete,
        "status": status,
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
    report_name = (
        f"{first.reference.device_type}-vs-{first.candidate.device_type}-"
        f"{precision_label(first)}-topology-suite.html"
    )
    report = suite_directory / report_name
    report.write_text(
        f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GLM-5.2 topology precision suite</title><style>
body{{font:14px/1.45 system-ui,sans-serif;max-width:1400px;margin:auto;padding:24px;color:#17202a}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d8dee9;padding:9px;text-align:left}}
th{{background:#eef2f7;position:sticky;top:0}}.status{{display:inline-block;color:#fff;padding:3px 8px;border-radius:999px;font-weight:700}}
.pass{{background:#17803d}}.fail{{background:#b42318}}.pending{{background:#64748b}}a{{color:#155eef}}
</style></head><body><h1>GLM-5.2 topology precision suite</h1>
<p><span class="status {status_class}">{status}</span></p>
<p>All rows use fixture <code>{html.escape(first.fixture_storage_name)}</code>.
Each topology compares captures with the same token plan, checkpoint, optimizer-step
definition, and training configuration.</p>
<table><thead><tr><th>Result</th><th>Topology</th><th>Candidate repeats</th>
<th>Worst loss mean absolute</th><th>Worst loss mean absolute relative</th>
<th>Worst |grad norm mean relative|</th><th>Report</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></body></html>""",
        encoding="utf-8",
    )
    return report


def run_topology_suite_cli(
    base: FormalExperimentConfig,
    script_path: str,
    *,
    topologies: dict[str, ParallelTopology],
    topology_names: Sequence[str],
) -> None:
    """Run data, endpoint capture, and compare for one or all topologies."""

    parser = argparse.ArgumentParser(description="Formal topology precision suite")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--data", action="store_true")
    actions.add_argument("--capture", choices=("reference", "candidate"))
    actions.add_argument(
        "--capture-all",
        action="store_true",
        help="compatibility alias for --capture candidate --topology all",
    )
    actions.add_argument("--compare", action="store_true")
    actions.add_argument("--list-topologies", action="store_true")
    parser.add_argument("--topology", choices=("all", *topology_names), default="all")
    parser.add_argument("--topologies", help="comma-separated topology subset or all")
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--precision", choices=("fp32", "bf16", "full-bf16"))
    parser.add_argument("--data-device", choices=("cuda", "npu"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    selected = select_topologies(
        available=topology_names,
        topology=None if args.topologies else args.topology,
        topologies=args.topologies,
    )
    if args.list_topologies:
        for name in topology_names:
            topology = topologies[name]
            try:
                schedule: dict[str, Any] = asdict(
                    training_topology_plan(base.training, topology)
                )
            except ValueError as error:
                schedule = {"invalid": str(error)}
            print(f"{name}: topology={asdict(topology)}, batch_schedule={schedule}")
        return

    role = "candidate" if args.capture_all else args.capture
    environment_role = role if role is not None else None
    configs = {
        name: topology_config(
            base,
            topologies[name],
            precision=args.precision,
            environment_role=environment_role,
        )
        for name in selected
    }
    root = _root(script_path)
    first = configs[selected[0]]
    if args.data:
        endpoint = _fixture_endpoint_from_environment(first, args.data_device)
        print(
            "Prepared fixture:",
            prepare_fixture(root, first, endpoint=endpoint, force=args.force),
        )
        return
    if role is not None:
        seen: set[Path] = set()
        for name, config in configs.items():
            endpoint = config.reference if role == "reference" else config.candidate
            repeats = (args.repeat,) if args.repeat else range(1, endpoint.repeats + 1)
            for repeat in repeats:
                artifact = _artifact_directory(root, config, role, endpoint, repeat)
                if artifact in seen:
                    print(f"Reuse shared {role} capture for {name}: {artifact}")
                    continue
                seen.add(artifact)
                path = capture_endpoint(
                    root,
                    config,
                    role=role,
                    repeat=repeat,
                    force=args.force,
                )
                print(f"Captured {role} {name} repeat {repeat}: {path}")
        return
    print(
        "Topology suite report:",
        compare_topology_suite(
            root,
            base,
            topologies=topologies,
            selected=selected,
            precision=args.precision,
            require_all=args.require_all,
        ),
    )
