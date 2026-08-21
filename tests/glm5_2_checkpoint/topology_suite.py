# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Resumable topology-suite lifecycle for checkpoint validation."""

from __future__ import annotations

from dataclasses import asdict
import html
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

from tests.glm5_2_common.cli import archive_previous_output, replace_topology
from tests.glm5_2_common.naming import config_name
from tests.glm5_2_common.topology import standard_topologies


CHECKPOINT_SCHEMA = "torchtitan.glm5_2.checkpoint_fault_recovery"


def _normalize_run_tag(value: str) -> str:
    tag = "-".join(
        part
        for part in "".join(
            character if character.isalnum() else "-" for character in value.lower()
        ).split("-")
        if part
    )
    if not tag:
        raise ValueError("run-tag must contain a letter or number")
    return tag


def _failure_mode_label(requested: Sequence[str]) -> str:
    if not requested:
        return "graceful"
    if "all" in requested:
        return "all-faults"
    unique = tuple(dict.fromkeys(requested))
    return unique[0] if len(unique) == 1 else "faults-" + "-".join(unique)


def checkpoint_output_names(
    *,
    device: str,
    topology_slug: str,
    precision: str,
    total_steps: int,
    split_step: int,
    local_batch_size: int,
    global_batch_size: int,
    sequence_length: int,
    seed: int,
    async_mode: str,
    comparison: str,
    requested_failure_modes: Sequence[str],
    run_tag: str | None,
    extra_train_args: Sequence[str] = (),
    interrupt_step: int | None = None,
    failure_rank: int | None = None,
    failure_timeout: float | None = None,
    restart_delay: float | None = None,
    topology_identity: dict[str, Any] | None = None,
) -> tuple[str, str]:
    save_mode = {
        "disabled": "sync",
        "async": "async",
        "async_with_pinned_mem": "pinned",
    }[async_mode]
    suffix = (
        f"{precision}-s{total_steps}-k{split_step}-l{local_batch_size}-"
        f"b{global_batch_size}-seq{sequence_length}-seed{seed}-{save_mode}-"
        f"{comparison}-{_failure_mode_label(requested_failure_modes)}"
    )
    suite_base = f"{device}-{suffix}"
    member_base = f"{device}-{topology_slug}-{suffix}"
    if run_tag:
        normalized_tag = _normalize_run_tag(run_tag)
        suite_base += f"-{normalized_tag}"
        member_base += f"-{normalized_tag}"
    identity = {
        "device": device,
        "precision": precision,
        "total_steps": total_steps,
        "split_step": split_step,
        "local_batch_size": local_batch_size,
        "global_batch_size": global_batch_size,
        "sequence_length": sequence_length,
        "seed": seed,
        "async_mode": async_mode,
        "comparison": comparison,
        "failure_modes": tuple(requested_failure_modes),
        "run_tag": run_tag,
        "extra_train_args": tuple(extra_train_args),
        "interrupt_step": interrupt_step,
        "failure_rank": failure_rank,
        "failure_timeout": failure_timeout,
        "restart_delay": restart_delay,
    }
    suite_name = config_name(suite_base, identity)
    member_name = config_name(
        member_base,
        {**identity, "topology": topology_identity or topology_slug},
    )
    return suite_name, member_name


def legacy_checkpoint_output_names(
    *,
    device: str,
    topology_slug: str,
    precision: str,
    total_steps: int,
    split_step: int,
    local_batch_size: int,
    global_batch_size: int,
    sequence_length: int,
    seed: int,
    async_mode: str,
    comparison: str,
    requested_failure_modes: Sequence[str],
    run_tag: str | None,
    **_: Any,
) -> tuple[str, str]:
    save_mode = {
        "disabled": "sync",
        "async": "async",
        "async_with_pinned_mem": "pinned",
    }[async_mode]
    suffix = (
        f"{precision}-s{total_steps}-k{split_step}-l{local_batch_size}-"
        f"b{global_batch_size}-q{sequence_length}-seed{seed}-{save_mode}-"
        f"{comparison}-{_failure_mode_label(requested_failure_modes)}"
    )
    suite_name = f"ckpt-{device}-{suffix}"
    member_name = f"ckpt-{device}-{topology_slug}-{suffix}"
    if run_tag:
        normalized_tag = _normalize_run_tag(run_tag)
        suite_name += f"-{normalized_tag}"
        member_name += f"-{normalized_tag}"
    return suite_name, member_name


def adopt_legacy_checkpoint_outputs(
    root: Path,
    *,
    suite_name: str,
    member_name: str,
    legacy_suite_name: str,
    legacy_member_name: str,
    topology_slug: str,
) -> None:
    for root_name in ("checkpoint_runs", "checkpoint_reports"):
        parent = root / root_name
        source = parent / legacy_suite_name
        destination = parent / suite_name
        if destination.exists() or not source.is_dir():
            continue
        parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        print(f"Adopted matching legacy output:\n  {source}\n  -> {destination}")
    report_directory = root / "checkpoint_reports" / suite_name / topology_slug
    old_summary = report_directory / f"{legacy_member_name}.json"
    new_summary = report_directory / f"{member_name}.json"
    if old_summary.is_file() and not new_summary.exists():
        summary = json.loads(old_summary.read_text(encoding="utf-8"))
        def rename_value(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: rename_value(item) for key, item in value.items()}
            if isinstance(value, list):
                return [rename_value(item) for item in value]
            if isinstance(value, str):
                return value.replace(legacy_suite_name, suite_name).replace(
                    legacy_member_name, member_name
                )
            return value

        summary = rename_value(summary)
        summary["run_name"] = member_name
        new_summary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        old_summary.unlink()
    old_report = report_directory / f"{legacy_member_name}.html"
    new_report = report_directory / f"{member_name}.html"
    if old_report.is_file() and not new_report.exists():
        report_text = old_report.read_text(encoding="utf-8")
        new_report.write_text(
            report_text.replace(legacy_suite_name, suite_name).replace(
                legacy_member_name, member_name
            ),
            encoding="utf-8",
        )
        old_report.unlink()


def checkpoint_member_paths(
    root: Path,
    *,
    suite_name: str,
    topology_slug: str,
    member_name: str,
) -> tuple[Path, Path, Path, Path]:
    run_directory = root / "checkpoint_runs" / suite_name / topology_slug
    report_directory = root / "checkpoint_reports" / suite_name / topology_slug
    summary_path = report_directory / f"{member_name}.json"
    report_path = report_directory / f"{member_name}.html"
    return run_directory, report_directory, summary_path, report_path


def checkpoint_member_complete(summary_path: Path, *, member_name: str) -> bool:
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return False
    return (
        summary.get("schema") == CHECKPOINT_SCHEMA
        and summary.get("schema_version") == 2
        and summary.get("run_name") == member_name
        and summary.get("passed") is True
    )


def _write_suite_report(
    root: Path,
    *,
    suite_name: str,
    members: Sequence[tuple[str, str]],
) -> tuple[Path, bool]:
    report_root = root / "checkpoint_reports" / suite_name
    report_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    rows: list[str] = []
    for topology_slug, member_name in members:
        _, _, summary_path, report_path = checkpoint_member_paths(
            root,
            suite_name=suite_name,
            topology_slug=topology_slug,
            member_name=member_name,
        )
        summary: dict[str, Any] | None = None
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError):
                summary = None
        passed = bool(
            summary
            and summary.get("schema") == CHECKPOINT_SCHEMA
            and summary.get("schema_version") == 2
            and summary.get("run_name") == member_name
            and summary.get("passed") is True
        )
        status = "PASS" if passed else ("FAIL" if summary else "INCOMPLETE")
        relative_report = report_path.relative_to(report_root).as_posix()
        detail = (
            f'<a href="{html.escape(relative_report)}">open report</a>'
            if report_path.is_file()
            else "-"
        )
        entries.append(
            {
                "topology": topology_slug,
                "member_name": member_name,
                "status": status,
                "passed": passed,
                "summary": summary_path.relative_to(report_root).as_posix(),
                "report": report_path.relative_to(report_root).as_posix(),
            }
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(topology_slug)}</td>"
            f"<td class={'pass' if passed else 'fail'}>{status}</td>"
            f"<td>{detail}</td></tr>"
        )
    passed = bool(entries) and all(entry["passed"] for entry in entries)
    suite_summary = {
        "schema": "torchtitan.glm5_2.checkpoint_topology_suite",
        "schema_version": 1,
        "suite_name": suite_name,
        "passed": passed,
        "members": entries,
    }
    (report_root / "suite_summary.json").write_text(
        json.dumps(suite_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = report_root / f"{suite_name}.html"
    report_path.write_text(
        f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(suite_name)}</title><style>
body{{font:14px/1.45 system-ui,sans-serif;max-width:1100px;margin:auto;padding:24px;color:#17202a}}
table{{border-collapse:collapse;width:100%;margin:16px 0}}th,td{{border:1px solid #d8dee9;padding:8px;text-align:left}}
th{{background:#eef2f7;position:sticky;top:0}}.pass{{color:#16723c;font-weight:800}}.fail{{color:#b42318;font-weight:800}}
</style></head><body><h1>GLM 5.2 checkpoint topology suite</h1>
<p>Suite status: <strong class="{'pass' if passed else 'fail'}">{'PASS' if passed else 'INCOMPLETE / FAIL'}</strong></p>
<table><thead><tr><th>Topology</th><th>Status</th><th>Details</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></body></html>""",
        encoding="utf-8",
    )
    return report_path, passed


def run_checkpoint_topology_suite(
    root: Path,
    *,
    script_path: str,
    argv: Sequence[str],
    topology_names: Sequence[str],
    device: str,
    output_name_args: dict[str, Any],
) -> int:
    members: list[tuple[str, str]] = []
    return_codes: list[int] = []
    suite_name: str | None = None
    topologies = standard_topologies()
    for topology_name in topology_names:
        topology_slug = topologies[topology_name].slug
        current_suite, member_name = checkpoint_output_names(
            device=device,
            topology_slug=topology_slug,
            topology_identity=asdict(topologies[topology_name]),
            **output_name_args,
        )
        if suite_name is None:
            suite_name = current_suite
        elif current_suite != suite_name:
            raise RuntimeError(
                "checkpoint topology members resolved different suites"
            )
        members.append((topology_slug, member_name))
    for topology_name in topology_names:
        command = [
            sys.executable,
            str(Path(script_path).resolve()),
            *replace_topology(argv, topology_name),
        ]
        print(f"Starting topology suite member: {topology_name}", flush=True)
        process = subprocess.run(command, check=False)
        return_codes.append(process.returncode)
        if process.returncode:
            print(
                f"Topology suite member failed: {topology_name} "
                f"(exit code {process.returncode}). Rerun the same command to retry "
                "this member and continue the suite.",
                flush=True,
            )
            break
    assert suite_name is not None
    report_path, suite_passed = _write_suite_report(
        root, suite_name=suite_name, members=members
    )
    print(f"Checkpoint topology suite report: {report_path}", flush=True)
    return 0 if suite_passed and not any(return_codes) else 1
