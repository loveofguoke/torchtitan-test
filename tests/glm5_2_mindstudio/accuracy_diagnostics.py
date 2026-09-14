# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Stateful diagnosis cases for the official MindStudio accuracy workflows."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
from typing import Any, Literal, Sequence

from tests.glm5_2_common.cli import (
    RunAttempt,
    archive_previous_output,
    assert_run_not_active,
)
from tests.glm5_2_common.topology import select_topologies, standard_topologies
from tests.glm5_2_mindstudio.artifacts import (
    artifact_is_complete,
    sha256_file,
    write_json,
)
from tests.glm5_2_mindstudio.configuration_check_benchmark import (
    CONFIG as CONFIG_CHECK_CONFIG,
)
from tests.glm5_2_mindstudio.migration_benchmark import (
    CONFIG as MIGRATION_CONFIG,
)
from tests.glm5_2_mindstudio.msprobe_adapter import (
    compare_command,
    find_dump_compare_input,
    summarize_official_results,
    write_compare_invocation,
)
from tests.glm5_2_mindstudio.training_monitor_benchmark import (
    CONFIG as MONITOR_CONFIG,
)
from tests.glm5_2_mindstudio.training_observation import compare_training_metrics
from tests.glm5_2_mindstudio.workflow import (
    _experiment_digest,
    _run_process,
    _stage_scoped_config,
)


Symptom = Literal[
    "unknown",
    "nan-or-overflow",
    "first-step-loss",
    "long-term-loss",
    "spike",
    "downstream-metric",
    "unstable",
]
Stage = Literal[
    "checklist",
    "reproduce",
    "observe",
    "localize",
    "verify",
    "validate",
]

SCHEMA = "torchtitan.glm5_2.mindstudio_diagnostic_case"
SCHEMA_VERSION = 1
CASE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
STAGES: tuple[Stage, ...] = (
    "checklist",
    "reproduce",
    "observe",
    "localize",
    "verify",
    "validate",
)
STAGE_TITLES = {
    "checklist": "CheckList and experiment contract",
    "reproduce": "same-endpoint reproduction",
    "observe": "whole-training observation and classification",
    "localize": "first-incident localization",
    "verify": "root-cause verification",
    "validate": "fix and delivery validation",
}
CONCLUSIONS = {
    "checklist": {"pass", "fail", "inconclusive"},
    "reproduce": {"stable", "unstable", "not-reproduced", "inconclusive"},
    "observe": {"normal", "abnormal", "inconclusive"},
    "localize": {"localized", "not-localized", "inconclusive"},
    "verify": {"confirmed", "rejected", "partial", "inconclusive"},
    "validate": {"pass", "fail", "inconclusive"},
}
ADVANCE_CONCLUSIONS = {
    "checklist": {"pass"},
    "reproduce": {"stable", "unstable"},
    "observe": {"abnormal"},
    "localize": {"localized"},
    "verify": {"confirmed"},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _quote_command(arguments: Sequence[str]) -> str:
    return shlex.join(arguments)


def _python_command(script: str, *arguments: str) -> str:
    return _quote_command(("python", script, *arguments))


def _case_root(repository_root: Path, case_id: str) -> Path:
    return (
        repository_root
        / MIGRATION_CONFIG.artifact_root
        / case_id
    )


def _legacy_case_root(repository_root: Path, case_id: str) -> Path:
    return repository_root / "mindstudio_cases" / "accuracy" / case_id


def _nested_case_root(repository_root: Path, case_id: str) -> Path:
    return (
        repository_root
        / MIGRATION_CONFIG.artifact_root
        / MIGRATION_CONFIG.storage_name
        / "cases"
        / case_id
    )


def _case_path(repository_root: Path, case_id: str) -> Path:
    return _case_root(repository_root, case_id) / "case.json"


def _load_case(repository_root: Path, case_id: str) -> dict[str, Any]:
    path = _case_path(repository_root, case_id)
    legacy_roots = (
        _nested_case_root(repository_root, case_id),
        _legacy_case_root(repository_root, case_id),
    )
    for legacy_root in legacy_roots:
        if not path.exists() and legacy_root.is_dir():
            path.parent.parent.mkdir(parents=True, exist_ok=True)
            legacy_root.rename(path.parent)
            print(f"Adopted legacy diagnostic case:\n  {legacy_root}\n  -> {path.parent}")
    if not path.is_file():
        raise FileNotFoundError(f"diagnostic case does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported diagnostic case schema: {path}")
    return value


def _write_case(repository_root: Path, value: dict[str, Any]) -> Path:
    value["updated_at"] = _utc_now()
    path = _case_path(repository_root, value["case_id"])
    write_json(path, value)
    _write_case_report(repository_root, value)
    return path


def create_case(
    repository_root: Path,
    *,
    case_id: str,
    title: str,
    symptom: Symptom,
    topologies: Sequence[str],
    repeat: int,
    notes: str,
) -> Path:
    if not CASE_ID_PATTERN.fullmatch(case_id):
        raise ValueError(
            "case ID must start with a lowercase letter or digit and contain "
            "only lowercase letters, digits, '.', '_' or '-'"
        )
    if repeat < 1:
        raise ValueError("repeat must be positive")
    path = _case_path(repository_root, case_id)
    if path.exists():
        raise FileExistsError(f"diagnostic case already exists: {path}")
    registry = standard_topologies()
    unknown = sorted(set(topologies) - set(registry))
    if unknown:
        raise ValueError(f"unknown topologies: {', '.join(unknown)}")
    now = _utc_now()
    value: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "title": title,
        "status": "open",
        "symptom": symptom,
        "topologies": list(topologies),
        "repeat": repeat,
        "notes": notes,
        "created_at": now,
        "updated_at": now,
        "official_guidance": {
            "first_or_early_loss_mean_relative_error": "greater than 1%",
            "long_term_late_window_mean_relative_error": "greater than 1%",
            "source": (
                "MindStudio Large Model Training Accuracy Debugging Guide"
            ),
            "warning": (
                "The guide does not define a universal training length, "
                "window aggregation, grad-norm threshold, or task metric tolerance."
            ),
        },
        "experiment": {
            "storage_name": case_id,
            "stages": {
                "dump": MIGRATION_CONFIG.output_subdirectory,
                "configuration_check": CONFIG_CHECK_CONFIG.output_subdirectory,
                "monitor": "diagnostics/monitor/<configuration>",
            },
        },
        "stages": {
            stage: {
                "status": "pending",
                "conclusion": None,
                "notes": "",
                "evidence": [],
                "updated_at": None,
            }
            for stage in STAGES
        },
        "incident": {
            "step": None,
            "rank": None,
            "phase": None,
            "module": None,
            "api": None,
        },
        "hypotheses": [],
        "events": [
            {
                "at": now,
                "action": "created",
                "detail": f"symptom={symptom}",
            }
        ],
    }
    return _write_case(repository_root, value)


def _resolved_evidence(repository_root: Path, values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        path = Path(value)
        resolved = path if path.is_absolute() else repository_root / path
        if not resolved.exists():
            raise FileNotFoundError(f"diagnostic evidence does not exist: {resolved}")
        try:
            result.append(
                resolved.resolve()
                .relative_to(repository_root.resolve())
                .as_posix()
            )
        except ValueError:
            result.append(str(resolved.resolve()))
    return result


def record_stage(
    repository_root: Path,
    *,
    case_id: str,
    stage: Stage,
    conclusion: str,
    evidence: Sequence[str],
    notes: str,
    incident: dict[str, Any],
    symptom: Symptom | None = None,
) -> Path:
    if conclusion not in CONCLUSIONS[stage]:
        expected = ", ".join(sorted(CONCLUSIONS[stage]))
        raise ValueError(f"invalid {stage} conclusion; choose one of: {expected}")
    value = _load_case(repository_root, case_id)
    index = STAGES.index(stage)
    if index:
        previous_stage = STAGES[index - 1]
        previous = value["stages"][previous_stage]
        if previous["status"] != "complete":
            raise ValueError(f"complete stage '{previous_stage}' before '{stage}'")
        allowed = ADVANCE_CONCLUSIONS[previous_stage]
        if previous["conclusion"] not in allowed:
            raise ValueError(
                f"stage '{previous_stage}' conclusion '{previous['conclusion']}' "
                f"does not permit advancing to '{stage}'"
            )
    resolved = _resolved_evidence(repository_root, evidence)
    if not resolved:
        raise ValueError("a completed diagnostic stage requires evidence")
    now = _utc_now()
    if stage == "localize" and conclusion == "localized":
        merged_incident = {
            **value["incident"],
            **{key: item for key, item in incident.items() if item is not None},
        }
        if (
            merged_incident.get("step") is None
            or merged_incident.get("rank") is None
            or merged_incident.get("phase") is None
            or not (merged_incident.get("module") or merged_incident.get("api"))
        ):
            raise ValueError(
                "a localized incident requires step, rank, phase, and module or API"
            )
    if symptom is not None:
        value["symptom"] = symptom
    value["stages"][stage] = {
        "status": "complete",
        "conclusion": conclusion,
        "notes": notes,
        "evidence": resolved,
        "updated_at": now,
    }
    for key, item in incident.items():
        if item is not None:
            value["incident"][key] = item
    value["events"].append(
        {
            "at": now,
            "action": "record_stage",
            "detail": f"{stage}={conclusion}",
        }
    )
    return _write_case(repository_root, value)


def add_hypothesis(
    repository_root: Path,
    *,
    case_id: str,
    statement: str,
    experiment: str,
) -> Path:
    value = _load_case(repository_root, case_id)
    hypothesis_id = len(value["hypotheses"]) + 1
    value["hypotheses"].append(
        {
            "id": hypothesis_id,
            "statement": statement,
            "experiment": experiment,
            "verdict": "pending",
            "notes": "",
            "evidence": [],
            "updated_at": _utc_now(),
        }
    )
    value["events"].append(
        {
            "at": _utc_now(),
            "action": "add_hypothesis",
            "detail": f"hypothesis={hypothesis_id}",
        }
    )
    return _write_case(repository_root, value)


def record_hypothesis(
    repository_root: Path,
    *,
    case_id: str,
    hypothesis_id: int,
    verdict: str,
    evidence: Sequence[str],
    notes: str,
) -> Path:
    value = _load_case(repository_root, case_id)
    matches = [item for item in value["hypotheses"] if item["id"] == hypothesis_id]
    if not matches:
        raise ValueError(f"unknown hypothesis ID: {hypothesis_id}")
    if verdict not in {"supported", "rejected", "partial", "inconclusive"}:
        raise ValueError("invalid hypothesis verdict")
    resolved = _resolved_evidence(repository_root, evidence)
    if not resolved:
        raise ValueError("a hypothesis verdict requires evidence")
    item = matches[0]
    item.update(
        verdict=verdict,
        notes=notes,
        evidence=resolved,
        updated_at=_utc_now(),
    )
    value["events"].append(
        {
            "at": _utc_now(),
            "action": "record_hypothesis",
            "detail": f"hypothesis={hypothesis_id}, verdict={verdict}",
        }
    )
    return _write_case(repository_root, value)


def close_case(repository_root: Path, *, case_id: str) -> Path:
    value = _load_case(repository_root, case_id)
    incomplete = [
        stage for stage in STAGES if value["stages"][stage]["status"] != "complete"
    ]
    if incomplete:
        raise ValueError(
            f"cannot close case; incomplete stages: {', '.join(incomplete)}"
        )
    required = {
        "checklist": "pass",
        "observe": "abnormal",
        "localize": "localized",
        "verify": "confirmed",
        "validate": "pass",
    }
    failed = [
        f"{stage}={value['stages'][stage]['conclusion']}"
        for stage, conclusion in required.items()
        if value["stages"][stage]["conclusion"] != conclusion
    ]
    if failed:
        raise ValueError(
            "cannot close case without a confirmed and validated root cause: "
            + ", ".join(failed)
        )
    if not any(item["verdict"] == "supported" for item in value["hypotheses"]):
        raise ValueError("cannot close case without a supported hypothesis")
    value["status"] = "closed"
    value["events"].append(
        {"at": _utc_now(), "action": "closed", "detail": "validation passed"}
    )
    return _write_case(repository_root, value)


def _selected_topology_args(value: dict[str, Any]) -> tuple[str, ...]:
    names = tuple(value["topologies"])
    if len(names) == 1:
        return ("--topology", names[0])
    return ("--topologies", ",".join(names))


def _migration_command(
    value: dict[str, Any],
    *arguments: str,
    repeat: int | None = None,
) -> str:
    return _python_command(
        "tests/glm5_2_mindstudio/accuracy_benchmark.py",
        "--stage",
        "dump",
        "--experiment",
        value["case_id"],
        *arguments,
        *_selected_topology_args(value),
        "--repeat",
        str(value["repeat"] if repeat is None else repeat),
    )


def _monitor_command(value: dict[str, Any], *arguments: str) -> str:
    return _python_command(
        "tests/glm5_2_mindstudio/accuracy_benchmark.py",
        "--stage",
        "monitor",
        "--experiment",
        value["case_id"],
        *arguments,
        *_selected_topology_args(value),
        "--repeat",
        str(value["repeat"]),
    )


def _checklist_plan(value: dict[str, Any]) -> dict[str, Any]:
    topology_args = _selected_topology_args(value)
    script = "tests/glm5_2_mindstudio/accuracy_benchmark.py"
    return {
        "stage": "checklist",
        "goal": "Prove that both endpoints execute the same experiment contract.",
        "commands": [
            _python_command(
                script, "--stage", "dump", "--experiment", value["case_id"],
                "--data", *topology_args,
            ),
            _python_command(
                script,
                "--stage",
                "config-check",
                "--experiment",
                value["case_id"],
                "--capture",
                "reference",
                *topology_args,
            ),
            _python_command(
                script,
                "--stage",
                "config-check",
                "--experiment",
                value["case_id"],
                "--capture",
                "candidate",
                *topology_args,
            ),
            _python_command(
                script,
                "--stage",
                "config-check",
                "--experiment",
                value["case_id"],
                "--compare",
                *topology_args,
            ),
        ],
        "inspect": [
            "Review every rank's config-check summary and detailed sheets.",
            "Attach model-structure, checkpoint, token-plan, source, and "
            "rank-map evidence.",
        ],
        "decision": (
            "Record pass only after every precision-relevant difference is explained."
        ),
    }


def _reproduce_plan(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": "reproduce",
        "goal": (
            "Establish same-endpoint repeatability before cross-device attribution."
        ),
        "commands": [
            _migration_command(
                value,
                "--data",
                "--data-device",
                "npu",
                "--summary-mode",
                "md5",
                "--level",
                "mix",
                repeat=1,
            ),
            _migration_command(
                value,
                "--capture",
                "candidate",
                "--summary-mode",
                "md5",
                "--level",
                "mix",
                repeat=1,
            ),
            _migration_command(
                value,
                "--capture",
                "candidate",
                "--summary-mode",
                "md5",
                "--level",
                "mix",
                repeat=2,
            ),
            _python_command(
                "tests/glm5_2_mindstudio/accuracy_diagnostic_benchmark.py",
                "compare-repeats",
                value["case_id"],
                "--role",
                "candidate",
                "--baseline-repeat",
                "1",
                "--target-repeat",
                "2",
            ),
        ],
        "inspect": [
            "Repeat the selected endpoint with the same fixture and compare "
            "Loss/Grad Norm.",
            "For an unstable incident, collect two MD5 runs and find the "
            "first unequal node.",
            "Record deterministic settings and whether instrumentation "
            "changes the symptom.",
        ],
        "decision": "Classify the symptom as stable, unstable, or not reproduced.",
    }


def compare_repeats(
    repository_root: Path,
    *,
    case_id: str,
    role: str,
    baseline_repeat: int,
    target_repeat: int,
) -> Path:
    if baseline_repeat == target_repeat:
        raise ValueError("baseline and target repeats must be different")
    value = _load_case(repository_root, case_id)
    config = replace(
        MIGRATION_CONFIG,
        dump=replace(MIGRATION_CONFIG.dump, level="mix", summary_mode="md5"),
    )
    config = _stage_scoped_config(config, MIGRATION_CONFIG, case_id)
    endpoint = config.reference if role == "reference" else config.candidate
    for repeat_value in (baseline_repeat, target_repeat):
        if not 1 <= repeat_value <= endpoint.repeats:
            raise ValueError(
                f"{role} repeat must be in [1, {endpoint.repeats}]"
            )
    output_root = (
        _case_root(repository_root, case_id)
        / "02_reproduce"
        / f"{role}-r{baseline_repeat}-vs-r{target_repeat}"
    )
    registry = standard_topologies()
    comparison_inputs: list[dict[str, Any]] = []
    jobs: list[tuple[str, Path, Path, Path]] = []
    for topology_name in value["topologies"]:
        topology = registry[topology_name]
        artifact_root = (
            repository_root
            / config.artifact_root
            / config.output_relative_root
            / topology.slug
        )
        baseline = artifact_root / f"{role}-r{baseline_repeat}"
        target = artifact_root / f"{role}-r{target_repeat}"
        manifests: list[dict[str, Any]] = []
        for artifact in (baseline, target):
            manifest_path = artifact / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    "repeat capture is incomplete; run the planned capture first: "
                    f"{artifact}"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            generation = manifest.get("fixture_generation_id")
            if not isinstance(generation, str) or not artifact_is_complete(
                artifact,
                experiment_digest=_experiment_digest(config, topology, role),
                fixture_generation_id=generation,
            ):
                raise FileNotFoundError(
                    "repeat capture is incomplete; run the planned capture first: "
                    f"{artifact}"
                )
            manifests.append(manifest)
        baseline_manifest, target_manifest = manifests
        if (
            baseline_manifest.get("fixture_generation_id")
            != target_manifest.get("fixture_generation_id")
        ):
            raise ValueError("repeat captures belong to different fixture generations")
        comparison_inputs.append(
            {
                "topology": topology_name,
                "baseline_manifest_sha256": sha256_file(baseline / "manifest.json"),
                "target_manifest_sha256": sha256_file(target / "manifest.json"),
            }
        )
        topology_output = output_root / topology.slug
        for step in config.dump.steps:
            jobs.append(
                (
                    topology_name,
                    find_dump_compare_input(baseline, step),
                    find_dump_compare_input(target, step),
                    topology_output / f"step{step}",
                )
            )
    completion = {
        "schema": "torchtitan.glm5_2.mindstudio_repeat_comparison",
        "schema_version": 1,
        "case_id": case_id,
        "role": role,
        "baseline_repeat": baseline_repeat,
        "target_repeat": target_repeat,
        "inputs": comparison_inputs,
    }
    complete_path = output_root / "complete.json"
    assert_run_not_active(
        output_root,
        state_name="repeat_compare_state.json",
    )
    if complete_path.is_file():
        if json.loads(complete_path.read_text(encoding="utf-8")) == completion:
            print(f"Skip completed repeat comparison: {output_root}")
            return output_root
        raise FileExistsError(
            "repeat comparison output belongs to different inputs: "
            f"{output_root}"
        )
    if output_root.exists():
        archived = archive_previous_output(output_root)
        print(f"Retry incomplete repeat comparison; archived: {archived}")
    output_root.mkdir(parents=True, exist_ok=True)
    attempt = RunAttempt.start(
        output_root,
        kind="mindstudio_repeat_compare",
        state_name="repeat_compare_state.json",
        context={
            "case_id": case_id,
            "role": role,
            "baseline_repeat": baseline_repeat,
            "target_repeat": target_repeat,
        },
    )
    try:
        for topology_name, baseline_input, target_input, step_output in jobs:
            step = int(step_output.name.removeprefix("step"))
            command = compare_command(
                target=target_input,
                golden=baseline_input,
                output=step_output,
            )
            step_output.mkdir(parents=True, exist_ok=True)
            write_compare_invocation(step_output / "invocation.json", command)
            _run_process(
                command,
                root=repository_root,
                environment=os.environ.copy(),
                log_path=step_output / "runtime.log",
                context={
                    **attempt.log_context,
                    "Topology": topology_name,
                    "Step": step,
                },
            )
        for topology_name in value["topologies"]:
            summarize_official_results(
                output_root / registry[topology_name].slug
            )
        write_json(complete_path, completion)
        attempt.update("completed", output=str(output_root.resolve()))
    except BaseException as error:
        attempt.update("failed", error=repr(error))
        raise
    return output_root


def analyze_training_observation(
    repository_root: Path,
    *,
    case_id: str,
    workflow: str,
    training_steps: int | None,
    loss_relative_threshold: float,
    grad_norm_relative_threshold: float | None,
    spike_relative_threshold: float | None,
    force: bool,
) -> Path:
    value = _load_case(repository_root, case_id)
    if workflow == "migration":
        config = replace(
            MIGRATION_CONFIG,
            dump=replace(MIGRATION_CONFIG.dump, level="mix"),
        )
        output_root = (
            _case_root(repository_root, case_id) / "03_observe/migration"
        )
    else:
        steps = (
            MONITOR_CONFIG.training.steps
            if training_steps is None
            else training_steps
        )
        if steps < 1:
            raise ValueError("training steps must be positive")
        config = replace(
            MONITOR_CONFIG,
            training=replace(MONITOR_CONFIG.training, steps=steps),
        )
        output_root = (
            _case_root(repository_root, case_id)
            / "03_observe/monitor"
            / f"s{steps}"
        )
    config = _stage_scoped_config(config, MIGRATION_CONFIG, case_id)
    inputs: list[dict[str, Any]] = []
    jobs: list[tuple[str, Path, Path, Path]] = []
    registry = standard_topologies()
    for topology_name in value["topologies"]:
        topology = registry[topology_name]
        run_root = repository_root / config.run_root / config.output_relative_root
        reference = (
            run_root
            / topology.slug
            / f"reference-r{value['repeat']}"
            / "training_metrics.jsonl"
        )
        candidate = (
            run_root
            / topology.slug
            / f"candidate-r{value['repeat']}"
            / "training_metrics.jsonl"
        )
        for path in (reference, candidate):
            if not path.is_file():
                raise FileNotFoundError(
                    "whole-training metrics are missing; run both planned "
                    f"captures first: {path}"
                )
        inputs.append(
            {
                "topology": topology_name,
                "reference": sha256_file(reference),
                "candidate": sha256_file(candidate),
            }
        )
        jobs.append(
            (topology_name, reference, candidate, output_root / topology.slug)
        )
    completion = {
        "schema": "torchtitan.glm5_2.mindstudio_training_observation_index",
        "schema_version": 1,
        "case_id": case_id,
        "workflow": workflow,
        "repeat": value["repeat"],
        "loss_relative_threshold": loss_relative_threshold,
        "grad_norm_relative_threshold": grad_norm_relative_threshold,
        "spike_relative_threshold": spike_relative_threshold,
        "inputs": inputs,
    }
    complete_path = output_root / "complete.json"
    assert_run_not_active(output_root, state_name="observation_state.json")
    if complete_path.is_file():
        if json.loads(complete_path.read_text(encoding="utf-8")) == completion:
            print(f"Skip completed training observation: {output_root}")
            return output_root
        if not force:
            raise FileExistsError(
                "training observation uses different inputs or thresholds; "
                f"pass --force to replace derived output only: {output_root}"
            )
    if output_root.exists():
        archived = archive_previous_output(output_root)
        print(f"Replace training observation; archived: {archived}")
    output_root.mkdir(parents=True, exist_ok=True)
    attempt = RunAttempt.start(
        output_root,
        kind="mindstudio_training_observation",
        state_name="observation_state.json",
        context={"case_id": case_id, "workflow": workflow},
    )
    try:
        summaries = []
        for topology_name, reference, candidate, output in jobs:
            summary_path = compare_training_metrics(
                reference_path=reference,
                candidate_path=candidate,
                output_directory=output,
                loss_relative_threshold=loss_relative_threshold,
                grad_norm_relative_threshold=grad_norm_relative_threshold,
                spike_relative_threshold=spike_relative_threshold,
            )
            summaries.append(
                {
                    "topology": topology_name,
                    "summary": summary_path.relative_to(output_root).as_posix(),
                }
            )
        write_json(output_root / "index.json", {**completion, "outputs": summaries})
        write_json(complete_path, completion)
        attempt.update("completed", output=str(output_root.resolve()))
    except BaseException as error:
        attempt.update("failed", error=repr(error))
        raise
    return output_root


def _observe_plan(value: dict[str, Any]) -> dict[str, Any]:
    symptom = value["symptom"]
    if symptom in {"long-term-loss", "spike", "downstream-metric", "unknown"}:
        commands = [
            _monitor_command(
                value,
                "--data",
                "--data-device",
                "npu",
                "--training-steps",
                "100",
            ),
            _monitor_command(
                value,
                "--capture",
                "reference",
                "--training-steps",
                "100",
            ),
            _monitor_command(
                value,
                "--capture",
                "candidate",
                "--training-steps",
                "100",
            ),
            _monitor_command(
                value,
                "--trend",
                "reference",
                "--training-steps",
                "100",
            ),
            _monitor_command(
                value,
                "--trend",
                "candidate",
                "--training-steps",
                "100",
            ),
            _python_command(
                "tests/glm5_2_mindstudio/accuracy_diagnostic_benchmark.py",
                "training-observation",
                value["case_id"],
                "--workflow",
                "monitor",
                "--training-steps",
                "100",
            ),
        ]
        inspect = [
            "Choose the reproduction window explicitly; 100 is a placeholder, "
            "not a standard.",
            "Compare Loss, Grad Norm, NaN/Inf, spikes, and the task metric.",
            "Use the trend database to identify the earliest sustained "
            "step/rank/layer drift.",
        ]
    else:
        commands = [
            _migration_command(value, "--data", "--data-device", "npu"),
            _migration_command(value, "--capture", "reference", "--level", "mix"),
            _migration_command(value, "--capture", "candidate", "--level", "mix"),
            _migration_command(value, "--compare", "--level", "mix"),
            _migration_command(value, "--graph-visualize", "--level", "mix"),
            _python_command(
                "tests/glm5_2_mindstudio/accuracy_diagnostic_benchmark.py",
                "training-observation",
                value["case_id"],
                "--workflow",
                "migration",
            ),
        ]
        inspect = [
            "Find the earliest differentiating step, rank, forward/backward "
            "phase, and module.",
            "Distinguish an already-different input from a normal-input "
            "abnormal output.",
        ]
    return {
        "stage": "observe",
        "goal": (
            "Classify the whole-training symptom and identify the first "
            "incident window."
        ),
        "commands": commands,
        "inspect": inspect,
        "decision": (
            "Record normal, abnormal, or inconclusive with whole-training evidence."
        ),
    }


def _localize_plan(value: dict[str, Any]) -> dict[str, Any]:
    symptom = value["symptom"]
    incident = value["incident"]
    step = incident["step"] if incident["step"] is not None else 0
    rank = incident["rank"] if incident["rank"] is not None else 0
    common = (
        "--dump-steps",
        str(step),
        "--dump-ranks",
        str(rank),
    )
    commands: list[str]
    inspect: list[str]
    if symptom == "nan-or-overflow":
        commands = [
            _migration_command(
                value,
                "--data",
                "--data-device",
                "npu",
                "--dump-task",
                "statistics",
                "--level",
                "mix",
                *common,
            ),
            _migration_command(
                value,
                "--capture",
                "candidate",
                "--dump-task",
                "statistics",
                "--level",
                "mix",
                *common,
            ),
            _migration_command(
                value,
                "--overflow-check",
                "candidate",
                "--dump-task",
                "statistics",
                "--level",
                "mix",
                *common,
            ),
            _migration_command(
                value,
                "--graph-visualize",
                "--dump-task",
                "statistics",
                "--level",
                "mix",
                "--graph-overflow-check",
                "--graph-side",
                "candidate",
                *common,
            ),
        ]
        inspect = [
            "If weight is first abnormal, move to the previous step's backward pass.",
            "If input is first abnormal, follow stack evidence upstream.",
            "If normal input creates an abnormal output, verify that API.",
        ]
    elif (
        symptom == "unstable"
        or value["stages"]["reproduce"]["conclusion"] == "unstable"
    ):
        commands = [
            _migration_command(
                value,
                "--data",
                "--data-device",
                "npu",
                "--dump-task",
                "statistics",
                "--summary-mode",
                "md5",
                "--level",
                "mix",
                *common,
            ),
            _migration_command(
                value,
                "--data",
                "--data-device",
                "npu",
                "--dump-task",
                "statistics",
                "--level",
                "L1",
                "--async-dump",
                *common,
            ),
            _migration_command(
                value,
                "--capture",
                "candidate",
                "--dump-task",
                "statistics",
                "--summary-mode",
                "md5",
                "--level",
                "mix",
                *common,
            ),
            _migration_command(
                value,
                "--capture",
                "candidate",
                "--dump-task",
                "statistics",
                "--level",
                "L1",
                "--async-dump",
                *common,
            ),
        ]
        inspect = [
            "Use MD5 repeats to find the first same-input/different-output node.",
            "Compare blocking, synchronous dump, and async statistics as "
            "separate A/B runs.",
            "Add --module-or-api before changing the async run to a tensor dump.",
            "Inspect regular corruption boundaries before escalating to "
            "profiler or sanitizer.",
        ]
    else:
        commands = [
            _migration_command(
                value,
                "--data",
                "--data-device",
                "npu",
                "--dump-task",
                "statistics",
                "--level",
                "L1",
                *common,
            ),
            _migration_command(
                value,
                "--capture",
                "reference",
                "--dump-task",
                "statistics",
                "--level",
                "L1",
                *common,
            ),
            _migration_command(
                value,
                "--capture",
                "candidate",
                "--dump-task",
                "statistics",
                "--level",
                "L1",
                *common,
            ),
            _migration_command(
                value,
                "--compare",
                "--dump-task",
                "statistics",
                "--level",
                "L1",
                *common,
            ),
        ]
        inspect = [
            "Narrow --scope or --module-or-api after the first module is known.",
            "Dump real tensors only for the small suspect set.",
            "For a later Loss difference, include previous backward plus "
            "current forward.",
        ]
    return {
        "stage": "localize",
        "goal": (
            "Reduce the first incident to a module/API without collecting "
            "the whole model."
        ),
        "commands": commands,
        "inspect": inspect,
        "decision": (
            "Record localized only with step, rank, phase, module/API, and "
            "stack evidence."
        ),
    }


def _verify_plan(value: dict[str, Any]) -> dict[str, Any]:
    incident = value["incident"]
    step = incident["step"] if incident["step"] is not None else 0
    rank = incident["rank"] if incident["rank"] is not None else 0
    module_or_api = incident["api"] or incident["module"]
    selection: tuple[str, ...] = ()
    if module_or_api:
        selection = ("--module-or-api", module_or_api)
    common = (
        "--dump-step",
        str(step),
        "--dump-ranks",
        str(rank),
        "--dump-task",
        "tensor",
        "--level",
        "L1",
        *selection,
    )
    return {
        "stage": "verify",
        "goal": "Prove that the suspect is a source of error, not an upstream victim.",
        "commands": [
            _migration_command(
                value,
                "--data",
                "--data-device",
                "npu",
                *common,
            ),
            _migration_command(value, "--capture", "reference", *common),
            _migration_command(value, "--capture", "candidate", *common),
            _migration_command(value, "--compare", "--tensor-log", *common),
            _migration_command(value, "--precheck", "candidate", *common),
        ],
        "inspect": [
            "Replay the real input against device and CPU high-precision references.",
            "Run one-variable A/B: FP32, move only this API to CPU, or "
            "decompose a fused op.",
            "Add each explanation as a hypothesis and attach supporting or "
            "rejecting evidence.",
        ],
        "decision": (
            "Confirm only when a one-variable experiment changes the original symptom."
        ),
    }


def _validate_plan(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": "validate",
        "goal": "Validate the fix from the local API back to the delivery target.",
        "commands": [
            _migration_command(
                value,
                "--data",
                "--data-device",
                "npu",
                "--level",
                "mix",
            ),
            _migration_command(value, "--capture", "reference", "--level", "mix"),
            _migration_command(value, "--capture", "candidate", "--level", "mix"),
            _migration_command(value, "--compare", "--level", "mix"),
            _monitor_command(
                value,
                "--data",
                "--data-device",
                "npu",
                "--training-steps",
                "100",
            ),
            _monitor_command(
                value,
                "--capture",
                "reference",
                "--training-steps",
                "100",
            ),
            _monitor_command(
                value,
                "--capture",
                "candidate",
                "--training-steps",
                "100",
            ),
        ],
        "inspect": [
            "Recheck the suspect API and the original first-incident window.",
            "Use the declared project window for long-training validation; "
            "100 is a placeholder.",
            "Run every target topology and final task metric before delivery.",
        ],
        "decision": (
            "Pass only when local, incident, long-term, topology, and task checks pass."
        ),
    }


def build_plan(value: dict[str, Any]) -> dict[str, Any]:
    for stage in STAGES:
        item = value["stages"][stage]
        retry = (
            stage in ADVANCE_CONCLUSIONS
            and item["status"] == "complete"
            and item["conclusion"] not in ADVANCE_CONCLUSIONS[stage]
        )
        if item["status"] != "complete" or retry:
            return {
                "checklist": _checklist_plan,
                "reproduce": _reproduce_plan,
                "observe": _observe_plan,
                "localize": _localize_plan,
                "verify": _verify_plan,
                "validate": _validate_plan,
            }[stage](value)
    return {
        "stage": "complete",
        "goal": "Review closure gates.",
        "commands": [],
        "inspect": [
            "Close the case only after a supported hypothesis and validation pass."
        ],
        "decision": "Run the close action when every closure gate is satisfied.",
    }


def _write_case_report(repository_root: Path, value: dict[str, Any]) -> Path:
    case_root = _case_root(repository_root, value["case_id"])
    case_root.mkdir(parents=True, exist_ok=True)
    plan = build_plan(value)
    write_json(case_root / "next_plan.json", plan)
    lines = [
        f"# {value['title']}",
        "",
        f"- Case: `{value['case_id']}`",
        f"- Status: `{value['status']}`",
        f"- Symptom: `{value['symptom']}`",
        f"- Topologies: `{', '.join(value['topologies'])}`",
        f"- Updated: `{value['updated_at']}`",
        "",
        "## Diagnostic gates",
        "",
        "| Stage | Status | Conclusion | Evidence |",
        "| --- | --- | --- | --- |",
    ]
    for stage in STAGES:
        item = value["stages"][stage]
        evidence = "<br>".join(f"`{path}`" for path in item["evidence"]) or "-"
        lines.append(
            f"| {STAGE_TITLES[stage]} | {item['status']} | "
            f"{item['conclusion'] or '-'} | {evidence} |"
        )
    lines.extend(
        [
            "",
            "## First incident",
            "",
            "```json",
            json.dumps(value["incident"], indent=2, sort_keys=True),
            "```",
            "",
            "## Hypotheses",
            "",
        ]
    )
    if value["hypotheses"]:
        for item in value["hypotheses"]:
            lines.extend(
                [
                    f"### H{item['id']}: {item['statement']}",
                    "",
                    f"- Experiment: {item['experiment']}",
                    f"- Verdict: `{item['verdict']}`",
                    f"- Notes: {item['notes'] or '-'}",
                    "- Evidence: "
                    + (
                        ", ".join(f"`{path}`" for path in item["evidence"])
                        or "-"
                    ),
                    "",
                ]
            )
    else:
        lines.extend(["No hypotheses recorded.", ""])
    lines.extend(
        [
            "## Next recipe",
            "",
            f"Goal: {plan['goal']}",
            "",
        ]
    )
    for command in plan["commands"]:
        lines.extend(["```bash", command, "```", ""])
    lines.extend(["Inspect:", ""])
    lines.extend(f"- {item}" for item in plan["inspect"])
    lines.extend(["", f"Decision: {plan['decision']}", ""])
    path = case_root / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _print_plan(value: dict[str, Any]) -> None:
    print(json.dumps(build_plan(value), indent=2, sort_keys=True))


def run_diagnostic_cli(
    repository_root: Path,
    arguments: Sequence[str] | None = None,
) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stateful diagnosis cases over existing MindStudio accuracy workflows"
        )
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    init = subparsers.add_parser("init")
    init.add_argument("case_id")
    init.add_argument("--title", required=True)
    init.add_argument(
        "--symptom",
        choices=(
            "unknown",
            "nan-or-overflow",
            "first-step-loss",
            "long-term-loss",
            "spike",
            "downstream-metric",
            "unstable",
        ),
        default="unknown",
    )
    init.add_argument("--topology", default="single")
    init.add_argument("--topologies")
    init.add_argument("--repeat", type=int, default=1)
    init.add_argument("--notes", default="")

    for name in ("status", "plan", "close"):
        command = subparsers.add_parser(name)
        command.add_argument("case_id")

    repeat_compare = subparsers.add_parser("compare-repeats")
    repeat_compare.add_argument("case_id")
    repeat_compare.add_argument(
        "--role", choices=("reference", "candidate"), required=True
    )
    repeat_compare.add_argument("--baseline-repeat", type=int, default=1)
    repeat_compare.add_argument("--target-repeat", type=int, default=2)

    observation = subparsers.add_parser("training-observation")
    observation.add_argument("case_id")
    observation.add_argument(
        "--workflow", choices=("migration", "monitor"), required=True
    )
    observation.add_argument("--training-steps", type=int)
    observation.add_argument("--loss-relative-threshold", type=float, default=0.01)
    observation.add_argument("--grad-norm-relative-threshold", type=float)
    observation.add_argument("--spike-relative-threshold", type=float)
    observation.add_argument("--force", action="store_true")

    record = subparsers.add_parser("record")
    record.add_argument("case_id")
    record.add_argument("--stage", choices=STAGES, required=True)
    record.add_argument("--conclusion", required=True)
    record.add_argument("--evidence", action="append", default=[])
    record.add_argument("--notes", default="")
    record.add_argument(
        "--symptom",
        choices=(
            "unknown",
            "nan-or-overflow",
            "first-step-loss",
            "long-term-loss",
            "spike",
            "downstream-metric",
            "unstable",
        ),
        help="refine the case classification while recording this stage",
    )
    record.add_argument("--step", type=int)
    record.add_argument("--rank", type=int)
    record.add_argument("--phase", choices=("forward", "backward", "optimizer"))
    record.add_argument("--module")
    record.add_argument("--api")

    hypothesis = subparsers.add_parser("hypothesis")
    hypothesis_actions = hypothesis.add_subparsers(
        dest="hypothesis_action", required=True
    )
    add = hypothesis_actions.add_parser("add")
    add.add_argument("case_id")
    add.add_argument("--statement", required=True)
    add.add_argument("--experiment", required=True)
    verdict = hypothesis_actions.add_parser("verdict")
    verdict.add_argument("case_id")
    verdict.add_argument("hypothesis_id", type=int)
    verdict.add_argument(
        "--verdict",
        choices=("supported", "rejected", "partial", "inconclusive"),
        required=True,
    )
    verdict.add_argument("--evidence", action="append", default=[])
    verdict.add_argument("--notes", default="")

    args = parser.parse_args(arguments)
    if args.action == "init":
        registry = standard_topologies()
        selected = select_topologies(
            available=tuple(registry),
            topology=args.topology if args.topologies is None else None,
            topologies=args.topologies,
            default=("single",),
        )
        path = create_case(
            repository_root,
            case_id=args.case_id,
            title=args.title,
            symptom=args.symptom,
            topologies=selected,
            repeat=args.repeat,
            notes=args.notes,
        )
        print(f"Diagnostic case: {path}")
        _print_plan(_load_case(repository_root, args.case_id))
        return
    if args.action == "record":
        path = record_stage(
            repository_root,
            case_id=args.case_id,
            stage=args.stage,
            conclusion=args.conclusion,
            evidence=args.evidence,
            notes=args.notes,
            incident={
                "step": args.step,
                "rank": args.rank,
                "phase": args.phase,
                "module": args.module,
                "api": args.api,
            },
            symptom=args.symptom,
        )
        print(f"Diagnostic case: {path}")
        _print_plan(_load_case(repository_root, args.case_id))
        return
    if args.action == "compare-repeats":
        path = compare_repeats(
            repository_root,
            case_id=args.case_id,
            role=args.role,
            baseline_repeat=args.baseline_repeat,
            target_repeat=args.target_repeat,
        )
        print(f"Repeat comparison: {path}")
        return
    if args.action == "training-observation":
        path = analyze_training_observation(
            repository_root,
            case_id=args.case_id,
            workflow=args.workflow,
            training_steps=args.training_steps,
            loss_relative_threshold=args.loss_relative_threshold,
            grad_norm_relative_threshold=args.grad_norm_relative_threshold,
            spike_relative_threshold=args.spike_relative_threshold,
            force=args.force,
        )
        print(f"Training observation: {path}")
        return
    if args.action == "hypothesis":
        if args.hypothesis_action == "add":
            path = add_hypothesis(
                repository_root,
                case_id=args.case_id,
                statement=args.statement,
                experiment=args.experiment,
            )
        else:
            path = record_hypothesis(
                repository_root,
                case_id=args.case_id,
                hypothesis_id=args.hypothesis_id,
                verdict=args.verdict,
                evidence=args.evidence,
                notes=args.notes,
            )
        print(f"Diagnostic case: {path}")
        return
    if args.action == "close":
        print(f"Diagnostic case: {close_case(repository_root, case_id=args.case_id)}")
        return
    value = _load_case(repository_root, args.case_id)
    if args.action == "plan":
        _print_plan(value)
    else:
        print(json.dumps(value, indent=2, sort_keys=True))
        report = _case_root(repository_root, args.case_id) / "README.md"
        print(f"Diagnostic report: {report}")
