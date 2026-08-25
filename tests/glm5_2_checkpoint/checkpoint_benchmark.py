#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Validate TorchTitan checkpoint save, load, and process-restart resume."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, replace
import html
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_common.cli import archive_previous_output
from tests.glm5_2_common.device import resolve_accelerator
from tests.glm5_2_common.naming import config_name
from tests.glm5_2_common.topology import (
    DISTRIBUTED_TOPOLOGY_NAMES,
    training_command_args,
)

from tests.glm5_2_checkpoint.fault_injection import (  # noqa: E402
    BOUNDARY_STEP_ENV,
    EVENTS_DIR_ENV,
    FAILURE_MODES,
    FAULT_MODE_ENV,
    FAULT_RANK_ENV,
    FAULT_STEP_ENV,
    READY_PATH_ENV,
)
from tests.glm5_2_checkpoint.state_compare import (  # noqa: E402
    compare_checkpoints,
    summarize_checkpoint,
)
from tests.glm5_2_checkpoint.topology_suite import (  # noqa: E402
    CHECKPOINT_SCHEMA,
    adopt_legacy_checkpoint_outputs,
    checkpoint_member_complete,
    checkpoint_member_paths,
    checkpoint_output_names,
    legacy_checkpoint_output_names,
    run_checkpoint_topology_suite,
)
from tests.glm5_2_graph.config import validate_graph_training_args  # noqa: E402
from tests.glm5_2_precision.artifacts import read_captured_metrics  # noqa: E402
from tests.glm5_2_precision.standards import (  # noqa: E402
    compute_error_metrics,
    exact_series_match,
)
from tests.glm5_2_precision.workflow import (  # noqa: E402
    FormalExperimentConfig,
    FormalTrainingConfig,
    ParallelTopology,
    TrainingEndpoint,
    _fixture_directory,
    fixed_input_environment,
    prepare_fixture,
    resolve_fixture_inputs,
    standard_topologies,
    training_topology_plan,
)


LOSS_KEY = "loss"
GRAD_NORM_KEY = "grad_norm"
_PR_SET_CHILD_SUBREAPER = 36
_PROCESS_GROUP_GRACE_SECONDS = 5.0


class CheckpointFixtureConfig(FormalExperimentConfig):
    """Use a compact checkpoint-specific name for shared test inputs."""

    @property
    def storage_name(self) -> str:
        precision = {
            "mixed-bfloat16": "bf16",
            "mixed-float32": "fp32",
            "full-bf16": "full-bf16",
        }.get(self.training.precision_name, self.training.precision_name)
        base = (
            f"{self.reference.device_type}-{precision}-"
            f"s{self.training.steps}-b{self.training.global_batch_size}-"
            f"seq{self.training.sequence_length}-seed{self.training.seed}"
        )
        return config_name(
            base,
            {
                "device": self.reference.device_type,
                "training": asdict(self.training),
            },
        )


def _precision_training(
    precision: str,
    *,
    steps: int,
    local_batch_size: int,
    global_batch_size: int,
    sequence_length: int,
    seed: int,
    extra_args: tuple[str, ...] = (),
) -> FormalTrainingConfig:
    values: dict[str, Any] = {
        "steps": steps,
        "local_batch_size": local_batch_size,
        "global_batch_size": global_batch_size,
        "sequence_length": sequence_length,
        "seed": seed,
        "deterministic": True,
        "extra_args": extra_args,
    }
    if precision == "fp32":
        values.update(training_dtype="float32", mixed_precision_param="float32")
    elif precision == "full-bf16":
        values.update(training_dtype="bfloat16", mixed_precision_param="bfloat16")
    else:
        values.update(training_dtype="float32", mixed_precision_param="bfloat16")
    return FormalTrainingConfig(**values)


def _device_from_environment(requested: str | None) -> tuple[str, str, str]:
    selection = resolve_accelerator(requested)
    return (
        selection.device_type,
        selection.visibility_variable,
        selection.visible_devices,
    )


def _metrics_rank(topology: ParallelTopology) -> int:
    if (
        topology.pipeline_parallel_degree == 1
        or topology.pipeline_parallel_schedule == "ZBVZeroBubble"
    ):
        return 0
    return (
        topology.world_size // topology.pipeline_parallel_degree
    ) * (topology.pipeline_parallel_degree - 1)


def _enable_child_subreaper() -> bool:
    """Make this Linux controller reap orphaned torchrun worker processes."""

    if not sys.platform.startswith("linux"):
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return True


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    return True


def _signal_process_group(process: subprocess.Popen[str], value: int) -> bool:
    if os.name != "posix":
        raise RuntimeError(
            "checkpoint signal-failure tests require a POSIX training host"
        )
    try:
        os.killpg(process.pid, value)
    except ProcessLookupError:
        return False
    return True


def _reap_process_group_children(process_group_id: int) -> int:
    reaped = 0
    while True:
        try:
            child_pid, _ = os.waitpid(-process_group_id, os.WNOHANG)
        except ChildProcessError:
            break
        if child_pid == 0:
            break
        reaped += 1
    return reaped


def _wait_for_process_group_exit(
    process_group_id: int,
    *,
    timeout: float,
    reap_children: bool,
) -> tuple[bool, int]:
    deadline = time.monotonic() + timeout
    reaped = 0
    while True:
        if reap_children:
            reaped += _reap_process_group_children(process_group_id)
        if not _process_group_exists(process_group_id):
            return True, reaped
        if time.monotonic() >= deadline:
            return False, reaped
        time.sleep(0.05)


def _cleanup_process_group(
    process: subprocess.Popen[str],
    *,
    reap_children: bool,
) -> dict[str, Any]:
    """Stop and reap only the process group created for one training run."""

    if os.name != "posix":
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=_PROCESS_GROUP_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        return {
            "term_sent": False,
            "kill_sent": False,
            "reaped_children": 0,
            "group_remaining": False,
        }

    process_group_id = process.pid
    reaped = (
        _reap_process_group_children(process_group_id) if reap_children else 0
    )
    term_sent = False
    kill_sent = False
    if _process_group_exists(process_group_id):
        term_sent = _signal_process_group(process, signal.SIGTERM)
    if process.poll() is None:
        try:
            process.wait(timeout=_PROCESS_GROUP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            pass
    exited, newly_reaped = _wait_for_process_group_exit(
        process_group_id,
        timeout=_PROCESS_GROUP_GRACE_SECONDS,
        reap_children=reap_children,
    )
    reaped += newly_reaped
    if not exited:
        kill_sent = _signal_process_group(process, signal.SIGKILL)
        if process.poll() is None:
            process.wait()
        exited, newly_reaped = _wait_for_process_group_exit(
            process_group_id,
            timeout=_PROCESS_GROUP_GRACE_SECONDS,
            reap_children=reap_children,
        )
        reaped += newly_reaped
    if process.poll() is None:
        process.wait()
    return {
        "term_sent": term_sent,
        "kill_sent": kill_sent,
        "reaped_children": reaped,
        "group_remaining": not exited,
    }


def _run_process(
    command: list[str],
    *,
    root: Path,
    environment: dict[str, str],
    log_path: Path,
    expected_failure: bool = False,
    launcher_signal: int | None = None,
    ready_path: Path | None = None,
    timeout: float = 1800,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("Command: " + " ".join(command) + "\n\n")
        stream.flush()
        try:
            reap_children = _enable_child_subreaper()
        except OSError as error:
            reap_children = False
            stream.write(f"Child subreaper unavailable: {error}\n")
            stream.flush()
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=os.name == "posix",
        )
        signal_sent = None
        try:
            if launcher_signal is not None:
                if ready_path is None:
                    raise ValueError(
                        "ready_path is required when signaling the launcher"
                    )
                deadline = time.monotonic() + timeout
                while not ready_path.is_file():
                    return_code = process.poll()
                    if return_code is not None:
                        raise RuntimeError(
                            "training exited before reaching the fault injection point; "
                            f"see {log_path}"
                        )
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "timed out waiting for checkpoint fault marker: "
                            f"{ready_path}"
                        )
                    time.sleep(0.1)
                signal_sent = signal.Signals(launcher_signal).name
                _signal_process_group(process, launcher_signal)
            try:
                return_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as error:
                raise TimeoutError(
                    f"training did not exit after fault; see {log_path}"
                ) from error
        finally:
            cleanup = _cleanup_process_group(
                process,
                reap_children=reap_children,
            )
            stream.write(
                "\nProcess cleanup: "
                + json.dumps(cleanup, sort_keys=True)
                + "\n"
            )
            stream.flush()

    if expected_failure:
        if return_code == 0:
            raise RuntimeError(
                f"fault-injected training exited successfully; see {log_path}"
            )
    elif return_code:
        raise subprocess.CalledProcessError(return_code, command)
    if cleanup["group_remaining"]:
        raise RuntimeError(
            "training process group still exists after TERM/KILL cleanup; "
            f"see {log_path}"
        )
    return {
        "return_code": return_code,
        "launcher_signal": signal_sent,
        "cleanup": cleanup,
    }


def _checkpoint_command(
    *,
    training: FormalTrainingConfig,
    topology: ParallelTopology,
    dump_folder: Path,
    initial_checkpoint: Path,
    checkpoint_interval: int,
    async_mode: str,
) -> list[str]:
    torchrun = shutil.which("torchrun") or str(
        Path(sys.executable).with_name("torchrun")
    )
    metrics_rank = _metrics_rank(topology)
    return [
        torchrun,
        f"--nproc_per_node={topology.world_size}",
        "--rdzv_backend=c10d",
        "--rdzv_endpoint=localhost:0",
        f"--local-ranks-filter={metrics_rank}",
        "--role=rank",
        "--tee=3",
        "-m",
        "tests.glm5_2_checkpoint.capture_resume",
        "--module",
        training.module,
        "--config",
        training.config,
        f"--dump_folder={dump_folder}",
        f"--training.steps={training.steps}",
        *training_command_args(
            local_batch_size=training.local_batch_size,
            global_batch_size=training.global_batch_size,
            sequence_length=training.sequence_length,
            topology=topology,
        ),
        f"--training.dtype={training.training_dtype}",
        f"--training.mixed_precision_param={training.mixed_precision_param}",
        f"--training.mixed_precision_reduce={training.mixed_precision_reduce}",
        f"--debug.seed={training.seed}",
        "--debug.deterministic",
        "--debug.no-enable-structured-logging",
        "--metrics.log_freq=1",
        "--metrics.enable_tensorboard",
        "--metrics.disable_color_printing",
        "--metrics.save_tb_folder=tensorboard",
        *topology.command_args(),
        "--checkpoint.enable",
        f"--checkpoint.interval={checkpoint_interval}",
        f"--checkpoint.async_mode={async_mode}",
        "--checkpoint.no-last-save-model-only",
        f"--checkpoint.initial_load_path={initial_checkpoint}",
        *training.extra_args,
    ]


def _capture_environment(
    *,
    base: dict[str, str],
    device: str,
    visible_env: str,
    visible_devices: str,
    training: FormalTrainingConfig,
    topology: ParallelTopology,
    token_plan: Path,
    metrics: Path,
    input_contract: Path,
    stop_after: int | None,
    events: Path,
    boundary_step: int,
    failure_mode: str | None = None,
    failure_step: int | None = None,
    failure_rank: int = 0,
    ready_path: Path | None = None,
) -> dict[str, str]:
    environment = dict(base)
    environment[visible_env] = visible_devices
    environment["TORCHTITAN_DEVICE"] = "npu" if device == "npu" else "gpu"
    environment["GLM5_PRECISION_METRICS_PATH"] = str(metrics)
    environment.update(
        fixed_input_environment(
            token_plan_path=token_plan,
            input_contract_directory=input_contract,
            training=training,
            topology=topology,
        )
    )
    environment["LOG_RANK"] = str(_metrics_rank(topology))
    environment[EVENTS_DIR_ENV] = str(events)
    environment[BOUNDARY_STEP_ENV] = str(boundary_step)
    if stop_after is None:
        environment.pop("GLM5_CHECKPOINT_STOP_AFTER", None)
    else:
        environment["GLM5_CHECKPOINT_STOP_AFTER"] = str(stop_after)
    if failure_mode is None:
        for name in (
            FAULT_MODE_ENV,
            FAULT_STEP_ENV,
            FAULT_RANK_ENV,
            READY_PATH_ENV,
        ):
            environment.pop(name, None)
    else:
        if failure_step is None or ready_path is None:
            raise ValueError("failure_step and ready_path are required for a failure")
        environment[FAULT_MODE_ENV] = failure_mode
        environment[FAULT_STEP_ENV] = str(failure_step)
        environment[FAULT_RANK_ENV] = str(failure_rank)
        environment[READY_PATH_ENV] = str(ready_path)
    return environment


def _read_contract(directory: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(directory.glob("rank-*.jsonl")):
        result[path.name] = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return result


def _combine_contracts(*directories: Path) -> dict[str, list[dict[str, Any]]]:
    combined: dict[str, list[dict[str, Any]]] = {}
    for directory in directories:
        for name, records in _read_contract(directory).items():
            combined.setdefault(name, []).extend(records)
    return combined


def _combine_recovered_contracts(
    interrupted: Path,
    resumed: Path,
    *,
    restored_step: int,
) -> dict[str, list[dict[str, Any]]]:
    """Drop interrupted work after the checkpoint, then append replayed work."""

    combined: dict[str, list[dict[str, Any]]] = {}
    first = _read_contract(interrupted)
    second = _read_contract(resumed)
    for name in sorted(first.keys() | second.keys()):
        retained = [
            record
            for record in first.get(name, [])
            if int(record["step"]) <= restored_step
        ]
        combined[name] = [*retained, *second.get(name, [])]
    return combined


def _combine_recovered_metrics(
    interrupted: Path,
    resumed: Path,
    destination: Path,
    *,
    restored_step: int,
) -> None:
    records: list[dict[str, Any]] = []
    for path, retain_interrupted in ((interrupted, True), (resumed, False)):
        seen_steps: set[int] = set()
        previous_step: int | None = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            step = int(record["step"])
            if step in seen_steps or (
                previous_step is not None and step <= previous_step
            ):
                raise RuntimeError(
                    "checkpoint metric phase is not strictly increasing: "
                    f"path={path}, previous={previous_step}, current={step}"
                )
            seen_steps.add(step)
            previous_step = step
            if retain_interrupted and step > restored_step:
                continue
            if not retain_interrupted and step <= restored_step:
                raise RuntimeError(
                    "resumed metrics include work at or before the restored "
                    f"checkpoint: path={path}, restored_step={restored_step}, "
                    f"observed_step={step}"
                )
            records.append(record)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _read_events(directory: Path) -> dict[int, list[dict[str, Any]]]:
    events: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(directory.glob("rank-*.jsonl")):
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if records:
            events[int(records[0]["rank"])] = records
    return events


def _loaded_steps(directory: Path) -> dict[int, int | None]:
    result: dict[int, int | None] = {}
    for rank, records in _read_events(directory).items():
        loads = [record for record in records if record["event"] == "load"]
        if len(loads) == 1:
            value = loads[0]["step"]
            result[rank] = int(value) if value is not None else None
    return result


def _optimizer_state_gaps(directory: Path, *, step: int) -> dict[int, list[str]]:
    gaps: dict[int, list[str]] = {}
    for rank, records in _read_events(directory).items():
        audits = [
            record
            for record in records
            if record["event"] == "optimizer-state-audit"
            and int(record["step"]) == step
        ]
        if len(audits) == 1:
            gaps[rank] = [str(name) for name in audits[0]["missing"]]
    return gaps


def _state_fingerprint_snapshots(
    directory: Path,
    *,
    phase: str,
    step: int,
) -> dict[int, dict[str, Any]]:
    snapshots: dict[int, dict[str, Any]] = {}
    for rank, records in _read_events(directory).items():
        matches = [
            record
            for record in records
            if record["event"] == "state-fingerprint"
            and record.get("phase") == phase
            and int(record["step"]) == step
        ]
        if len(matches) == 1:
            snapshots[rank] = matches[0]["scopes"]
    return snapshots


def _compare_scope_fingerprints(
    saved: dict[str, Any],
    loaded: dict[str, Any],
) -> dict[str, Any]:
    saved_leaves = saved.get("leaves", {})
    loaded_leaves = loaded.get("leaves", {})
    saved_paths = set(saved_leaves)
    loaded_paths = set(loaded_leaves)
    missing_loaded = sorted(saved_paths - loaded_paths)
    missing_saved = sorted(loaded_paths - saved_paths)
    changed = sorted(
        path
        for path in saved_paths & loaded_paths
        if saved_leaves[path] != loaded_leaves[path]
    )
    return {
        "passed": not missing_loaded and not missing_saved and not changed,
        "saved_sha256": saved.get("sha256"),
        "loaded_sha256": loaded.get("sha256"),
        "saved_leaf_count": saved.get("leaf_count"),
        "loaded_leaf_count": loaded.get("leaf_count"),
        "missing_after_load": missing_loaded[:20],
        "unexpected_after_load": missing_saved[:20],
        "changed_leaves": changed[:20],
        "changed_leaf_count": len(changed),
    }


def _boundary_memory_comparison(
    saved_directory: Path,
    loaded_directory: Path,
    *,
    step: int,
    expected_ranks: set[int],
) -> dict[str, Any]:
    """Compare live local state immediately before and after process restart."""

    saved = _state_fingerprint_snapshots(
        saved_directory, phase="save", step=step
    )
    loaded = _state_fingerprint_snapshots(
        loaded_directory, phase="load", step=step
    )
    rank_results: dict[int, dict[str, Any]] = {}
    mismatch_examples: list[str] = []
    for rank in sorted(expected_ranks & saved.keys() & loaded.keys()):
        saved_scopes = saved[rank]
        loaded_scopes = loaded[rank]
        scope_names = sorted(set(saved_scopes) | set(loaded_scopes))
        scopes: dict[str, Any] = {}
        for scope in scope_names:
            if scope not in saved_scopes or scope not in loaded_scopes:
                result = {
                    "passed": False,
                    "missing_before_save": scope not in saved_scopes,
                    "missing_after_load": scope not in loaded_scopes,
                }
            else:
                result = _compare_scope_fingerprints(
                    saved_scopes[scope], loaded_scopes[scope]
                )
            scopes[scope] = result
            if not result["passed"] and len(mismatch_examples) < 20:
                changed = result.get("changed_leaves", [])
                detail = (
                    f"; changed leaves: {', '.join(changed[:5])}"
                    if changed
                    else ""
                )
                mismatch_examples.append(
                    f"rank {rank}, {scope}: state differs after load{detail}"
                )
        rank_results[rank] = {
            "passed": all(result["passed"] for result in scopes.values()),
            "scopes": scopes,
        }
    missing_saved_ranks = sorted(expected_ranks - saved.keys())
    missing_loaded_ranks = sorted(expected_ranks - loaded.keys())
    if missing_saved_ranks:
        mismatch_examples.append(
            f"missing pre-restart fingerprints on ranks {missing_saved_ranks}"
        )
    if missing_loaded_ranks:
        mismatch_examples.append(
            f"missing post-load fingerprints on ranks {missing_loaded_ranks}"
        )
    passed = (
        not missing_saved_ranks
        and not missing_loaded_ranks
        and all(result["passed"] for result in rank_results.values())
    )
    return {
        "passed": bool(passed),
        "step": step,
        "missing_saved_ranks": missing_saved_ranks,
        "missing_loaded_ranks": missing_loaded_ranks,
        "ranks": rank_results,
        "mismatch_examples": mismatch_examples,
    }


def _selected_failure_modes(requested: list[str], *, world_size: int) -> list[str]:
    if not requested:
        return ["graceful"]
    if "all" in requested:
        modes = ["graceful", "sigint", "sigterm", "sigkill", "incomplete"]
        if world_size > 1:
            modes.extend(("rank-error", "rank-sigkill"))
        return modes
    result: list[str] = []
    for mode in requested:
        if mode not in result:
            result.append(mode)
    return result


def _metric_series(path: Path) -> dict[str, dict[int, float]]:
    metrics = read_captured_metrics(path)
    return {
        LOSS_KEY: {record.step: record.loss for record in metrics},
        GRAD_NORM_KEY: {record.step: record.grad_norm for record in metrics},
    }


def _metric_window_summary(
    steps: tuple[int, ...],
    reference: dict[int, float],
    candidate: dict[int, float],
) -> dict[str, Any]:
    if not steps:
        return {"count": 0, "bitwise_equal": True}
    reference_values = tuple(reference[step] for step in steps)
    candidate_values = tuple(candidate[step] for step in steps)
    metrics = compute_error_metrics(reference_values, candidate_values)
    return {
        "count": metrics.count,
        "bitwise_equal": exact_series_match(
            {step: reference[step] for step in steps},
            {step: candidate[step] for step in steps},
        ),
        "mean_absolute": metrics.absolute,
        "mean_absolute_relative": metrics.relative_absolute,
        "max_absolute": metrics.max_absolute,
    }


def _metrics_comparison(
    reference_path: Path,
    candidate_path: Path,
    *,
    exact: bool,
    loss_absolute_limit: float,
    loss_relative_limit: float,
    grad_relative_limit: float,
    restored_step: int | None = None,
) -> dict[str, Any]:
    reference = _metric_series(reference_path)
    candidate = _metric_series(candidate_path)
    if reference[LOSS_KEY].keys() != candidate[LOSS_KEY].keys():
        return {"passed": False, "error": "metric step sets differ"}
    result: dict[str, Any] = {"mode": "exact" if exact else "tolerance"}
    for name in (LOSS_KEY, GRAD_NORM_KEY):
        steps = tuple(sorted(reference[name]))
        reference_values = tuple(reference[name][step] for step in steps)
        candidate_values = tuple(candidate[name][step] for step in steps)
        metrics = compute_error_metrics(reference_values, candidate_values)
        differing_steps = [
            step
            for step in steps
            if float.hex(reference[name][step])
            != float.hex(candidate[name][step])
        ]
        first_mismatch = differing_steps[0] if differing_steps else None
        result[name] = {
            "count": metrics.count,
            "normal": metrics.normal,
            "mean_absolute": metrics.absolute,
            "mean_relative": metrics.relative_normal,
            "mean_absolute_relative": metrics.relative_absolute,
            "max_absolute": metrics.max_absolute,
            "bitwise_equal": exact_series_match(reference[name], candidate[name]),
            "first_bitwise_mismatch_step": first_mismatch,
            "first_bitwise_mismatch": (
                {
                    "reference": reference[name][first_mismatch],
                    "candidate": candidate[name][first_mismatch],
                    "absolute": abs(
                        reference[name][first_mismatch]
                        - candidate[name][first_mismatch]
                    ),
                }
                if first_mismatch is not None
                else None
            ),
        }
        if restored_step is not None:
            result[name]["through_checkpoint"] = _metric_window_summary(
                tuple(step for step in steps if step <= restored_step),
                reference[name],
                candidate[name],
            )
            result[name]["after_restart"] = _metric_window_summary(
                tuple(step for step in steps if step > restored_step),
                reference[name],
                candidate[name],
            )
    if exact:
        passed = (
            result[LOSS_KEY]["bitwise_equal"]
            and result[GRAD_NORM_KEY]["bitwise_equal"]
        )
    else:
        loss = result[LOSS_KEY]
        grad = result[GRAD_NORM_KEY]
        passed = (
            loss["mean_absolute"] <= loss_absolute_limit
            or loss["mean_absolute_relative"] <= loss_relative_limit
        ) and abs(grad["mean_relative"]) <= grad_relative_limit
    result["passed"] = bool(passed)
    return result


def _format_metric_window(window: dict[str, Any]) -> str:
    if not window or not window.get("count"):
        return "no observations"
    return (
        f"{window['count']} steps; "
        f"bitwise={'yes' if window['bitwise_equal'] else 'no'}; "
        f"mean abs={window['mean_absolute']:.9g}; "
        f"max abs={window['max_absolute']:.9g}"
    )


def _format_metric_window_html(window: dict[str, Any]) -> str:
    return html.escape(_format_metric_window(window))


def _format_first_mismatch(metric: dict[str, Any]) -> str:
    step = metric.get("first_bitwise_mismatch_step")
    values = metric.get("first_bitwise_mismatch")
    if step is None or values is None:
        return "none"
    return (
        f"step {step}; reference={values['reference']:.9g}; "
        f"resumed={values['candidate']:.9g}; "
        f"absolute={values['absolute']:.9g}"
    )


def _resume_failure_location(
    *,
    split_state: dict[str, Any],
    boundary_memory: dict[str, Any],
    metrics: dict[str, Any],
    final_state: dict[str, Any],
    restored_step: int,
) -> dict[str, Any]:
    """Classify the first observed resume boundary where equivalence is lost."""

    if not split_state["passed"]:
        return {
            "stage": "checkpoint-save",
            "description": (
                "The saved recovery-boundary checkpoint already differs from "
                "the uninterrupted baseline. Inspect the boundary DCP scope table."
            ),
        }
    if not boundary_memory["passed"]:
        return {
            "stage": "checkpoint-load",
            "description": (
                "The saved checkpoint is correct, but at least one rank-local "
                "state leaf differs immediately after TorchTitan loads it."
            ),
        }
    differing_steps = [
        metric["first_bitwise_mismatch_step"]
        for name, metric in metrics.items()
        if name in {LOSS_KEY, GRAD_NORM_KEY}
        and metric.get("first_bitwise_mismatch_step") is not None
    ]
    first_difference = min(differing_steps) if differing_steps else None
    if first_difference is not None and first_difference <= restored_step:
        return {
            "stage": "pre-checkpoint-trajectory",
            "first_differing_step": first_difference,
            "description": (
                "The combined interrupted trajectory differs before or at the "
                "restored checkpoint, so the failure is not isolated to resume."
            ),
        }
    if first_difference is not None:
        return {
            "stage": "post-load-training",
            "first_differing_step": first_difference,
            "description": (
                "All persisted state is exact immediately after load, but the "
                f"training trajectory first differs at step {first_difference}. "
                "Investigate non-checkpointed runtime state or TP execution."
            ),
        }
    if not final_state["passed"]:
        return {
            "stage": "post-load-state-update",
            "description": (
                "Logged loss and grad norm remain identical, but the final logical "
                "checkpoint differs. Inspect the final per-scope state table."
            ),
        }
    return {
        "stage": "exact",
        "description": "No checkpoint or resumed-training divergence was observed.",
    }


def _checkpoint_scope_rows(comparison: dict[str, Any]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(scope)}</td>"
        f"<td>{stats['exact_tensor_mismatches']}</td>"
        f"<td>{stats['tolerance_tensor_mismatches']}</td>"
        f"<td>{stats['non_tensor_mismatches']}</td>"
        f"<td>{stats['max_absolute']:.9g}</td>"
        f"<td>{stats['mean_absolute']:.9g}</td>"
        f"<td>{stats['mean_absolute_relative']:.6%}</td></tr>"
        for scope, stats in comparison["scopes"].items()
    )
    return rows or '<tr><td colspan="7">No checkpoint scopes found</td></tr>'


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    status = "PASS" if summary["passed"] else "FAIL"
    completed = summary["completed_restart"]
    completed_loaded = ", ".join(
        f"rank {rank}: step {step}"
        for rank, step in sorted(completed["loaded_steps"].items())
    )
    scenario_rows = (
        "<tr>"
        "<td>completed</td>"
        "<td>normal training completion followed by a no-op restart</td>"
        f"<td>{completed['return_code']}</td>"
        f"<td>{html.escape(completed_loaded or 'missing')}</td>"
        "<td>not applicable</td>"
        f"<td class={'pass' if completed['passed'] else 'fail'}>"
        f"{'PASS' if completed['passed'] else 'FAIL'}</td></tr>"
    )
    scenario_details = ""
    for mode, values in summary["scenarios"].items():
        scenario_status = "PASS" if values["passed"] else "FAIL"
        diagnosis = values["failure_location"]
        loaded = ", ".join(
            f"rank {rank}: step {step}"
            for rank, step in sorted(values["loaded_steps"].items())
        )
        scenario_rows += (
            "<tr>"
            f"<td>{html.escape(mode)}</td>"
            f"<td>{html.escape(values['interruption']['description'])}</td>"
            f"<td>{values['interruption']['return_code']}</td>"
            f"<td>{html.escape(loaded or 'missing')}</td>"
            f"<td>{'yes' if values['incomplete_checkpoint_ignored'] else 'no'}</td>"
            f"<td class={'pass' if values['passed'] else 'fail'}>"
            f"{scenario_status}</td></tr>"
        )
        check_rows = "".join(
            f"<tr><td>{html.escape(name)}</td>"
            f"<td class={'pass' if passed else 'fail'}>"
            f"{'PASS' if passed else 'FAIL'}</td></tr>"
            for name, passed in values["checks"].items()
        )
        metric_rows = "".join(
            "<tr>"
            f"<td>{name}</td><td>{metric['count']}</td>"
            f"<td>{str(metric['bitwise_equal'])}</td>"
            f"<td>{html.escape(_format_first_mismatch(metric))}</td>"
            f"<td>{metric['mean_absolute']:.9g}</td>"
            f"<td>{metric['mean_absolute_relative']:.6%}</td>"
            f"<td>{metric['max_absolute']:.9g}</td>"
            "<td>"
            f"{_format_metric_window_html(metric.get('through_checkpoint', {}))}"
            "</td><td>"
            f"{_format_metric_window_html(metric.get('after_restart', {}))}"
            "</td>"
            "</tr>"
            for name, metric in values["metrics"].items()
            if name in {LOSS_KEY, GRAD_NORM_KEY}
        )
        boundary_rows = ""
        for rank, rank_values in sorted(
            values["boundary_memory"]["ranks"].items(), key=lambda item: int(item[0])
        ):
            for scope, scope_values in sorted(rank_values["scopes"].items()):
                changed = scope_values.get("changed_leaves", [])
                missing = scope_values.get("missing_after_load", [])
                unexpected = scope_values.get("unexpected_after_load", [])
                examples = [
                    *(f"changed: {path}" for path in changed[:3]),
                    *(f"missing: {path}" for path in missing[:3]),
                    *(f"unexpected: {path}" for path in unexpected[:3]),
                ]
                boundary_rows += (
                    "<tr>"
                    f"<td>{rank}</td><td>{html.escape(scope)}</td>"
                    f"<td class={'pass' if scope_values['passed'] else 'fail'}>"
                    f"{'PASS' if scope_values['passed'] else 'FAIL'}</td>"
                    f"<td>{scope_values.get('changed_leaf_count', '-')}</td>"
                    f"<td>{len(missing)}</td>"
                    f"<td>{len(unexpected)}</td>"
                    f"<td>{html.escape('; '.join(examples) or '-')}</td></tr>"
                )
        if not boundary_rows:
            boundary_rows = (
                '<tr><td colspan="7">No boundary fingerprints found</td></tr>'
            )
        split_scope_rows = _checkpoint_scope_rows(values["split_state"])
        final_scope_rows = _checkpoint_scope_rows(values["final_state"])
        split_mismatch = "\n".join(values["split_state"]["mismatch_examples"])
        final_mismatch = "\n".join(values["final_state"]["mismatch_examples"])
        boundary_mismatch = "\n".join(
            values["boundary_memory"]["mismatch_examples"]
        )
        scenario_details += f"""
<details><summary>{html.escape(mode)}: {scenario_status}</summary>
<h3>First observed divergence</h3><p><strong>{html.escape(diagnosis['stage'])}</strong>: {html.escape(diagnosis['description'])}</p>
<h3>Checks</h3><table><thead><tr><th>Check</th><th>Result</th></tr></thead><tbody>{check_rows}</tbody></table>
<h3>Training metrics</h3><table><thead><tr><th>Series</th><th>Steps</th><th>Bitwise equal</th><th>First differing step</th><th>Mean absolute</th><th>Mean absolute relative</th><th>Max absolute</th><th>Through checkpoint</th><th>After restart</th></tr></thead><tbody>{metric_rows}</tbody></table>
<h3>Saved recovery-boundary checkpoint by scope</h3><table><thead><tr><th>Scope</th><th>Exact tensor mismatches</th><th>Tolerance tensor mismatches</th><th>Non-tensor mismatches</th><th>Max absolute</th><th>Mean absolute</th><th>Mean absolute relative</th></tr></thead><tbody>{split_scope_rows}</tbody></table>
<pre>{html.escape(split_mismatch or 'Saved checkpoint matches the baseline')}</pre>
<h3>Immediate post-load memory audit</h3><table><thead><tr><th>Rank</th><th>State scope</th><th>Result</th><th>Changed leaf hashes</th><th>Missing after load</th><th>Unexpected after load</th><th>Examples</th></tr></thead><tbody>{boundary_rows}</tbody></table>
<pre>{html.escape(boundary_mismatch or 'No in-memory state mismatch')}</pre>
<h3>Final logical checkpoint by scope</h3><table><thead><tr><th>Scope</th><th>Exact tensor mismatches</th><th>Tolerance tensor mismatches</th><th>Non-tensor mismatches</th><th>Max absolute</th><th>Mean absolute</th><th>Mean absolute relative</th></tr></thead><tbody>{final_scope_rows}</tbody></table>
<h3>Final checkpoint mismatch examples</h3><pre>{html.escape(final_mismatch or 'None')}</pre>
</details>"""
    path.write_text(
        f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(summary['run_name'])}</title><style>
body{{font:14px/1.45 system-ui,sans-serif;max-width:1280px;margin:auto;padding:24px;color:#17202a}}
table{{border-collapse:collapse;width:100%;margin:16px 0}}th,td{{border:1px solid #d8dee9;padding:8px;text-align:left}}
th{{background:#eef2f7;position:sticky;top:0}}.pass{{color:#16723c;font-weight:800}}.fail{{color:#b42318;font-weight:800}}
.status{{display:inline-block;background:{'#17803d' if status == 'PASS' else '#b42318'};color:white;padding:6px 12px;border-radius:999px;font-weight:800}}
pre{{white-space:pre-wrap;background:#f8fafc;padding:12px;border:1px solid #d8dee9;border-radius:8px}}details{{border:1px solid #d8dee9;border-radius:8px;padding:12px;margin:16px 0}}summary{{font-weight:800;cursor:pointer}}</style></head>
<body><h1>GLM 5.2 checkpoint resume report</h1><p><span class="status">{status}</span></p>
<p>One uninterrupted baseline is shared by every scenario. Each interrupted job must select the last complete step-{summary['split_step']} DCP checkpoint, replay discarded work, and match the baseline at step {summary['total_steps']}.</p>
<h2>Failure recovery matrix</h2><table><thead><tr><th>Mode</th><th>Interruption</th><th>Exit code</th><th>Loaded checkpoint</th><th>Ignored incomplete latest</th><th>Result</th></tr></thead><tbody>{scenario_rows}</tbody></table>
{scenario_details}
<h2>Configuration</h2><pre>{html.escape(json.dumps(summary['configuration'], indent=2, sort_keys=True))}</pre>
</body></html>""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        action="store_true",
        help="prepare the shared step-0 checkpoint and fixed token plan, then exit",
    )
    parser.add_argument("--device", choices=("cuda", "npu"))
    suite_topologies = ("single", *DISTRIBUTED_TOPOLOGY_NAMES)
    parser.add_argument(
        "--topology", default="single", choices=("all", *suite_topologies)
    )
    parser.add_argument(
        "--precision", default="bf16", choices=("fp32", "bf16", "full-bf16")
    )
    parser.add_argument("--total-steps", type=int, default=20)
    parser.add_argument("--split-step", type=int, default=10)
    parser.add_argument("--local-batch-size", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument(
        "--comparison", choices=("exact", "tolerance"), default="exact"
    )
    parser.add_argument("--state-atol", type=float, default=0.01)
    parser.add_argument("--state-rtol", type=float, default=0.01)
    parser.add_argument("--loss-atol", type=float, default=0.01)
    parser.add_argument("--loss-rtol", type=float, default=0.01)
    parser.add_argument("--grad-relative-limit", type=float, default=0.05)
    parser.add_argument(
        "--extra-train-arg",
        action="append",
        default=[],
        help="append a raw TorchTitan CLI argument",
    )
    parser.add_argument(
        "--run-tag",
        help="short label added to output names, for example inductor",
    )
    parser.add_argument(
        "--async-mode",
        choices=("disabled", "async", "async_with_pinned_mem"),
        default="disabled",
    )
    parser.add_argument(
        "--failure-mode",
        action="append",
        choices=("all", *sorted(FAILURE_MODES)),
        help=(
            "failure scenario to validate; repeat the option or use all. "
            "The default preserves the original graceful-resume test"
        ),
    )
    parser.add_argument(
        "--failure-rank",
        type=int,
        help="rank used by rank-error/rank-sigkill (default: rank 1, or 0 single-card)",
    )
    parser.add_argument(
        "--interrupt-step",
        type=int,
        help="step at which abrupt failures occur (default: split-step + 1)",
    )
    parser.add_argument("--failure-timeout", type=float, default=1800)
    parser.add_argument(
        "--restart-delay",
        type=float,
        default=2.0,
        help="seconds allowed for accelerator/process cleanup after an abrupt exit",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 0 < args.split_step < args.total_steps:
        raise ValueError("split-step must be between zero and total-steps")
    if args.failure_timeout <= 0 or args.restart_delay < 0:
        raise ValueError(
            "failure-timeout must be positive and restart-delay nonnegative"
        )
    interrupt_step = args.interrupt_step or args.split_step + 1
    if not args.split_step < interrupt_step <= args.total_steps:
        raise ValueError(
            "interrupt-step must be greater than split-step and no greater "
            "than total-steps"
        )

    root = Path(__file__).resolve().parents[2]
    device, visible_env, visible_devices = _device_from_environment(args.device)
    validate_graph_training_args(
        device_type=device,
        arguments=args.extra_train_arg,
    )
    output_name_args = {
        "precision": args.precision,
        "total_steps": args.total_steps,
        "split_step": args.split_step,
        "local_batch_size": args.local_batch_size,
        "global_batch_size": args.global_batch_size,
        "sequence_length": args.sequence_length,
        "seed": args.seed,
        "async_mode": args.async_mode,
        "comparison": args.comparison,
        "requested_failure_modes": args.failure_mode or [],
        "run_tag": args.run_tag,
        "extra_train_args": tuple(args.extra_train_arg),
        "interrupt_step": interrupt_step,
        "failure_rank": args.failure_rank,
        "failure_timeout": args.failure_timeout,
        "restart_delay": args.restart_delay,
    }
    if args.topology == "all" and not args.data:
        return run_checkpoint_topology_suite(
            root,
            script_path=__file__,
            argv=sys.argv[1:],
            topology_names=suite_topologies,
            device=device,
            output_name_args=output_name_args,
        )
    # Fixtures contain global inputs and an unsharded seed checkpoint. They are
    # shared by every topology, so `--data --topology all` prepares only once.
    topology_name = "single" if args.data else args.topology
    topology = standard_topologies()[topology_name]
    failure_modes = _selected_failure_modes(
        args.failure_mode or [], world_size=topology.world_size
    )
    failure_rank = args.failure_rank
    if failure_rank is None:
        failure_rank = 1 if topology.world_size > 1 else 0
    if not 0 <= failure_rank < topology.world_size:
        raise ValueError(
            f"failure-rank {failure_rank} is outside topology world size "
            f"{topology.world_size}"
        )
    training = _precision_training(
        args.precision,
        steps=args.total_steps,
        local_batch_size=args.local_batch_size,
        global_batch_size=args.global_batch_size,
        sequence_length=args.sequence_length,
        seed=args.seed,
        extra_args=tuple(args.extra_train_arg),
    )
    batch_schedule = training_topology_plan(training, topology)
    visible_count = len([item for item in visible_devices.split(",") if item.strip()])
    if visible_count < topology.world_size:
        raise ValueError(
            f"topology {topology.name} needs {topology.world_size} visible devices, "
            f"but {visible_env} contains {visible_count}"
        )
    endpoint = TrainingEndpoint(
        name=device,
        device_type=device,
        visible_devices=visible_devices,
        topology=topology,
        repeats=1,
    )
    suite_name, run_name = checkpoint_output_names(
        device=device,
        topology_slug=topology.slug,
        topology_identity=asdict(topology),
        **output_name_args,
    )
    fixture_training = replace(training, extra_args=())
    fixture_config = CheckpointFixtureConfig(
        name="checkpoint",
        kind="self_consistency",
        reference=endpoint,
        candidate=endpoint,
        training=fixture_training,
        fixture_root="checkpoint_fixtures",
        artifact_root="checkpoint_artifacts",
        report_root="checkpoint_reports",
        run_root="checkpoint_runs",
        fixture_name=suite_name,
    )
    fixture = _fixture_directory(root, fixture_config)
    fixture_manifest = fixture / "fixture.json"
    if args.data:
        path = prepare_fixture(
            root,
            fixture_config,
            endpoint=endpoint,
            force=args.force,
        )
        print(f"Prepared checkpoint fixture: {path}", flush=True)
        return 0
    fixture_generation_id = None
    if fixture_manifest.is_file():
        fixture_generation_id = json.loads(
            fixture_manifest.read_text(encoding="utf-8")
        ).get("generation_id")

    legacy_suite_name, legacy_run_name = legacy_checkpoint_output_names(
        device=device,
        topology_slug=topology.slug,
        **output_name_args,
    )
    if not training.extra_args:
        adopt_legacy_checkpoint_outputs(
            root,
            suite_name=suite_name,
            member_name=run_name,
            legacy_suite_name=legacy_suite_name,
            legacy_member_name=legacy_run_name,
            topology_slug=topology.slug,
        )
    run_root, report_root, summary_path, report_path = checkpoint_member_paths(
        root,
        suite_name=suite_name,
        topology_slug=topology.slug,
        member_name=run_name,
    )
    if run_root.exists() or report_root.exists():
        if not args.force and checkpoint_member_complete(
            summary_path,
            member_name=run_name,
            fixture_generation_id=fixture_generation_id,
        ):
            print(
                f"Skip completed checkpoint topology: {topology.name}\n"
                f"Report: {report_path}",
                flush=True,
            )
            return 0
        if args.force:
            shutil.rmtree(run_root, ignore_errors=True)
            shutil.rmtree(report_root, ignore_errors=True)
        else:
            archived_run = archive_previous_output(run_root)
            archived_report = archive_previous_output(report_root)
            archived = [
                str(path)
                for path in (archived_run, archived_report)
                if path is not None
            ]
            print(
                "Retry incomplete or failed checkpoint topology; archived previous "
                f"output: {', '.join(archived)}",
                flush=True,
            )
    run_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    if not fixture_manifest.is_file():
        if fixture.exists():
            raise RuntimeError(
                "checkpoint fixture is incomplete; recreate it with --data "
                f"--force: {fixture}"
            )
        print(f"Preparing shared seed checkpoint and token plan: {fixture}", flush=True)
        prepare_fixture(root, fixture_config, endpoint=endpoint, force=False)
    initial_checkpoint, token_plan = resolve_fixture_inputs(root, fixture_config)

    common_environment = os.environ.copy()
    baseline_root = run_root / "baseline"
    baseline_log = baseline_root / "runtime.log"
    print(
        f"Starting uninterrupted baseline: topology={topology.name}, "
        f"target_step={args.total_steps}\nRuntime log: {baseline_log}",
        flush=True,
    )
    baseline_environment = _capture_environment(
        base=common_environment,
        device=device,
        visible_env=visible_env,
        visible_devices=visible_devices,
        training=training,
        topology=topology,
        token_plan=token_plan,
        metrics=baseline_root / "metrics.jsonl",
        input_contract=baseline_root / "input_contract",
        stop_after=None,
        events=baseline_root / "events",
        boundary_step=args.split_step,
    )
    command = _checkpoint_command(
        training=training,
        topology=topology,
        dump_folder=baseline_root / "trainer_output",
        initial_checkpoint=initial_checkpoint,
        checkpoint_interval=args.split_step,
        async_mode=args.async_mode,
    )
    _run_process(
        command,
        root=root,
        environment=baseline_environment,
        log_path=baseline_log,
        timeout=args.failure_timeout,
    )

    baseline_checkpoint = baseline_root.joinpath(
        "trainer_output", "checkpoint", f"step-{args.total_steps}"
    )
    baseline_split_checkpoint = baseline_root.joinpath(
        "trainer_output", "checkpoint", f"step-{args.split_step}"
    )
    for path in (baseline_checkpoint, baseline_split_checkpoint):
        if not path.is_dir():
            raise FileNotFoundError(
                f"expected baseline checkpoint was not saved: {path}"
            )

    completed_restart_root = baseline_root / "completed_restart"
    print(
        f"Validating restart after normal completion: topology={topology.name}, "
        f"expected_load_step={args.total_steps}\n"
        f"Runtime log: {completed_restart_root / 'runtime.log'}",
        flush=True,
    )
    completed_environment = _capture_environment(
        base=common_environment,
        device=device,
        visible_env=visible_env,
        visible_devices=visible_devices,
        training=training,
        topology=topology,
        token_plan=token_plan,
        metrics=completed_restart_root / "metrics.jsonl",
        input_contract=completed_restart_root / "input_contract",
        stop_after=None,
        events=completed_restart_root / "events",
        boundary_step=args.split_step,
    )
    completed_outcome = _run_process(
        command,
        root=root,
        environment=completed_environment,
        log_path=completed_restart_root / "runtime.log",
        timeout=args.failure_timeout,
    )
    completed_loaded_steps = _loaded_steps(completed_restart_root / "events")
    expected_ranks = set(range(topology.world_size))
    completed_restart_exact = (
        set(completed_loaded_steps) == expected_ranks
        and set(completed_loaded_steps.values()) == {args.total_steps}
    )

    baseline_contract = _read_contract(baseline_root / "input_contract")
    exact = args.comparison == "exact"
    scenarios: dict[str, Any] = {}
    signal_by_mode = {
        "sigint": signal.SIGINT,
        "sigterm": signal.SIGTERM,
        "sigkill": signal.SIGKILL,
        "incomplete": signal.SIGKILL,
    }
    descriptions = {
        "graceful": "normal controlled exit immediately after a completed save",
        "sigint": "Ctrl+C equivalent sent to the torchrun process group",
        "sigterm": "SIGTERM sent to the torchrun process group",
        "sigkill": "SIGKILL sent to the torchrun process group",
        "rank-error": f"uncaught exception injected on rank {failure_rank}",
        "rank-sigkill": f"SIGKILL injected on rank {failure_rank}",
        "incomplete": (
            "SIGKILL after a higher step directory appears without DCP metadata"
        ),
    }
    for mode in failure_modes:
        scenario_root = run_root / mode
        ready_path = scenario_root / "fault-ready.json"
        phase_1 = scenario_root / "interrupted"
        phase_2 = scenario_root / "resumed"
        fault_step = args.split_step if mode == "graceful" else interrupt_step
        print(
            f"Starting checkpoint scenario: mode={mode}, topology={topology.name}, "
            f"checkpoint_step={args.split_step}, interrupt_step={fault_step}\n"
            f"Runtime log: {phase_1 / 'runtime.log'}",
            flush=True,
        )
        phase_1_environment = _capture_environment(
            base=common_environment,
            device=device,
            visible_env=visible_env,
            visible_devices=visible_devices,
            training=training,
            topology=topology,
            token_plan=token_plan,
            metrics=phase_1 / "metrics.jsonl",
            input_contract=phase_1 / "input_contract",
            stop_after=args.split_step if mode == "graceful" else None,
            events=phase_1 / "events",
            boundary_step=args.split_step,
            failure_mode=mode,
            failure_step=fault_step,
            failure_rank=failure_rank,
            ready_path=ready_path,
        )
        scenario_command = _checkpoint_command(
            training=training,
            topology=topology,
            dump_folder=scenario_root / "trainer_output",
            initial_checkpoint=initial_checkpoint,
            checkpoint_interval=args.split_step,
            async_mode=args.async_mode,
        )
        interruption = _run_process(
            scenario_command,
            root=root,
            environment=phase_1_environment,
            log_path=phase_1 / "runtime.log",
            expected_failure=mode != "graceful",
            launcher_signal=signal_by_mode.get(mode),
            ready_path=ready_path,
            timeout=args.failure_timeout,
        )
        interruption["description"] = descriptions[mode]

        split_checkpoint = scenario_root.joinpath(
            "trainer_output", "checkpoint", f"step-{args.split_step}"
        )
        if not split_checkpoint.is_dir():
            raise FileNotFoundError(
                f"last complete checkpoint is missing after {mode}: {split_checkpoint}"
            )
        optimizer_state_gaps = _optimizer_state_gaps(
            phase_1 / "events", step=args.split_step
        )
        optimizer_schema_complete = (
            set(optimizer_state_gaps) == expected_ranks
            and not any(optimizer_state_gaps.values())
        )
        if not optimizer_schema_complete:
            examples = {
                rank: names[:10]
                for rank, names in optimizer_state_gaps.items()
                if names
            }
            raise RuntimeError(
                "completed checkpoint has missing optimizer state; restart would "
                f"not be loadable. Missing state by rank: {examples}. "
                f"Worker evidence: {phase_1 / 'events'}"
            )
        partial_checkpoint = scenario_root.joinpath(
            "trainer_output", "checkpoint", f"step-{interrupt_step}"
        )
        partial_before_resume = (
            partial_checkpoint.is_dir()
            and not (partial_checkpoint / ".metadata").is_file()
        )
        if mode != "graceful" and args.restart_delay:
            time.sleep(args.restart_delay)

        print(
            f"Restarting checkpoint scenario: mode={mode}, "
            f"expected_load_step={args.split_step}\n"
            f"Runtime log: {phase_2 / 'runtime.log'}",
            flush=True,
        )
        phase_2_environment = _capture_environment(
            base=common_environment,
            device=device,
            visible_env=visible_env,
            visible_devices=visible_devices,
            training=training,
            topology=topology,
            token_plan=token_plan,
            metrics=phase_2 / "metrics.jsonl",
            input_contract=phase_2 / "input_contract",
            stop_after=None,
            events=phase_2 / "events",
            boundary_step=args.split_step,
        )
        _run_process(
            scenario_command,
            root=root,
            environment=phase_2_environment,
            log_path=phase_2 / "runtime.log",
            timeout=args.failure_timeout,
        )

        loaded_steps = _loaded_steps(phase_2 / "events")
        selection_exact = (
            set(loaded_steps) == expected_ranks
            and set(loaded_steps.values()) == {args.split_step}
        )
        incomplete_checkpoint_ignored = mode != "incomplete" or (
            partial_before_resume and selection_exact
        )
        resumed_final_checkpoint = scenario_root.joinpath(
            "trainer_output", "checkpoint", f"step-{args.total_steps}"
        )
        if not resumed_final_checkpoint.is_dir():
            raise FileNotFoundError(
                f"resumed final checkpoint was not saved: {resumed_final_checkpoint}"
            )

        resumed_contract = _combine_recovered_contracts(
            phase_1 / "input_contract",
            phase_2 / "input_contract",
            restored_step=args.split_step,
        )
        input_contract_exact = baseline_contract == resumed_contract
        combined_metrics = scenario_root / "metrics.jsonl"
        _combine_recovered_metrics(
            phase_1 / "metrics.jsonl",
            phase_2 / "metrics.jsonl",
            combined_metrics,
            restored_step=args.split_step,
        )
        metrics = _metrics_comparison(
            baseline_root / "metrics.jsonl",
            combined_metrics,
            exact=exact,
            loss_absolute_limit=args.loss_atol,
            loss_relative_limit=args.loss_rtol,
            grad_relative_limit=args.grad_relative_limit,
            restored_step=args.split_step,
        )
        resume_summary = summarize_checkpoint(split_checkpoint)
        split_state = compare_checkpoints(
            baseline_split_checkpoint,
            split_checkpoint,
            exact=exact,
            absolute_tolerance=args.state_atol,
            relative_tolerance=args.state_rtol,
        )
        boundary_memory = _boundary_memory_comparison(
            phase_1 / "events",
            phase_2 / "events",
            step=args.split_step,
            expected_ranks=expected_ranks,
        )
        final_state = compare_checkpoints(
            baseline_checkpoint,
            resumed_final_checkpoint,
            exact=exact,
            absolute_tolerance=args.state_atol,
            relative_tolerance=args.state_rtol,
        )
        failure_location = _resume_failure_location(
            split_state=split_state,
            boundary_memory=boundary_memory,
            metrics=metrics,
            final_state=final_state,
            restored_step=args.split_step,
        )
        phase_1_events = _read_events(phase_1 / "events")
        observed_events = {
            record["event"]
            for records in phase_1_events.values()
            for record in records
        }
        fault_observed = (
            "graceful-stop" in observed_events
            if mode == "graceful"
            else "fault-ready" in observed_events
        )
        checks = {
            "Requested interruption was observed": fault_observed,
            "Optimizer state schema is complete on every rank": (
                optimizer_schema_complete
            ),
            "Last complete checkpoint contains all training state": not resume_summary[
                "missing_required_scopes"
            ],
            "Every rank loaded the last complete checkpoint": selection_exact,
            "Higher incomplete checkpoint was ignored": incomplete_checkpoint_ignored,
            "Saved recovery-boundary checkpoint matches baseline": split_state[
                "passed"
            ],
            "Loaded in-memory state matches the pre-restart state": boundary_memory[
                "passed"
            ],
            "Replayed fixed-token input sequence is exact": input_contract_exact,
            "Loss and grad norm meet the selected standard": metrics["passed"],
            "Final logical distributed checkpoint matches": final_state["passed"],
        }
        scenario_passed = all(checks.values())
        scenarios[mode] = {
            "passed": scenario_passed,
            "fault_step": fault_step,
            "interruption": interruption,
            "loaded_steps": loaded_steps,
            "partial_checkpoint_before_resume": partial_before_resume,
            "incomplete_checkpoint_ignored": incomplete_checkpoint_ignored,
            "checkpoint_summary": resume_summary,
            "optimizer_state_gaps": optimizer_state_gaps,
            "split_state": split_state,
            "boundary_memory": boundary_memory,
            "input_contract_exact": input_contract_exact,
            "metrics": metrics,
            "final_state": final_state,
            "failure_location": failure_location,
            "checks": checks,
        }

    passed = completed_restart_exact and all(
        values["passed"] for values in scenarios.values()
    )
    summary = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": 2,
        "run_name": run_name,
        "fixture_generation_id": fixture_generation_id,
        "passed": bool(passed),
        "total_steps": args.total_steps,
        "split_step": args.split_step,
        "interrupt_step": interrupt_step,
        "baseline_contract_ranks": sorted(baseline_contract),
        "completed_restart": {
            "passed": completed_restart_exact,
            "loaded_steps": completed_loaded_steps,
            **completed_outcome,
        },
        "scenarios": scenarios,
        "configuration": {
            "device": device,
            "visible_devices": visible_devices,
            "topology": asdict(topology),
            "batch_schedule": asdict(batch_schedule),
            "training": asdict(training),
            "comparison": args.comparison,
            "state_atol": args.state_atol,
            "state_rtol": args.state_rtol,
            "loss_atol": args.loss_atol,
            "loss_rtol": args.loss_rtol,
            "grad_relative_limit": args.grad_relative_limit,
            "async_mode": args.async_mode,
            "failure_modes": failure_modes,
            "failure_rank": failure_rank,
            "failure_timeout": args.failure_timeout,
            "restart_delay": args.restart_delay,
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(report_path, summary)
    print(
        f"Checkpoint resume status: {'PASS' if passed else 'FAIL'}\n"
        f"Report: {report_path}",
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
