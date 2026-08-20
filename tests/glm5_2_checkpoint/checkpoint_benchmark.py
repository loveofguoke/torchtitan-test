#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Validate TorchTitan checkpoint save, load, and process-restart resume."""

from __future__ import annotations

import argparse
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

from tests.glm5_2_common.cli import run_all_topologies
from tests.glm5_2_common.device import resolve_accelerator
from tests.glm5_2_common.topology import DISTRIBUTED_TOPOLOGY_NAMES

from tests.glm5_2_checkpoint.fault_injection import (  # noqa: E402
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
    prepare_fixture,
    standard_topologies,
    training_topology_plan,
)


LOSS_KEY = "loss"
GRAD_NORM_KEY = "grad_norm"


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


class CheckpointFixtureConfig(FormalExperimentConfig):
    """Use a compact checkpoint-specific name for shared test inputs."""

    @property
    def storage_name(self) -> str:
        precision = {
            "mixed-bfloat16": "bf16",
            "mixed-float32": "fp32",
            "full-bf16": "full-bf16",
        }.get(self.training.precision_name, self.training.precision_name)
        return (
            f"ckpt-{self.reference.device_type}-{precision}-"
            f"s{self.training.steps}-b{self.training.global_batch_size}-"
            f"q{self.training.sequence_length}-seed{self.training.seed}"
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


def _signal_process_group(process: subprocess.Popen[str], value: int) -> None:
    if os.name != "posix":
        raise RuntimeError(
            "checkpoint signal-failure tests require a POSIX training host"
        )
    os.killpg(process.pid, value)


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
        if launcher_signal is not None:
            if ready_path is None:
                raise ValueError("ready_path is required when signaling the launcher")
            deadline = time.monotonic() + timeout
            while not ready_path.is_file():
                return_code = process.poll()
                if return_code is not None:
                    raise RuntimeError(
                        "training exited before reaching the fault injection point; "
                        f"see {log_path}"
                    )
                if time.monotonic() >= deadline:
                    _signal_process_group(process, signal.SIGKILL)
                    process.wait()
                    raise TimeoutError(
                        f"timed out waiting for checkpoint fault marker: {ready_path}"
                    )
                time.sleep(0.1)
            signal_sent = signal.Signals(launcher_signal).name
            _signal_process_group(process, launcher_signal)
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            _signal_process_group(process, signal.SIGKILL)
            process.wait()
            raise TimeoutError(
                f"training did not exit after fault; see {log_path}"
            ) from error

    if expected_failure:
        if return_code == 0:
            raise RuntimeError(
                f"fault-injected training exited successfully; see {log_path}"
            )
    elif return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return {"return_code": return_code, "launcher_signal": signal_sent}


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
        f"--training.local_batch_size={training.local_batch_size}",
        f"--training.global_batch_size={training.global_batch_size}",
        f"--training.seq_len={training.sequence_length}",
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
    failure_mode: str | None = None,
    failure_step: int | None = None,
    failure_rank: int = 0,
    ready_path: Path | None = None,
) -> dict[str, str]:
    environment = dict(base)
    environment[visible_env] = visible_devices
    environment["TORCHTITAN_DEVICE"] = "npu" if device == "npu" else "gpu"
    environment["GLM5_PRECISION_METRICS_PATH"] = str(metrics)
    environment["GLM5_PRECISION_TOKEN_PLAN_PATH"] = str(token_plan)
    environment["GLM5_PRECISION_INPUT_CONTRACT_DIR"] = str(input_contract)
    environment["GLM5_PRECISION_GLOBAL_BATCH_SIZE"] = str(
        training.global_batch_size
    )
    environment["GLM5_PRECISION_TRAINING_LOCAL_BATCH_SIZE"] = str(
        training.local_batch_size
    )
    environment["GLM5_PRECISION_TENSOR_PARALLEL_DEGREE"] = str(
        topology.tensor_parallel_degree
    )
    environment["GLM5_PRECISION_CONTEXT_PARALLEL_DEGREE"] = str(
        topology.context_parallel_degree
    )
    environment["LOG_RANK"] = str(_metrics_rank(topology))
    environment[EVENTS_DIR_ENV] = str(events)
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
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if retain_interrupted and int(record["step"]) > restored_step:
                continue
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


def _metrics_comparison(
    reference_path: Path,
    candidate_path: Path,
    *,
    exact: bool,
    loss_absolute_limit: float,
    loss_relative_limit: float,
    grad_relative_limit: float,
) -> dict[str, Any]:
    reference = _metric_series(reference_path)
    candidate = _metric_series(candidate_path)
    if reference[LOSS_KEY].keys() != candidate[LOSS_KEY].keys():
        return {"passed": False, "error": "metric step sets differ"}
    result: dict[str, Any] = {"mode": "exact" if exact else "tolerance"}
    for name in (LOSS_KEY, GRAD_NORM_KEY):
        reference_values = tuple(
            reference[name][step] for step in sorted(reference[name])
        )
        candidate_values = tuple(
            candidate[name][step] for step in sorted(candidate[name])
        )
        metrics = compute_error_metrics(reference_values, candidate_values)
        result[name] = {
            "count": metrics.count,
            "normal": metrics.normal,
            "mean_absolute": metrics.absolute,
            "mean_relative": metrics.relative_normal,
            "mean_absolute_relative": metrics.relative_absolute,
            "max_absolute": metrics.max_absolute,
            "bitwise_equal": exact_series_match(reference[name], candidate[name]),
        }
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
            f"<td>{metric['mean_absolute']:.9g}</td>"
            f"<td>{metric['mean_absolute_relative']:.6%}</td>"
            f"<td>{metric['max_absolute']:.9g}</td></tr>"
            for name, metric in values["metrics"].items()
            if name in {LOSS_KEY, GRAD_NORM_KEY}
        )
        mismatch = "\n".join(values["final_state"]["mismatch_examples"])
        scenario_details += f"""
<details><summary>{html.escape(mode)}: {scenario_status}</summary>
<h3>Checks</h3><table><thead><tr><th>Check</th><th>Result</th></tr></thead><tbody>{check_rows}</tbody></table>
<h3>Training metrics</h3><table><thead><tr><th>Series</th><th>Steps</th><th>Bitwise equal</th><th>Mean absolute</th><th>Mean absolute relative</th><th>Max absolute</th></tr></thead><tbody>{metric_rows}</tbody></table>
<h3>Mismatch examples</h3><pre>{html.escape(mismatch or 'None')}</pre>
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
    if args.topology == "all":
        return run_all_topologies(
            __file__, argv=sys.argv[1:], topology_names=suite_topologies
        )
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
    topology = standard_topologies()[args.topology]
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
    device, visible_env, visible_devices = _device_from_environment(args.device)
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
    )
    save_mode = {
        "disabled": "sync",
        "async": "async",
        "async_with_pinned_mem": "pinned",
    }[args.async_mode]
    run_name = (
        f"ckpt-{device}-{topology.slug}-{args.precision}-"
        f"s{args.total_steps}-k{args.split_step}-"
        f"b{args.global_batch_size}-q{args.sequence_length}-"
        f"seed{args.seed}-{save_mode}"
    )
    if failure_modes != ["graceful"]:
        mode_name = "all-faults" if len(failure_modes) > 1 else failure_modes[0]
        run_name += f"-{mode_name}"
    if args.run_tag:
        run_name += "-" + _normalize_run_tag(args.run_tag)
    run_root = root / "checkpoint_runs" / run_name
    report_root = root / "checkpoint_reports" / run_name
    if run_root.exists() or report_root.exists():
        if not args.force:
            raise FileExistsError(
                f"checkpoint test output exists; pass --force to replace it: {run_name}"
            )
        shutil.rmtree(run_root, ignore_errors=True)
        shutil.rmtree(report_root, ignore_errors=True)
    run_root.mkdir(parents=True)
    report_root.mkdir(parents=True)

    fixture = (
        root / fixture_config.fixture_root / fixture_config.fixture_storage_name
    )
    if not (fixture / "fixture.json").is_file():
        print(f"Preparing shared seed checkpoint and token plan: {fixture}", flush=True)
        prepare_fixture(root, fixture_config, endpoint=endpoint, force=False)
    manifest = json.loads((fixture / "fixture.json").read_text(encoding="utf-8"))
    initial_checkpoint = fixture / manifest["checkpoint_relative_path"]
    token_plan = fixture / manifest["token_plan_relative_path"]

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
        )
        resume_summary = summarize_checkpoint(split_checkpoint)
        split_state = compare_checkpoints(
            baseline_split_checkpoint,
            split_checkpoint,
            exact=exact,
            absolute_tolerance=args.state_atol,
            relative_tolerance=args.state_rtol,
        )
        final_state = compare_checkpoints(
            baseline_checkpoint,
            resumed_final_checkpoint,
            exact=exact,
            absolute_tolerance=args.state_atol,
            relative_tolerance=args.state_rtol,
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
            "Checkpoint state at the recovery boundary matches": split_state["passed"],
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
            "input_contract_exact": input_contract_exact,
            "metrics": metrics,
            "final_state": final_state,
            "checks": checks,
        }

    passed = completed_restart_exact and all(
        values["passed"] for values in scenarios.values()
    )
    summary = {
        "schema": "torchtitan.glm5_2.checkpoint_fault_recovery",
        "schema_version": 2,
        "run_name": run_name,
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
    summary_path = report_root / f"{run_name}.json"
    report_path = report_root / f"{run_name}.html"
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
