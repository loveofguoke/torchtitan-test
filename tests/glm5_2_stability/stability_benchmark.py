#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Reusable single-node GLM 5.2 training stability benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_common.cli import (
    archive_previous_output,
    reset_output_generation,
    run_all_topologies,
)
from tests.glm5_2_common.device import resolve_accelerator
from tests.glm5_2_common.naming import config_name
from tests.glm5_2_common.topology import (
    DISTRIBUTED_TOPOLOGY_NAMES,
    training_command_args,
)

from tests.glm5_2_graph.config import validate_graph_training_args  # noqa: E402
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
from tests.glm5_2_precision.token_data import (  # noqa: E402
    TokenDataError,
    load_token_plan,
    validate_runtime_input_contract,
)


LOSS_KEY = "loss_metrics/global_avg_loss"
MAX_LOSS_KEY = "loss_metrics/global_max_loss"
GRAD_NORM_KEY = "grad_norm"
DEFAULT_STEPS = 20_000
DEFAULT_MINIMUM_HOURS = 1.0
DEFAULT_STALL_TIMEOUT_MINUTES = 20.0
STABILITY_SCHEMA = "torchtitan.glm5_2.stability"


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


def _precision_training(precision: str, **overrides: Any) -> FormalTrainingConfig:
    values: dict[str, Any] = {
        "steps": overrides["steps"],
        "local_batch_size": overrides["local_batch_size"],
        "global_batch_size": overrides["global_batch_size"],
        "sequence_length": overrides["sequence_length"],
        "seed": overrides["seed"],
        "deterministic": True,
        "extra_args": tuple(overrides.get("extra_args", ())),
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


def _command(
    *,
    training: FormalTrainingConfig,
    topology: ParallelTopology,
    dump_folder: Path,
    initial_checkpoint: Path,
) -> list[str]:
    torchrun = shutil.which("torchrun") or str(
        Path(sys.executable).with_name("torchrun")
    )
    rank = _metrics_rank(topology)
    return [
        torchrun,
        f"--nproc_per_node={topology.world_size}",
        "--rdzv_backend=c10d",
        "--rdzv_endpoint=localhost:0",
        f"--local-ranks-filter={rank}",
        "--role=rank",
        "--tee=3",
        "-m",
        "tests.glm5_2_stability.capture_metrics",
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
        "--checkpoint.load_only",
        f"--checkpoint.initial_load_path={initial_checkpoint}",
        *training.extra_args,
    ]


def _metric_progress(path: Path) -> tuple[int, int]:
    if not path.is_file():
        return 0, 0
    lines = path.read_text(encoding="utf-8").splitlines()
    valid_count = 0
    last_step = 0
    for line in lines:
        try:
            last_step = int(json.loads(line)["step"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            # The monitor can observe the final line between append and flush.
            # Final report parsing remains strict and diagnoses malformed data.
            break
        valid_count += 1
    return valid_count, last_step


def _stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.terminate()
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            record = json.loads(line)
            values = record["metrics"]
            for key in (LOSS_KEY, MAX_LOSS_KEY, GRAD_NORM_KEY):
                value = float(values[key])
                if not math.isfinite(value):
                    raise ValueError(f"non-finite {key}")
            records.append(record)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"invalid stability metric at {path}:{line_number}: {error}"
            ) from error
    return records


def _write_report(
    *,
    report_path: Path,
    summary_path: Path,
    summary: dict[str, Any],
) -> None:
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    status = summary["status"]
    status_color = {
        "PASS": "#17803d",
        "INSUFFICIENT_DURATION": "#b7791f",
    }.get(status, "#b42318")
    rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th>"
        f"<td>{html.escape(str(value))}</td></tr>"
        for key, value in summary.items()
        if key not in {"command", "metric_ranges"}
    )
    ranges = "".join(
        f"<tr><th>{html.escape(key)}</th><td>{value['minimum']:.9g}</td>"
        f"<td>{value['maximum']:.9g}</td><td>{value['final']:.9g}</td></tr>"
        for key, value in summary.get("metric_ranges", {}).items()
    )
    report_path.write_text(
        f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(summary['run_name'])}</title><style>
body{{font:14px/1.45 system-ui,sans-serif;max-width:1200px;margin:auto;padding:24px;color:#17202a}}
.status{{display:inline-block;background:{status_color};color:white;padding:6px 12px;border-radius:999px;font-weight:800}}
table{{border-collapse:collapse;width:100%;margin:18px 0}}th,td{{border:1px solid #d8dee9;padding:8px;text-align:left}}
th{{background:#eef2f7}}pre{{white-space:pre-wrap;background:#f8fafc;padding:12px;border:1px solid #d8dee9;border-radius:8px}}
</style></head><body><h1>GLM 5.2 training stability report</h1>
<p><span class="status">{html.escape(status)}</span></p>
<p>PASS requires normal process exit, every requested step, finite loss and grad norm, no detected stall, and the configured minimum wall-clock duration.</p>
<table>{rows}</table><h2>Metric ranges</h2>
<table><thead><tr><th>Metric</th><th>Minimum</th><th>Maximum</th><th>Final</th></tr></thead><tbody>{ranges}</tbody></table>
<h2>Command</h2><pre>{html.escape(' '.join(summary['command']))}</pre>
</body></html>""",
        encoding="utf-8",
    )


def _stability_output_names(
    *,
    device: str,
    topology_slug: str,
    precision: str,
    steps: int,
    local_batch_size: int,
    global_batch_size: int,
    sequence_length: int,
    seed: int,
    run_tag: str | None,
    extra_args: tuple[str, ...] = (),
    topology_identity: dict[str, Any] | None = None,
) -> tuple[str, str]:
    suffix = (
        f"{precision}-s{steps}-l{local_batch_size}-b{global_batch_size}-"
        f"seq{sequence_length}-seed{seed}"
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
        "steps": steps,
        "local_batch_size": local_batch_size,
        "global_batch_size": global_batch_size,
        "sequence_length": sequence_length,
        "seed": seed,
        "run_tag": run_tag,
        "extra_args": extra_args,
    }
    suite_name = config_name(suite_base, identity)
    member_name = config_name(
        member_base,
        {**identity, "topology": topology_identity or topology_slug},
    )
    return suite_name, member_name


def _legacy_stability_output_names(
    *,
    device: str,
    topology_slug: str,
    precision: str,
    steps: int,
    local_batch_size: int,
    global_batch_size: int,
    sequence_length: int,
    seed: int,
    run_tag: str | None,
    extra_args: tuple[str, ...] = (),
    topology_identity: dict[str, Any] | None = None,
) -> tuple[str, str]:
    suffix = (
        f"{precision}-s{steps}-l{local_batch_size}-b{global_batch_size}-"
        f"seq{sequence_length}-seed{seed}"
    )
    suite_name = f"stability-{device}-{suffix}"
    member_name = f"stability-{device}-{topology_slug}-{suffix}"
    if run_tag:
        normalized_tag = _normalize_run_tag(run_tag)
        suite_name += f"-{normalized_tag}"
        member_name += f"-{normalized_tag}"
    return suite_name, member_name


def _completed_stability_member(
    summary_path: Path,
    *,
    member_name: str,
    fixture_generation_id: str | None = None,
) -> bool:
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return False
    generation_matches = (
        fixture_generation_id is None
        or summary.get("fixture_generation_id") == fixture_generation_id
    )
    return generation_matches and (
        summary.get("schema") == STABILITY_SCHEMA
        and summary.get("schema_version") == 1
        and summary.get("run_name") == member_name
        and summary.get("status") == "PASS"
    )


def _adopt_legacy_stability_outputs(
    root: Path,
    *,
    suite_name: str,
    member_name: str,
    legacy_suite_name: str,
    legacy_member_name: str,
    topology_slug: str,
) -> None:
    for root_name in ("stability_runs", "stability_reports"):
        parent = root / root_name
        source = parent / legacy_suite_name
        destination = parent / suite_name
        if destination.exists() or not source.is_dir():
            continue
        parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        print(f"Adopted matching legacy output:\n  {source}\n  -> {destination}")
    report_directory = root / "stability_reports" / suite_name / topology_slug
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
    parser.add_argument("--precision", default="bf16", choices=("fp32", "bf16", "full-bf16"))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--minimum-hours", type=float, default=DEFAULT_MINIMUM_HOURS)
    parser.add_argument(
        "--stall-timeout-minutes",
        type=float,
        default=DEFAULT_STALL_TIMEOUT_MINUTES,
    )
    parser.add_argument("--local-batch-size", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=61)
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
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.minimum_hours < 0 or args.stall_timeout_minutes <= 0:
        raise ValueError("steps and timing limits must be positive")

    root = Path(__file__).resolve().parents[2]
    # Stability fixtures are global-input contracts, not topology-specific
    # outputs. `--data --topology all` therefore prepares one single-card copy.
    topology_name = "single" if args.data or args.topology == "all" else args.topology
    topology = standard_topologies()[topology_name]
    training = _precision_training(
        args.precision,
        steps=args.steps,
        local_batch_size=args.local_batch_size,
        global_batch_size=args.global_batch_size,
        sequence_length=args.sequence_length,
        seed=args.seed,
        extra_args=tuple(args.extra_train_arg),
    )
    batch_schedule = training_topology_plan(training, topology)
    device, visible_env, visible_devices = _device_from_environment(args.device)
    validate_graph_training_args(
        device_type=device,
        arguments=args.extra_train_arg,
    )
    if args.topology == "all" and not args.data:
        child_argv = [value for value in sys.argv[1:] if value != "--force"]
        if args.force:
            selected_paths: list[Path] = []
            for member_topology_name in suite_topologies:
                member_topology = standard_topologies()[member_topology_name]
                suite_name, _ = _stability_output_names(
                    device=device,
                    topology_slug=member_topology.slug,
                    precision=args.precision,
                    steps=training.steps,
                    local_batch_size=training.local_batch_size,
                    global_batch_size=training.global_batch_size,
                    sequence_length=training.sequence_length,
                    seed=training.seed,
                    run_tag=args.run_tag,
                    extra_args=training.extra_args,
                    topology_identity=asdict(member_topology),
                )
                selected_paths.extend(
                    (
                        root
                        / "stability_runs"
                        / suite_name
                        / member_topology.slug,
                        root
                        / "stability_reports"
                        / suite_name
                        / member_topology.slug,
                    )
                )
            reset_output_generation(selected_paths)
        return run_all_topologies(
            __file__, argv=child_argv, topology_names=suite_topologies
        )
    visible_count = len([value for value in visible_devices.split(",") if value.strip()])
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
    suite_name, run_name = _stability_output_names(
        device=device,
        topology_slug=topology.slug,
        precision=args.precision,
        steps=training.steps,
        local_batch_size=training.local_batch_size,
        global_batch_size=training.global_batch_size,
        sequence_length=training.sequence_length,
        seed=training.seed,
        run_tag=args.run_tag,
        extra_args=training.extra_args,
        topology_identity=asdict(topology),
    )
    fixture_config = FormalExperimentConfig(
        name="stability",
        kind="self_consistency",
        reference=endpoint,
        candidate=endpoint,
        training=replace(training, extra_args=()),
        fixture_root="stability_fixtures",
        artifact_root="stability_artifacts",
        report_root="stability_reports",
        run_root="stability_runs",
        fixture_name=suite_name,
    )
    fixture_directory = _fixture_directory(root, fixture_config)
    fixture_manifest = fixture_directory / "fixture.json"
    if args.data:
        path = prepare_fixture(
            root,
            fixture_config,
            endpoint=endpoint,
            force=args.force,
        )
        print(f"Prepared stability fixture: {path}")
        return 0
    fixture_generation_id = None
    if fixture_manifest.is_file():
        fixture_generation_id = json.loads(
            fixture_manifest.read_text(encoding="utf-8")
        ).get("generation_id")
    if not fixture_manifest.is_file():
        if fixture_directory.exists():
            raise RuntimeError(
                "stability fixture is incomplete; recreate it with --data "
                f"--force: {fixture_directory}"
            )
        print(
            f"Preparing shared seed checkpoint and token plan: {fixture_directory}",
            flush=True,
        )
        prepare_fixture(
            root,
            fixture_config,
            endpoint=endpoint,
            force=False,
        )
    initial_checkpoint, token_plan = resolve_fixture_inputs(root, fixture_config)

    legacy_suite_name, legacy_run_name = _legacy_stability_output_names(
        device=device,
        topology_slug=topology.slug,
        precision=args.precision,
        steps=training.steps,
        local_batch_size=training.local_batch_size,
        global_batch_size=training.global_batch_size,
        sequence_length=training.sequence_length,
        seed=training.seed,
        run_tag=args.run_tag,
        extra_args=training.extra_args,
        topology_identity=asdict(topology),
    )
    if not training.extra_args:
        _adopt_legacy_stability_outputs(
            root,
            suite_name=suite_name,
            member_name=run_name,
            legacy_suite_name=legacy_suite_name,
            legacy_member_name=legacy_run_name,
            topology_slug=topology.slug,
        )
    run_directory = root / "stability_runs" / suite_name / topology.slug
    report_directory = root / "stability_reports" / suite_name / topology.slug
    report_path = report_directory / f"{run_name}.html"
    summary_path = report_directory / f"{run_name}.json"
    if run_directory.exists() or report_directory.exists():
        if not args.force and _completed_stability_member(
            summary_path,
            member_name=run_name,
            fixture_generation_id=fixture_generation_id,
        ):
            print(
                f"Skip completed stability topology: {topology.name}\n"
                f"Report: {report_path}",
                flush=True,
            )
            return 0
        if args.force:
            shutil.rmtree(run_directory, ignore_errors=True)
            shutil.rmtree(report_directory, ignore_errors=True)
        else:
            archived = [
                str(path)
                for path in (
                    archive_previous_output(run_directory),
                    archive_previous_output(report_directory),
                )
                if path is not None
            ]
            print(
                "Retry incomplete stability topology; archived previous output: "
                + ", ".join(archived),
                flush=True,
            )
    run_directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = run_directory / "raw_metrics.jsonl"
    runtime_log = run_directory / "runtime.log"
    command = _command(
        training=training,
        topology=topology,
        dump_folder=run_directory / "trainer_output",
        initial_checkpoint=initial_checkpoint,
    )
    environment = os.environ.copy()
    environment[visible_env] = visible_devices
    environment["TORCHTITAN_DEVICE"] = "npu" if device == "npu" else "gpu"
    environment["GLM5_STABILITY_METRICS_PATH"] = str(metrics_path)
    environment.update(
        fixed_input_environment(
            token_plan_path=token_plan,
            input_contract_directory=run_directory / "input_contract",
            training=training,
            topology=topology,
        )
    )
    environment["LOG_RANK"] = str(_metrics_rank(topology))

    print(
        f"Starting stability run: device={device}, topology={topology.name}, "
        f"steps={training.steps}, minimum_hours={args.minimum_hours}\n"
        f"Runtime log: {runtime_log}",
        flush=True,
    )
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    last_progress = started
    last_report = started
    observed_count = 0
    observed_step = 0
    stalled = False
    interrupted = False
    with runtime_log.open("w", encoding="utf-8") as log:
        log.write("Command: " + " ".join(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=os.name != "nt",
        )
        try:
            while process.poll() is None:
                time.sleep(5)
                count, step = _metric_progress(metrics_path)
                now = time.monotonic()
                if count > observed_count:
                    observed_count, observed_step = count, step
                    last_progress = now
                if now - last_report >= 60:
                    print(
                        f"Stability progress: step={observed_step}/{training.steps}, "
                        f"elapsed_hours={(now - started) / 3600:.3f}",
                        flush=True,
                    )
                    last_report = now
                if now - last_progress > args.stall_timeout_minutes * 60:
                    stalled = True
                    print(
                        f"No new metric for {args.stall_timeout_minutes:g} minutes; "
                        "terminating the stalled run.",
                        file=sys.stderr,
                        flush=True,
                    )
                    _stop_process(process)
                    break
        except KeyboardInterrupt:
            interrupted = True
            _stop_process(process)
        return_code = process.wait()
    ended = time.monotonic()
    ended_at = datetime.now(timezone.utc)
    records = _read_metrics(metrics_path)
    steps = [int(record["step"]) for record in records]
    contiguous = steps == list(range(1, len(steps) + 1))
    completed = len(records) == training.steps and steps[-1:] == [training.steps]
    input_contract: dict[str, Any]
    if return_code == 0 and completed:
        try:
            input_contract = validate_runtime_input_contract(
                contract_directory=run_directory / "input_contract",
                plan=load_token_plan(token_plan),
                steps=training.steps,
                global_batch_size=training.global_batch_size,
                training_local_batch_size=training.local_batch_size,
                dp_world_size=topology.data_parallel_degree,
                context_parallel_degree=topology.context_parallel_degree,
                tensor_parallel_degree=topology.tensor_parallel_degree,
                pipeline_parallel_degree=topology.pipeline_parallel_degree,
                node_rank=0,
                num_processes_per_node=topology.world_size,
            )
        except TokenDataError as error:
            input_contract = {"valid": False, "error": str(error)}
    else:
        input_contract = {
            "valid": False,
            "error": "training did not complete every requested step",
        }
    elapsed_seconds = ended - started
    sufficient_duration = elapsed_seconds >= args.minimum_hours * 3600
    if interrupted:
        status = "INTERRUPTED"
    elif (
        return_code != 0
        or stalled
        or not contiguous
        or not completed
        or not input_contract["valid"]
    ):
        status = "FAIL"
    elif not sufficient_duration:
        status = "INSUFFICIENT_DURATION"
    else:
        status = "PASS"

    def metric_range(key: str) -> dict[str, float]:
        values = [float(record["metrics"][key]) for record in records]
        return {
            "minimum": min(values),
            "maximum": max(values),
            "final": values[-1],
        }

    ranges = (
        {
            "loss": metric_range(LOSS_KEY),
            "global_max_loss": metric_range(MAX_LOSS_KEY),
            "grad_norm": metric_range(GRAD_NORM_KEY),
        }
        if records
        else {}
    )
    summary = {
        "schema": STABILITY_SCHEMA,
        "schema_version": 1,
        "run_name": run_name,
        "fixture_generation_id": fixture_generation_id,
        "status": status,
        "device": device,
        "visible_devices": visible_devices,
        "topology": asdict(topology),
        "batch_schedule": asdict(batch_schedule),
        "requested_steps": training.steps,
        "completed_metric_steps": len(records),
        "last_step": steps[-1] if steps else 0,
        "steps_contiguous": contiguous,
        "process_return_code": return_code,
        "stalled": stalled,
        "minimum_hours": args.minimum_hours,
        "elapsed_hours": elapsed_seconds / 3600,
        "average_seconds_per_completed_step": (
            elapsed_seconds / len(records) if records else None
        ),
        "started_at_utc": started_at.isoformat(),
        "ended_at_utc": ended_at.isoformat(),
        "runtime_log": str(runtime_log),
        "raw_metrics": str(metrics_path),
        "fixture": str(fixture_directory),
        "initial_checkpoint": str(initial_checkpoint),
        "token_plan": str(token_plan),
        "input_contract": input_contract,
        "metric_ranges": ranges,
        "command": command,
    }
    _write_report(report_path=report_path, summary_path=summary_path, summary=summary)
    print(f"Stability status: {status}\nReport: {report_path}", flush=True)
    return 0 if status == "PASS" else 2 if status == "INSUFFICIENT_DURATION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
