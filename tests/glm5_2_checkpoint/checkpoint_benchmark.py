#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Validate TorchTitan checkpoint save, load, and process-restart resume."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
    TrainingEndpoint,
    ParallelTopology,
    prepare_fixture,
    standard_topologies,
    training_topology_plan,
)


LOSS_KEY = "loss"
GRAD_NORM_KEY = "grad_norm"


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
) -> FormalTrainingConfig:
    values: dict[str, Any] = {
        "steps": steps,
        "local_batch_size": local_batch_size,
        "global_batch_size": global_batch_size,
        "sequence_length": sequence_length,
        "seed": seed,
        "deterministic": True,
    }
    if precision == "fp32":
        values.update(training_dtype="float32", mixed_precision_param="float32")
    elif precision == "full-bf16":
        values.update(training_dtype="bfloat16", mixed_precision_param="bfloat16")
    else:
        values.update(training_dtype="float32", mixed_precision_param="bfloat16")
    return FormalTrainingConfig(**values)


def _device_from_environment(requested: str | None) -> tuple[str, str, str]:
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    npu = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").strip()
    if requested == "cuda":
        if not cuda:
            raise ValueError("CUDA_VISIBLE_DEVICES must be exported")
        return "cuda", "CUDA_VISIBLE_DEVICES", cuda
    if requested == "npu":
        if not npu:
            raise ValueError("ASCEND_RT_VISIBLE_DEVICES must be exported")
        return "npu", "ASCEND_RT_VISIBLE_DEVICES", npu
    if bool(cuda) == bool(npu):
        raise ValueError(
            "export exactly one visibility variable or pass --device cuda|npu"
        )
    return (
        ("cuda", "CUDA_VISIBLE_DEVICES", cuda)
        if cuda
        else ("npu", "ASCEND_RT_VISIBLE_DEVICES", npu)
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


def _run_process(
    command: list[str],
    *,
    root: Path,
    environment: dict[str, str],
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("Command: " + " ".join(command) + "\n\n")
        stream.flush()
        process = subprocess.run(
            command,
            cwd=root,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


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
    environment["LOG_RANK"] = str(_metrics_rank(topology))
    if stop_after is None:
        environment.pop("GLM5_CHECKPOINT_STOP_AFTER", None)
    else:
        environment["GLM5_CHECKPOINT_STOP_AFTER"] = str(stop_after)
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
    scope_rows = "".join(
        "<tr>"
        f"<td>{html.escape(scope)}</td>"
        f"<td>{values['tensor_leaves']}</td>"
        f"<td>{values['tensor_elements']}</td>"
        f"<td>{values['exact_tensor_mismatches']}</td>"
        f"<td>{values['tolerance_tensor_mismatches']}</td>"
        f"<td>{values['non_tensor_mismatches']}</td>"
        f"<td>{values['max_absolute']:.9g}</td>"
        f"<td>{values['mean_absolute']:.9g}</td>"
        "</tr>"
        for scope, values in summary["final_state"]["scopes"].items()
    )
    metric_rows = "".join(
        "<tr>"
        f"<td>{name}</td><td>{values['count']}</td>"
        f"<td>{str(values['bitwise_equal'])}</td>"
        f"<td>{values['mean_absolute']:.9g}</td>"
        f"<td>{values['mean_absolute_relative']:.6%}</td>"
        f"<td>{values['max_absolute']:.9g}</td></tr>"
        for name, values in summary["metrics"].items()
        if name in {LOSS_KEY, GRAD_NORM_KEY}
    )
    checks = {
        "Resume checkpoint contains full training state": not summary[
            "resume_checkpoint"
        ]["missing_required_scopes"],
        "Baseline and segmented runs match before restart": summary[
            "split_state"
        ]["passed"],
        "Loaded step and fixed-token data sequence are exact": summary[
            "input_contract_exact"
        ],
        "Loss and grad norm meet the selected standard": summary["metrics"][
            "passed"
        ],
        "Final model, optimizer, scheduler, dataloader, and step match": summary[
            "final_state"
        ]["passed"],
    }
    check_rows = "".join(
        f"<tr><td>{html.escape(name)}</td>"
        f"<td class={'pass' if passed else 'fail'}>"
        f"{'PASS' if passed else 'FAIL'}</td></tr>"
        for name, passed in checks.items()
    )
    split_mismatch = "\n".join(summary["split_state"]["mismatch_examples"])
    final_mismatch = "\n".join(summary["final_state"]["mismatch_examples"])
    mismatch = (
        "Before restart (baseline step K vs segmented step K):\n"
        f"{split_mismatch or 'None'}\n\n"
        "After resume (baseline step N vs resumed step N):\n"
        f"{final_mismatch or 'None'}"
    )
    path.write_text(
        f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(summary['run_name'])}</title><style>
body{{font:14px/1.45 system-ui,sans-serif;max-width:1280px;margin:auto;padding:24px;color:#17202a}}
table{{border-collapse:collapse;width:100%;margin:16px 0}}th,td{{border:1px solid #d8dee9;padding:8px;text-align:left}}
th{{background:#eef2f7;position:sticky;top:0}}.pass{{color:#16723c;font-weight:800}}.fail{{color:#b42318;font-weight:800}}
.status{{display:inline-block;background:{'#17803d' if status == 'PASS' else '#b42318'};color:white;padding:6px 12px;border-radius:999px;font-weight:800}}
pre{{white-space:pre-wrap;background:#f8fafc;padding:12px;border:1px solid #d8dee9;border-radius:8px}}</style></head>
<body><h1>GLM 5.2 checkpoint resume report</h1><p><span class="status">{status}</span></p>
<p>The baseline trains without interruption. The candidate stops at step {summary['split_step']}, exits, reloads a full DCP checkpoint in a new process, and continues to step {summary['total_steps']}.</p>
<h2>Decisive checks</h2><table><thead><tr><th>Check</th><th>Result</th></tr></thead><tbody>{check_rows}</tbody></table>
<h2>Training metrics</h2><table><thead><tr><th>Series</th><th>Steps</th><th>Bitwise equal</th><th>Mean absolute</th><th>Mean absolute relative</th><th>Max absolute</th></tr></thead><tbody>{metric_rows}</tbody></table>
<h2>Final checkpoint state</h2><table><thead><tr><th>Scope</th><th>Tensor leaves</th><th>Elements</th><th>Exact tensor mismatches</th><th>Tolerance mismatches</th><th>Metadata mismatches</th><th>Max absolute</th><th>Mean absolute</th></tr></thead><tbody>{scope_rows}</tbody></table>
<h2>Mismatch examples</h2><pre>{html.escape(mismatch)}</pre>
<h2>Configuration</h2><pre>{html.escape(json.dumps(summary['configuration'], indent=2, sort_keys=True))}</pre>
</body></html>""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "npu"))
    parser.add_argument("--topology", default="single", choices=standard_topologies())
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
        "--async-mode",
        choices=("disabled", "async", "async_with_pinned_mem"),
        default="disabled",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 0 < args.split_step < args.total_steps:
        raise ValueError("split-step must be between zero and total-steps")

    root = Path(__file__).resolve().parents[2]
    topology = standard_topologies()[args.topology]
    training = _precision_training(
        args.precision,
        steps=args.total_steps,
        local_batch_size=args.local_batch_size,
        global_batch_size=args.global_batch_size,
        sequence_length=args.sequence_length,
        seed=args.seed,
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
    fixture_config = CheckpointFixtureConfig(
        name="checkpoint",
        kind="self_consistency",
        reference=endpoint,
        candidate=endpoint,
        training=training,
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

    fixture = root / fixture_config.fixture_root / fixture_config.storage_name
    if not (fixture / "fixture.json").is_file():
        print(f"Preparing shared seed checkpoint and token plan: {fixture}", flush=True)
        prepare_fixture(root, fixture_config, endpoint=endpoint, force=False)
    manifest = json.loads((fixture / "fixture.json").read_text(encoding="utf-8"))
    initial_checkpoint = fixture / manifest["checkpoint_relative_path"]
    token_plan = fixture / manifest["token_plan_relative_path"]

    common_environment = os.environ.copy()
    phases = (
        (
            "baseline",
            args.total_steps,
            None,
            run_root / "baseline" / "trainer_output",
            run_root / "baseline" / "metrics.jsonl",
            run_root / "baseline" / "input_contract",
        ),
        (
            "resume phase 1",
            args.split_step,
            args.split_step,
            run_root / "resumed" / "trainer_output",
            run_root / "resumed" / "metrics.jsonl",
            run_root / "resumed" / "input_contract_phase_1",
        ),
        (
            "resume phase 2",
            args.total_steps,
            None,
            run_root / "resumed" / "trainer_output",
            run_root / "resumed" / "metrics.jsonl",
            run_root / "resumed" / "input_contract_phase_2",
        ),
    )
    for (
        label,
        target_step,
        stop_after,
        dump_folder,
        metrics_path,
        contract_path,
    ) in phases:
        print(
            f"Starting {label}: topology={topology.name}, target_step={target_step}\n"
            f"Runtime log: {run_root / (label.replace(' ', '_') + '.log')}",
            flush=True,
        )
        environment = _capture_environment(
            base=common_environment,
            device=device,
            visible_env=visible_env,
            visible_devices=visible_devices,
            training=training,
            topology=topology,
            token_plan=token_plan,
            metrics=metrics_path,
            input_contract=contract_path,
            stop_after=stop_after,
        )
        command = _checkpoint_command(
            training=training,
            topology=topology,
            dump_folder=dump_folder,
            initial_checkpoint=initial_checkpoint,
            checkpoint_interval=args.split_step,
            async_mode=args.async_mode,
        )
        _run_process(
            command,
            root=root,
            environment=environment,
            log_path=run_root / (label.replace(" ", "_") + ".log"),
        )

    baseline_checkpoint = run_root.joinpath(
        "baseline", "trainer_output", "checkpoint", f"step-{args.total_steps}"
    )
    baseline_split_checkpoint = run_root.joinpath(
        "baseline", "trainer_output", "checkpoint", f"step-{args.split_step}"
    )
    resume_checkpoint = run_root.joinpath(
        "resumed", "trainer_output", "checkpoint", f"step-{args.split_step}"
    )
    resumed_final_checkpoint = run_root.joinpath(
        "resumed", "trainer_output", "checkpoint", f"step-{args.total_steps}"
    )
    for path in (
        baseline_checkpoint,
        baseline_split_checkpoint,
        resume_checkpoint,
        resumed_final_checkpoint,
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"expected checkpoint was not saved: {path}")

    baseline_contract = _read_contract(run_root / "baseline" / "input_contract")
    resumed_contract = _combine_contracts(
        run_root / "resumed" / "input_contract_phase_1",
        run_root / "resumed" / "input_contract_phase_2",
    )
    input_contract_exact = baseline_contract == resumed_contract
    exact = args.comparison == "exact"
    metrics = _metrics_comparison(
        run_root / "baseline" / "metrics.jsonl",
        run_root / "resumed" / "metrics.jsonl",
        exact=exact,
        loss_absolute_limit=args.loss_atol,
        loss_relative_limit=args.loss_rtol,
        grad_relative_limit=args.grad_relative_limit,
    )
    resume_summary = summarize_checkpoint(resume_checkpoint)
    split_state = compare_checkpoints(
        baseline_split_checkpoint,
        resume_checkpoint,
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
    passed = (
        not resume_summary["missing_required_scopes"]
        and split_state["passed"]
        and input_contract_exact
        and metrics["passed"]
        and final_state["passed"]
    )
    summary = {
        "schema": "torchtitan.glm5_2.checkpoint_resume",
        "schema_version": 1,
        "run_name": run_name,
        "passed": bool(passed),
        "total_steps": args.total_steps,
        "split_step": args.split_step,
        "resume_checkpoint": resume_summary,
        "split_state": split_state,
        "input_contract_exact": input_contract_exact,
        "baseline_contract_ranks": sorted(baseline_contract),
        "resumed_contract_ranks": sorted(resumed_contract),
        "metrics": metrics,
        "final_state": final_state,
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
