"""Capture, analyze, and report bounded CUDA profiler runs."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Sequence

from gpu_analysis import build_analysis, render_html_report
from gpu_config import GpuPerformanceConfig, gpu_topologies
from tests.glm5_2_common.topology import training_command_args


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _repository_root(script_path: str) -> Path:
    return Path(script_path).resolve().parents[2]


def _visible_devices() -> list[str]:
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not value:
        raise ValueError("CUDA_VISIBLE_DEVICES must be exported")
    return [part.strip() for part in value.split(",") if part.strip()]


def _run_name(config: GpuPerformanceConfig) -> str:
    precision = "bf16" if config.mixed_precision_param == "bfloat16" else config.mixed_precision_param
    return (
        f"gpu-{config.topology}-{precision}-s{config.steps}-"
        f"b{config.global_batch_size}-seq{config.sequence_length}-seed{config.seed}"
    )


def _metrics_rank(world_size: int, pp: int, pp_schedule: str) -> int:
    if pp == 1 or pp_schedule == "ZBVZeroBubble":
        return 0
    return (world_size // pp) * (pp - 1)


def _training_command(
    *, root: Path, config: GpuPerformanceConfig, run_directory: Path
) -> list[str]:
    topology = gpu_topologies()[config.topology]
    visible = _visible_devices()
    if len(visible) < topology.world_size:
        raise ValueError(
            f"topology {topology.name} needs {topology.world_size} GPUs, "
            f"but CUDA_VISIBLE_DEVICES exposes {len(visible)}"
        )
    torchrun = shutil.which("torchrun") or str(Path(sys.executable).with_name("torchrun"))
    profile_frequency = config.warmup_steps + config.active_steps
    return [
        torchrun,
        "--nnodes=1",
        "--node_rank=0",
        f"--nproc_per_node={topology.world_size}",
        "--rdzv_backend=c10d",
        "--rdzv_endpoint=localhost:0",
        f"--rdzv_id={_run_name(config)}",
        f"--local-ranks-filter={_metrics_rank(topology.world_size, topology.pp, topology.pp_schedule)}",
        "--role=rank",
        "--tee=3",
        str(root / "tests" / "glm5_2_performance-gpu" / "capture_metrics.py"),
        "--module",
        config.module,
        "--config",
        config.model_config,
        f"--dump_folder={run_directory / 'trainer_output'}",
        f"--training.steps={config.steps}",
        *training_command_args(
            local_batch_size=config.local_batch_size,
            global_batch_size=config.global_batch_size,
            sequence_length=config.sequence_length,
            topology=topology,
        ),
        f"--training.dtype={config.training_dtype}",
        f"--training.mixed_precision_param={config.mixed_precision_param}",
        f"--debug.seed={config.seed}",
        "--debug.no-enable-structured-logging",
        "--metrics.log_freq=1",
        "--metrics.enable_tensorboard",
        "--metrics.disable_color_printing",
        "--metrics.save_tb_folder=tensorboard",
        "--profiler.enable_profiling",
        "--profiler.save_traces_folder=profiling/traces",
        f"--profiler.profile_freq={profile_frequency}",
        "--profiler.profiler_repeat=1",
        f"--profiler.profiler_skip_first={config.skip_steps}",
        f"--profiler.profiler_warmup={config.warmup_steps}",
        f"--profiler.profiler_active={config.active_steps}",
        *topology.command_args(),
        *config.extra_args,
    ]


def _run_process(
    command: Sequence[str], *, root: Path, environment: dict[str, str], log_path: Path
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("Command: " + " ".join(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            list(command), cwd=root, env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def capture(
    root: Path, config: GpuPerformanceConfig, *, force: bool = False
) -> tuple[Path, Path]:
    run_name = _run_name(config)
    run_directory = root / config.run_root / config.topology / run_name
    artifact_directory = root / config.artifact_root / config.topology / run_name
    if run_directory.exists():
        if not force:
            raise FileExistsError(f"capture already exists: {run_directory}; use --force")
        expected_parent = (root / config.run_root / config.topology).resolve()
        if run_directory.resolve().parent != expected_parent:
            raise RuntimeError(f"refusing to replace unexpected path: {run_directory}")
        shutil.rmtree(run_directory)
    if artifact_directory.exists() and force:
        expected_parent = (root / config.artifact_root / config.topology).resolve()
        if artifact_directory.resolve().parent != expected_parent:
            raise RuntimeError(f"refusing to replace unexpected path: {artifact_directory}")
        shutil.rmtree(artifact_directory)

    run_directory.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update(
        {
            "TORCHTITAN_DEVICE": "gpu",
            # A shared environment may still have torch_npu installed. Do not let
            # PyTorch auto-load unrelated private-use backends in this CUDA-only
            # subprocess; a missing CANN/HCCL runtime must not break torch import.
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
            "GLM5_GPU_PERFORMANCE_METRICS_PATH": str(run_directory / "metrics.jsonl"),
            "GLM5_GPU_PROFILE_RANKS": config.profile_ranks,
            "GLM5_GPU_PROFILE_RECORD_SHAPES": str(config.record_shapes).lower(),
            "GLM5_GPU_PROFILE_MEMORY": str(config.profile_memory).lower(),
            "GLM5_GPU_PROFILE_WITH_STACK": str(config.with_stack).lower(),
            "GLM5_GPU_PROFILE_WITH_FLOPS": str(config.with_flops).lower(),
        }
    )
    topology = gpu_topologies()[config.topology]
    environment["LOG_RANK"] = str(
        _metrics_rank(topology.world_size, topology.pp, topology.pp_schedule)
    )
    command = _training_command(root=root, config=config, run_directory=run_directory)
    try:
        _run_process(
            command, root=root, environment=environment,
            log_path=run_directory / "runtime.log",
        )
    except Exception as error:
        _write_json(run_directory / "failure.json", {"error": repr(error), "command": command})
        raise

    manifest = {
        "format_version": 1,
        "run_name": run_name,
        "device": "cuda",
        "visible_devices": _visible_devices(),
        "topology": config.topology,
        "config": config.as_dict(),
        "command": command,
        "run_directory": str(run_directory),
    }
    _write_json(run_directory / "manifest.json", manifest)
    artifact_directory.mkdir(parents=True, exist_ok=True)
    _write_json(artifact_directory / "manifest.json", manifest)
    metrics = run_directory / "metrics.jsonl"
    if metrics.is_file():
        shutil.copy2(metrics, artifact_directory / metrics.name)
    return run_directory, artifact_directory


def analyze(root: Path, config: GpuPerformanceConfig) -> Path:
    run_name = _run_name(config)
    run_directory = root / config.run_root / config.topology / run_name
    artifact_directory = root / config.artifact_root / config.topology / run_name
    manifest_path = run_directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"capture manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    analysis = build_analysis(run_directory)
    _write_json(artifact_directory / "analysis.json", analysis)
    report = root / config.report_root / config.topology / f"{run_name}.html"
    render_html_report(manifest=manifest, analysis=analysis, output_path=report)
    print(f"GPU performance report: {report}")
    return report


def run_gpu_profiler_cli(config: GpuPerformanceConfig, script_path: str) -> None:
    parser = argparse.ArgumentParser(description="Capture and analyze TorchTitan CUDA performance")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--probe", action="store_true")
    actions.add_argument("--capture", action="store_true")
    actions.add_argument("--analyze", action="store_true")
    parser.add_argument("--topology", choices=tuple(gpu_topologies()))
    parser.add_argument("--visible-devices")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--skip-steps", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--active-steps", type=int)
    parser.add_argument("--local-batch-size", type=int)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--profile-ranks")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--extra-train-arg", action="append", default=[])
    args = parser.parse_args()
    if args.visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_devices
    overrides = {
        "topology": args.topology,
        "steps": args.steps,
        "skip_steps": args.skip_steps,
        "warmup_steps": args.warmup_steps,
        "active_steps": args.active_steps,
        "local_batch_size": args.local_batch_size,
        "global_batch_size": args.global_batch_size,
        "sequence_length": args.sequence_length,
        "profile_ranks": args.profile_ranks,
    }
    effective = replace(
        config,
        **{key: value for key, value in overrides.items() if value is not None},
        profile_memory=config.profile_memory or args.profile_memory,
        with_stack=config.with_stack or args.with_stack,
        extra_args=config.extra_args + tuple(args.extra_train_arg),
    )
    root = _repository_root(script_path)
    if args.capture or args.probe:
        capture(root, effective, force=args.force)
    if args.analyze or args.probe:
        analyze(root, effective)
