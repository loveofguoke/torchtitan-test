"""Run, post-process, and report independent profiler experiments."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from .analysis import build_analysis, render_html_report
from .config import (
    PerformanceConfig,
    ProfilerPreset,
    profiler_presets,
    standard_topologies,
)


def _slug(value: str) -> str:
    result = "".join(
        character if character.isalnum() else "-" for character in value.lower()
    )
    return "-".join(part for part in result.split("-") if part)


def _repository_root(script_path: str) -> Path:
    return Path(script_path).resolve().parents[2]


def _git_value(root: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _source_metadata(root: Path) -> dict[str, str | None]:
    metadata: dict[str, str | None] = {
        "test_commit": _git_value(root, "rev-parse", "HEAD"),
        "test_branch": _git_value(root, "branch", "--show-current"),
        "test_status": _git_value(root, "status", "--short"),
    }
    try:
        import torchtitan

        package_root = Path(torchtitan.__file__).resolve().parents[1]
        metadata.update(
            {
                "torchtitan_root": str(package_root),
                "torchtitan_commit": _git_value(package_root, "rev-parse", "HEAD"),
                "torchtitan_status": _git_value(package_root, "status", "--short"),
            }
        )
    except (ImportError, OSError):
        metadata["torchtitan_root"] = None
    return metadata


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") is not None:
        return "npu"
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        return "cuda"
    try:
        import torch

        if getattr(torch, "npu", None) is not None and torch.npu.is_available():
            return "npu"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    raise RuntimeError(
        "cannot infer device; export ASCEND_RT_VISIBLE_DEVICES or "
        "CUDA_VISIBLE_DEVICES, or pass --device"
    )


def _visible_devices(device: str) -> list[str] | None:
    variable = (
        "ASCEND_RT_VISIBLE_DEVICES" if device == "npu" else "CUDA_VISIBLE_DEVICES"
    )
    value = os.environ.get(variable)
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _run_name(config: PerformanceConfig, device: str) -> str:
    return _slug(f"{config.name}-{device}-{config.topology}-{config.preset}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metrics_rank(world_size: int, pp: int, pp_schedule: str) -> int:
    if pp == 1 or pp_schedule == "ZBVZeroBubble":
        return 0
    return (world_size // pp) * (pp - 1)


def _training_command(
    *,
    root: Path,
    config: PerformanceConfig,
    device: str,
    run_directory: Path,
) -> list[str]:
    topology = standard_topologies()[config.topology]
    visible = _visible_devices(device)
    if visible is not None and len(visible) < topology.world_size:
        raise ValueError(
            f"topology {topology.name} needs {topology.world_size} devices, "
            f"but only {len(visible)} are visible"
        )
    if config.global_batch_size % topology.dp_replicate:
        raise ValueError(
            "global_batch_size must be divisible by data-parallel replicate degree"
        )

    torchrun = shutil.which("torchrun") or str(
        Path(sys.executable).with_name("torchrun")
    )
    metrics_rank = _metrics_rank(
        topology.world_size, topology.pp, topology.pp_schedule
    )
    profile_frequency = config.warmup_steps + config.active_steps
    command = [
        torchrun,
        "--nnodes=1",
        "--node_rank=0",
        f"--nproc_per_node={topology.world_size}",
        "--rdzv_backend=c10d",
        "--rdzv_endpoint=localhost:0",
        f"--rdzv_id={_run_name(config, device)}",
        f"--local-ranks-filter={metrics_rank}",
        "--role=rank",
        "--tee=3",
        "-m",
        "tests.glm5_2_performance.capture_metrics",
        "--module",
        config.module,
        "--config",
        config.model_config,
        f"--dump_folder={run_directory / 'trainer_output'}",
        f"--training.steps={config.steps}",
        f"--training.local_batch_size={config.local_batch_size}",
        f"--training.global_batch_size={config.global_batch_size}",
        f"--training.seq_len={config.sequence_length}",
        f"--training.dtype={config.training_dtype}",
        f"--training.mixed_precision_param={config.mixed_precision_param}",
        f"--training.mixed_precision_reduce={config.mixed_precision_reduce}",
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
    if config.deterministic:
        command.append("--debug.deterministic")
    return command


def _run_process(
    command: Sequence[str],
    *,
    root: Path,
    environment: dict[str, str],
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("Command: " + " ".join(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            list(command),
            cwd=root,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, list(command))


def _recover_partial_run(run_directory: Path) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    destination = run_directory.with_name(f"{run_directory.name}.failed-{timestamp}")
    suffix = 1
    while destination.exists():
        destination = run_directory.with_name(
            f"{run_directory.name}.failed-{timestamp}-{suffix}"
        )
        suffix += 1
    run_directory.rename(destination)
    return destination


def _remove_known_output(path: Path, parent: Path) -> None:
    resolved_path = path.resolve()
    resolved_parent = parent.resolve()
    if resolved_path.parent != resolved_parent:
        raise RuntimeError(f"refusing to remove unexpected output path: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def capture(
    root: Path,
    config: PerformanceConfig,
    *,
    device: str,
    preset: ProfilerPreset,
    force: bool,
) -> tuple[Path, Path]:
    run_name = _run_name(config, device)
    run_parent = root / config.run_root
    artifact_parent = root / config.artifact_root
    run_directory = run_parent / run_name
    artifact_directory = artifact_parent / run_name
    complete_manifest = artifact_directory / "manifest.json"

    run_manifest = run_directory / "manifest.json"
    if complete_manifest.is_file() and run_manifest.is_file() and not force:
        print(f"Skip completed performance capture: {artifact_directory}")
        return run_directory, artifact_directory
    if force:
        _remove_known_output(run_directory, run_parent)
        _remove_known_output(artifact_directory, artifact_parent)
    elif run_directory.exists():
        recovered = _recover_partial_run(run_directory)
        print(f"Archived incomplete run before retry: {recovered}")
    if artifact_directory.exists() and not force:
        recovered = _recover_partial_run(artifact_directory)
        print(f"Archived incomplete artifact before retry: {recovered}")

    run_directory.mkdir(parents=True, exist_ok=False)
    metrics_path = run_directory / "metrics.jsonl"
    runtime_log = run_directory / "runtime.log"
    environment = os.environ.copy()
    environment.update(config.environment)
    environment.update(preset.environment())
    environment["TORCHTITAN_DEVICE"] = "npu" if device == "npu" else "gpu"
    environment["GLM5_PERFORMANCE_METRICS_PATH"] = str(metrics_path)
    environment["GLM5_PERFORMANCE_PROFILE_RANKS"] = preset.profile_ranks
    topology = standard_topologies()[config.topology]
    environment["LOG_RANK"] = str(
        _metrics_rank(topology.world_size, topology.pp, topology.pp_schedule)
    )
    command = _training_command(
        root=root,
        config=config,
        device=device,
        run_directory=run_directory,
    )
    print(
        f"Starting profiler capture: device={device}, topology={config.topology}, "
        f"preset={config.preset}, output={run_directory}"
    )
    print(f"Runtime log: {runtime_log}")
    try:
        _run_process(
            command,
            root=root,
            environment=environment,
            log_path=runtime_log,
        )
    except Exception as error:
        _write_json(
            run_directory / "failure.json",
            {"error": repr(error), "command": command},
        )
        raise

    manifest = {
        "format_version": 1,
        "run_name": run_name,
        "device": device,
        "topology": config.topology,
        "preset": config.preset,
        "config": config.as_dict(),
        "profiler_environment": preset.environment(),
        "command": command,
        "source": _source_metadata(root),
        "run_directory": str(run_directory),
    }
    _write_json(run_directory / "manifest.json", manifest)
    artifact_directory.mkdir(parents=True, exist_ok=True)
    _write_json(artifact_directory / "manifest.json", manifest)
    if metrics_path.is_file():
        shutil.copy2(metrics_path, artifact_directory / "metrics.jsonl")
    print(f"Captured performance artifact: {artifact_directory}")
    return run_directory, artifact_directory


def _find_profiler_directory(run_directory: Path) -> Path:
    candidates = (
        run_directory / "trainer_output" / "profiling" / "traces",
        run_directory / "trainer_output" / "profiling",
    )
    return next((path for path in candidates if path.is_dir()), candidates[0])


def offline_parse(run_directory: Path) -> list[str]:
    profiler_directory = _find_profiler_directory(run_directory)
    raw_directories = sorted(
        path
        for path in profiler_directory.rglob("*_ascend_pt")
        if path.is_dir()
    )
    if not raw_directories:
        raise FileNotFoundError(
            f"no *_ascend_pt raw profiler directories under {profiler_directory}"
        )
    try:
        import torch_npu
    except ImportError as error:
        raise RuntimeError("offline parsing requires torch_npu") from error
    parsed = []
    for raw_directory in raw_directories:
        print(f"Parsing Ascend profiler data: {raw_directory}")
        torch_npu.profiler.profiler.analyse(profiler_path=str(raw_directory))
        parsed.append(str(raw_directory))
    return parsed


def run_advisor(run_directory: Path) -> dict[str, Any]:
    executable = shutil.which("msprof-analyze")
    if executable is None:
        raise FileNotFoundError(
            "msprof-analyze is not installed or is not on PATH"
        )
    profiler_directory = _find_profiler_directory(run_directory)
    output_directory = run_directory / "advisor"
    command = [
        executable,
        "advisor",
        "all",
        "-d",
        str(profiler_directory),
        "-o",
        str(output_directory),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    status = {
        "command": command,
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    _write_json(run_directory / "advisor.json", status)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    return status


def analyze(
    root: Path,
    config: PerformanceConfig,
    *,
    device: str,
    parse_offline: bool,
    advisor: bool,
) -> Path:
    run_name = _run_name(config, device)
    run_directory = root / config.run_root / run_name
    artifact_directory = root / config.artifact_root / run_name
    manifest_path = artifact_directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"capture artifact not found: {manifest_path}")
    if parse_offline:
        offline_parse(run_directory)
    if advisor:
        run_advisor(run_directory)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    analysis = build_analysis(run_directory)
    _write_json(artifact_directory / "analysis.json", analysis)
    report_path = root / config.report_root / f"{run_name}.html"
    render_html_report(
        manifest=manifest,
        analysis=analysis,
        output_path=report_path,
    )
    print(f"Performance report: {report_path}")
    return report_path


def run_profiler_cli(config: PerformanceConfig, script_path: str) -> None:
    topologies = standard_topologies()
    presets = profiler_presets()
    parser = argparse.ArgumentParser(
        description="Capture and visualize a bounded TorchTitan profiler window"
    )
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument(
        "--probe",
        action="store_true",
        help="capture and immediately generate the HTML report",
    )
    actions.add_argument("--capture", action="store_true")
    actions.add_argument("--analyze", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "npu"))
    parser.add_argument("--topology", choices=tuple(topologies))
    parser.add_argument("--preset", choices=tuple(presets))
    parser.add_argument("--visible-devices")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--skip-steps", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--active-steps", type=int)
    parser.add_argument("--local-batch-size", type=int)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--offline-parse", action="store_true")
    parser.add_argument("--advisor", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--extra-train-arg",
        action="append",
        default=[],
        help="append a raw TorchTitan CLI argument",
    )
    args = parser.parse_args()

    overrides = {
        "device": args.device,
        "topology": args.topology,
        "preset": args.preset,
        "steps": args.steps,
        "skip_steps": args.skip_steps,
        "warmup_steps": args.warmup_steps,
        "active_steps": args.active_steps,
        "local_batch_size": args.local_batch_size,
        "global_batch_size": args.global_batch_size,
        "sequence_length": args.sequence_length,
    }
    effective = replace(
        config,
        **{key: value for key, value in overrides.items() if value is not None},
        extra_args=config.extra_args + tuple(args.extra_train_arg),
    )
    root = _repository_root(script_path)
    device = _resolve_device(effective.device)
    if args.visible_devices:
        variable = (
            "ASCEND_RT_VISIBLE_DEVICES"
            if device == "npu"
            else "CUDA_VISIBLE_DEVICES"
        )
        os.environ[variable] = args.visible_devices
    preset = presets[effective.preset]
    if args.offline:
        preset = replace(preset, online_parse=False)

    if args.capture or args.probe:
        capture(
            root,
            effective,
            device=device,
            preset=preset,
            force=args.force,
        )
    if args.analyze or args.probe:
        analyze(
            root,
            effective,
            device=device,
            parse_offline=args.offline_parse,
            advisor=args.advisor,
        )
