"""Run, post-process, and report independent profiler experiments."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from tests.glm5_2_common.device import resolve_device_type
from tests.glm5_2_common.naming import config_name, slug
from tests.glm5_2_common.topology import select_topologies, training_command_args

from .analysis import build_analysis, render_html_report
from .config import (
    PerformanceConfig,
    ProfilerPreset,
    profiler_presets,
    standard_topologies,
)


def _slug(value: str) -> str:
    return slug(value)


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
    return resolve_device_type(requested)


def _visible_devices(device: str) -> list[str] | None:
    variable = (
        "ASCEND_RT_VISIBLE_DEVICES" if device == "npu" else "CUDA_VISIBLE_DEVICES"
    )
    value = os.environ.get(variable)
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _profiling_preflight(output_directory: Path, device: str) -> dict[str, Any]:
    """Record official environment and storage recommendations."""

    warnings = []
    effective_umask = None
    if os.name == "posix":
        effective_umask = os.umask(0)
        os.umask(effective_umask)
        if effective_umask & 0o027 != 0o027:
            warnings.append(
                f"umask {effective_umask:04o} is weaker than the recommended 0027"
            )
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            warnings.append("profiling as root is not recommended")
    output_parent = output_directory.parent
    storage = shutil.disk_usage(output_parent)
    if output_parent.is_symlink():
        warnings.append("profiler output parent is a symbolic link")
    if device == "npu" and not any(
        os.environ.get(name)
        for name in ("ASCEND_HOME_PATH", "ASCEND_OPP_PATH", "ASCEND_TOOLKIT_HOME")
    ):
        warnings.append("no standard CANN environment path variable was detected")
    result = {
        "effective_umask": (
            f"{effective_umask:04o}" if effective_umask is not None else None
        ),
        "running_as_root": bool(
            os.name == "posix"
            and hasattr(os, "geteuid")
            and os.geteuid() == 0
        ),
        "output_parent": str(output_parent.resolve()),
        "output_parent_is_symlink": output_parent.is_symlink(),
        "storage_free_bytes_before_capture": storage.free,
        "storage_total_bytes": storage.total,
        "warnings": warnings,
    }
    for warning in warnings:
        print(f"Profiler preflight warning: {warning}")
    return result


def _run_name(
    config: PerformanceConfig,
    device: str,
    preset: ProfilerPreset | None = None,
) -> str:
    default_preset = profiler_presets()[config.preset]
    effective_preset = preset or default_preset
    preset_overridden = effective_preset.environment() != default_preset.environment()
    precision = {
        ("float32", "bfloat16"): "bf16",
        ("float32", "float32"): "fp32",
        ("bfloat16", "bfloat16"): "full-bf16",
    }.get(
        (config.training_dtype, config.mixed_precision_param),
        f"{config.training_dtype}-{config.mixed_precision_param}",
    )
    base = (
        f"{device}-{config.topology}-{precision}-s{config.steps}-"
        f"l{config.local_batch_size}-b{config.global_batch_size}-"
        f"seq{config.sequence_length}-seed{config.seed}-{config.preset}"
    )
    if preset_overridden:
        base += f"-{effective_preset.parse_mode}"
    identity = config.as_dict()
    for key in ("name", "device", "run_root", "artifact_root", "report_root"):
        identity.pop(key, None)
    identity["resolved_device"] = device
    if preset_overridden:
        identity["profiler_environment"] = effective_preset.environment()
    return config_name(base, identity)


def _adopt_legacy_performance_output(
    parent: Path,
    *,
    config: PerformanceConfig,
    device: str,
    destination_name: str,
) -> Path:
    destination = parent / destination_name
    legacy = parent / _slug(
        f"{config.name}-{device}-{config.topology}-{config.preset}"
    )
    if destination.exists() or not legacy.is_dir():
        return destination
    manifest_path = legacy / "manifest.json"
    if not manifest_path.is_file():
        return destination
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return destination
    if manifest.get("config") != config.as_dict():
        return destination
    parent.mkdir(parents=True, exist_ok=True)
    legacy.rename(destination)
    def rename_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: rename_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rename_value(item) for item in value]
        if isinstance(value, str):
            return value.replace(legacy.name, destination_name)
        return value

    manifest = rename_value(manifest)
    manifest["run_name"] = destination_name
    _write_json(destination / "manifest.json", manifest)
    print(f"Adopted matching legacy output:\n  {legacy}\n  -> {destination}")
    return destination


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
    preset: ProfilerPreset,
    run_directory: Path,
) -> list[str]:
    topology = standard_topologies()[config.topology]
    visible = _visible_devices(device)
    if visible is not None and len(visible) < topology.world_size:
        raise ValueError(
            f"topology {topology.name} needs {topology.world_size} devices, "
            f"but only {len(visible)} are visible"
        )
    data_parallel_batch = config.local_batch_size * topology.data_parallel_degree
    if config.global_batch_size % data_parallel_batch:
        raise ValueError(
            "global_batch_size must be divisible by local_batch_size * "
            "data_parallel_degree"
        )
    if config.local_batch_size % topology.pipeline_parallel_microbatch_size:
        raise ValueError(
            "local_batch_size must be divisible by pipeline microbatch size"
        )
    pipeline_microbatches = (
        config.local_batch_size // topology.pipeline_parallel_microbatch_size
    )
    if (
        topology.pipeline_parallel_degree > 1
        and pipeline_microbatches < topology.pipeline_parallel_degree
    ):
        raise ValueError(
            "local batch does not provide enough pipeline microbatches"
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
        f"--rdzv_id={_run_name(config, device, preset)}",
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
        *training_command_args(
            local_batch_size=config.local_batch_size,
            global_batch_size=config.global_batch_size,
            sequence_length=config.sequence_length,
            topology=topology,
        ),
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
    if device == "npu" and os.environ.get("PROF_CONFIG_PATH"):
        raise ValueError(
            "PROF_CONFIG_PATH enables dynamic_profile and cannot be combined "
            "with this scheduled profiler capture"
        )
    run_name = _run_name(config, device, preset)
    run_parent = root / config.run_root
    artifact_parent = root / config.artifact_root
    run_directory = _adopt_legacy_performance_output(
        run_parent,
        config=config,
        device=device,
        destination_name=run_name,
    )
    artifact_directory = _adopt_legacy_performance_output(
        artifact_parent,
        config=config,
        device=device,
        destination_name=run_name,
    )
    complete_manifest = artifact_directory / "manifest.json"

    run_manifest = run_directory / "manifest.json"
    if complete_manifest.is_file() and run_manifest.is_file() and not force:
        existing = json.loads(complete_manifest.read_text(encoding="utf-8"))
        if (
            existing.get("config") != config.as_dict()
            or existing.get("profiler_environment") != preset.environment()
        ):
            raise FileExistsError(
                f"completed capture uses different settings: {artifact_directory}; "
                "pass --force or choose a distinct script name"
            )
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
    preflight = _profiling_preflight(run_directory, device)
    command = _training_command(
        root=root,
        config=config,
        device=device,
        preset=preset,
        run_directory=run_directory,
    )
    print(
        f"Starting profiler capture: device={device}, topology={config.topology}, "
        f"preset={config.preset}, output={run_directory}"
    )
    print(f"Runtime log: {runtime_log}")
    capture_started = time.monotonic()
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

    storage_after = shutil.disk_usage(run_directory.parent)
    output_bytes = sum(
        path.stat().st_size
        for path in run_directory.rglob("*")
        if path.is_file()
    )
    preflight.update(
        {
            "capture_duration_seconds": time.monotonic() - capture_started,
            "storage_free_bytes_after_capture": storage_after.free,
            "run_output_bytes": output_bytes,
            "recommended_free_bytes_for_same_capture": output_bytes * 20,
        }
    )
    if preflight["storage_free_bytes_before_capture"] < output_bytes * 20:
        warning = "free storage was below 20 times the captured run size"
        preflight["warnings"].append(warning)
        print(f"Profiler preflight warning: {warning}")

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
        "preflight": preflight,
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


def offline_parse(
    run_directory: Path,
    *,
    max_process_number: int | None = None,
) -> list[str]:
    if max_process_number is not None and max_process_number < 1:
        raise ValueError("offline parse workers must be positive")
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
    print(
        f"Parsing {len(raw_directories)} Ascend profile root(s) under "
        f"{profiler_directory}"
    )
    arguments: dict[str, Any] = {"profiler_path": str(profiler_directory)}
    if max_process_number is not None:
        arguments["max_process_number"] = max_process_number
    torch_npu.profiler.profiler.analyse(**arguments)
    return [str(path) for path in raw_directories]


def run_advisor(run_directory: Path) -> dict[str, Any]:
    profiler_directory = _find_profiler_directory(run_directory)
    output_directory = run_directory / "advisor"
    command = [
        _msprof_analyze_executable(),
        "advisor",
        "all",
        "-d",
        str(profiler_directory),
        "-o",
        str(output_directory),
    ]
    return _run_msprof_analyze(
        run_directory, name="advisor", command=command
    )


def _run_msprof_analyze(
    run_directory: Path,
    *,
    name: str,
    command: list[str],
) -> dict[str, Any]:
    print(f"Running msprof-analyze {name}: {' '.join(command)}")
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    status = {
        "command": command,
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    _write_json(run_directory / f"{name}.json", status)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    return status


def _msprof_analyze_executable() -> str:
    executable = shutil.which("msprof-analyze")
    if executable is None:
        raise FileNotFoundError(
            "msprof-analyze is not installed or is not on PATH"
        )
    return executable


def run_cluster_analysis(run_directory: Path) -> dict[str, Any]:
    """Build the official multi-rank communication analysis deliverable."""

    profiler_directory = _find_profiler_directory(run_directory)
    output_directory = run_directory / "cluster"
    command = [
        _msprof_analyze_executable(),
        "cluster",
        "-m",
        "all",
        "-d",
        str(profiler_directory),
        "-o",
        str(output_directory),
    ]
    return _run_msprof_analyze(
        run_directory, name="cluster", command=command
    )


def _comparison_input(run_directory: Path) -> Path:
    profiler_directory = _find_profiler_directory(run_directory)
    ascend_roots = sorted(
        path
        for path in profiler_directory.rglob("*_ascend_pt")
        if path.is_dir()
    )
    return ascend_roots[0] if len(ascend_roots) == 1 else profiler_directory


def run_performance_compare(
    run_directory: Path,
    baseline: Path,
) -> dict[str, Any]:
    """Compare one profile against an NPU or GPU baseline profile."""

    if not baseline.exists():
        raise FileNotFoundError(f"performance baseline not found: {baseline}")
    output_directory = run_directory / "compare"
    command = [
        _msprof_analyze_executable(),
        "compare",
        "-d",
        str(_comparison_input(run_directory)),
        "-bp",
        str(baseline.resolve()),
        "-o",
        str(output_directory),
    ]
    return _run_msprof_analyze(
        run_directory, name="compare", command=command
    )


def analyze(
    root: Path,
    config: PerformanceConfig,
    *,
    device: str,
    preset: ProfilerPreset,
    parse_offline: bool,
    parse_workers: int | None,
    advisor: bool,
    cluster: bool,
    compare_baseline: Path | None,
) -> Path:
    run_name = _run_name(config, device, preset)
    run_directory = _adopt_legacy_performance_output(
        root / config.run_root,
        config=config,
        device=device,
        destination_name=run_name,
    )
    artifact_directory = _adopt_legacy_performance_output(
        root / config.artifact_root,
        config=config,
        device=device,
        destination_name=run_name,
    )
    manifest_path = artifact_directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"capture artifact not found: {manifest_path}")
    if parse_offline:
        offline_parse(run_directory, max_process_number=parse_workers)
    if advisor:
        run_advisor(run_directory)
    if cluster:
        run_cluster_analysis(run_directory)
    if compare_baseline is not None:
        run_performance_compare(run_directory, compare_baseline)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    analysis = build_analysis(
        run_directory,
        config=manifest.get("config"),
        profiler_environment=manifest.get("profiler_environment"),
    )
    _write_json(artifact_directory / "analysis.json", analysis)
    report_path = root / config.report_root / f"{run_name}.html"
    render_html_report(
        manifest=manifest,
        analysis=analysis,
        output_path=report_path,
    )
    print(f"Performance report: {report_path}")
    return report_path


def run_profiler_cli(
    config: PerformanceConfig,
    script_path: str,
) -> None:
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
    suite_topologies = tuple(
        name for name, topology in topologies.items() if topology.world_size <= 8
    )
    parser.add_argument("--topology", choices=("all", *suite_topologies))
    parser.add_argument("--topologies", help="comma-separated topology subset or all")
    parser.add_argument("--preset", choices=tuple(presets))
    parser.add_argument("--visible-devices")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--skip-steps", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--active-steps", type=int)
    parser.add_argument("--local-batch-size", type=int)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument(
        "--run-root",
        help="raw output root; use node-local storage for large cluster captures",
    )
    parser.add_argument(
        "--parse-mode",
        choices=("sync", "async", "offline"),
        help=(
            "Ascend parsing mode: synchronous, asynchronous, or after "
            "capture"
        ),
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--offline-parse", action="store_true")
    parser.add_argument("--parse-workers", type=int)
    parser.add_argument("--advisor", action="store_true")
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument(
        "--compare-baseline",
        type=Path,
        help="GPU trace or NPU profile path for msprof-analyze compare",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--extra-train-arg",
        action="append",
        default=[],
        help="append a raw TorchTitan CLI argument",
    )
    args = parser.parse_args()

    requested_topology = None if args.topology == "all" else args.topology
    overrides = {
        "device": args.device,
        "topology": requested_topology,
        "preset": args.preset,
        "steps": args.steps,
        "skip_steps": args.skip_steps,
        "warmup_steps": args.warmup_steps,
        "active_steps": args.active_steps,
        "local_batch_size": args.local_batch_size,
        "global_batch_size": args.global_batch_size,
        "sequence_length": args.sequence_length,
        "run_root": args.run_root,
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
    if args.offline and args.parse_mode not in {None, "offline"}:
        parser.error("--offline conflicts with a non-offline --parse-mode")
    if args.offline:
        preset = replace(preset, parse_mode="offline")
    elif args.parse_mode:
        preset = replace(preset, parse_mode=args.parse_mode)

    selected = select_topologies(
        available=suite_topologies,
        topology="all" if args.topology == "all" else requested_topology,
        topologies=args.topologies,
        default=(effective.topology,),
    )
    reports: list[tuple[str, Path]] = []
    for topology_name in selected:
        topology_config = replace(effective, topology=topology_name)
        print(
            f"Starting performance topology: {topology_name} "
            f"({topologies[topology_name].world_size} ranks)",
            flush=True,
        )
        if args.capture or args.probe:
            capture(
                root,
                topology_config,
                device=device,
                preset=preset,
                force=args.force,
            )
        if args.analyze or args.probe:
            report = analyze(
                root,
                topology_config,
                device=device,
                preset=preset,
                parse_offline=(
                    args.offline_parse
                    or (
                        args.probe
                        and device == "npu"
                        and preset.parse_mode == "offline"
                    )
                ),
                parse_workers=args.parse_workers,
                advisor=args.advisor,
                cluster=args.cluster,
                compare_baseline=args.compare_baseline,
            )
            reports.append((topology_name, report))
    if len(reports) > 1:
        links = "".join(
            f'<li><a href="{path.name}">{name}</a></li>' for name, path in reports
        )
        index = root / effective.report_root / f"{_slug(effective.name)}-{device}-suite.html"
        index.write_text(
            "<!doctype html><html><head><meta charset=\"utf-8\">"
            "<title>GLM-5.2 performance topology suite</title></head>"
            "<body><h1>GLM-5.2 performance topology suite</h1>"
            f"<ul>{links}</ul></body></html>",
            encoding="utf-8",
        )
        print(f"Performance suite report: {index}")
