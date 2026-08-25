"""Run, post-process, and report independent profiler experiments."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
import json
import os
import re
import shlex
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
    performance_topologies,
    profiler_presets,
)
from .documentation import card_scope, write_run_readme


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


def _device_snapshot(device: str) -> dict[str, Any]:
    executable = shutil.which("npu-smi" if device == "npu" else "nvidia-smi")
    if executable is None:
        return {"available": False}
    command = [executable, "info"] if device == "npu" else [executable]
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": True, "error": repr(error), "command": command}
    return {
        "available": True,
        "command": command,
        "return_code": result.returncode,
        "output": result.stdout,
    }


def _profiling_preflight(
    output_directory: Path, device: str, *, profiler_enabled: bool = True
) -> dict[str, Any]:
    """Record official environment and storage recommendations."""

    warnings = []
    effective_umask = None
    if os.name == "posix":
        effective_umask = os.umask(0)
        os.umask(effective_umask)
        if profiler_enabled and effective_umask & 0o027 != 0o027:
            warnings.append(
                f"umask {effective_umask:04o} is weaker than the recommended 0027"
            )
        if profiler_enabled and hasattr(os, "geteuid") and os.geteuid() == 0:
            warnings.append("profiling as root is not recommended")
    output_parent = output_directory.parent
    storage = shutil.disk_usage(output_parent)
    if output_parent.is_symlink():
        warnings.append("profiler output parent is a symbolic link")
    if profiler_enabled and device == "npu" and not any(
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
        "device_snapshot_before_capture": _device_snapshot(device),
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
    if config.mixed_precision_reduce != "float32":
        reduce_precision = {
            "bfloat16": "bf16",
            "float16": "fp16",
        }.get(config.mixed_precision_reduce, config.mixed_precision_reduce)
        base += f"-reduce-{reduce_precision}"
    if not config.profiler_enabled:
        base += "-profiler-off"
    if config.replicate:
        base += f"-r{config.replicate}"
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
    legacy_parent: Path | None = None,
) -> Path:
    destination = parent / destination_name
    legacy_name = _slug(f"{config.name}-{device}-{config.topology}-{config.preset}")
    candidates = [parent / legacy_name]
    if legacy_parent is not None and legacy_parent != parent:
        candidates = [
            legacy_parent / destination_name,
            legacy_parent / legacy_name,
            *candidates,
        ]
    legacy = next((path for path in candidates if path.is_dir()), None)
    if destination.exists() or legacy is None:
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
            return value.replace(str(legacy), str(destination)).replace(
                legacy.name, destination_name
            )
        return value

    manifest = rename_value(manifest)
    manifest["run_name"] = destination_name
    _write_json(destination / "manifest.json", manifest)
    print(f"Adopted matching legacy output:\n  {legacy}\n  -> {destination}")
    return destination


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _scoped_parent(root: Path, configured_root: str, topology: str) -> Path:
    return root / configured_root / card_scope(topology) / topology


def _exploration_directory(root: Path, topology: str, run_name: str) -> Path:
    return (
        root
        / "tests"
        / "glm5_2_performance"
        / "explorations"
        / "runs"
        / card_scope(topology)
        / topology
        / run_name
    )


def _refresh_exploration_documents(root: Path) -> None:
    """Refresh topology/run indexes after an exploration changes state."""

    from .explorations.tools.organize import generate_documents

    generate_documents(root)


def _record_exploration_command(
    root: Path,
    *,
    run_name: str,
    action: str,
    device: str,
    topology: str,
) -> Path:
    """Append the exact driver invocation and relevant environment as JSONL."""

    directory = _exploration_directory(root, topology, run_name)
    directory.mkdir(parents=True, exist_ok=True)
    visible_variable = (
        "ASCEND_RT_VISIBLE_DEVICES" if device == "npu" else "CUDA_VISIBLE_DEVICES"
    )
    record = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "action": action,
        "cwd": str(root),
        "argv": [sys.executable, *sys.argv],
        "shell_command": shlex.join([sys.executable, *sys.argv]),
        "device": device,
        "topology": topology,
        "environment": {
            name: os.environ[name]
            for name in (
                visible_variable,
                "TORCHTITAN_MSPROF_ANALYZE",
                "PATH",
                "PYTHONPATH",
                "ASCEND_HOME_PATH",
                "ASCEND_OPP_PATH",
                "ASCEND_TOOLKIT_HOME",
                "CONDA_DEFAULT_ENV",
                "HCCL_NPU_SOCKET_PORT_RANGE",
                "HCCL_IF_BASE_PORT",
                "HCCL_CONNECT_TIMEOUT",
                "TORCHTITAN_MSPROF_ANALYZE_WORKERS",
            )
            if name in os.environ
        },
        "source": _source_metadata(root),
    }
    history = directory / "command_history.jsonl"
    with history.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    write_run_readme(directory)
    return directory


def _sync_exploration_bundle(
    root: Path,
    *,
    topology: str,
    run_name: str,
    run_directory: Path,
    artifact_directory: Path,
    report_path: Path | None = None,
) -> Path:
    """Keep compact evidence beside the experiment's reproducibility log."""

    directory = _exploration_directory(root, topology, run_name)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "analysis.json", "metrics.jsonl"):
        source = artifact_directory / name
        if source.is_file():
            shutil.copy2(source, directory / name)
    command_directory = directory / "tool_commands"
    for status_path in sorted(run_directory.glob("*.json")):
        if status_path.name == "manifest.json":
            continue
        command_directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(status_path, command_directory / status_path.name)
    index = {
        "run_name": run_name,
        "raw_run_directory": str(run_directory),
        "artifact_directory": str(artifact_directory),
        "report_path": str(report_path) if report_path is not None else None,
        "mindstudio_input": str(_find_profiler_directory(run_directory)),
    }
    _write_json(directory / "artifacts.json", index)
    write_run_readme(directory)
    _refresh_exploration_documents(root)
    return directory


def _config_is_compatible(
    saved: dict[str, Any], current: PerformanceConfig
) -> bool:
    """Match manifests written before additive config fields existed."""

    current_values = json.loads(json.dumps(current.as_dict()))
    defaults = json.loads(json.dumps(PerformanceConfig(name=current.name).as_dict()))
    for key, value in current_values.items():
        if key in saved:
            if saved[key] != value:
                return False
        elif defaults.get(key) != value:
            return False
    return not any(key not in current_values for key in saved)


def _find_matching_capture(
    artifact_parent: Path,
    *,
    config: PerformanceConfig,
    device: str,
    profiler_environment: dict[str, str],
) -> tuple[Path, dict[str, Any]] | None:
    matches = []
    if not artifact_parent.is_dir():
        return None
    for manifest_path in artifact_parent.rglob("manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            continue
        saved_config = manifest.get("config")
        if (
            manifest.get("device") == device
            and manifest.get("topology") == config.topology
            and manifest.get("preset") == config.preset
            and manifest.get("profiler_environment") == profiler_environment
            and isinstance(saved_config, dict)
            and _config_is_compatible(saved_config, config)
        ):
            matches.append((manifest_path.parent, manifest))
    if len(matches) > 1:
        names = ", ".join(path.name for path, _ in matches)
        raise RuntimeError(f"multiple compatible captures found: {names}")
    return matches[0] if matches else None


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
    topology = performance_topologies()[config.topology]
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
    profiler_arguments = (
        [
            "--profiler.enable_profiling",
            "--profiler.save_traces_folder=profiling/traces",
            f"--profiler.profile_freq={profile_frequency}",
            "--profiler.profiler_repeat=1",
            f"--profiler.profiler_skip_first={config.skip_steps}",
            f"--profiler.profiler_warmup={config.warmup_steps}",
            f"--profiler.profiler_active={config.active_steps}",
        ]
        if config.profiler_enabled
        else []
    )
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
        *profiler_arguments,
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
    if (
        config.profiler_enabled
        and device == "npu"
        and os.environ.get("PROF_CONFIG_PATH")
    ):
        raise ValueError(
            "PROF_CONFIG_PATH enables dynamic_profile and cannot be combined "
            "with this scheduled profiler capture"
        )
    run_name = _run_name(config, device, preset)
    run_parent = _scoped_parent(root, config.run_root, config.topology)
    artifact_parent = _scoped_parent(root, config.artifact_root, config.topology)
    run_directory = _adopt_legacy_performance_output(
        run_parent,
        config=config,
        device=device,
        destination_name=run_name,
        legacy_parent=root / config.run_root,
    )
    artifact_directory = _adopt_legacy_performance_output(
        artifact_parent,
        config=config,
        device=device,
        destination_name=run_name,
        legacy_parent=root / config.artifact_root,
    )
    complete_manifest = artifact_directory / "manifest.json"

    run_manifest = run_directory / "manifest.json"
    profiler_environment = preset.environment() if config.profiler_enabled else {}
    if complete_manifest.is_file() and run_manifest.is_file() and not force:
        existing = json.loads(complete_manifest.read_text(encoding="utf-8"))
        if (
            not _config_is_compatible(existing.get("config", {}), config)
            or existing.get("profiler_environment") != profiler_environment
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
    environment.update(profiler_environment)
    environment["TORCHTITAN_DEVICE"] = "npu" if device == "npu" else "gpu"
    environment["GLM5_PERFORMANCE_METRICS_PATH"] = str(metrics_path)
    if config.profiler_enabled:
        environment["GLM5_PERFORMANCE_PROFILE_RANKS"] = preset.profile_ranks
    if config.mixed_precision_reduce != "float32":
        environment["GLM5_PERFORMANCE_REDUCE_DTYPE_OVERRIDE"] = (
            config.mixed_precision_reduce
        )
    topology = performance_topologies()[config.topology]
    environment["LOG_RANK"] = str(
        _metrics_rank(topology.world_size, topology.pp, topology.pp_schedule)
    )
    preflight = _profiling_preflight(
        run_directory, device, profiler_enabled=config.profiler_enabled
    )
    command = _training_command(
        root=root,
        config=config,
        device=device,
        preset=preset,
        run_directory=run_directory,
    )
    capture_kind = "profiler" if config.profiler_enabled else "performance"
    print(
        f"Starting {capture_kind} capture: device={device}, topology={config.topology}, "
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
        failure = {"error": repr(error), "command": command}
        failure_path = run_directory / "failure.json"
        _write_json(failure_path, failure)
        exploration_directory = _exploration_directory(
            root, config.topology, run_name
        )
        exploration_directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(failure_path, exploration_directory / "failure.json")
        if runtime_log.is_file():
            shutil.copy2(
                runtime_log, exploration_directory / "failed_runtime.log"
            )
        write_run_readme(exploration_directory)
        _refresh_exploration_documents(root)
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
            "device_snapshot_after_capture": _device_snapshot(device),
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
        "visible_devices": _visible_devices(device),
        "topology": config.topology,
        "preset": config.preset,
        "config": config.as_dict(),
        "profiler_environment": profiler_environment,
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
    _sync_exploration_bundle(
        root,
        topology=config.topology,
        run_name=run_name,
        run_directory=run_directory,
        artifact_directory=artifact_directory,
    )
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
    override = os.environ.get("TORCHTITAN_MSPROF_ANALYZE")
    if override:
        executable = str(Path(override).expanduser().resolve())
        if not Path(executable).is_file() or not os.access(executable, os.X_OK):
            raise FileNotFoundError(
                "TORCHTITAN_MSPROF_ANALYZE must name an executable file: "
                f"{executable}"
            )
        return executable
    executable = shutil.which("msprof-analyze")
    if executable is None:
        raise FileNotFoundError(
            "msprof-analyze is not installed or is not on PATH"
        )
    return executable


def _msprof_analyze_workers() -> int:
    value = os.environ.get("TORCHTITAN_MSPROF_ANALYZE_WORKERS", "1")
    try:
        workers = int(value)
    except ValueError as error:
        raise ValueError(
            "TORCHTITAN_MSPROF_ANALYZE_WORKERS must be an integer"
        ) from error
    if not 1 <= workers <= 8:
        raise ValueError(
            "TORCHTITAN_MSPROF_ANALYZE_WORKERS must be between 1 and 8"
        )
    return workers


def run_cluster_analysis(run_directory: Path) -> dict[str, Any]:
    """Run the official cluster, time, bottleneck, and idle recipes."""

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
    results = {"cluster": _run_msprof_analyze(
        run_directory, name="cluster", command=command
    )}
    executable = _msprof_analyze_executable()
    recipes = {
        "cluster_time_summary": [
            executable,
            "-m",
            "cluster_time_summary",
            "-d",
            str(profiler_directory),
            "-o",
            str(run_directory / "cluster_time_summary"),
            "--export_type",
            "text",
        ],
        "free_analysis": [
            executable,
            "-m",
            "free_analysis",
            "-d",
            str(profiler_directory),
            "-o",
            str(run_directory / "free_analysis"),
            "--top_num",
            "20",
            "--export_type",
            "text",
        ],
    }
    rank_ids = sorted(
        {
            match.group(1)
            for path in profiler_directory.iterdir()
            if (match := re.match(r"rank_(\d+)_", path.name))
        },
        key=int,
    )
    communication_recipes = {}
    for rank_id in rank_ids:
        communication_recipes[f"communication_bottleneck_rank_{rank_id}"] = [
            executable,
            "-m",
            "communication_bottleneck",
            "-d",
            str(profiler_directory),
            "-o",
            str(run_directory / f"communication_bottleneck_rank_{rank_id}"),
            "--rank_id",
            rank_id,
            "--top_num",
            "20",
            "--export_type",
            "text",
        ]
    for name, recipe in recipes.items():
        results[name] = _run_msprof_analyze(
            run_directory, name=name, command=recipe
        )
    workers = _msprof_analyze_workers()
    if workers == 1:
        for name, recipe in communication_recipes.items():
            results[name] = _run_msprof_analyze(
                run_directory, name=name, command=recipe
            )
    else:
        print(
            "Running communication_bottleneck recipes with "
            f"{workers} workers"
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                name: executor.submit(
                    _run_msprof_analyze,
                    run_directory,
                    name=name,
                    command=recipe,
                )
                for name, recipe in communication_recipes.items()
            }
            for name, future in futures.items():
                results[name] = future.result()
    return results


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
    run_parent = _scoped_parent(root, config.run_root, config.topology)
    artifact_parent = _scoped_parent(root, config.artifact_root, config.topology)
    run_directory = _adopt_legacy_performance_output(
        run_parent,
        config=config,
        device=device,
        destination_name=run_name,
        legacy_parent=root / config.run_root,
    )
    artifact_directory = _adopt_legacy_performance_output(
        artifact_parent,
        config=config,
        device=device,
        destination_name=run_name,
        legacy_parent=root / config.artifact_root,
    )
    manifest_path = artifact_directory / "manifest.json"
    if not manifest_path.is_file():
        match = _find_matching_capture(
            artifact_parent,
            config=config,
            device=device,
            profiler_environment=(
                preset.environment() if config.profiler_enabled else {}
            ),
        )
        if match is None:
            raise FileNotFoundError(f"capture artifact not found: {manifest_path}")
        artifact_directory, manifest = match
        run_name = str(manifest["run_name"])
        run_directory = run_parent / run_name
        manifest_path = artifact_directory / "manifest.json"
        print(f"Using compatible capture manifest: {manifest_path}")
    if parse_offline and config.profiler_enabled:
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
    report_path = (
        _scoped_parent(root, config.report_root, config.topology)
        / f"{run_name}.html"
    )
    render_html_report(
        manifest=manifest,
        analysis=analysis,
        output_path=report_path,
    )
    _sync_exploration_bundle(
        root,
        topology=config.topology,
        run_name=run_name,
        run_directory=run_directory,
        artifact_directory=artifact_directory,
        report_path=report_path,
    )
    print(f"Performance report: {report_path}")
    return report_path


def run_profiler_cli(
    config: PerformanceConfig,
    script_path: str,
) -> None:
    topologies = performance_topologies()
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
    topology_selection = parser.add_mutually_exclusive_group()
    topology_selection.add_argument(
        "--topology", choices=("all", *suite_topologies)
    )
    topology_selection.add_argument(
        "--topologies", help="comma-separated topology subset or all"
    )
    parser.add_argument("--preset", choices=tuple(presets))
    parser.add_argument("--visible-devices")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--skip-steps", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--active-steps", type=int)
    parser.add_argument(
        "--profiler-off",
        action="store_true",
        help="disable collection for authoritative steady-state throughput",
    )
    parser.add_argument("--local-batch-size", type=int)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument(
        "--replicate",
        type=int,
        help="repeat index included in run identity without changing training",
    )
    parser.add_argument("--training-dtype")
    parser.add_argument("--mixed-precision-param")
    parser.add_argument("--mixed-precision-reduce")
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
        "profiler_enabled": False if args.profiler_off else None,
        "local_batch_size": args.local_batch_size,
        "global_batch_size": args.global_batch_size,
        "sequence_length": args.sequence_length,
        "replicate": args.replicate,
        "training_dtype": args.training_dtype,
        "mixed_precision_param": args.mixed_precision_param,
        "mixed_precision_reduce": args.mixed_precision_reduce,
        "run_root": args.run_root,
    }
    effective = replace(
        config,
        **{key: value for key, value in overrides.items() if value is not None},
        extra_args=config.extra_args + tuple(args.extra_train_arg),
    )
    root = _repository_root(script_path)
    device = _resolve_device(effective.device)
    if device != "npu":
        raise NotImplementedError(
            "the performance profiler currently supports only Ascend NPU; "
            "the CUDA interface is reserved until the torch.profiler workflow "
            "and report contract are defined"
        )
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
    if args.force and (args.capture or args.probe):
        # Clear every selected member before launching the first capture. A
        # mid-suite failure can then be resumed without mixing generations.
        for topology_name in selected:
            topology_config = replace(effective, topology=topology_name)
            run_name = _run_name(topology_config, device, preset)
            run_parent = _scoped_parent(
                root, topology_config.run_root, topology_config.topology
            )
            artifact_parent = _scoped_parent(
                root, topology_config.artifact_root, topology_config.topology
            )
            report_parent = _scoped_parent(
                root, topology_config.report_root, topology_config.topology
            )
            _remove_known_output(run_parent / run_name, run_parent)
            _remove_known_output(artifact_parent / run_name, artifact_parent)
            _remove_known_output(report_parent / f"{run_name}.html", report_parent)
    reports: list[tuple[str, Path]] = []
    for topology_name in selected:
        topology_config = replace(effective, topology=topology_name)
        run_name = _run_name(topology_config, device, preset)
        action = "probe" if args.probe else "capture" if args.capture else "analyze"
        if args.capture or args.probe:
            _record_exploration_command(
                root,
                run_name=run_name,
                action=action,
                device=device,
                topology=topology_name,
            )
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
                force=False,
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
            if args.analyze:
                _record_exploration_command(
                    root,
                    run_name=report.stem,
                    action=action,
                    device=device,
                    topology=topology_name,
                )
            reports.append((topology_name, report))
    if len(reports) > 1:
        index = (
            root
            / effective.report_root
            / "suites"
            / f"{_slug(effective.name)}-{device}-suite.html"
        )
        index.parent.mkdir(parents=True, exist_ok=True)
        links = "".join(
            f'<li><a href="{Path(os.path.relpath(path, index.parent)).as_posix()}">'
            f"{name}</a></li>"
            for name, path in reports
        )
        index.write_text(
            "<!doctype html><html><head><meta charset=\"utf-8\">"
            "<title>GLM-5.2 performance topology suite</title></head>"
            "<body><h1>GLM-5.2 performance topology suite</h1>"
            f"<ul>{links}</ul></body></html>",
            encoding="utf-8",
        )
        print(f"Performance suite report: {index}")
