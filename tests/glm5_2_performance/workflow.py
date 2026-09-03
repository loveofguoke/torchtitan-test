"""Run, post-process, and report independent profiler experiments.

The pipeline is intentionally staged:

``capture -> optional offline parse -> msprof-analyze advisor/cluster/compare
-> portable analysis.json -> flamegraph/memory/graph links -> HTML report``.

Training output, raw profiler data, compact artifacts, and reports live in
separate roots. Profiler-active traces diagnose causes; repeated profiler-off
runs are the authoritative step-time and throughput baseline.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

from tests.glm5_2_common.cli import (
    RunAttempt,
    assert_run_not_active,
    reset_output_generation,
)
from tests.glm5_2_common.device import resolve_device_type
from tests.glm5_2_common.naming import config_name, slug
from tests.glm5_2_common.topology import (
    ParallelTopology,
    select_topologies,
    training_command_args,
)

from .advanced_analysis import (
    MUTATING_RECIPES,
    recipe_arguments,
    resolve_recipe_plan,
)
from .analysis import build_analysis, render_html_report
from .collectors import (
    PerformanceCollector,
    require_training_collector,
    resolve_msprof_executable,
)
from .config import (
    PerformanceConfig,
    ProfilerPreset,
    all_profiler_presets,
    performance_topologies,
    profiler_presets,
)
from .documentation import card_scope, write_run_readme
from .visualization import (
    find_flamegraph_script,
    find_mindstudio_flamegraph_script,
    mindstudio_insight_handoff,
    render_flamegraphs,
    render_mindstudio_flamegraphs,
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


def _file_content_identity(path: Path) -> dict[str, Any]:
    """Return a portable identity for one executable or adapter file."""

    resolved = path.resolve()
    status = resolved.stat()
    return {
        "name": resolved.name,
        "size": status.st_size,
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def _package_file_identity(
    package: str,
    relative_path: str,
) -> dict[str, Any]:
    """Hash one source-installed package file without importing the package."""

    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ModuleNotFoundError, ValueError) as error:
        return {"available": False, "error": repr(error)}
    if spec is None:
        return {"available": False, "error": f"package not found: {package}"}
    if spec.submodule_search_locations:
        locations = list(spec.submodule_search_locations)
        root = Path(locations[0]).resolve() if locations else None
    elif spec.origin and spec.origin not in {"built-in", "frozen"}:
        root = Path(spec.origin).resolve().parent
    else:
        root = None
    if root is None:
        return {"available": False, "error": f"package root unavailable: {package}"}
    source = root / Path(relative_path)
    if not source.is_file():
        return {
            "available": False,
            "relative_path": relative_path,
            "error": f"package file not found: {source}",
        }
    return {
        "available": True,
        "package": package,
        "relative_path": relative_path,
        "file": _file_content_identity(source),
    }


def _torch_npu_profiler_adapter_identity() -> dict[str, Any]:
    """Identify every repository-owned layer in the NPU profiler adapter."""

    return {
        "torchtitan": _package_file_identity(
            "torchtitan", "tools/profiler.py"
        ),
        "torchtitanturbo": _package_file_identity(
            "torchtitanturbo", "tools/profiler.py"
        ),
        "capture_metrics": _file_content_identity(
            Path(__file__).with_name("capture_metrics.py")
        ),
    }


def _collector_toolchain_metadata(
    config: PerformanceConfig,
) -> dict[str, Any] | None:
    """Bind active official collection to its framework and adapter identity."""

    if not config.profiler_enabled:
        return None
    if config.collector == PerformanceCollector.MSPROF.value:
        scope = "performance-capture"
        tool = "capture"
    elif config.collector == PerformanceCollector.TORCH_NPU_PROFILER.value:
        scope = "performance-torch-npu-capture"
        tool = "torch-npu-capture"
    else:
        return None
    metadata = _performance_stage_metadata(scope)
    doctor = metadata.get("doctor", {})
    if not doctor.get("ok", False):
        raise RuntimeError(
            "MindStudio performance doctor failed: "
            f"{doctor.get('required_failures', [])}"
        )
    identity = _performance_toolchain_identity(metadata, tool=tool)
    if tool == "torch-npu-capture":
        identity["adapters"] = _torch_npu_profiler_adapter_identity()
    return identity


def _analysis_toolchain_metadata() -> dict[str, Any]:
    """Return the analyzer-only identity used by derived official outputs."""

    metadata = _performance_stage_metadata("performance-analysis")
    doctor = metadata.get("doctor", {})
    if not doctor.get("ok", False):
        raise RuntimeError(
            "MindStudio performance analysis doctor failed: "
            f"{doctor.get('required_failures', [])}"
        )
    return _performance_toolchain_identity(metadata, tool="analysis")


def _performance_stage_metadata(scope: str) -> dict[str, Any]:
    """Collect one performance stage without validating unrelated tools."""

    from tests.glm5_2_mindstudio import toolchain

    report = toolchain.doctor_report(scope=scope)
    lock_bytes = toolchain.LOCK_PATH.read_bytes()
    lock = toolchain.load_toolchain_lock()
    tool_name = {
        "performance-capture": "msprof",
        "performance-torch-npu-capture": None,
        "performance-analysis": "msprof-analyze",
    }[scope]
    stage_lock = {
        "profile_name": lock.get("profile_name"),
        "compatibility": (
            lock.get("compatibility")
            if scope
            in {"performance-capture", "performance-torch-npu-capture"}
            else None
        ),
        "tool": (
            lock.get("tools", {}).get(tool_name)
            if tool_name is not None
            else None
        ),
    }
    checks = report.get("checks", {})
    framework = checks.get("framework", {})
    source_checkouts = checks.get("source_checkouts", {})
    sources = source_checkouts.get("tools", {})
    return {
        "schema_version": 1,
        "profile_name": report.get("profile_name"),
        "scope": report.get("scope"),
        "lock": {
            "sha256": _json_identity(stage_lock),
            "full_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        },
        "compatibility": lock.get("compatibility"),
        "versions": {
            "torch": framework.get("torch", {}).get("version"),
            "torch_npu": framework.get("torch_npu", {}).get("version"),
            "msprof": checks.get("msprof", {}).get("reported_version"),
            "msprof-analyze": checks.get("msprof-analyze", {})
            .get("import", {})
            .get("version"),
        },
        "sources": sources if isinstance(sources, dict) else {},
        "doctor": report,
    }


def _performance_toolchain_identity(
    metadata: dict[str, Any],
    *,
    tool: str,
) -> dict[str, Any]:
    """Select only inputs that can change one performance pipeline stage.

    Capture and offline analysis can run on different hosts.  The capture
    cache therefore records torch/torch_npu/CANN/msprof only, while derived
    advisor output records msprof-analyze only.  In particular, installing or
    upgrading msprof-analyze must never invalidate a raw capture.
    """

    if tool not in {"capture", "torch-npu-capture", "analysis"}:
        raise ValueError(f"unsupported performance toolchain stage: {tool}")
    versions = metadata.get("versions", {})
    sources = metadata.get("sources", {})
    doctor = metadata.get("doctor", {})
    checks = doctor.get("checks", {}) if isinstance(doctor, dict) else {}
    lock = metadata.get("lock", {})
    if tool == "capture":
        version_names = ("torch", "torch_npu", "msprof")
        check_names = ("framework", "cann", "msprof")
        source_names = ("msprof",)
        scope = "performance-capture"
    elif tool == "torch-npu-capture":
        version_names = ("torch", "torch_npu")
        check_names = ("framework", "cann")
        source_names = ()
        scope = "performance-torch-npu-capture"
    else:
        version_names = ("msprof-analyze",)
        check_names = ("msprof-analyze",)
        source_names = ("msprof-analyze",)
        scope = "performance-analysis"
    return {
        "schema_version": metadata.get("schema_version"),
        "profile_name": metadata.get("profile_name"),
        "scope": scope,
        "lock_sha256": lock.get("sha256") if isinstance(lock, dict) else None,
        "compatibility": (
            metadata.get("compatibility")
            if tool in {"capture", "torch-npu-capture"}
            else None
        ),
        "versions": {
            name: versions.get(name)
            for name in version_names
            if isinstance(versions, dict)
        },
        "sources": {
            name: sources.get(name)
            for name in source_names
            if isinstance(sources, dict)
        },
        "doctor": {
            "scope": doctor.get("scope") if isinstance(doctor, dict) else scope,
            "runtime": doctor.get("runtime") if isinstance(doctor, dict) else None,
            "checks": {
                name: checks.get(name)
                for name in check_names
                if isinstance(checks, dict)
            },
        },
    }


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
    preset_label = (
        "basic-msprof"
        if config.collector == PerformanceCollector.MSPROF.value
        else config.preset
    )
    base = (
        f"{device}-{config.topology}-{precision}-s{config.steps}-"
        f"l{config.local_batch_size}-b{config.global_batch_size}-"
        f"seq{config.sequence_length}-seed{config.seed}-{preset_label}"
    )
    if config.mixed_precision_reduce != "float32":
        reduce_precision = {
            "bfloat16": "bf16",
            "float16": "fp16",
        }.get(config.mixed_precision_reduce, config.mixed_precision_reduce)
        base += f"-reduce-{reduce_precision}"
    if not config.profiler_enabled:
        base += "-profiler-off"
    elif (
        config.collector
        == PerformanceCollector.TORCH_NPU_PROFILER.value
        and effective_preset.record_shapes
    ):
        # Shape capture materially changes both storage cost and availability
        # of analyses such as GLM5 MoE load balance.  Keep the full profiler
        # contract in the hash, but expose this important distinction in the
        # human-readable run name as well.
        base += "-shapes"
    if config.replicate:
        base += f"-r{config.replicate}"
    if preset_overridden:
        base += f"-{effective_preset.parse_mode}"
    identity = config.as_dict()
    for key in ("name", "device", "run_root", "artifact_root", "report_root"):
        identity.pop(key, None)
    if config.collector == PerformanceCollector.TORCH_NPU_PROFILER.value:
        # Preserve names from before collector selection became explicit.
        identity.pop("collector", None)
    if not config.collector_args:
        identity.pop("collector_args", None)
    identity["resolved_device"] = device
    if (
        config.profiler_enabled
        and config.collector
        == PerformanceCollector.TORCH_NPU_PROFILER.value
    ):
        # The profiler contract changes both collected evidence and which
        # offline recipes are valid.  Bind it unconditionally instead of only
        # for CLI overrides: otherwise changing a preset default can collide
        # with an older completed capture that has the same human-readable
        # name but a different environment.
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
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
    for name in (
        "manifest.json",
        "analysis.json",
        "metrics.jsonl",
        "mindstudio_insight_handoff.json",
    ):
        source = artifact_directory / name
        if source.is_file():
            shutil.copy2(source, directory / name)
    command_directory = directory / "tool_commands"
    for status_path in sorted(run_directory.glob("*.json")):
        if status_path.name == "manifest.json":
            continue
        command_directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(status_path, command_directory / status_path.name)
    analysis_path = artifact_directory / "analysis.json"
    analysis = (
        json.loads(analysis_path.read_text(encoding="utf-8"))
        if analysis_path.is_file()
        else {}
    )
    index = {
        "run_name": run_name,
        "raw_run_directory": str(run_directory),
        "artifact_directory": str(artifact_directory),
        "report_path": str(report_path) if report_path is not None else None,
        "mindstudio_input": str(_find_profiler_directory(run_directory)),
        "tensorboard_root": analysis.get("tensorboard", {}).get("root"),
        "flamegraph_root": analysis.get("profiler_directory"),
        "graph_visualization_root": analysis.get("graph_visualizations", {}).get(
            "root"
        ),
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
    collector = require_training_collector(config.collector)
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
        and collector is PerformanceCollector.TORCH_NPU_PROFILER
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
    if config.profiler_enabled and collector is PerformanceCollector.MSPROF:
        executable = resolve_msprof_executable()
        output_directory = (
            run_directory / "trainer_output" / "profiling" / "msprof"
        )
        default_type = (
            []
            if any(
                argument == "--type" or argument.startswith("--type=")
                for argument in config.collector_args
            )
            else ["--type=text"]
        )
        command = [
            executable,
            f"--output={output_directory}",
            *default_type,
            *config.collector_args,
            *command,
        ]
    return command


def _run_process(
    command: Sequence[str],
    *,
    root: Path,
    environment: dict[str, str],
    log_path: Path,
    log_context: dict[str, Any] | None = None,
) -> None:
    """Execute one stage while teeing terminal output into its runtime log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        for key, value in (log_context or {}).items():
            log.write(f"{key}: {value}\n")
        if log_context:
            log.write("\n")
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
    """Archive incomplete raw output so a retry starts from an empty path."""
    assert_run_not_active(run_directory)
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


def capture(
    root: Path,
    config: PerformanceConfig,
    *,
    device: str,
    preset: ProfilerPreset,
    force: bool,
) -> tuple[Path, Path]:
    """Run one topology/preset capture and publish its compact manifest.

    The run directory owns large trainer/profiler output. The artifact directory
    owns the small identity and metrics needed for indexing. A completed matching
    manifest is reusable; partial raw output is archived before relaunch.
    """
    collector = require_training_collector(config.collector)
    collector_toolchain = _collector_toolchain_metadata(config)
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
    report_parent = _scoped_parent(root, config.report_root, config.topology)
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
    profiler_environment = (
        preset.environment()
        if config.profiler_enabled
        and collector is PerformanceCollector.TORCH_NPU_PROFILER
        else {}
    )
    if complete_manifest.is_file() and run_manifest.is_file() and not force:
        existing = json.loads(complete_manifest.read_text(encoding="utf-8"))
        if (
            not _config_is_compatible(existing.get("config", {}), config)
            or existing.get("profiler_environment") != profiler_environment
            or existing.get("collector_toolchain") != collector_toolchain
        ):
            raise FileExistsError(
                f"completed capture uses different settings: {artifact_directory}; "
                "pass --force or choose a distinct script name"
            )
        print(f"Skip completed performance capture: {artifact_directory}")
        return run_directory, artifact_directory
    if force:
        reset_output_generation(
            (
                run_directory,
                artifact_directory,
                report_parent / f"{run_name}.html",
                _exploration_directory(root, config.topology, run_name),
            ),
            active_run_directories=(run_directory,),
            label="performance capture",
        )
    elif run_directory.exists():
        recovered = _recover_partial_run(run_directory)
        print(f"Archived incomplete run before retry: {recovered}")
    if artifact_directory.exists() and not force:
        recovered = _recover_partial_run(artifact_directory)
        print(f"Archived incomplete artifact before retry: {recovered}")

    run_directory.mkdir(parents=True, exist_ok=False)
    attempt = RunAttempt.start(
        run_directory,
        kind="performance",
        context={
            "device": device,
            "topology": config.topology,
            "preset": config.preset,
        },
    )
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
        f"Starting {capture_kind} capture: device={device}, "
        f"topology={config.topology}, "
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
            log_context=attempt.log_context,
        )
    except Exception as error:
        attempt.update("failed", error=repr(error))
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
        "collector": config.collector,
        "config": config.as_dict(),
        "profiler_environment": profiler_environment,
        "command": command,
        "source": _source_metadata(root),
        "collector_toolchain": collector_toolchain,
        "preflight": preflight,
        "run_directory": str(run_directory),
        "attempt_id": attempt.attempt_id,
    }
    _write_json(run_directory / "manifest.json", manifest)
    artifact_directory.mkdir(parents=True, exist_ok=True)
    _write_json(artifact_directory / "manifest.json", manifest)
    if metrics_path.is_file():
        shutil.copy2(metrics_path, artifact_directory / "metrics.jsonl")
    attempt.update("completed", artifact=str(artifact_directory))
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
        run_directory / "trainer_output" / "profiling" / "msprof",
        run_directory / "trainer_output" / "profiling" / "traces",
        run_directory / "trainer_output" / "profiling",
    )
    return next((path for path in candidates if path.is_dir()), candidates[0])


def _merge_legacy_analysis_tree(source: Path, destination: Path) -> None:
    """Move derived files without mixing conflicting experiment generations."""

    if not source.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*"), key=lambda item: len(item.parts)):
        target = destination / path.relative_to(source)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if path.read_bytes() != target.read_bytes():
                raise RuntimeError(
                    "refusing to merge conflicting performance analysis "
                    f"outputs: {path} -> {target}"
                )
            path.unlink()
        else:
            path.replace(target)
    shutil.rmtree(source)


def _adopt_legacy_cluster_outputs(run_directory: Path) -> Path:
    """Move historical cluster results into the official Insight import root."""

    profiler_directory = _find_profiler_directory(run_directory)
    delivery = profiler_directory / "cluster_analysis_output"
    legacy_advanced = delivery / "advanced"
    if legacy_advanced.is_dir():
        for recipe_root in tuple(legacy_advanced.iterdir()):
            if recipe_root.is_dir():
                nested = recipe_root / "cluster_analysis_output"
                _merge_legacy_analysis_tree(
                    nested if nested.is_dir() else recipe_root,
                    delivery,
                )
        if legacy_advanced.is_dir() and not any(legacy_advanced.iterdir()):
            legacy_advanced.rmdir()
    _merge_legacy_analysis_tree(
        run_directory / "cluster" / "cluster_analysis_output", delivery
    )
    run_level_advanced = run_directory / "cluster" / "advanced"
    if run_level_advanced.is_dir():
        for recipe_root in tuple(run_level_advanced.iterdir()):
            if recipe_root.is_dir():
                nested = recipe_root / "cluster_analysis_output"
                _merge_legacy_analysis_tree(
                    nested if nested.is_dir() else recipe_root,
                    delivery,
                )
        if run_level_advanced.is_dir() and not any(run_level_advanced.iterdir()):
            run_level_advanced.rmdir()
    legacy_text_state = run_directory / "cluster" / "cluster_text.json"
    current_text_state = run_directory / "cluster_text.json"
    if legacy_text_state.is_file():
        if current_text_state.is_file():
            if legacy_text_state.read_bytes() != current_text_state.read_bytes():
                raise RuntimeError(
                    "refusing to merge conflicting cluster text state files: "
                    f"{legacy_text_state} -> {current_text_state}"
                )
            legacy_text_state.unlink()
        else:
            legacy_text_state.replace(current_text_state)
    for name in ("cluster_time_summary", "free_analysis"):
        source_root = run_directory / name
        _merge_legacy_analysis_tree(
            source_root / "cluster_analysis_output", delivery
        )
        if source_root.is_dir() and not any(source_root.iterdir()):
            source_root.rmdir()
    for source in run_directory.glob("communication_bottleneck_rank_*"):
        if source.is_dir():
            _merge_legacy_analysis_tree(
                source / "cluster_analysis_output", delivery
            )
            if source.is_dir() and not any(source.iterdir()):
                source.rmdir()
    legacy_cluster = run_directory / "cluster"
    if legacy_cluster.is_dir() and not any(legacy_cluster.iterdir()):
        legacy_cluster.rmdir()
    return delivery


def offline_parse(
    run_directory: Path,
    *,
    max_process_number: int | None = None,
) -> list[str]:
    """Parse retained Ascend raw roots after the training process has ended.

    This moves expensive CANN conversion off the training critical path and can
    use multiple CPU workers. It cannot reconstruct events that collection did
    not enable.
    """
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


def run_advisor(
    run_directory: Path,
    *,
    analysis_toolchain: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
        run_directory,
        name="advisor",
        command=command,
        analysis_toolchain=analysis_toolchain,
    )


def _run_msprof_analyze(
    run_directory: Path,
    *,
    name: str,
    command: list[str],
    analysis_toolchain: dict[str, Any] | None = None,
    status_path: Path | None = None,
) -> dict[str, Any]:
    print(f"Running msprof-analyze {name}: {' '.join(command)}")
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    status = {
        "command": command,
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "analysis_toolchain": analysis_toolchain,
    }
    _write_json(status_path or (run_directory / f"{name}.json"), status)
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


def _safe_distributed_parse_preset(
    preset: ProfilerPreset,
    *,
    device: str,
    world_size: int,
) -> ProfilerPreset:
    """Keep synchronous CANN parsing out of a live distributed job.

    A sync parse pauses only the profiled rank. Other ranks can enter the next
    collective and hit TorchTitan's training process-group timeout while CANN
    is still exporting a large trace. Offline parsing preserves the same raw
    capture and analysis while moving the pause after every rank has exited.
    """

    if device == "npu" and world_size > 1 and preset.parse_mode == "sync":
        return replace(preset, parse_mode="offline")
    return preset


def run_cluster_analysis(
    run_directory: Path,
    *,
    extended: bool = True,
    advanced_recipe_policy: str = "none",
    topology: ParallelTopology | None = None,
    record_shapes: bool = False,
    mstx_enabled: bool = False,
    with_flops: bool = False,
    cluster_summary_baseline: Path | None = None,
    mode: str = "all",
    agent_output: bool = False,
    bypass_input_safety_checks: bool = False,
    analysis_toolchain: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the official cluster entry and selected advanced recipes."""

    if mode not in {"all", "communication_time", "communication_matrix"}:
        raise ValueError(
            "cluster mode must be all, communication_time, or "
            "communication_matrix"
        )

    profiler_directory = _find_profiler_directory(run_directory)
    _adopt_legacy_cluster_outputs(run_directory)
    # Keep the official msprof-analyze delivery beside the per-rank profiler
    # data.  MindStudio Insight expects users to import this profiler root; a
    # second run-level ``cluster/`` tree is neither official nor portable.
    output_directory = profiler_directory
    command = [
        _msprof_analyze_executable(),
        "cluster",
        "-m",
        mode,
        "-d",
        str(profiler_directory),
        "-o",
        str(output_directory),
    ]
    if agent_output:
        command.append("--agent")
    if bypass_input_safety_checks:
        command.append("--force")
    results = {
        "cluster": _run_msprof_analyze(
            run_directory,
            name="cluster",
            command=command,
            analysis_toolchain=analysis_toolchain,
        )
    }
    results["cluster_text"] = _run_cluster_text_delivery(
        run_directory,
        profiler_directory=profiler_directory,
        output_directory=output_directory,
        mode=mode,
        agent_output=agent_output,
        bypass_input_safety_checks=bypass_input_safety_checks,
        analysis_toolchain=analysis_toolchain,
    )
    cluster_delivery = profiler_directory / "cluster_analysis_output"
    if cluster_delivery.is_dir():
        results["mindstudio_insight_bundle"] = {
            "import_root": str(profiler_directory),
            "cluster_delivery": str(cluster_delivery),
            "files": sorted(
                path.relative_to(cluster_delivery).as_posix()
                for path in cluster_delivery.rglob("*")
                if path.is_file()
            ),
        }
    if not extended:
        if advanced_recipe_policy == "none":
            return results
        if topology is None:
            raise ValueError("advanced cluster analysis requires a topology")
        advanced = _run_advanced_cluster_analysis(
            run_directory,
            profiler_directory=profiler_directory,
            topology=topology,
            record_shapes=record_shapes,
            policy=advanced_recipe_policy,
            mstx_enabled=mstx_enabled,
            with_flops=with_flops,
            cluster_summary_baseline=cluster_summary_baseline,
            analysis_toolchain=analysis_toolchain,
        )
        results["advanced"] = advanced
        return results
    executable = _msprof_analyze_executable()
    recipes = ("cluster_time_summary", "free_analysis")
    rank_ids = sorted(
        {
            match.group(1)
            for path in profiler_directory.iterdir()
            if (match := re.match(r"rank_(\d+)_", path.name))
        },
        key=int,
    )
    for name in recipes:
        results[name] = _run_recipe_exports(
            run_directory,
            executable=executable,
            recipe=name,
            profiler_directory=profiler_directory,
            status_prefix=name,
            analysis_toolchain=analysis_toolchain,
        )
    # This recipe has a required rank argument. Run both official export modes
    # for every captured rank against the same canonical output root.
    for rank_id in rank_ids:
        name = f"communication_bottleneck_rank_{rank_id}"
        results[name] = _run_recipe_exports(
            run_directory,
            executable=executable,
            recipe="communication_bottleneck",
            profiler_directory=profiler_directory,
            rank_id=int(rank_id),
            status_prefix=name,
            analysis_toolchain=analysis_toolchain,
        )
    return results


def _profiler_rank_ids(profiler_directory: Path) -> list[int]:
    """Discover rank IDs from supported Ascend PyTorch Profiler layouts."""

    rank_ids: set[int] = set()
    for path in profiler_directory.rglob("profiler_info_*.json"):
        match = re.search(r"profiler_info_(\d+)\.json$", path.name)
        if match:
            rank_ids.add(int(match.group(1)))
    if not rank_ids:
        for path in profiler_directory.iterdir():
            match = re.search(r"(?:rank[_-]?)(\d+)", path.name)
            if match:
                rank_ids.add(int(match.group(1)))
    return sorted(rank_ids)


def _advanced_recipe_command(
    executable: str,
    *,
    recipe: str,
    profiler_directory: Path,
    output_directory: Path,
    rank_id: int | None = None,
    export_type: str | None = None,
) -> list[str]:
    command = [
        executable,
        "-m",
        recipe,
        "-d",
        str(profiler_directory),
        "-o",
        str(output_directory),
    ]
    command.extend(
        recipe_arguments(
            recipe,
            rank_id=rank_id,
            export_type=export_type,
        )
    )
    return command


def _run_recipe_exports(
    run_directory: Path,
    *,
    executable: str,
    recipe: str,
    profiler_directory: Path,
    rank_id: int | None = None,
    status_prefix: str,
    analysis_toolchain: dict[str, Any] | None,
) -> dict[str, Any]:
    """Generate both official DB and Text deliveries in one output root."""

    results: dict[str, Any] = {}
    for export_type in ("db", "text"):
        name = f"{status_prefix}_{export_type}"
        results[export_type] = _run_msprof_analyze(
            run_directory,
            name=name,
            command=_advanced_recipe_command(
                executable,
                recipe=recipe,
                profiler_directory=profiler_directory,
                output_directory=profiler_directory,
                rank_id=rank_id,
                export_type=export_type,
            ),
            status_path=run_directory / f"{name}.json",
            analysis_toolchain=analysis_toolchain,
        )
    return results


def _run_advanced_cluster_analysis(
    run_directory: Path,
    *,
    profiler_directory: Path,
    topology: ParallelTopology,
    policy: str,
    record_shapes: bool,
    mstx_enabled: bool,
    with_flops: bool,
    cluster_summary_baseline: Path | None,
    analysis_toolchain: dict[str, Any] | None,
) -> dict[str, Any]:
    """Run official advanced recipes without altering the raw capture."""

    plan = resolve_recipe_plan(
        policy,
        topology=topology,
        record_shapes=record_shapes,
        mstx_enabled=mstx_enabled,
        with_flops=with_flops,
        cluster_summary_baseline=cluster_summary_baseline is not None,
    )
    executable = _msprof_analyze_executable()
    delivery = profiler_directory / "cluster_analysis_output"
    delivery.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"policy": policy, "skipped": plan.skipped}
    rank_ids = _profiler_rank_ids(profiler_directory)

    for recipe in plan.recipes:
        if recipe == "cluster_time_compare_summary":
            assert cluster_summary_baseline is not None
            compare_root = Path(tempfile.mkdtemp(prefix="glm5-cluster-compare-"))
            candidate_summary = compare_root / "candidate"
            baseline_summary = compare_root / "baseline"
            # The compare recipe requires ClusterTimeSummary DB tables. The
            # ordinary summary recipe intentionally exports a readable CSV,
            # so generate dedicated DB summaries for both inputs here.
            results["cluster_time_summary_candidate_db"] = _run_msprof_analyze(
                run_directory,
                name="advanced_cluster_time_summary_candidate_db",
                command=[
                    executable,
                    "-m",
                    "cluster_time_summary",
                    "-d",
                    str(profiler_directory),
                    "-o",
                    str(candidate_summary),
                ],
                status_path=run_directory / "cluster_time_summary_candidate_db.json",
                analysis_toolchain=analysis_toolchain,
            )
            results["cluster_time_summary_baseline"] = _run_msprof_analyze(
                run_directory,
                name="advanced_cluster_time_summary_baseline",
                command=[
                    executable,
                    "-m",
                    "cluster_time_summary",
                    "-d",
                    str(cluster_summary_baseline),
                    "-o",
                    str(baseline_summary),
                ],
                status_path=run_directory / "cluster_time_summary_baseline_db.json",
                analysis_toolchain=analysis_toolchain,
            )
            command = [
                executable,
                "-m",
                recipe,
                "-d",
                str(candidate_summary),
                "--bp",
                str(baseline_summary),
                "-o",
                str(profiler_directory),
            ]
            results[recipe] = _run_msprof_analyze(
                run_directory,
                name=f"advanced_{recipe}",
                command=command,
                status_path=run_directory / "cluster_time_compare_summary.json",
                analysis_toolchain=analysis_toolchain,
            )
            shutil.rmtree(compare_root)
            continue
        if recipe == "communication_bottleneck":
            if not rank_ids:
                raise RuntimeError(
                    "communication_bottleneck could not discover profiler rank IDs"
                )
            rank_results: dict[str, Any] = {}
            for rank_id in rank_ids:
                name = f"communication_bottleneck_rank_{rank_id}"
                rank_results[str(rank_id)] = _run_recipe_exports(
                    run_directory,
                    executable=executable,
                    recipe=recipe,
                    profiler_directory=profiler_directory,
                    rank_id=rank_id,
                    status_prefix=f"advanced_{name}",
                    analysis_toolchain=analysis_toolchain,
                )
            results[recipe] = rank_results
            continue
        if recipe in MUTATING_RECIPES:
            print(
                f"Running explicitly requested source-mutating recipe: {recipe}",
                flush=True,
            )
        results[recipe] = _run_recipe_exports(
            run_directory,
            executable=executable,
            recipe=recipe,
            profiler_directory=profiler_directory,
            status_prefix=f"advanced_{recipe}",
            analysis_toolchain=analysis_toolchain,
        )

    inventory = {
        "policy": policy,
        "topology": topology.name,
        "recipes": list(plan.recipes),
        "skipped": plan.skipped,
        "rank_ids": rank_ids,
        "outputs": sorted(
            path.relative_to(delivery).as_posix()
            for path in delivery.rglob("*")
            if path.is_file()
        ),
    }
    results["index"] = inventory
    return results


def _run_cluster_text_delivery(
    run_directory: Path,
    *,
    profiler_directory: Path,
    output_directory: Path,
    mode: str,
    agent_output: bool,
    bypass_input_safety_checks: bool,
    analysis_toolchain: dict[str, Any] | None,
) -> dict[str, Any]:
    """Generate the official text delivery beside the official cluster DB.

    Ascend PyTorch Profiler commonly exports both DB and text under each rank.
    msprof-analyze selects one input format per invocation, so a mixed root can
    yield only ``cluster_analysis.db``.  A temporary text-only view makes the
    second official invocation deterministic; its CSV/JSON files are merged
    into the same ``cluster_analysis_output`` and the staging tree is removed.
    """

    rank_roots = sorted(
        path for path in profiler_directory.glob("*_ascend_pt") if path.is_dir()
    )
    text_names = {
        "step_trace_time.csv",
        "communication.json",
        "communication_matrix.json",
    }
    with tempfile.TemporaryDirectory(prefix="glm5-cluster-text-") as directory:
        temporary = Path(directory)
        text_input = temporary / "input"
        copied: list[str] = []
        for rank_root in rank_roots:
            destination = text_input / rank_root.name
            destination.mkdir(parents=True)
            for info in rank_root.glob("profiler_info_*.json"):
                shutil.copy2(info, destination / info.name)
                copied.append(str(info))
            source_output = rank_root / "ASCEND_PROFILER_OUTPUT"
            destination_output = destination / "ASCEND_PROFILER_OUTPUT"
            destination_output.mkdir()
            for name in text_names:
                source = source_output / name
                if source.is_file():
                    shutil.copy2(source, destination_output / name)
                    copied.append(str(source))
        if not copied or not any(
            (
                text_input
                / rank.name
                / "ASCEND_PROFILER_OUTPUT"
                / "step_trace_time.csv"
            ).is_file()
            for rank in rank_roots
        ):
            return {"status": "missing_text_input", "copied": copied}
        text_output = temporary / "output"
        command = [
            _msprof_analyze_executable(),
            "cluster",
            "-m",
            mode,
            "-d",
            str(text_input),
            "-o",
            str(text_output),
        ]
        if agent_output:
            command.append("--agent")
        if bypass_input_safety_checks:
            command.append("--force")
        status_path = run_directory / "cluster_text.json"
        result = _run_msprof_analyze(
            run_directory,
            name="cluster_text",
            command=command,
            analysis_toolchain=analysis_toolchain,
            status_path=status_path,
        )
        generated = text_output / "cluster_analysis_output"
        destination = output_directory / "cluster_analysis_output"
        if not generated.is_dir():
            raise FileNotFoundError(
                "msprof-analyze text cluster output is missing: "
                f"{generated}"
            )
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(generated, destination, dirs_exist_ok=True)
        result["merged_output"] = str(destination)
        result["files"] = sorted(
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*")
            if path.is_file()
        )
        _write_json(status_path, result)
        return result


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
    *,
    standard_cli: bool = False,
    analysis_toolchain: dict[str, Any] | None = None,
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
        "--output_path" if standard_cli else "-o",
        str(output_directory),
    ]
    return _run_msprof_analyze(
        run_directory,
        name="compare",
        command=command,
        analysis_toolchain=analysis_toolchain,
    )


def _json_identity(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _path_tree_identity(path: Path) -> dict[str, Any]:
    """Fingerprint a file or directory without hashing multi-gigabyte traces.

    Official profiler baselines are usually directory trees containing large
    databases. The recursive relative path, size, and nanosecond mtime tuple
    detects an in-place rewrite while keeping compare startup bounded.
    """

    resolved = path.resolve()
    if resolved.is_file():
        status = resolved.stat()
        entries = [
            {
                "path": ".",
                "size": status.st_size,
                "mtime_ns": status.st_mtime_ns,
            }
        ]
        kind = "file"
    elif resolved.is_dir():
        entries = []
        for candidate in sorted(
            (item for item in resolved.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(resolved).as_posix(),
        ):
            status = candidate.stat()
            entries.append(
                {
                    "path": candidate.relative_to(resolved).as_posix(),
                    "size": status.st_size,
                    "mtime_ns": status.st_mtime_ns,
                }
            )
        kind = "directory"
    else:
        raise FileNotFoundError(f"analysis input does not exist: {resolved}")
    return {
        "kind": kind,
        "file_count": len(entries),
        "total_bytes": sum(entry["size"] for entry in entries),
        "tree_stat_sha256": _json_identity(entries),
    }


def _optional_visualizer_identity(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return _file_content_identity(Path(path))


def _visualization_toolchain_identity() -> dict[str, Any]:
    """Bind derived visualizations to the scripts that render them."""

    return {
        "mindstudio_insight_flamegraph": _optional_visualizer_identity(
            find_mindstudio_flamegraph_script()
        ),
        "brendangregg_flamegraph": _optional_visualizer_identity(
            find_flamegraph_script()
        ),
    }


def _capture_analysis_identity(manifest: dict[str, Any]) -> str:
    """Identify captured evidence without binding analysis to local paths."""

    source = manifest.get("source", {})
    portable_source = {
        key: value
        for key, value in source.items()
        if not key.endswith("_root")
    } if isinstance(source, dict) else source
    return _json_identity(
        {
            "format_version": manifest.get("format_version"),
            "run_name": manifest.get("run_name"),
            "attempt_id": manifest.get("attempt_id"),
            "device": manifest.get("device"),
            "topology": manifest.get("topology"),
            "preset": manifest.get("preset"),
            "collector": manifest.get("collector"),
            "config": manifest.get("config"),
            "profiler_environment": manifest.get("profiler_environment"),
            "source": portable_source,
            "collector_toolchain": manifest.get("collector_toolchain"),
        }
    )


def _analysis_request(
    *,
    manifest: dict[str, Any],
    parse_offline: bool,
    parse_workers: int | None,
    advisor: bool,
    cluster: bool,
    cluster_mode: str = "all",
    cluster_agent_output: bool = False,
    cluster_bypass_input_safety_checks: bool = False,
    extended_cluster_analysis: bool = True,
    advanced_recipe_policy: str = "none",
    cluster_summary_baseline: Path | None = None,
    compare_baseline: Path | None = None,
    analysis_toolchain: dict[str, Any] | None = None,
) -> dict[str, Any]:
    baseline: dict[str, Any] | None = None
    if compare_baseline is not None:
        resolved = compare_baseline.resolve()
        baseline = {
            "path": str(resolved),
            **_path_tree_identity(resolved),
        }
    request = {
        "capture_identity": _capture_analysis_identity(manifest),
        "operations": {
            "offline_parse": parse_offline,
            "parse_workers": parse_workers if parse_offline else None,
            "advisor": advisor,
            "cluster": cluster,
            "cluster_mode": cluster_mode if cluster else None,
            "cluster_agent_output": (
                cluster_agent_output if cluster else False
            ),
            "cluster_bypass_input_safety_checks": (
                cluster_bypass_input_safety_checks if cluster else False
            ),
            "extended_cluster_analysis": (
                extended_cluster_analysis if cluster else False
            ),
            "advanced_recipe_policy": (
                advanced_recipe_policy if cluster else "none"
            ),
            "cluster_summary_baseline": (
                {
                    "path": str(cluster_summary_baseline.resolve()),
                    **_path_tree_identity(cluster_summary_baseline.resolve()),
                }
                if cluster and cluster_summary_baseline is not None
                else None
            ),
            "compare_baseline": baseline,
        },
        "analysis_toolchain": analysis_toolchain,
        "visualization_toolchain": _visualization_toolchain_identity(),
    }
    request["identity"] = _json_identity(request)
    return request


def _analysis_output_paths(
    *,
    run_directory: Path,
    artifact_directory: Path,
    report_path: Path,
    exploration_directory: Path,
) -> list[Path]:
    """List derived analysis output while intentionally excluding raw data."""

    paths = [
        artifact_directory / "analysis.json",
        artifact_directory / "mindstudio_insight_handoff.json",
        artifact_directory / "analysis_state.json",
        artifact_directory / "analysis_state.json.lock",
        report_path,
        exploration_directory / "analysis.json",
        exploration_directory / "mindstudio_insight_handoff.json",
        exploration_directory / "artifacts.json",
        run_directory / "advisor",
        run_directory / "advisor.json",
        run_directory / "cluster.json",
        run_directory / "cluster_time_summary",
        run_directory / "cluster_time_summary.json",
        run_directory / "free_analysis",
        run_directory / "free_analysis.json",
        run_directory / "compare",
        run_directory / "compare.json",
    ]
    paths.extend(run_directory.glob("communication_bottleneck_rank_*"))
    paths.extend(run_directory.glob("advanced_*.json"))
    paths.append(run_directory / "cluster_text.json")
    command_directory = exploration_directory / "tool_commands"
    for name in (
        "advisor.json",
        "cluster.json",
        "cluster_time_summary.json",
        "free_analysis.json",
        "compare.json",
    ):
        paths.append(command_directory / name)
    paths.extend(command_directory.glob("communication_bottleneck_rank_*.json"))
    profiler_directory = _find_profiler_directory(run_directory)
    paths.append(profiler_directory / "mindstudio_flamegraphs")
    if profiler_directory.is_dir():
        paths.extend(profiler_directory.rglob("*_flamegraph.svg"))
    return list(dict.fromkeys(paths))


def _analysis_outputs_exist(
    *,
    run_directory: Path,
    artifact_directory: Path,
    report_path: Path,
) -> bool:
    """Return whether provenance-sensitive derived output already exists."""

    candidates = [
        artifact_directory / "analysis.json",
        artifact_directory / "mindstudio_insight_handoff.json",
        report_path,
        run_directory / "advisor",
        run_directory / "advisor.json",
        run_directory / "cluster.json",
        run_directory / "cluster_time_summary",
        run_directory / "cluster_time_summary.json",
        run_directory / "free_analysis",
        run_directory / "free_analysis.json",
        run_directory / "compare",
        run_directory / "compare.json",
    ]
    candidates.extend(run_directory.glob("communication_bottleneck_rank_*"))
    candidates.extend(run_directory.glob("advanced_*.json"))
    candidates.append(run_directory / "cluster_text.json")
    profiler_directory = _find_profiler_directory(run_directory)
    candidates.append(profiler_directory / "mindstudio_flamegraphs")
    if profiler_directory.is_dir():
        candidates.extend(profiler_directory.rglob("*_flamegraph.svg"))
    return any(path.exists() for path in candidates)


def _analysis_outputs_complete(
    run_directory: Path,
    request: dict[str, Any],
) -> bool:
    operations = request.get("operations", {})
    required = []
    if operations.get("advisor"):
        required.extend(
            (run_directory / "advisor", run_directory / "advisor.json")
        )
    if operations.get("cluster"):
        cluster_delivery = (
            _find_profiler_directory(run_directory) / "cluster_analysis_output"
        )
        required.extend(
            (cluster_delivery, run_directory / "cluster.json")
        )
    if operations.get("compare_baseline") is not None:
        required.extend(
            (run_directory / "compare", run_directory / "compare.json")
        )
    return all(path.exists() for path in required)


def _analysis_stage_context(
    request: dict[str, Any], stage: str
) -> dict[str, Any]:
    """Return the independently versioned identity for one analysis tool.

    Capture data is shared, but offline parsing, Advisor, cluster analysis and
    comparison are separate official operations with separate output roots.
    Keeping their identities independent allows a later operation to be added
    without invalidating or deleting an earlier operation.
    """

    operations = request["operations"]
    stage_operations: dict[str, Any]
    if stage == "offline_parse":
        stage_operations = {
            "parse_workers": operations.get("parse_workers"),
        }
    elif stage == "advisor":
        stage_operations = {}
    elif stage == "cluster":
        stage_operations = {
            "output_layout": "profiler-root-cluster-analysis-output-v2",
            "recipe_export_types": ["db", "text"],
            "cluster_mode": operations.get("cluster_mode"),
            "cluster_agent_output": operations.get("cluster_agent_output"),
            "cluster_bypass_input_safety_checks": operations.get(
                "cluster_bypass_input_safety_checks"
            ),
            "extended_cluster_analysis": operations.get(
                "extended_cluster_analysis"
            ),
            "advanced_recipe_policy": operations.get(
                "advanced_recipe_policy", "none"
            ),
            "cluster_summary_baseline": operations.get(
                "cluster_summary_baseline"
            ),
        }
    elif stage == "compare":
        stage_operations = {
            "compare_baseline": operations.get("compare_baseline"),
        }
    else:
        raise ValueError(f"unknown performance analysis stage: {stage}")
    context = {
        "capture_identity": request["capture_identity"],
        "stage": stage,
        "operations": stage_operations,
        "analysis_toolchain": (
            request.get("analysis_toolchain")
            if stage in {"advisor", "cluster", "compare"}
            else None
        ),
    }
    context["identity"] = _json_identity(context)
    return context


def _analysis_stage_paths(
    run_directory: Path,
    artifact_directory: Path,
    stage: str,
) -> list[Path]:
    state = artifact_directory / f"analysis_{stage}_state.json"
    paths = [state, state.with_suffix(state.suffix + ".lock")]
    if stage == "advisor":
        paths.extend((run_directory / "advisor", run_directory / "advisor.json"))
    elif stage == "cluster":
        paths.extend(
            (
                run_directory / "cluster.json",
                run_directory / "cluster_time_summary",
                run_directory / "cluster_time_summary.json",
                run_directory / "free_analysis",
                run_directory / "free_analysis.json",
            )
        )
        paths.extend(run_directory.glob("communication_bottleneck_rank_*"))
        paths.extend(run_directory.glob("cluster_time_summary_*.json"))
        paths.extend(run_directory.glob("free_analysis_*.json"))
        paths.extend(run_directory.glob("advanced_*.json"))
        paths.append(run_directory / "cluster_text.json")
        try:
            paths.append(
                _find_profiler_directory(run_directory)
                / "cluster_analysis_output"
            )
        except FileNotFoundError:
            pass
    elif stage == "compare":
        paths.extend((run_directory / "compare", run_directory / "compare.json"))
    return list(dict.fromkeys(paths))


def _analysis_stage_outputs_complete(run_directory: Path, stage: str) -> bool:
    if stage == "offline_parse":
        profiler_directory = _find_profiler_directory(run_directory)
        return any(profiler_directory.rglob("*.db"))
    required = {
        "advisor": (run_directory / "advisor", run_directory / "advisor.json"),
        "cluster": (
            _find_profiler_directory(run_directory) / "cluster_analysis_output",
            run_directory / "cluster.json",
        ),
        "compare": (run_directory / "compare", run_directory / "compare.json"),
    }[stage]
    return all(path.exists() for path in required)


def _prepare_analysis_stage(
    *,
    run_directory: Path,
    artifact_directory: Path,
    request: dict[str, Any],
    stage: str,
    force: bool,
) -> tuple[RunAttempt | None, bool]:
    """Prepare one analysis stage and return ``(attempt, should_run)``."""

    if stage == "cluster" and not force:
        # Layout-only adoption preserves a completed historical analysis and
        # lets the normal state/identity checks decide whether it can be reused.
        _adopt_legacy_cluster_outputs(run_directory)
    context = _analysis_stage_context(request, stage)
    state_name = f"analysis_{stage}_state.json"
    state_path = artifact_directory / state_name
    if force:
        reset_output_generation(
            _analysis_stage_paths(run_directory, artifact_directory, stage),
            label=f"performance {stage} analysis",
            include_archives=False,
        )
    elif state_path.is_file():
        assert_run_not_active(artifact_directory, state_name=state_name)
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"{stage} analysis state is unreadable; pass --force to "
                f"replace only that stage: {state_path}"
            ) from error
        if (
            state.get("status") == "completed"
            and state.get("context", {}).get("identity") == context["identity"]
            and _analysis_stage_outputs_complete(run_directory, stage)
        ):
            print(f"Skip completed performance {stage} analysis: {state_path}")
            return None, False
        if (
            state.get("status") == "completed"
            and state.get("context", {}).get("identity") != context["identity"]
        ):
            # Analysis outputs are derived from an immutable capture.  A tool
            # upgrade or option change must invalidate only this stage; making
            # users force the whole topology suite would either recapture the
            # model unnecessarily or strand an interrupted ``--topology all``
            # run behind stale derived state.  Resetting the stage paths keeps
            # the capture and every independent completed analysis intact.
            print(
                f"Rebuild performance {stage} analysis because its inputs or "
                f"options changed: {state_path}",
                flush=True,
            )
            reset_output_generation(
                _analysis_stage_paths(
                    run_directory,
                    artifact_directory,
                    stage,
                ),
                label=f"stale performance {stage} analysis",
                include_archives=False,
            )
        else:
            reset_output_generation(
                _analysis_stage_paths(
                    run_directory,
                    artifact_directory,
                    stage,
                ),
                label=f"incomplete performance {stage} analysis",
                include_archives=False,
            )
    elif _analysis_stage_outputs_complete(run_directory, stage):
        # Adopt output produced by the former monolithic analysis lifecycle
        # only when its recorded request proves that this exact stage ran.
        legacy_path = artifact_directory / "analysis_state.json"
        legacy = None
        if legacy_path.is_file():
            try:
                legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError):
                legacy = None
        legacy_operations = (
            legacy.get("context", {}).get("operations", {})
            if isinstance(legacy, dict)
            else {}
        )
        legacy_key = "compare_baseline" if stage == "compare" else stage
        if (
            isinstance(legacy, dict)
            and legacy.get("status") == "completed"
            and legacy_operations.get(legacy_key)
            and legacy.get("context", {}).get("capture_identity")
            == request["capture_identity"]
        ):
            attempt = RunAttempt.start(
                artifact_directory,
                kind=f"performance-{stage}-analysis",
                context=context,
                state_name=state_name,
            )
            attempt.update("completed", adopted_legacy_output=True)
            print(f"Adopted completed legacy {stage} output: {state_path}")
            return None, False
        raise FileExistsError(
            f"{stage} output exists without matching stage provenance; pass "
            f"--force to replace only that stage: {state_path}"
        )
    attempt = RunAttempt.start(
        artifact_directory,
        kind=f"performance-{stage}-analysis",
        context=context,
        state_name=state_name,
    )
    return attempt, True


def _reset_analysis_outputs(
    *,
    run_directory: Path,
    artifact_directory: Path,
    report_path: Path,
    exploration_directory: Path,
) -> None:
    """Reset one selected analysis generation without touching its capture."""

    assert_run_not_active(
        artifact_directory,
        state_name="analysis_state.json",
    )
    reset_output_generation(
        _analysis_output_paths(
            run_directory=run_directory,
            artifact_directory=artifact_directory,
            report_path=report_path,
            exploration_directory=exploration_directory,
        ),
        label="performance analysis",
        include_archives=False,
    )


def _validate_analysis_capabilities(
    config: PerformanceConfig,
    *,
    advisor: bool,
    cluster: bool,
    compare_baseline: Path | None,
    run_directory: Path | None = None,
) -> None:
    """Enforce the input formats supported by msprof-analyze 26.1."""

    if config.collector != PerformanceCollector.MSPROF.value:
        return
    unsupported = []
    if advisor:
        unsupported.append("advisor")
    if compare_baseline is not None:
        unsupported.append("compare")
    if unsupported:
        raise ValueError(
            "msprof-analyze 26.1 "
            + "/".join(unsupported)
            + " requires an Ascend PyTorch Profiler *_ascend_pt capture; "
            "the msprof collector supports cluster analysis and MindStudio "
            "Insight only"
        )
    if cluster and run_directory is not None:
        profiler_directory = _find_profiler_directory(run_directory)
        if not any(profiler_directory.rglob("*.db")):
            raise ValueError(
                "msprof-analyze cluster requires a database produced by the "
                "msprof capture, but no *.db file was found"
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
    cluster_mode: str = "all",
    cluster_agent_output: bool = False,
    cluster_bypass_input_safety_checks: bool = False,
    extended_cluster_analysis: bool = True,
    advanced_recipe_policy: str = "none",
    cluster_summary_baseline: Path | None = None,
    force: bool = False,
) -> Path:
    """Attach official analyses and build a portable report for one capture.

    Analysis may parse, advise, compare, and render existing evidence; it never
    reruns training or changes the profiler-off performance baseline.
    """
    _validate_analysis_capabilities(
        config,
        advisor=advisor,
        cluster=cluster,
        compare_baseline=compare_baseline,
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
    manifest_path = artifact_directory / "manifest.json"
    if not manifest_path.is_file():
        match = _find_matching_capture(
            artifact_parent,
            config=config,
            device=device,
            profiler_environment=(
                preset.environment()
                if config.profiler_enabled
                and config.collector
                == PerformanceCollector.TORCH_NPU_PROFILER.value
                else {}
            ),
        )
        if match is None:
            raise FileNotFoundError(f"capture artifact not found: {manifest_path}")
        artifact_directory, manifest = match
        run_name = str(manifest["run_name"])
        run_directory = run_parent / run_name
        manifest_path = artifact_directory / "manifest.json"
        print(f"Using compatible capture manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report_path = (
        _scoped_parent(root, config.report_root, config.topology)
        / f"{run_name}.html"
    )
    exploration_directory = _exploration_directory(
        root, config.topology, run_name
    )
    _validate_analysis_capabilities(
        config,
        advisor=advisor,
        cluster=cluster,
        compare_baseline=compare_baseline,
        run_directory=run_directory,
    )
    if (
        parse_offline
        and config.profiler_enabled
        and config.collector == PerformanceCollector.MSPROF.value
    ):
        raise ValueError(
            "torch_npu offline parsing cannot process an msprof collector run"
        )
    requires_official_analyzer = (
        advisor or cluster or compare_baseline is not None
    )
    analysis_toolchain = (
        _analysis_toolchain_metadata() if requires_official_analyzer else None
    )
    request = _analysis_request(
        manifest=manifest,
        parse_offline=parse_offline,
        parse_workers=parse_workers,
        advisor=advisor,
        cluster=cluster,
        cluster_mode=cluster_mode,
        cluster_agent_output=cluster_agent_output,
        cluster_bypass_input_safety_checks=(
            cluster_bypass_input_safety_checks
        ),
        extended_cluster_analysis=extended_cluster_analysis,
        advanced_recipe_policy=advanced_recipe_policy,
        cluster_summary_baseline=cluster_summary_baseline,
        compare_baseline=compare_baseline,
        analysis_toolchain=analysis_toolchain,
    )
    assert_run_not_active(
        artifact_directory,
        state_name="analysis_state.json",
    )
    try:
        if parse_offline and config.profiler_enabled:
            parse_attempt, should_parse = _prepare_analysis_stage(
                run_directory=run_directory,
                artifact_directory=artifact_directory,
                request=request,
                stage="offline_parse",
                force=force,
            )
            if should_parse:
                assert parse_attempt is not None
                try:
                    offline_parse(
                        run_directory,
                        max_process_number=parse_workers,
                    )
                except BaseException as error:
                    parse_attempt.update("failed", error=repr(error))
                    raise
                parse_attempt.update("completed")
        if advisor:
            advisor_attempt, should_advise = _prepare_analysis_stage(
                run_directory=run_directory,
                artifact_directory=artifact_directory,
                request=request,
                stage="advisor",
                force=force,
            )
            if should_advise:
                assert advisor_attempt is not None
                try:
                    run_advisor(
                        run_directory,
                        analysis_toolchain=analysis_toolchain,
                    )
                except BaseException as error:
                    advisor_attempt.update("failed", error=repr(error))
                    raise
                advisor_attempt.update("completed")
        if cluster:
            cluster_attempt, should_cluster = _prepare_analysis_stage(
                run_directory=run_directory,
                artifact_directory=artifact_directory,
                request=request,
                stage="cluster",
                force=force,
            )
            if should_cluster:
                assert cluster_attempt is not None
                try:
                    run_cluster_analysis(
                        run_directory,
                        extended=(
                            extended_cluster_analysis
                            and config.collector
                            != PerformanceCollector.MSPROF.value
                        ),
                        mode=cluster_mode,
                        agent_output=cluster_agent_output,
                        bypass_input_safety_checks=(
                            cluster_bypass_input_safety_checks
                        ),
                        analysis_toolchain=analysis_toolchain,
                        advanced_recipe_policy=advanced_recipe_policy,
                        topology=performance_topologies()[config.topology],
                        record_shapes=preset.record_shapes,
                        mstx_enabled=preset.mstx,
                        with_flops=preset.with_flops,
                        cluster_summary_baseline=cluster_summary_baseline,
                    )
                except BaseException as error:
                    cluster_attempt.update("failed", error=repr(error))
                    raise
                cluster_attempt.update("completed")
        if compare_baseline is not None:
            compare_attempt, should_compare = _prepare_analysis_stage(
                run_directory=run_directory,
                artifact_directory=artifact_directory,
                request=request,
                stage="compare",
                force=force,
            )
            if should_compare:
                assert compare_attempt is not None
                try:
                    run_performance_compare(
                        run_directory,
                        compare_baseline,
                        standard_cli=True,
                        analysis_toolchain=analysis_toolchain,
                    )
                except BaseException as error:
                    compare_attempt.update("failed", error=repr(error))
                    raise
                compare_attempt.update("completed")

        # The report is a replaceable index over all stage directories. It is
        # intentionally rebuilt after every analysis invocation so adding
        # Cluster later makes the existing Advisor visible without rerunning it.
        attempt = RunAttempt.start(
            artifact_directory,
            kind="performance-analysis-report",
            context=request,
            state_name="analysis_state.json",
        )
        profiler_directory = _find_profiler_directory(run_directory)
        render_mindstudio_flamegraphs(profiler_directory)
        render_flamegraphs(profiler_directory)
        analysis = build_analysis(
            run_directory,
            config=manifest.get("config"),
            profiler_environment=manifest.get("profiler_environment"),
        )
        analysis["analysis_toolchain"] = analysis_toolchain
        analysis["analysis_operation"] = request
        handoff_path = artifact_directory / "mindstudio_insight_handoff.json"
        handoff = mindstudio_insight_handoff(
            profiler_directory,
            analysis_root=run_directory,
            portable_base=root,
        )
        _write_json(handoff_path, handoff)
        analysis["mindstudio_insight_handoff"] = {
            "path": str(handoff_path),
            "import_targets": handoff["import_targets"],
            "portable_import_targets": handoff["portable_import_targets"],
        }
        _write_json(artifact_directory / "analysis.json", analysis)
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
    except BaseException as error:
        if "attempt" in locals():
            attempt.update("failed", error=repr(error))
        raise
    attempt.update(
        "completed",
        report=str(report_path),
        analysis=str(artifact_directory / "analysis.json"),
    )
    print(f"Performance report: {report_path}")
    return report_path


def _suite_report_path(
    root: Path,
    config: PerformanceConfig,
    device: str,
    *,
    multiple_presets: bool,
) -> Path:
    suffix = "all-presets-suite" if multiple_presets else "suite"
    return (
        root
        / config.report_root
        / "suites"
        / f"{_slug(config.name)}-{device}-{suffix}.html"
    )


def _write_suite_report(
    output_path: Path,
    reports: list[tuple[str, str, Path]],
) -> None:
    rows = "".join(
        "<tr>"
        f"<td>{topology}</td><td>{preset_name}</td>"
        f'<td><a href="{Path(os.path.relpath(path, output_path.parent)).as_posix()}">'
        f"{path.name}</a></td></tr>"
        for topology, preset_name, path in reports
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        "<title>GLM-5.2 performance suite</title>"
        "<style>body{font:14px/1.55 system-ui;max-width:980px;margin:32px auto;"
        "padding:0 18px;color:#172033}table{border-collapse:collapse;width:100%}"
        "th,td{padding:9px;border-bottom:1px solid #dfe5ef;text-align:left}"
        "th{background:#f8fafc;position:sticky;top:0}</style></head><body>"
        "<h1>GLM-5.2 性能采集套件</h1>"
        "<p>每一行是一组独立的 topology × preset 采集。不同 preset 不混合："
        "浅层吞吐、通信、kernel、算子参数、显存、调用栈和系统数据分别落盘，"
        "避免一次重型采集同时打开所有开关后无法判断测量扰动来源。</p>"
        "<table><thead><tr><th>Topology</th><th>Preset</th><th>Report</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></body></html>",
        encoding="utf-8",
    )


def _profiled_rank_count(preset: ProfilerPreset, world_size: int) -> int:
    """Return how many ranks from this job the preset actually records."""

    value = preset.profile_ranks.strip().lower()
    if value in {"all", "*", "-1"}:
        return world_size
    ranks = {
        int(part.strip())
        for part in value.split(",")
        if part.strip()
    }
    return len({rank for rank in ranks if 0 <= rank < world_size})


def run_profiler_cli(
    config: PerformanceConfig,
    script_path: str,
    *,
    standard_analysis: bool = False,
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
    parser.add_argument(
        "--preset",
        choices=(*tuple(presets), "all"),
        help=(
            "one acquisition policy or all non-redundant policies; all runs "
            "each policy as an independent experiment"
        ),
    )
    parser.add_argument("--visible-devices")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--skip-steps", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--active-steps", type=int)
    parser.add_argument(
        "--collector",
        choices=tuple(collector.value for collector in PerformanceCollector),
        help=(
            "collection entry; msprof and torch_npu_profiler are executable "
            "for training, while other registered MindStudio collectors "
            "raise an audited not-implemented error"
        ),
    )
    parser.add_argument(
        "--collector-arg",
        action="append",
        default=[],
        help=(
            "append one version-specific msprof option before the training "
            "application; use --collector-arg=--option=value"
        ),
    )
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
        "--cluster-mode",
        choices=("all", "communication_time", "communication_matrix"),
        default="all",
        help=(
            "official msprof-analyze cluster mode; all produces both "
            "communication time and communication matrix evidence"
        ),
    )
    parser.add_argument(
        "--cluster-agent-output",
        action="store_true",
        help="request the official cluster JSON summary on stdout",
    )
    parser.add_argument(
        "--cluster-bypass-input-safety-checks",
        action="store_true",
        help=(
            "pass msprof-analyze cluster --force to bypass its ownership, "
            "permission, and large-file checks; unrelated to experiment "
            "lifecycle --force"
        ),
    )
    parser.add_argument(
        "--cluster-recipes",
        default="necessary" if standard_analysis else "none",
        help=(
            "advanced msprof-analyze policy: none, necessary, all, or a "
            "comma-separated official recipe list; MindStudio defaults to "
            "necessary"
        ),
    )
    parser.add_argument(
        "--cluster-summary-baseline",
        type=Path,
        help=(
            "baseline multi-rank profiler root for "
            "cluster_time_compare_summary"
        ),
    )
    parser.add_argument(
        "--analysis-tools",
        choices=("none", "offline", "advisor", "cluster", "all"),
        default="none",
        help=(
            "collector-aware offline analysis policy; msprof all selects "
            "cluster, while torch_npu_profiler all selects offline parsing, "
            "advisor, and multi-rank cluster analysis"
        ),
    )
    parser.add_argument(
        "--profiler-level",
        choices=("level_none", "level0", "level1", "level2"),
        help="override the selected preset's Ascend collection level",
    )
    parser.add_argument(
        "--profile-ranks",
        help="all or comma-separated global ranks to profile",
    )
    parser.add_argument(
        "--aic-metrics",
        choices=(
            "none",
            "pipe_utilization",
            "arithmetic_utilization",
            "memory",
            "memory_l0",
            "memory_ub",
            "resource_conflict_ratio",
            "l2_cache",
            "memory_access",
        ),
    )
    for option, destination, help_text in (
        ("record-shapes", "record_shapes", "operator input shapes and dtypes"),
        ("profile-memory", "profile_memory", "framework and CANN memory"),
        ("with-stack", "with_stack", "Python/framework call stacks"),
        ("with-modules", "with_modules", "module hierarchy"),
        ("with-flops", "with_flops", "raw FLOPs field (not parsed upstream)"),
        ("export-stacks", "export_stacks", "CPU/NPU folded stacks"),
        (
            "export-memory-timeline",
            "export_memory_timeline",
            "official memory timeline HTML and JSON",
        ),
        ("l2-cache", "l2_cache", "L2 cache counters"),
        ("op-attr", "op_attr", "aclnn operator attributes"),
        (
            "data-simplification",
            "data_simplification",
            "delete redundant raw profiling data after parsing",
        ),
        ("record-op-args", "record_op_args", "operator argument statistics"),
        ("msprof-tx", "msprof_tx", "legacy msprof TX range collection"),
        ("mstx", "mstx", "MSTX ranges"),
        ("system-io", "system_io", "NIC and RoCE system I/O"),
        (
            "system-interconnection",
            "system_interconnection",
            "HCCS and PCIe interconnection data",
        ),
    ):
        parser.add_argument(
            f"--{option}",
            dest=destination,
            action=argparse.BooleanOptionalAction,
            default=None,
            help=help_text,
        )
    parser.add_argument("--gc-detect-threshold", type=float)
    parser.add_argument(
        "--host-system",
        help="comma-separated cpu,mem,disk,network,osrt,numa collectors",
    )
    parser.add_argument("--mstx-domain-include")
    parser.add_argument("--mstx-domain-exclude")
    parser.add_argument(
        "--export-types",
        help="comma-separated official parsed formats: text,db",
    )
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
    requested_preset = args.preset or config.preset
    overrides = {
        "device": args.device,
        "topology": requested_topology,
        "preset": None if requested_preset == "all" else requested_preset,
        "steps": args.steps,
        "skip_steps": args.skip_steps,
        "warmup_steps": args.warmup_steps,
        "active_steps": args.active_steps,
        "profiler_enabled": False if args.profiler_off else None,
        "collector": args.collector,
        "collector_args": tuple(args.collector_arg) if args.collector_arg else None,
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
    if effective.profiler_enabled:
        # Reject registered-but-unverified collectors before ``--force`` can
        # clear any prior generation.
        require_training_collector(effective.collector)
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
    if requested_preset == "all" and args.profiler_off:
        parser.error("--preset all cannot be combined with --profiler-off")
    if (
        effective.collector == PerformanceCollector.MSPROF.value
        and requested_preset == "all"
    ):
        parser.error(
            "--collector msprof performs one full-job CANN/NPU capture; "
            "--preset all is specific to scheduled torch_npu_profiler captures"
        )
    if (
        effective.collector == PerformanceCollector.MSPROF.value
        and requested_preset != "overview"
    ):
        parser.error(
            "msprof uses the independent basic collection policy; keep "
            "--preset overview or select --collector torch_npu_profiler"
        )
    if requested_preset == "all" and (args.offline or args.parse_mode):
        parser.error(
            "--preset all keeps each preset's audited parse mode; select one "
            "preset before overriding --parse-mode"
        )
    advanced_overrides: dict[str, Any] = {
        "level": args.profiler_level,
        "profile_ranks": args.profile_ranks,
        "record_shapes": args.record_shapes,
        "profile_memory": args.profile_memory,
        "with_stack": args.with_stack,
        "with_modules": args.with_modules,
        "with_flops": args.with_flops,
        "export_stacks": args.export_stacks,
        "export_memory_timeline": args.export_memory_timeline,
        "aic_metrics": args.aic_metrics,
        "l2_cache": args.l2_cache,
        "op_attr": args.op_attr,
        "data_simplification": args.data_simplification,
        "record_op_args": args.record_op_args,
        "gc_detect_threshold": args.gc_detect_threshold,
        "msprof_tx": args.msprof_tx,
        "mstx": args.mstx,
        "system_io": args.system_io,
        "system_interconnection": args.system_interconnection,
    }
    for field, value in (
        ("host_system", args.host_system),
        ("mstx_domain_include", args.mstx_domain_include),
        ("mstx_domain_exclude", args.mstx_domain_exclude),
        ("export_types", args.export_types),
    ):
        if value is not None:
            advanced_overrides[field] = tuple(
                part.strip().lower()
                for part in value.split(",")
                if part.strip()
            )
    advanced_overrides = {
        key: value for key, value in advanced_overrides.items() if value is not None
    }
    if (
        effective.collector == PerformanceCollector.MSPROF.value
        and advanced_overrides
    ):
        parser.error(
            "advanced --profiler-* overrides belong to torch_npu_profiler; "
            "use the basic msprof launch or select "
            "--collector torch_npu_profiler"
        )
    if requested_preset == "all" and advanced_overrides:
        parser.error(
            "advanced profiler overrides apply to one preset; select a single "
            "--preset instead of all"
        )
    preset_names = (
        all_profiler_presets()
        if requested_preset == "all"
        else (requested_preset,)
    )
    selected_presets: list[tuple[str, ProfilerPreset]] = []
    for preset_name in preset_names:
        preset = presets[preset_name]
        if advanced_overrides:
            preset = replace(preset, **advanced_overrides)
        if args.offline and args.parse_mode not in {None, "offline"}:
            parser.error("--offline conflicts with a non-offline --parse-mode")
        if args.offline:
            preset = replace(preset, parse_mode="offline")
        elif args.parse_mode:
            preset = replace(preset, parse_mode=args.parse_mode)
        selected_presets.append((preset_name, preset))

    analysis_tools = set()
    if args.analysis_tools == "all":
        if effective.collector == PerformanceCollector.MSPROF.value:
            analysis_tools.add("cluster")
        else:
            analysis_tools.update(("offline", "advisor", "cluster"))
    elif args.analysis_tools != "none":
        analysis_tools.add(args.analysis_tools)
    if args.offline_parse:
        analysis_tools.add("offline")
    if args.advisor:
        analysis_tools.add("advisor")
    if args.cluster:
        analysis_tools.add("cluster")
    if (
        args.cluster_mode != "all"
        or args.cluster_agent_output
        or args.cluster_bypass_input_safety_checks
    ) and "cluster" not in analysis_tools:
        parser.error(
            "cluster-specific options require --cluster or an "
            "--analysis-tools policy that includes cluster"
        )
    if (
        effective.collector == PerformanceCollector.MSPROF.value
        and "offline" in analysis_tools
    ):
        parser.error(
            "offline parsing through torch_npu.profiler.analyse is not "
            "applicable to an msprof launch; use cluster or MindStudio "
            "Insight on the msprof output"
        )
    try:
        _validate_analysis_capabilities(
            effective,
            advisor="advisor" in analysis_tools,
            cluster="cluster" in analysis_tools,
            compare_baseline=args.compare_baseline,
        )
    except ValueError as error:
        parser.error(str(error))

    selected = select_topologies(
        available=suite_topologies,
        topology="all" if args.topology == "all" else requested_topology,
        topologies=args.topologies,
        default=(effective.topology,),
    )
    if args.compare_baseline is not None and len(selected) != 1:
        parser.error(
            "--compare-baseline requires exactly one topology; run compare "
            "separately for each topology with its matching baseline"
        )
    if (
        args.force
        and (args.analyze or args.probe)
        and (
            "advisor" in analysis_tools
            or "cluster" in analysis_tools
            or args.compare_baseline is not None
        )
    ):
        # Validate the offline host before deleting any prior derived report.
        _analysis_toolchain_metadata()
    if args.force and (args.capture or args.probe):
        # Clear every selected member before launching the first capture. A
        # mid-suite failure can then be resumed without mixing generations.
        selected_paths: list[Path] = []
        selected_runs: list[Path] = []
        selected_paths.append(
            _suite_report_path(
                root,
                effective,
                device,
                multiple_presets=len(selected_presets) > 1,
            )
        )
        for preset_name, preset in selected_presets:
            for topology_name in selected:
                topology = topologies[topology_name]
                effective_preset = (
                    preset
                    if effective.collector == PerformanceCollector.MSPROF.value
                    else _safe_distributed_parse_preset(
                        preset,
                        device=device,
                        world_size=topology.world_size,
                    )
                )
                topology_config = replace(
                    effective, topology=topology_name, preset=preset_name
                )
                run_name = _run_name(
                    topology_config, device, effective_preset
                )
                run_parent = _scoped_parent(
                    root, topology_config.run_root, topology_config.topology
                )
                artifact_parent = _scoped_parent(
                    root, topology_config.artifact_root, topology_config.topology
                )
                report_parent = _scoped_parent(
                    root, topology_config.report_root, topology_config.topology
                )
                run_directory = run_parent / run_name
                selected_runs.append(run_directory)
                selected_paths.extend(
                    (
                        run_directory,
                        artifact_parent / run_name,
                        report_parent / f"{run_name}.html",
                        _exploration_directory(root, topology_name, run_name),
                    )
                )
        reset_output_generation(
            selected_paths,
            active_run_directories=selected_runs,
            label="performance suite",
        )
    elif args.force and args.analyze:
        reset_output_generation(
            (
                _suite_report_path(
                    root,
                    effective,
                    device,
                    multiple_presets=len(selected_presets) > 1,
                ),
            ),
            label="performance analysis suite index",
            include_archives=False,
        )
    reports: list[tuple[str, str, Path]] = []
    for preset_name, preset in selected_presets:
        for topology_name in selected:
            topology = topologies[topology_name]
            effective_preset = (
                preset
                if effective.collector == PerformanceCollector.MSPROF.value
                else _safe_distributed_parse_preset(
                    preset,
                    device=device,
                    world_size=topology.world_size,
                )
            )
            if effective_preset is not preset:
                print(
                    "Using offline parse for distributed NPU capture: "
                    f"topology={topology_name}, preset={preset_name}; "
                    "synchronous parsing can stall one rank past the "
                    "training process-group timeout.",
                    flush=True,
                )
            topology_config = replace(
                effective, topology=topology_name, preset=preset_name
            )
            run_name = _run_name(topology_config, device, effective_preset)
            action = (
                "probe" if args.probe else "capture" if args.capture else "analyze"
            )
            if args.capture or args.probe:
                _record_exploration_command(
                    root,
                    run_name=run_name,
                    action=action,
                    device=device,
                    topology=topology_name,
                )
            phase = "probe" if args.probe else "capture" if args.capture else "analysis"
            print(
                f"Starting performance {phase}: topology={topology_name} "
                f"({topologies[topology_name].world_size} ranks), "
                f"preset={preset_name}",
                flush=True,
            )
            if args.capture or args.probe:
                capture(
                    root,
                    topology_config,
                    device=device,
                    preset=effective_preset,
                    force=False,
                )
            if args.analyze or args.probe:
                cluster_requested = "cluster" in analysis_tools
                profiled_rank_count = (
                    topology.world_size
                    if effective.collector == PerformanceCollector.MSPROF.value
                    else _profiled_rank_count(
                        effective_preset, topology.world_size
                    )
                )
                cluster_enabled = (
                    cluster_requested and profiled_rank_count > 1
                )
                if cluster_requested and not cluster_enabled:
                    print(
                        "Skipping cluster analysis because this preset captured "
                        f"{profiled_rank_count} rank(s); the tool requires a "
                        "multi-rank capture.",
                        flush=True,
                    )
                report = analyze(
                    root,
                    topology_config,
                    device=device,
                    preset=effective_preset,
                    parse_offline=(
                        (
                            "offline" in analysis_tools
                            and effective_preset.parse_mode == "offline"
                        )
                        or (
                            args.probe
                            and device == "npu"
                            and effective_preset.parse_mode == "offline"
                        )
                    ),
                    parse_workers=args.parse_workers,
                    advisor="advisor" in analysis_tools,
                    cluster=cluster_enabled,
                    compare_baseline=args.compare_baseline,
                    cluster_mode=args.cluster_mode,
                    cluster_agent_output=args.cluster_agent_output,
                    cluster_bypass_input_safety_checks=(
                        args.cluster_bypass_input_safety_checks
                    ),
                    extended_cluster_analysis=not standard_analysis,
                    advanced_recipe_policy=(
                        args.cluster_recipes
                        if standard_analysis
                        and effective.collector
                        == PerformanceCollector.TORCH_NPU_PROFILER.value
                        else "none"
                    ),
                    cluster_summary_baseline=args.cluster_summary_baseline,
                    force=args.force and args.analyze,
                )
                if args.analyze:
                    _record_exploration_command(
                        root,
                        run_name=report.stem,
                        action=action,
                        device=device,
                        topology=topology_name,
                    )
                reports.append((topology_name, preset_name, report))
    if len(reports) > 1:
        index = _suite_report_path(
            root,
            effective,
            device,
            multiple_presets=len(selected_presets) > 1,
        )
        _write_suite_report(index, reports)
        print(f"Performance suite report: {index}")
