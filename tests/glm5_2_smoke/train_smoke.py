#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Run short GPU or NPU training jobs across shared topology definitions.

Smoke answers one narrow question: can the current three source checkouts build
the model and complete a few optimizer steps for this device/topology/graph
mode? It records the exact command, dependency check, contract, and full log.
It is not a precision, convergence, performance, or stability claim.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_common.cli import (  # noqa: E402
    LoggedProcessError,
    RunAttempt,
    assert_run_not_active,
    print_output_path,
    print_runtime_log,
    reset_output_generation,
)
from tests.glm5_2_common.topology import (  # noqa: E402
    ParallelTopology,
    select_topologies,
    standard_topologies,
    training_command_args,
)
from tests.glm5_2_common.execution import compose_execution  # noqa: E402
from tests.glm5_2_graph.config import GraphFeatureConfig  # noqa: E402
from tests.glm5_2_graph.visualization import (  # noqa: E402
    generate_graph_compilation_report,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_suite_report(suite_root: Path, results: dict[str, Any]) -> None:
    """Persist the selected invocation, including members not reached yet."""
    suite_root.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": datetime.now(timezone.utc).isoformat(), "results": results}
    (suite_root / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = ["# Smoke run-through report", "",
             "Wall time includes process startup and compilation; this is not a performance benchmark.", "",
             "| Topology | Status | Started (UTC) | Finished (UTC) | Seconds | Log |",
             "|---|---|---|---|---|---|"]
    for name, record in results.items():
        lines.append(
            f"| {name} | {record['status']} | {record.get('started_at', 'unknown')} | "
            f"{record.get('finished_at', 'unknown')} | {record.get('elapsed_seconds', 'unknown')} | "
            f"[runtime.log]({name}/runtime.log) |"
        )
    lines.extend(["", "Full contracts, commands, device visibility and errors: [summary.json](summary.json).", ""])
    (suite_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _check_runtime_dependencies() -> None:
    try:
        import grain.python  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "the current TorchTitan requires grain==0.2.18; install the "
            "TorchTitan dependencies or run: "
            "python -m pip install 'grain==0.2.18'"
        ) from error


def _device(requested: str) -> str:
    if requested != "auto":
        return requested
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        return "npu"
    return "gpu"


def _visible_devices(device: str) -> str:
    variable = (
        "ASCEND_RT_VISIBLE_DEVICES" if device == "npu" else "CUDA_VISIBLE_DEVICES"
    )
    value = os.environ.get(variable, "").strip()
    if not value:
        raise RuntimeError(
            f"export {variable} before running the {device} smoke suite"
        )
    return value


def _default_log_rank(topology: ParallelTopology) -> int:
    """Select a rank that owns the real loss for the topology."""
    if (
        topology.pipeline_parallel_degree == 1
        or topology.pipeline_parallel_schedule == "ZBVZeroBubble"
    ):
        return 0
    ranks_per_pipeline_stage = (
        topology.world_size // topology.pipeline_parallel_degree
    )
    return ranks_per_pipeline_stage * (topology.pipeline_parallel_degree - 1)


def _normalize_log_ranks(value: int | str) -> int | list[int]:
    """Normalize torchrun's scalar or comma-separated local-rank filter."""

    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        raise ValueError("LOG_RANK must contain at least one local rank")
    ranks = [int(part) for part in parts]
    if any(rank < 0 for rank in ranks):
        raise ValueError("LOG_RANK values must be non-negative")
    return ranks[0] if len(ranks) == 1 else ranks


def _distribution_identity(distribution_name: str) -> dict[str, str | None]:
    """Identify installed wheel contents without importing the package."""
    try:
        distribution = importlib_metadata.distribution(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return {"version": None, "record_sha256": None}
    record = distribution.read_text("RECORD")
    normalized_record = "\n".join(
        sorted(
            line
            for line in (record or "").splitlines()
            if "__pycache__" not in line and not line.endswith(".pyc,,")
        )
    )
    return {
        "version": distribution.version,
        "record_sha256": (
            hashlib.sha256(normalized_record.encode("utf-8")).hexdigest()
            if normalized_record
            else None
        ),
    }


def _torch_npu_compiler_tree_sha256() -> str | None:
    try:
        distribution = importlib_metadata.distribution("torch-npu")
    except importlib_metadata.PackageNotFoundError:
        return None
    package_root = Path(distribution.locate_file("torch_npu"))
    roots = (
        package_root / "_inductor",
        package_root / "utils" / "patch_flexattention.py",
    )
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in sorted(files):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        digest.update(path.relative_to(package_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest() if files else None


def _npu_compiler_identity() -> dict[str, Any]:
    identity = {
        "schema_version": 1,
        "torch": _distribution_identity("torch"),
        "torch_npu": _distribution_identity("torch-npu"),
        "triton": _distribution_identity("triton"),
        "ascend_home_path": os.environ.get("ASCEND_HOME_PATH"),
    }
    identity["torch_npu"]["compiler_tree_sha256"] = (
        _torch_npu_compiler_tree_sha256()
    )
    identity["cache_key"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return identity


def _contract(
    *,
    device: str,
    topology: ParallelTopology,
    steps: int,
    local_batch_size: int,
    global_batch_size: int,
    sequence_length: int,
    seed: int,
    module: str,
    config: str,
    graph: GraphFeatureConfig = GraphFeatureConfig(),
    compiler_cache: str = "fresh",
    gradient_diagnostics: bool = False,
    nonfinite_diagnostics: bool = False,
    diagnostic_compiler_cache: str = "per-rank",
    diagnostic_flex_dsdp: bool = False,
    diagnostic_inplace_buffers: str = "default",
    diagnostic_rank: int | str = 6,
    diagnostic_layer: str = "layers.6.attention.inner_attention",
    npu_compiler_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    topology_contract = asdict(topology)
    # Manifests are JSON.  Normalize tuple-valued fields before comparing a
    # freshly built contract with one loaded back from disk.
    topology_contract["extra_args"] = list(topology_contract["extra_args"])
    contract = {
        "schema_version": 1,
        "device": device,
        "topology": topology_contract,
        "steps": steps,
        "local_batch_size": local_batch_size,
        "global_batch_size": global_batch_size,
        "sequence_length": sequence_length,
        "seed": seed,
        "module": module,
        "config": config,
    }
    if topology.pipeline_parallel_degree > 1:
        contract["log_rank"] = _normalize_log_ranks(
            os.environ.get("LOG_RANK", _default_log_rank(topology))
        )
    if graph.mode != "eager":
        contract["graph"] = {
            "mode": graph.mode,
            "components": list(graph.components),
            "diagnostics": graph.diagnostics,
        }
    if graph.npu_codegen or graph.npu_flexattention_mask_mode:
        contract["npu_compiler"] = {
            **({"codegen": graph.npu_codegen} if graph.npu_codegen else {}),
            **(
                {"flexattention_mask_mode": graph.npu_flexattention_mask_mode}
                if graph.npu_flexattention_mask_mode
                else {}
            ),
        }
    if npu_compiler_identity is not None:
        contract.setdefault("npu_compiler", {})["installation"] = (
            npu_compiler_identity
        )
        contract.setdefault("npu_compiler", {})["cache_policy"] = compiler_cache
        contract.setdefault("npu_compiler", {})["cache_scope"] = "run-shared"
    if gradient_diagnostics:
        contract["gradient_diagnostics"] = {
            "schema_version": 1,
            "max_nonfinite_parameters": 32,
        }
    if nonfinite_diagnostics:
        contract["nonfinite_diagnostics"] = {
            "rank": diagnostic_rank,
            "layer": diagnostic_layer,
            "capture_schema_version": 4,
            "compiler_cache": diagnostic_compiler_cache,
            "flex_dsdp": diagnostic_flex_dsdp,
            "inplace_buffers": diagnostic_inplace_buffers,
        }
    return contract


def _completed(path: Path, contract: dict[str, Any]) -> bool:
    manifest = path / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return record.get("status") == "passed" and record.get("contract") == contract


def _preserve_failed_run(path: Path) -> None:
    if not path.exists():
        return
    assert_run_not_active(path)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = path.with_name(f"{path.name}.failed-{stamp}")
    suffix = 1
    while target.exists():
        target = path.with_name(f"{path.name}.failed-{stamp}-{suffix}")
        suffix += 1
    path.rename(target)
    print_output_path("Preserved incomplete run", target)


def _automatic_replay_captures(capture_root: Path) -> list[Path]:
    """Select the call with the largest observed Q/K gradient per rank."""
    selected = []
    for rank_directory in sorted(capture_root.glob("rank*")):
        calls = sorted(rank_directory.glob("call*"))
        if not calls:
            continue
        representative = calls[0]
        representative_score = -1.0
        for call_directory in calls:
            statistics_path = call_directory / "actual_gradients.json"
            if not statistics_path.is_file():
                continue
            try:
                statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            score = 0.0
            for name in ("dq_QNH", "dk_KNH"):
                gradient = statistics.get(name, {})
                if gradient.get("finite_count") != gradient.get("numel"):
                    score = float("inf")
                    break
                score = max(score, float(gradient.get("max_abs") or 0.0))
            if score > representative_score:
                representative = call_directory
                representative_score = score
        selected.append(representative)
    return selected


def _capture_gradient_score(capture_directory: Path) -> float:
    statistics_path = capture_directory / "actual_gradients.json"
    if not statistics_path.is_file():
        return -1.0
    try:
        statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return -1.0
    score = 0.0
    for name in ("dq_QNH", "dk_KNH"):
        gradient = statistics.get(name, {})
        if gradient.get("finite_count") != gradient.get("numel"):
            return float("inf")
        score = max(score, float(gradient.get("max_abs") or 0.0))
    return score


def _capture_rank(capture_directory: Path) -> int:
    name = capture_directory.parent.name
    if not name.startswith("rank") or not name[4:].isdigit():
        raise ValueError(
            f"cannot determine rank from capture path: {capture_directory}"
        )
    return int(name[4:])


def _run_device_replays(
    *,
    root: Path,
    capture_root: Path,
    capture_directories: list[Path],
    environment: dict[str, str],
    visible_devices: str,
) -> dict[str, object]:
    """Replay the worst capture on its origin device and device zero."""
    worst_capture = max(capture_directories, key=_capture_gradient_score)
    worst_rank = _capture_rank(worst_capture)
    jobs = [(f"rank{worst_rank}-origin", worst_capture, worst_rank)]
    if worst_rank != 0:
        jobs.append((f"rank{worst_rank}-control-device0", worst_capture, 0))

    replay_root = capture_root / "device_replays"
    replay_root.mkdir(parents=True, exist_ok=True)
    records = []
    for name, capture_directory, logical_device in jobs:
        job_root = replay_root / name
        # Both jobs intentionally reuse one compiled kernel. Changing only the
        # execution device isolates a physical-device/runtime effect without
        # paying for or introducing another autotune decision.
        compiler_root = capture_root / "compiler" / "targeted_replay"
        replay_log = job_root / "replay.log"
        summary_path = job_root / "replay_summary.json"
        job_root.mkdir(parents=True, exist_ok=True)
        replay_environment = environment.copy()
        replay_environment.update(
            {
                "TORCH_TRACE": str(compiler_root / "trace"),
                "TORCH_COMPILE_DEBUG": "1",
                "TORCH_COMPILE_DEBUG_DIR": str(compiler_root / "debug"),
                "TORCHINDUCTOR_CACHE_DIR": str(compiler_root / "cache" / "inductor"),
                "TRITON_CACHE_DIR": str(compiler_root / "cache" / "triton"),
            }
        )
        for variable in (
            "TORCH_TRACE",
            "TORCH_COMPILE_DEBUG_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
        ):
            Path(replay_environment[variable]).mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).with_name("replay_glm5_flex.py")),
            str(capture_directory),
            "--device",
            f"npu:{logical_device}",
            "--visible-devices",
            visible_devices,
            "--result-directory",
            str(job_root),
            "--summary-path",
            str(summary_path),
            "--compact",
        ]
        print(
            f"Running FlexAttention replay {name} on npu:{logical_device}",
            flush=True,
        )
        print_output_path("FlexAttention replay log", replay_log)
        with replay_log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                command,
                cwd=root,
                env=replay_environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        result_path = job_root / "replay_result.json"
        replay_result = None
        if result_path.is_file():
            try:
                replay_result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        records.append(
            {
                "name": name,
                "capture": str(capture_directory),
                "capture_rank": _capture_rank(capture_directory),
                "logical_device": logical_device,
                "physical_device": visible_devices.split(",")[logical_device].strip(),
                "return_code": result.returncode,
                "log": str(replay_log),
                "summary": str(summary_path),
                "result": str(result_path),
                "replay_result": replay_result,
            }
        )
        partial_summary = {
            "status": "running",
            "visible_devices": visible_devices,
            "worst_capture": str(worst_capture),
            "worst_capture_rank": worst_rank,
            "jobs": records,
        }
        (capture_root / "device_replay_summary.json").write_text(
            json.dumps(partial_summary, indent=2) + "\n", encoding="utf-8"
        )
    origin = next(
        (job for job in records if job["name"] == f"rank{worst_rank}-origin"),
        None,
    )
    control = next(
        (
            job
            for job in records
            if job["name"] == f"rank{worst_rank}-control-device0"
        ),
        origin if worst_rank == 0 else None,
    )

    def qk_max(job):
        result_payload = job.get("replay_result") if job else None
        if not result_payload:
            return None
        return max(
            float(result_payload[name].get("max_abs") or 0.0)
            for name in ("dq", "dk")
        )

    origin_qk_max = qk_max(origin)
    control_qk_max = qk_max(control)
    captured_qk_max = _capture_gradient_score(worst_capture)
    if origin_qk_max is None or control_qk_max is None:
        diagnosis = "incomplete_replay"
    elif origin_qk_max > max(control_qk_max * 1000.0, 1.0):
        diagnosis = "origin_device_or_origin_compiler_choice"
    elif origin_qk_max > 1.0 and control_qk_max > 1.0:
        diagnosis = "reproducible_input_or_kernel_path"
    elif captured_qk_max > max(origin_qk_max * 1000.0, 1.0):
        diagnosis = "full_cp_runtime_only"
    else:
        diagnosis = "no_large_replay_divergence"
    summary = {
        "status": "completed",
        "visible_devices": visible_devices,
        "worst_capture": str(worst_capture),
        "worst_capture_rank": worst_rank,
        "captured_worst_qk_max_abs": captured_qk_max,
        "origin_replay_qk_max_abs": origin_qk_max,
        "control_replay_qk_max_abs": control_qk_max,
        "diagnosis": diagnosis,
        "jobs": records,
    }
    summary_path = capture_root / "device_replay_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (capture_root / "replay_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print_output_path("FlexAttention device replay summary", summary_path)
    return summary


def _run_topology(
    *,
    root: Path,
    suite_root: Path,
    device: str,
    visible_devices: str,
    topology: ParallelTopology,
    steps: int,
    local_batch_size: int,
    global_batch_size: int,
    sequence_length: int,
    seed: int,
    module: str,
    config: str,
    graph: GraphFeatureConfig = GraphFeatureConfig(),
    compiler_cache: str = "fresh",
    gradient_diagnostics: bool = False,
    nonfinite_diagnostics: bool = False,
    diagnostic_compiler_cache: str = "per-rank",
    diagnostic_flex_dsdp: bool = False,
    diagnostic_inplace_buffers: str = "default",
    diagnostic_rank: int | str = 6,
    diagnostic_layer: str = "layers.6.attention.inner_attention",
    force: bool,
) -> Path:
    execution = compose_execution(
        topology,
        [graph.feature(device_type="npu" if device == "npu" else "cuda")],
    )
    run_directory = suite_root / topology.slug
    npu_compiler_identity = _npu_compiler_identity() if device == "npu" else None
    contract = _contract(
        device=device,
        topology=topology,
        steps=steps,
        local_batch_size=local_batch_size,
        global_batch_size=global_batch_size,
        sequence_length=sequence_length,
        seed=seed,
        module=module,
        config=config,
        graph=graph,
        compiler_cache=compiler_cache,
        gradient_diagnostics=gradient_diagnostics,
        nonfinite_diagnostics=nonfinite_diagnostics,
        diagnostic_compiler_cache=diagnostic_compiler_cache,
        diagnostic_flex_dsdp=diagnostic_flex_dsdp,
        diagnostic_inplace_buffers=diagnostic_inplace_buffers,
        diagnostic_rank=diagnostic_rank,
        diagnostic_layer=diagnostic_layer,
        npu_compiler_identity=npu_compiler_identity,
    )
    if not force and _completed(run_directory, contract):
        print_output_path(f"Skip completed topology {topology.name}", run_directory)
        return run_directory
    if force and run_directory.exists():
        reset_output_generation(
            (run_directory,),
            active_run_directories=(run_directory,),
            label="smoke",
        )
    else:
        _preserve_failed_run(run_directory)
    run_directory.mkdir(parents=True)
    attempt = RunAttempt.start(
        run_directory,
        kind="smoke",
        context={
            "device": device,
            "topology": topology.slug,
            "steps": steps,
        },
    )

    runtime_log = run_directory / "runtime.log"
    command = [
        "bash",
        str(root / "run_train.sh"),
        f"--dump_folder={run_directory / 'trainer_output'}",
        f"--training.steps={steps}",
        "--training.disable_cuda_graphs",
        *training_command_args(
            local_batch_size=local_batch_size,
            global_batch_size=global_batch_size,
            sequence_length=sequence_length,
            topology=topology,
        ),
        f"--debug.seed={seed}",
        "--metrics.log_freq=1",
        *topology.command_args(),
        *execution.command_args(),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "TORCHTITAN_DEVICE": device,
            "TORCHTITAN_RUN_LOG": str(runtime_log),
            "NGPU": str(topology.world_size),
            "LOG_RANK": environment.get(
                "LOG_RANK", str(_default_log_rank(topology))
            ),
            "MODULE": module,
            "CONFIG": config,
        }
    )
    environment.update(execution.environment())
    if graph.diagnostics:
        environment["GLM5_EXPERIMENT_RUN_DIRECTORY"] = str(
            run_directory.resolve()
        )
    if npu_compiler_identity is not None:
        if compiler_cache == "fresh":
            compiler_cache_root = run_directory / "compiler_cache"
        else:
            compiler_cache_root = (
                suite_root
                / ".compiler_cache"
                / str(npu_compiler_identity["cache_key"])
                / topology.name
            )
        compiler_cache_root.mkdir(parents=True, exist_ok=True)
        environment["TORCHTITAN_COMPILER_CACHE_ROOT"] = str(compiler_cache_root)
    if gradient_diagnostics:
        environment.update(
            {
                "TORCHTITAN_DIAGNOSE_GRADIENTS": "1",
                "TORCHTITAN_GRADIENT_DIAGNOSTIC_DIR": str(
                    run_directory / "gradient_diagnostics"
                ),
            }
        )
    if nonfinite_diagnostics:
        environment.update(
            {
                "TORCHTITAN_DIAGNOSE_NONFINITE": "1",
                "TORCHTITAN_NONFINITE_CAPTURE_RANK": str(diagnostic_rank),
                "TORCHTITAN_NONFINITE_CAPTURE_LAYER": diagnostic_layer,
                "TORCHTITAN_NONFINITE_DUMP_DIR": str(
                    run_directory / "nonfinite_replay"
                ),
                "TORCHTITAN_NONFINITE_COMPILER_CACHE": diagnostic_compiler_cache,
            }
        )
        if diagnostic_flex_dsdp:
            environment.update(
                {
                    "TORCHNPU_FLEXATTENTION_DSDP_DIAGNOSTICS": "1",
                    "TORCHNPU_FLEXATTENTION_DSDP_DIAGNOSTIC_RANK": str(
                        diagnostic_rank
                    ),
                    "TRITON_DEVICE_PRINT": "1",
                }
            )
        if diagnostic_inplace_buffers != "default":
            environment["ENABLE_INPLACE_BUFFERS"] = (
                "1" if diagnostic_inplace_buffers == "enabled" else "0"
            )
    visible_variable = (
        "ASCEND_RT_VISIBLE_DEVICES"
        if device == "npu"
        else "CUDA_VISIBLE_DEVICES"
    )
    environment[visible_variable] = visible_devices

    print(
        f"Starting smoke topology: {topology.name}, device={device}, "
        f"world_size={topology.world_size}"
    )
    print_runtime_log(runtime_log)
    started_at = datetime.now(timezone.utc).isoformat()
    started_clock = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=root, env=environment, check=False)
    except (OSError, KeyboardInterrupt) as error:
        record = {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "contract": contract, "command": command,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started_clock,
            "visible_devices": visible_devices,
            "error": repr(error), "attempt_id": attempt.attempt_id,
        }
        (run_directory / "manifest.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        attempt.update("failed")
        print_runtime_log(runtime_log)
        raise
    record = {
        "status": "passed" if result.returncode == 0 else "failed",
        "return_code": result.returncode,
        "contract": contract,
        "command": command,
        "runtime_log": str(runtime_log),
        "attempt_id": attempt.attempt_id,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started_clock,
        "visible_devices": visible_devices,
        "error": None if result.returncode == 0 else f"Training exited with code {result.returncode}; see runtime.log",
    }
    if graph.diagnostics:
        graph_report = generate_graph_compilation_report(run_directory)
        record["graph_compilation_report"] = {
            "json": graph_report["json"],
            "html": graph_report["html"],
            "totals": graph_report["totals"],
        }
        print_output_path(
            "Graph compilation report",
            Path(graph_report["html"]),
        )
    (run_directory / "manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    attempt.update(
        "completed" if result.returncode == 0 else "failed",
        return_code=result.returncode,
    )
    if nonfinite_diagnostics and result.returncode:
        capture_root = run_directory / "nonfinite_replay"
        capture_directories = _automatic_replay_captures(capture_root)
        if capture_directories:
            replay_status = capture_root / "replay_status.json"
            replay_status.write_text(
                json.dumps(
                    {
                        "status": "starting",
                        "selected_captures": [str(path) for path in capture_directories],
                        "log": str(capture_root / "device_replay_summary.json"),
                        "summary": str(capture_root / "replay_summary.json"),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            device_replays = _run_device_replays(
                root=root,
                capture_root=capture_root,
                capture_directories=capture_directories,
                environment=environment,
                visible_devices=visible_devices,
            )
            comparison_log = capture_root / "compiler_comparison.log"
            comparison_command = [
                sys.executable,
                str(
                    Path(__file__).with_name(
                        "analyze_flex_compiler_artifacts.py"
                    )
                ),
                str(capture_root),
            ]
            with comparison_log.open("w", encoding="utf-8") as stream:
                comparison_result = subprocess.run(
                    comparison_command,
                    cwd=root,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            record["nonfinite_replay"] = {
                "return_code": max(
                    (job["return_code"] for job in device_replays["jobs"]),
                    default=0,
                ),
                "log": str(capture_root / "device_replay_summary.json"),
                "capture_count": len(capture_directories),
                "device_replays": device_replays,
                "compiler_comparison_return_code": comparison_result.returncode,
                "compiler_comparison_log": str(comparison_log),
                "compiler_comparison": str(
                    capture_root / "compiler_comparison.json"
                ),
            }
            replay_status.write_text(
                json.dumps(
                    {
                        "status": (
                            "completed"
                            if record["nonfinite_replay"]["return_code"] == 0
                            else "failed"
                        ),
                        **record["nonfinite_replay"],
                        "summary": str(capture_root / "replay_summary.json"),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (run_directory / "manifest.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print_output_path(
                "FlexAttention device replay summary",
                capture_root / "device_replay_summary.json",
            )
            summary_path = capture_root / "replay_summary.json"
            if summary_path.is_file():
                print_output_path("FlexAttention replay summary", summary_path)
    if result.returncode:
        print_runtime_log(runtime_log)
        raise LoggedProcessError(
            result.returncode, command, log_path=runtime_log
        )
    print_output_path(f"Passed smoke topology {topology.name}", run_directory)
    print_runtime_log(runtime_log)
    return run_directory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "gpu", "npu"), default="auto")
    topology_selection = parser.add_mutually_exclusive_group()
    topology_selection.add_argument("--topology")
    topology_selection.add_argument("--topologies")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--local-batch-size", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--module", default="glm5")
    parser.add_argument("--config", default="glm5_debugmodel")
    parser.add_argument(
        "--graph",
        choices=("eager", "inductor", "npugraphs"),
        default="eager",
    )
    parser.add_argument(
        "--compile-loss",
        action="store_true",
        help="compile the loss together with the model",
    )
    parser.add_argument(
        "--compiler-diagnostics",
        action="store_true",
        help="enable graph-break, recompile, and dynamic-shape diagnostics",
    )
    parser.add_argument(
        "--compiler-cache",
        choices=("fresh", "reuse"),
        default="fresh",
        help=(
            "use one fresh run-local cache shared by ranks by default, or "
            "explicitly reuse a persistent cache for the same compiler "
            "installation and topology"
        ),
    )
    parser.add_argument(
        "--gradient-diagnostics",
        action="store_true",
        help=(
            "record compact per-rank non-finite parameter-gradient summaries; "
            "does not enable FlexAttention capture or device replay"
        ),
    )
    parser.add_argument(
        "--nonfinite-diagnostics",
        action="store_true",
        help=(
            "capture GLM FlexAttention inputs, the forward-output lifetime "
            "check, and a reference backward DELTA"
        ),
    )
    parser.add_argument(
        "--diagnostic-rank",
        default="6",
        help="global rank to capture, or 'all' to compare every rank",
    )
    parser.add_argument(
        "--diagnostic-layer",
        default="layers.6.attention.inner_attention",
    )
    parser.add_argument(
        "--diagnostic-compiler-cache",
        choices=("shared", "per-rank"),
        default="per-rank",
        help="use a shared or rank-local Inductor/Triton cache for diagnostics",
    )
    parser.add_argument(
        "--diagnostic-flex-dsdp",
        action="store_true",
        help="print abnormal DELTA/dP/dS values inside NPU FlexAttention kernels",
    )
    parser.add_argument(
        "--diagnostic-inplace-buffers",
        choices=("default", "enabled", "disabled"),
        default="default",
        help=(
            "control Inductor buffer reuse for the FlexAttention non-finite "
            "A/B; recorded in the run identity and manifest"
        ),
    )
    parser.add_argument("--force", action="store_true")
    from tests.glm5_2_graph.config import (
        add_npu_codegen_argument,
        add_npu_flexattention_argument,
    )
    add_npu_codegen_argument(parser)
    add_npu_flexattention_argument(parser)
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.diagnostic_rank != "all":
        try:
            args.diagnostic_rank = int(args.diagnostic_rank)
        except ValueError:
            parser.error("--diagnostic-rank must be a non-negative integer or 'all'")
        if args.diagnostic_rank < 0:
            parser.error("--diagnostic-rank must be non-negative")
    device = _device(args.device)
    if args.nonfinite_diagnostics and device != "npu":
        parser.error("--nonfinite-diagnostics is available only for NPU runs")
    if args.gradient_diagnostics and device != "npu":
        parser.error("--gradient-diagnostics is available only for NPU runs")
    if args.diagnostic_flex_dsdp and not args.nonfinite_diagnostics:
        parser.error("--diagnostic-flex-dsdp requires --nonfinite-diagnostics")
    if (
        args.diagnostic_inplace_buffers != "default"
        and not args.nonfinite_diagnostics
    ):
        parser.error(
            "--diagnostic-inplace-buffers requires --nonfinite-diagnostics"
        )
    visible_devices = _visible_devices(device)
    graph = GraphFeatureConfig(
        mode=args.graph,
        components=("model", "loss") if args.compile_loss else ("model",),
        diagnostics=args.compiler_diagnostics,
        npu_codegen=args.npu_codegen,
        npu_flexattention_mask_mode=args.npu_flexattention_mask_mode,
    )
    graph.feature(device_type="npu" if device == "npu" else "cuda")
    _check_runtime_dependencies()
    topologies = standard_topologies()
    available = tuple(
        name for name, topology in topologies.items() if topology.world_size <= 8
    )
    selected = select_topologies(
        available=available,
        topology=args.topology,
        topologies=args.topologies,
        default=("single",),
    )
    num_visible_devices = len(
        [value for value in visible_devices.split(",") if value.strip()]
    )
    for name in selected:
        local_world_size = topologies[name].world_size
        if num_visible_devices < local_world_size:
            parser.error(
                f"topology {name} needs {local_world_size} visible {device} "
                f"devices, but only {num_visible_devices} were exported"
            )
        if (
            args.nonfinite_diagnostics
            and args.diagnostic_rank != "all"
            and args.diagnostic_rank >= local_world_size
        ):
            parser.error(
                f"--diagnostic-rank={args.diagnostic_rank} is outside topology "
                f"{name} with world size {local_world_size}"
            )
    suite_name = (
        f"{device}-{args.config}-s{args.steps}-b{args.global_batch_size}-"
        f"seq{args.sequence_length}-seed{args.seed}"
    )
    if graph.mode != "eager":
        suite_name += f"-{graph.mode}-{'-'.join(graph.components)}"
        if graph.diagnostics:
            suite_name += "-diag"
    if graph.npu_codegen:
        suite_name += f"-{graph.npu_codegen}"
    if graph.npu_flexattention_mask_mode:
        suite_name += f"-flex-{graph.npu_flexattention_mask_mode}"
    if device == "npu" and args.compiler_cache == "reuse":
        suite_name += "-cache-reuse"
    if args.gradient_diagnostics:
        suite_name += "-gradient-diag"
    if args.nonfinite_diagnostics:
        suite_name += f"-nonfinite-r{args.diagnostic_rank}"
        if args.diagnostic_compiler_cache == "shared":
            suite_name += "-cache-shared"
        if args.diagnostic_flex_dsdp:
            suite_name += "-dsdp"
        if args.diagnostic_inplace_buffers != "default":
            suite_name += f"-inplace-{args.diagnostic_inplace_buffers}"
    root = _root()
    suite_root = root / "smoke_runs" / suite_name
    if args.force:
        # Reset the complete selected generation before launching its first
        # topology, so a later resume cannot mix old and new suite members.
        reset_output_generation(
            [suite_root / topologies[name].slug for name in selected],
            active_run_directories=[
                suite_root / topologies[name].slug for name in selected
            ],
            label="smoke suite",
        )
    results = {topologies[name].slug: {"status": "not_run"} for name in selected}
    _write_suite_report(suite_root, results)
    print_output_path("Smoke report", suite_root / "README.md")
    for name in selected:
        slug = topologies[name].slug
        try:
            _run_topology(
                root=root,
                suite_root=suite_root,
                device=device,
                visible_devices=visible_devices,
                topology=topologies[name],
                steps=args.steps,
                local_batch_size=args.local_batch_size,
                global_batch_size=args.global_batch_size,
                sequence_length=args.sequence_length,
                seed=args.seed,
                module=args.module,
                config=args.config,
                graph=graph,
                compiler_cache=args.compiler_cache,
                gradient_diagnostics=args.gradient_diagnostics,
                nonfinite_diagnostics=args.nonfinite_diagnostics,
                diagnostic_compiler_cache=args.diagnostic_compiler_cache,
                diagnostic_flex_dsdp=args.diagnostic_flex_dsdp,
                diagnostic_inplace_buffers=args.diagnostic_inplace_buffers,
                diagnostic_rank=args.diagnostic_rank,
                diagnostic_layer=args.diagnostic_layer,
                force=False,
            )
        finally:
            manifest = suite_root / slug / "manifest.json"
            if manifest.is_file():
                results[slug] = json.loads(manifest.read_text(encoding="utf-8"))
            else:
                results[slug] = {"status": "incomplete"}
            _write_suite_report(suite_root, results)
    print_output_path("Smoke suite passed", suite_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
