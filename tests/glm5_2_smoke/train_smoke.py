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
    nonfinite_diagnostics: bool = False,
    diagnostic_rank: int | str = 6,
    diagnostic_layer: str = "layers.6.attention.inner_attention",
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
        contract["log_rank"] = int(
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
    if nonfinite_diagnostics:
        contract["nonfinite_diagnostics"] = {
            "rank": diagnostic_rank,
            "layer": diagnostic_layer,
            "capture_schema_version": 4,
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
    print(f"Preserved incomplete run: {target}")


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
    nonfinite_diagnostics: bool = False,
    diagnostic_rank: int | str = 6,
    diagnostic_layer: str = "layers.6.attention.inner_attention",
    force: bool,
) -> Path:
    execution = compose_execution(
        topology,
        [graph.feature(device_type="npu" if device == "npu" else "cuda")],
    )
    run_directory = suite_root / topology.slug
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
        nonfinite_diagnostics=nonfinite_diagnostics,
        diagnostic_rank=diagnostic_rank,
        diagnostic_layer=diagnostic_layer,
    )
    if not force and _completed(run_directory, contract):
        print(f"Skip completed topology {topology.name}: {run_directory}")
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
    if nonfinite_diagnostics:
        environment.update(
            {
                "TORCHTITAN_DIAGNOSE_NONFINITE": "1",
                "TORCHTITAN_NONFINITE_CAPTURE_RANK": str(diagnostic_rank),
                "TORCHTITAN_NONFINITE_CAPTURE_LAYER": diagnostic_layer,
                "TORCHTITAN_NONFINITE_DUMP_DIR": str(
                    run_directory / "nonfinite_replay"
                ),
            }
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
    print(f"Runtime log: {runtime_log}")
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
    (run_directory / "manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    attempt.update(
        "completed" if result.returncode == 0 else "failed",
        return_code=result.returncode,
    )
    if nonfinite_diagnostics:
        capture_root = run_directory / "nonfinite_replay"
        capture_directories = _automatic_replay_captures(capture_root)
        if capture_directories:
            replay_log = capture_root / "replay.log"
            replay_status = capture_root / "replay_status.json"
            replay_status.write_text(
                json.dumps(
                    {
                        "status": "starting",
                        "selected_captures": [str(path) for path in capture_directories],
                        "log": str(replay_log),
                        "summary": str(capture_root / "replay_summary.json"),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            replay_command = [
                sys.executable,
                str(Path(__file__).with_name("replay_glm5_flex.py")),
                *(str(path) for path in capture_directories),
                "--device",
                "npu:0",
                "--compact",
            ]
            replay_environment = environment.copy()
            replay_compiler_root = capture_root / "compiler" / "replay"
            replay_environment.update(
                {
                    "TORCH_TRACE": str(replay_compiler_root / "trace"),
                    "TORCH_COMPILE_DEBUG": "1",
                    "TORCH_COMPILE_DEBUG_DIR": str(
                        replay_compiler_root / "debug"
                    ),
                }
            )
            Path(replay_environment["TORCH_TRACE"]).mkdir(parents=True, exist_ok=True)
            Path(replay_environment["TORCH_COMPILE_DEBUG_DIR"]).mkdir(
                parents=True, exist_ok=True
            )
            print(f"Running automatic FlexAttention replay: {replay_log}")
            with replay_log.open("w", encoding="utf-8") as stream:
                replay_result = subprocess.run(
                    replay_command,
                    cwd=root,
                    env=replay_environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
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
                "return_code": replay_result.returncode,
                "log": str(replay_log),
                "capture_count": len(capture_directories),
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
                            "completed" if replay_result.returncode == 0 else "failed"
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
            print(f"FlexAttention replay log: {replay_log}")
            summary_path = capture_root / "replay_summary.json"
            if summary_path.is_file():
                print(f"FlexAttention replay summary: {summary_path}")
    if result.returncode:
        print_runtime_log(runtime_log)
        raise LoggedProcessError(
            result.returncode, command, log_path=runtime_log
        )
    print(f"Passed smoke topology {topology.name}: {run_directory}")
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
    if args.nonfinite_diagnostics:
        suite_name += f"-nonfinite-r{args.diagnostic_rank}"
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
    print(f"Smoke report: {suite_root / 'README.md'}")
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
                nonfinite_diagnostics=args.nonfinite_diagnostics,
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
    print(f"Smoke suite passed: {suite_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
