#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Run short GPU or NPU training jobs across shared topology definitions."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_common.cli import reset_output_generation  # noqa: E402
from tests.glm5_2_common.execution import compose_execution  # noqa: E402
from tests.glm5_2_common.full_dsa import (  # noqa: E402
    FULL_DSA_KERNELS,
    REFERENCE_KERNEL,
    full_dsa_override_targets,
    merge_override_imports,
    resolve_full_dsa_config,
)
from tests.glm5_2_common.topology import (  # noqa: E402
    ParallelTopology,
    select_topologies,
    standard_topologies,
    training_command_args,
)
from tests.glm5_2_graph.config import GraphFeatureConfig  # noqa: E402


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


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
    override_imports: str | None = None,
    extra_args: tuple[str, ...] = (),
    graph: GraphFeatureConfig = GraphFeatureConfig(),
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
    if extra_args:
        contract["extra_args"] = list(extra_args)
    if graph.mode != "eager":
        contract["graph"] = {
            "mode": graph.mode,
            "components": list(graph.components),
            "diagnostics": graph.diagnostics,
        }
    if override_imports:
        contract["override_imports"] = override_imports
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
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = path.with_name(f"{path.name}.failed-{stamp}")
    suffix = 1
    while target.exists():
        target = path.with_name(f"{path.name}.failed-{stamp}-{suffix}")
        suffix += 1
    path.rename(target)
    print(f"Preserved incomplete run: {target}")


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
    override_imports: str | None = None,
    extra_args: tuple[str, ...] = (),
    graph: GraphFeatureConfig = GraphFeatureConfig(),
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
        override_imports=override_imports,
        extra_args=extra_args,
        graph=graph,
    )
    if not force and _completed(run_directory, contract):
        print(f"Skip completed topology {topology.name}: {run_directory}")
        return run_directory
    if force and run_directory.exists():
        shutil.rmtree(run_directory)
    else:
        _preserve_failed_run(run_directory)
    run_directory.mkdir(parents=True)

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
        *extra_args,
    ]
    if override_imports:
        command.append(f"--override.imports={override_imports}")
    environment = os.environ.copy()
    environment.update(
        {
            "TORCHTITAN_DEVICE": device,
            "TORCHTITAN_RUN_LOG": str(runtime_log),
            "NGPU": str(topology.world_size),
            "LOG_RANK": environment.get("LOG_RANK", "0"),
            "MODULE": module,
            "CONFIG": config,
        }
    )
    environment.update(execution.environment())
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
    result = subprocess.run(command, cwd=root, env=environment, check=False)
    record = {
        "status": "passed" if result.returncode == 0 else "failed",
        "return_code": result.returncode,
        "contract": contract,
        "command": command,
        "runtime_log": str(runtime_log),
    }
    (run_directory / "manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    print(f"Passed smoke topology {topology.name}: {run_directory}")
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
    parser.add_argument("--config")
    parser.add_argument(
        "--full-dsa",
        action="store_true",
        help="select glm5_full_dsa_debugmodel without changing default smoke runs",
    )
    parser.add_argument(
        "--full-dsa-kernel",
        choices=FULL_DSA_KERNELS,
        default=REFERENCE_KERNEL,
        help="optional Full DSA implementation; auto selects by device",
    )
    parser.add_argument(
        "--override-imports",
        help="comma-separated explicit TorchTitan component overrides",
    )
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
        "--extra-train-arg",
        action="append",
        default=[],
        help="append one raw TorchTitan training argument",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be positive")
    device = _device(args.device)
    visible_devices = _visible_devices(device)
    if args.full_dsa_kernel != REFERENCE_KERNEL and not args.full_dsa:
        parser.error("--full-dsa-kernel requires --full-dsa")
    try:
        config = resolve_full_dsa_config(
            enabled=args.full_dsa,
            requested_config=args.config,
        )
        override_imports = merge_override_imports(
            args.override_imports,
            full_dsa_override_targets(
                device_type=device,
                kernel=args.full_dsa_kernel,
            )
            if args.full_dsa
            else (),
        )
    except ValueError as error:
        parser.error(str(error))
    extra_args = tuple(args.extra_train_arg)
    graph = GraphFeatureConfig(
        mode=args.graph,
        components=("model", "loss") if args.compile_loss else ("model",),
        diagnostics=args.compiler_diagnostics,
    )
    graph.feature(device_type="npu" if device == "npu" else "cuda")
    _check_runtime_dependencies()
    topologies = standard_topologies()
    available = tuple(
        name
        for name, topology in topologies.items()
        if topology.world_size <= 8
        and (not args.full_dsa or topology.pipeline_parallel_degree == 1)
    )
    try:
        selected = select_topologies(
            available=available,
            topology=args.topology,
            topologies=args.topologies,
            default=("single",),
        )
    except ValueError as error:
        parser.error(str(error))
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
    suite_name = (
        f"{device}-{config}-s{args.steps}-b{args.global_batch_size}-"
        f"seq{args.sequence_length}-seed{args.seed}"
    )
    if graph.mode != "eager":
        suite_name += f"-{graph.mode}-{'-'.join(graph.components)}"
        if graph.diagnostics:
            suite_name += "-diag"
    if override_imports:
        override_names = "-".join(
            target.rsplit(".", 1)[-1]
            for target in override_imports.split(",")
        )
        suite_name += f"-override-{override_names}"
    if extra_args:
        options_digest = hashlib.sha256(
            "\0".join(extra_args).encode("utf-8")
        ).hexdigest()[:8]
        suite_name += f"-opts-{options_digest}"
    root = _root()
    suite_root = root / "smoke_runs" / suite_name
    if args.force:
        # Reset the whole selected generation before launching its first
        # topology. This prevents a failed all-topology rerun from later
        # reusing untouched members from the previous generation.
        reset_output_generation(
            [suite_root / topologies[name].slug for name in selected]
        )
    for name in selected:
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
            config=config,
            override_imports=override_imports,
            extra_args=extra_args,
            graph=graph,
            force=False,
        )
    print(f"Smoke suite passed: {suite_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
