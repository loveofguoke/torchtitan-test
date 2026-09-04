# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Independent official msOpProf and msMemScope orchestration.

These tools intentionally do not share a capture with system profiling:
msOpProf profiles selected operator launches, while msMemScope hooks memory
events.  The launcher preserves each tool's native output tree under the run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any, Sequence

from tests.glm5_2_common.cli import RunAttempt, reset_output_generation
from tests.glm5_2_common.naming import config_name
from tests.glm5_2_common.topology import (
    ParallelTopology,
    select_topologies,
    standard_topologies,
    training_command_args,
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _run_logged(command: Sequence[str], *, log: Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    normalized = [str(part) for part in command]
    rendered = shlex.join(normalized)
    print(f"Runtime log: {log}")
    print(f"Command: {rendered}")
    with log.open("w", encoding="utf-8") as stream:
        stream.write(f"Command: {rendered}\n\n")
        stream.flush()
        process = subprocess.Popen(
            normalized,
            cwd=repository_root(),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            stream.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, normalized)


def _tool_version(executable: str) -> str:
    path = shutil.which(executable)
    if path is None:
        raise FileNotFoundError(f"required official tool is not on PATH: {executable}")
    result = subprocess.run(
        [path, "--version"], capture_output=True, text=True, check=False
    )
    output = (result.stdout or result.stderr).strip()
    return output or f"{path} (version output unavailable)"


def _write_manifest(
    directory: Path,
    *,
    kind: str,
    contract: dict[str, Any],
    output: Path,
) -> None:
    files = sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    )
    payload = {
        "schema": "torchtitan.glm5_2.mindstudio_specialized_tuning",
        "schema_version": 1,
        "kind": kind,
        "status": "completed",
        "contract": contract,
        "official_output": output.relative_to(directory).as_posix(),
        "official_files": files,
    }
    (directory / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _completed(directory: Path, contract: dict[str, Any]) -> bool:
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    official = directory / str(value.get("official_output", "missing"))
    return (
        value.get("status") == "completed"
        and value.get("contract") == contract
        and official.is_dir()
        and any(official.iterdir())
    )


def operator_command(args: argparse.Namespace, output: Path) -> list[str]:
    command = [args.executable]
    if args.mode == "simulator":
        command.append("simulator")
        if args.soc_version:
            command.append(f"--soc-version={args.soc_version}")
    command.extend([f"--output={output}", f"--aic-metrics={args.metrics}"])
    if args.kernel_name:
        command.append(f"--kernel-name={args.kernel_name}")
    command.append(f"--launch-count={args.launch_count}")
    if args.launch_skip:
        command.append(f"--launch-skip-before-match={args.launch_skip}")
    if args.kill:
        command.append("--kill=on")
    if args.core_id:
        command.append(f"--core-id={args.core_id}")
    if args.dump:
        command.append("--dump=on")
    command.extend(args.tool_arg)
    if args.config_file:
        command.append(f"--config={Path(args.config_file).resolve()}")
    else:
        application = list(args.application)
        if application and application[0] == "--":
            application = application[1:]
        if not application:
            raise ValueError("msOpProf requires --config-file or an application after --")
        command.extend(application)
    return command


def run_operator_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Official standalone msOpProf operator tuning workflow"
    )
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--name", required=True, help="stable operator experiment name")
    parser.add_argument("--mode", choices=("onboard", "simulator"), default="onboard")
    parser.add_argument("--executable", default="msopprof")
    parser.add_argument("--soc-version")
    parser.add_argument("--metrics", default="Default")
    parser.add_argument("--kernel-name")
    parser.add_argument("--launch-count", type=int, default=1)
    parser.add_argument("--launch-skip", type=int, default=0)
    parser.add_argument("--kill", action="store_true")
    parser.add_argument("--core-id")
    parser.add_argument("--dump", action="store_true")
    parser.add_argument("--tool-arg", action="append", default=[])
    parser.add_argument("--config-file")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("application", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.probe:
        parser.error("select --probe")
    if args.launch_count < 1 or args.launch_count > 5000:
        parser.error("--launch-count must be in [1, 5000]")
    version = "unverified-dry-run" if args.dry_run else _tool_version(args.executable)
    contract = {
        "kind": "operator",
        "name": args.name,
        "mode": args.mode,
        "version": version,
        "metrics": args.metrics,
        "kernel_name": args.kernel_name,
        "launch_count": args.launch_count,
        "launch_skip": args.launch_skip,
        "kill": args.kill,
        "core_id": args.core_id,
        "dump": args.dump,
        "soc_version": args.soc_version,
        "tool_arg": args.tool_arg,
        "config_file": args.config_file,
        "application": args.application,
    }
    identity = config_name(f"npu-{args.name}-{args.mode}", contract)
    run = repository_root() / "mindstudio_runs" / "operator" / identity
    output = run / "operator_profile"
    if _completed(run, contract) and not args.force:
        print(f"Skip completed msOpProf capture: {run}")
        return 0
    command = operator_command(args, output)
    print(f"Official operator output: {output}")
    print("Command: " + shlex.join(command))
    if args.dry_run:
        return 0
    if run.exists():
        reset_output_generation((run,), active_run_directories=(run,), label="msOpProf")
    output.mkdir(parents=True)
    attempt = RunAttempt.start(run, kind="msopprof", context={"name": args.name})
    try:
        _run_logged(command, log=run / "runtime.log", env=os.environ.copy())
        if not any(output.iterdir()):
            raise RuntimeError(f"msOpProf produced no official output: {output}")
        _write_manifest(run, kind="operator", contract=contract, output=output)
        attempt.update("completed")
    except BaseException as error:
        attempt.update("failed", error=repr(error))
        raise
    print(f"MindStudio Insight operator import: {output}")
    return 0


def memory_command(
    args: argparse.Namespace,
    output: Path,
    topology: ParallelTopology,
) -> list[str]:
    command = [
        args.executable,
        f"--device={args.mem_device}",
        f"--level={args.level}",
        f"--events={args.events}",
        f"--call-stack={args.call_stack}",
        f"--collect-mode={args.collect_mode}",
        f"--analysis={args.analysis}",
        f"--data-format={args.data_format}",
        f"--output={output}",
    ]
    if args.capture_steps:
        command.append(f"--steps={args.capture_steps}")
    command.extend(args.tool_arg)
    training = [
        "bash",
        str(repository_root() / "run_train.sh"),
        f"--dump_folder={output.parent / 'trainer_output'}",
        f"--training.steps={args.steps}",
        *training_command_args(
            local_batch_size=args.local_batch_size,
            global_batch_size=args.global_batch_size,
            sequence_length=args.sequence_length,
            topology=topology,
        ),
        f"--debug.seed={args.seed}",
        "--metrics.log_freq=1",
        *topology.command_args(),
    ]
    command.extend(["--", *training])
    return command


def _memory_contract(
    args: argparse.Namespace,
    topology: ParallelTopology,
    version: str,
) -> dict[str, Any]:
    topology_value = asdict(topology)
    topology_value["extra_args"] = list(topology_value["extra_args"])
    return {
        "kind": "memory",
        "version": version,
        "topology": topology_value,
        "steps": args.steps,
        "local_batch_size": args.local_batch_size,
        "global_batch_size": args.global_batch_size,
        "sequence_length": args.sequence_length,
        "seed": args.seed,
        "mem_device": args.mem_device,
        "level": args.level,
        "events": args.events,
        "call_stack": args.call_stack,
        "collect_mode": args.collect_mode,
        "analysis": args.analysis,
        "data_format": args.data_format,
        "capture_steps": args.capture_steps,
        "tool_arg": args.tool_arg,
    }


def run_memory_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Official standalone msMemScope memory tuning workflow"
    )
    parser.add_argument("--probe", action="store_true")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "CANDIDATE"),
        help="run official step-memory comparison on two prior output directories",
    )
    parser.add_argument("--topology", default="single")
    parser.add_argument("--topologies")
    parser.add_argument("--list-topologies", action="store_true")
    parser.add_argument("--executable", default="msmemscope")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--local-batch-size", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--mem-device", default="npu")
    parser.add_argument("--level", choices=("op", "kernel"), default="op")
    parser.add_argument("--events", default="alloc,free,launch,traceback")
    parser.add_argument("--call-stack", default="python:20,c:20")
    parser.add_argument("--collect-mode", choices=("immediate", "deferred"), default="immediate")
    parser.add_argument("--analysis", default="leaks,decompose")
    parser.add_argument("--data-format", choices=("db", "csv"), default="db")
    parser.add_argument("--capture-steps")
    parser.add_argument("--tool-arg", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    topologies = standard_topologies()
    if args.list_topologies:
        for topology in topologies.values():
            print(f"{topology.slug}: world_size={topology.world_size}")
        return 0
    if bool(args.probe) == bool(args.compare):
        parser.error("select exactly one of --probe or --compare BASELINE CANDIDATE")
    if args.compare:
        version = "unverified-dry-run" if args.dry_run else _tool_version(args.executable)
        inputs = [str(Path(value).expanduser().resolve()) for value in args.compare]
        for value in inputs:
            if not Path(value).is_dir():
                parser.error(f"memory compare input is not a directory: {value}")
        contract = {
            "kind": "memory-compare",
            "version": version,
            "inputs": inputs,
            "level": args.level,
        }
        identity = config_name("npu-memory-compare", contract)
        run = repository_root() / "mindstudio_runs" / "memory" / "compare" / identity
        output = run / "memory_compare"
        if _completed(run, contract) and not args.force:
            print(f"Skip completed msMemScope comparison: {run}")
            return 0
        command = [
            args.executable,
            "--compare",
            f"--input={','.join(inputs)}",
            f"--level={args.level}",
            f"--output={output}",
        ]
        print(f"Official memory comparison output: {output}")
        print("Command: " + shlex.join(command))
        if args.dry_run:
            return 0
        if run.exists():
            reset_output_generation((run,), active_run_directories=(run,), label="msMemScope compare")
        output.mkdir(parents=True)
        attempt = RunAttempt.start(run, kind="msmemscope-compare")
        try:
            _run_logged(command, log=run / "runtime.log", env=os.environ.copy())
            if not any(output.iterdir()):
                raise RuntimeError(f"msMemScope produced no compare output: {output}")
            _write_manifest(run, kind="memory-compare", contract=contract, output=output)
            attempt.update("completed")
        except BaseException as error:
            attempt.update("failed", error=repr(error))
            raise
        print(f"msMemScope comparison output: {output}")
        return 0
    selected_names = select_topologies(
        available=tuple(topologies),
        topology=args.topology,
        topologies=args.topologies,
        default=("single",),
    )
    selected = tuple(topologies[name] for name in selected_names)
    if args.capture_steps and args.analysis.strip():
        parser.error(
            "official msMemScope step selection cannot be combined with "
            "memory analysis; pass --analysis '' and compare the captures later"
        )
    version = "unverified-dry-run" if args.dry_run else _tool_version(args.executable)
    for topology in selected:
        contract = _memory_contract(args, topology, version)
        base = (
            f"npu-{topology.slug}-bf16-s{args.steps}-l{args.local_batch_size}-"
            f"b{args.global_batch_size}-seq{args.sequence_length}-seed{args.seed}-"
            f"{args.level}-{args.data_format}"
        )
        identity = config_name(base, contract)
        run = (
            repository_root()
            / "mindstudio_runs"
            / "memory"
            / f"{topology.world_size}-card"
            / topology.slug
            / identity
        )
        output = run / "memory_profile"
        if _completed(run, contract) and not args.force:
            print(f"Skip completed msMemScope capture: {run}")
            continue
        command = memory_command(args, output, topology)
        print(f"Starting msMemScope capture: topology={topology.slug}")
        print(f"Official memory output: {output}")
        print("Command: " + shlex.join(command))
        if args.dry_run:
            continue
        if run.exists():
            reset_output_generation(
                (run,), active_run_directories=(run,), label="msMemScope"
            )
        output.mkdir(parents=True)
        attempt = RunAttempt.start(
            run, kind="msmemscope", context={"topology": topology.slug}
        )
        env = os.environ.copy()
        env.update(
            {
                "TORCHTITAN_DEVICE": "npu",
                "NGPU": str(topology.world_size),
                "TORCHTITAN_RUN_LOG": str(run / "training.log"),
            }
        )
        try:
            _run_logged(command, log=run / "runtime.log", env=env)
            if not any(output.iterdir()):
                raise RuntimeError(f"msMemScope produced no official output: {output}")
            _write_manifest(run, kind="memory", contract=contract, output=output)
            attempt.update("completed")
        except BaseException as error:
            attempt.update("failed", error=repr(error))
            raise
        print(f"MindStudio Insight memory import: {output}")
    return 0
