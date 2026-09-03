"""Nsight Systems capture and post-collection analysis orchestration."""

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

from tests.glm5_2_common.cli import RunAttempt, reset_output_generation
from tests.glm5_2_common.naming import config_name
from tests.glm5_2_common.topology import (
    ParallelTopology,
    select_topologies,
    standard_topologies,
    training_command_args,
)


DEFAULT_TRACE = "cuda,nvtx,osrt,cublas,cudnn"
DEFAULT_PYTORCH = "functions-trace-shapes,autograd-nvtx"
DEFAULT_STATS = (
    "cuda_api_sum",
    "cuda_gpu_kern_sum",
    "cuda_gpu_mem_time_sum",
    "cuda_gpu_mem_size_sum",
    "nvtx_sum",
    "osrt_sum",
)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_logged(
    command: list[str], *, cwd: Path, log: Path, env: dict[str, str]
) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"Runtime log: {log}")
    print("Command: " + subprocess.list2cmdline(command))
    with log.open("w", encoding="utf-8") as stream:
        stream.write("Command: " + subprocess.list2cmdline(command) + "\n\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
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
        raise subprocess.CalledProcessError(return_code, command)


def _nsys_version(nsys: str) -> str:
    result = subprocess.run(
        [nsys, "--version"], capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise RuntimeError(f"failed to run {nsys} --version: {result.stderr.strip()}")
    return (result.stdout or result.stderr).strip()


def _contract(
    args: argparse.Namespace,
    topology: ParallelTopology,
    version: str,
) -> dict[str, Any]:
    topology_value = asdict(topology)
    topology_value["extra_args"] = list(topology_value["extra_args"])
    return {
        "schema_version": 1,
        "device": "cuda",
        "topology": topology_value,
        "module": args.module,
        "config": args.config,
        "steps": args.steps,
        "local_batch_size": args.local_batch_size,
        "global_batch_size": args.global_batch_size,
        "sequence_length": args.sequence_length,
        "seed": args.seed,
        "nsys_version": version,
        "trace": args.trace,
        "pytorch": args.pytorch,
        "sample": args.sample,
        "cuda_memory_usage": args.cuda_memory_usage,
        "cuda_graph_trace": args.cuda_graph_trace,
        "delay": args.delay,
        "duration": args.duration,
        "stats_reports": list(args.stats_reports),
    }


def _identity_name(
    args: argparse.Namespace,
    topology: ParallelTopology,
    contract: dict[str, Any],
) -> str:
    base = (
        f"cuda-{topology.slug}-bf16-s{args.steps}-l{args.local_batch_size}-"
        f"b{args.global_batch_size}-seq{args.sequence_length}-seed{args.seed}-standard"
    )
    return config_name(base, contract)


def _read_manifest(path: Path) -> dict[str, Any]:
    manifest = path / "manifest.json"
    if not manifest.is_file():
        return {}
    try:
        return json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _official_output_dir(run_dir: Path) -> Path:
    """Return the run-owned Nsight Systems output directory."""

    return run_dir / "trainer_output" / "profiling" / "nsys"


def _move_tree_without_overwrite(source: Path, destination: Path) -> None:
    """Move a legacy output tree, refusing to mix conflicting generations."""

    if not source.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*"), key=lambda item: len(item.parts)):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if path.read_bytes() != target.read_bytes():
                raise RuntimeError(
                    "refusing to merge conflicting legacy Nsight output: "
                    f"{path} -> {target}"
                )
            path.unlink()
        else:
            path.replace(target)
    shutil.rmtree(source)


def _adopt_legacy_outputs(run_dir: Path, artifact_dir: Path) -> Path:
    """Move pre-layout-fix official Nsight files from artifacts into runs."""

    output = _official_output_dir(run_dir)
    legacy_paths = (
        (artifact_dir / "profile.nsys-rep", output / "profile.nsys-rep"),
        (artifact_dir / "profile.sqlite", output / "profile.sqlite"),
        (artifact_dir / "export_sqlite.log", output / "export_sqlite.log"),
    )
    for source, destination in legacy_paths:
        if not source.is_file():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if source.read_bytes() != destination.read_bytes():
                raise RuntimeError(
                    "refusing to merge conflicting legacy Nsight output: "
                    f"{source} -> {destination}"
                )
            source.unlink()
        else:
            source.replace(destination)
    _move_tree_without_overwrite(artifact_dir / "stats", output / "stats")
    return output


def _capture(
    args: argparse.Namespace,
    *,
    topology: ParallelTopology,
    contract: dict[str, Any],
    run_dir: Path,
    artifact_dir: Path,
) -> None:
    output_dir = _adopt_legacy_outputs(run_dir, artifact_dir)
    record = _read_manifest(artifact_dir)
    report = output_dir / "profile.nsys-rep"
    if (
        record.get("capture_status") == "completed"
        and record.get("contract") == contract
        and report.is_file()
    ):
        print(f"Skip completed Nsight Systems capture: {artifact_dir}")
        return
    if artifact_dir.exists() or run_dir.exists():
        reset_output_generation(
            (run_dir, artifact_dir),
            active_run_directories=(run_dir,),
            label=f"Nsight Systems {topology.slug} capture",
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    attempt = RunAttempt.start(
        run_dir,
        kind="nsys-performance",
        context={"topology": topology.slug, "steps": args.steps},
    )
    train_log = run_dir / "training.log"
    train_command = [
        "bash",
        str(_root() / "run_train.sh"),
        f"--dump_folder={run_dir / 'trainer_output'}",
        f"--training.steps={args.steps}",
        "--training.disable_cuda_graphs",
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
    command = [
        args.nsys,
        "profile",
        f"--trace={args.trace}",
        f"--pytorch={args.pytorch}",
        f"--sample={args.sample}",
        f"--cuda-memory-usage={'true' if args.cuda_memory_usage else 'false'}",
        f"--cuda-graph-trace={args.cuda_graph_trace}",
        "--wait=all",
        "--stats=false",
        "--force-overwrite=true",
        f"--output={output_dir / 'profile'}",
    ]
    if args.delay is not None:
        command.append(f"--delay={args.delay}")
    if args.duration is not None:
        command.append(f"--duration={args.duration}")
    command.extend(train_command)
    env = os.environ.copy()
    env.update(
        {
            "TORCHTITAN_DEVICE": "gpu",
            "TORCHTITAN_RUN_LOG": str(train_log),
            "NGPU": str(topology.world_size),
            "LOG_RANK": env.get("LOG_RANK", "0"),
            "MODULE": args.module,
            "CONFIG": args.config,
            "CUDA_VISIBLE_DEVICES": args.visible_devices,
        }
    )
    manifest = {
        "capture_status": "running",
        "analysis_status": "pending",
        "contract": contract,
        "command": command,
        "training_command": train_command,
        "runtime_log": str(run_dir / "nsys_profile.log"),
        "training_log": str(train_log),
        "attempt_id": attempt.attempt_id,
        "official_output": str(output_dir),
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        _run_logged(command, cwd=_root(), log=run_dir / "nsys_profile.log", env=env)
    except BaseException:
        manifest["capture_status"] = "failed"
        (artifact_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        attempt.update("failed")
        raise
    if not report.is_file():
        attempt.update("failed")
        raise RuntimeError(f"nsys completed without producing {report}")
    manifest["capture_status"] = "completed"
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    attempt.update("completed")
    print(f"Nsight Systems capture: {report}")


def _analyze(
    args: argparse.Namespace,
    *,
    contract: dict[str, Any],
    run_dir: Path,
    artifact_dir: Path,
) -> None:
    output_dir = _adopt_legacy_outputs(run_dir, artifact_dir)
    manifest = _read_manifest(artifact_dir)
    report = output_dir / "profile.nsys-rep"
    if (
        manifest.get("capture_status") != "completed"
        or manifest.get("contract") != contract
        or not report.is_file()
    ):
        raise RuntimeError(f"matching completed capture not found: {artifact_dir}")
    stats_dir = output_dir / "stats"
    sqlite_path = output_dir / "profile.sqlite"
    expected = [stats_dir / f"{name}.csv" for name in args.stats_reports]
    if (
        manifest.get("analysis_status") == "completed"
        and sqlite_path.is_file()
        and all(path.is_file() for path in expected)
    ):
        print(f"Skip completed Nsight Systems analysis: {artifact_dir}")
        return
    stats_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    _run_logged(
        [
            args.nsys,
            "export",
            "--type=sqlite",
            "--force-overwrite=true",
            f"--output={sqlite_path}",
            str(report),
        ],
        cwd=_root(),
        log=output_dir / "export_sqlite.log",
        env=env,
    )
    for name, output in zip(args.stats_reports, expected, strict=True):
        command = [
            args.nsys,
            "stats",
            f"--report={name}",
            "--format=csv",
            "--output=-",
            str(report),
        ]
        result = subprocess.run(
            command,
            cwd=_root(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        stats_log = stats_dir / f"{name}.log"
        stats_log.write_text(
            "Command: "
            + subprocess.list2cmdline(command)
            + "\n\n"
            + result.stderr,
            encoding="utf-8",
        )
        print(f"Nsight Systems statistic log: {stats_log}")
        if result.returncode:
            raise subprocess.CalledProcessError(
                result.returncode, command, result.stdout, result.stderr
            )
        output.write_text(result.stdout, encoding="utf-8")
        print(f"Nsight Systems statistic: {output}")
    manifest["analysis_status"] = "completed"
    manifest["sqlite"] = str(sqlite_path)
    manifest["statistics"] = [str(path) for path in expected]
    manifest["official_output"] = str(output_dir)
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_report(
    report_path: Path,
    *,
    topology: ParallelTopology,
    run_dir: Path,
    artifact_dir: Path,
) -> None:
    output_dir = _adopt_legacy_outputs(run_dir, artifact_dir)
    manifest = _read_manifest(artifact_dir)
    rows = "".join(
        f"<li><code>{html.escape(path)}</code></li>"
        for path in manifest.get("statistics", [])
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>GLM Nsight Systems report</title>"
        f"<h1>GLM CUDA Nsight Systems: {html.escape(topology.slug)}</h1>"
        "<p>Capture: <code>"
        f"{html.escape(str(output_dir / 'profile.nsys-rep'))}</code></p>"
        "<p>SQLite: <code>"
        f"{html.escape(str(output_dir / 'profile.sqlite'))}</code></p>"
        "<p>Open the .nsys-rep in Nsight Systems UI for the interactive timeline. "
        "Use the CSV summaries for CUDA API, kernel, memory, NVTX, and OS "
        "runtime aggregation.</p>"
        f"<h2>Statistics</h2><ul>{rows}</ul>",
        encoding="utf-8",
    )
    print(f"Nsight Systems report: {report_path}")


def run_cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--capture", action="store_true", help="collect .nsys-rep only"
    )
    actions.add_argument(
        "--analyze", action="store_true", help="analyze an existing capture"
    )
    actions.add_argument(
        "--probe", action="store_true", help="capture and analyze (default)"
    )
    topology_group = parser.add_mutually_exclusive_group()
    topology_group.add_argument("--topology", default="single")
    topology_group.add_argument("--topologies")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--local-batch-size", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--module", default="glm5")
    parser.add_argument("--config", default="glm5_debugmodel")
    parser.add_argument(
        "--visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "")
    )
    parser.add_argument("--nsys", default=os.environ.get("NSYS", "nsys"))
    parser.add_argument("--trace", default=DEFAULT_TRACE)
    parser.add_argument("--pytorch", default=DEFAULT_PYTORCH)
    parser.add_argument(
        "--sample",
        choices=("none", "process-tree", "system-wide"),
        default="none",
    )
    parser.add_argument(
        "--cuda-memory-usage",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cuda-graph-trace",
        choices=("none", "graph", "node"),
        default="node",
    )
    parser.add_argument("--delay", type=float)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--stats-reports", default=",".join(DEFAULT_STATS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.stats_reports = tuple(
        value.strip()
        for value in args.stats_reports.split(",")
        if value.strip()
    )
    if (
        args.steps < 1
        or args.local_batch_size < 1
        or args.global_batch_size < 1
        or args.sequence_length < 1
    ):
        parser.error("steps, batch sizes, and sequence length must be positive")
    if not args.visible_devices:
        parser.error("export CUDA_VISIBLE_DEVICES or pass --visible-devices")
    version = "dry-run" if args.dry_run else _nsys_version(args.nsys)
    topologies = standard_topologies()
    available = tuple(
        name for name, value in topologies.items() if value.world_size <= 8
    )
    selected = select_topologies(
        available=available,
        topology=None if args.topologies else args.topology,
        topologies=args.topologies,
        default=("single",),
    )
    num_visible = len(
        [value for value in args.visible_devices.split(",") if value.strip()]
    )
    for name in selected:
        if topologies[name].world_size > num_visible:
            parser.error(
                f"topology {name} needs {topologies[name].world_size} "
                f"visible GPUs, found {num_visible}"
            )
    members: list[tuple[ParallelTopology, dict[str, Any], Path, Path, Path]] = []
    for name in selected:
        topology = topologies[name]
        contract = _contract(args, topology, version)
        identity = _identity_name(args, topology, contract)
        card_group = f"{topology.world_size}-card"
        members.append((
            topology,
            contract,
            _root() / "nsys_runs" / card_group / topology.slug / identity,
            _root() / "nsys_artifacts" / card_group / topology.slug / identity,
            _root() / "nsys_reports" / card_group / topology.slug / f"{identity}.html",
        ))
    if args.force:
        reset_output_generation(
            [
                path
                for _, _, run, artifact, report in members
                for path in (run, artifact, report)
            ],
            active_run_directories=[run for _, _, run, _, _ in members],
            label="Nsight Systems suite",
        )
    if args.dry_run:
        for topology, contract, run, artifact, report in members:
            print(
                json.dumps(
                    {
                        "topology": topology.name,
                        "contract": contract,
                        "run": str(run),
                        "artifact": str(artifact),
                        "report": str(report),
                    },
                    indent=2,
                )
            )
        return 0
    no_action = not (args.capture or args.analyze or args.probe)
    do_capture = args.capture or args.probe or no_action
    do_analyze = args.analyze or args.probe or no_action
    for topology, contract, run_dir, artifact_dir, report_path in members:
        print(
            f"Starting Nsight Systems probe: topology={topology.name}, "
            f"world_size={topology.world_size}"
        )
        if do_capture:
            _capture(
                args,
                topology=topology,
                contract=contract,
                run_dir=run_dir,
                artifact_dir=artifact_dir,
            )
        if do_analyze:
            _analyze(
                args,
                contract=contract,
                run_dir=run_dir,
                artifact_dir=artifact_dir,
            )
            _write_report(
                report_path,
                topology=topology,
                run_dir=run_dir,
                artifact_dir=artifact_dir,
            )
    return 0
