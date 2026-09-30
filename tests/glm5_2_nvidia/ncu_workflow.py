"""Targeted Nsight Compute replay after Nsight Systems triage."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import html
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from tests.glm5_2_common.cli import (
    LoggedProcessError,
    RunAttempt,
    display_repository_path,
    print_output_path,
    print_runtime_log,
    reset_output_generation,
    write_experiment_overview,
)
from tests.glm5_2_common.naming import config_name
from tests.glm5_2_common.topology import standard_topologies, training_command_args


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _version(ncu: str) -> str:
    result = subprocess.run([ncu, "--version"], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"failed to run {ncu} --version: {result.stderr.strip()}")
    return (result.stdout or result.stderr).strip()


def _contract(args: argparse.Namespace, topology: Any, version: str) -> dict[str, Any]:
    topology_value = asdict(topology)
    topology_value["extra_args"] = list(topology_value["extra_args"])
    return {
        "schema_version": 1,
        "tool": "nsight-compute",
        "ncu_version": version,
        "topology": topology_value,
        "module": args.module,
        "config": args.config,
        "steps": args.steps,
        "local_batch_size": args.local_batch_size,
        "global_batch_size": args.global_batch_size,
        "sequence_length": args.sequence_length,
        "seed": args.seed,
        "section_set": args.section_set,
        "kernel_name": args.kernel_name,
        "nvtx_include": args.nvtx_include,
        "launch_skip": args.launch_skip,
        "launch_count": args.launch_count,
        "replay_mode": args.replay_mode,
    }


def _completed(manifest_path: Path, report_dir: Path, contract: dict[str, Any]) -> bool:
    if not manifest_path.is_file() or not list(report_dir.glob("*.ncu-rep")):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("status") == "completed" and manifest.get("contract") == contract


def _run(command: list[str], log: Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print_runtime_log(log)
    with log.open("w", encoding="utf-8") as stream:
        stream.write("Command: " + subprocess.list2cmdline(command) + "\n\n")
        process = subprocess.Popen(
            command, cwd=_root(), env=env, stdout=stream,
            stderr=subprocess.STDOUT, text=True,
        )
        return_code = process.wait()
    print_runtime_log(log)
    if return_code:
        raise LoggedProcessError(return_code, command, log_path=log)


def _write_report(
    path: Path,
    *,
    contract: dict[str, Any],
    reports: list[Path],
    runtime_log: Path,
) -> None:
    report_rows = "".join(
        f"<tr><td><code>{html.escape(report.name)}</code></td>"
        f"<td><code>{html.escape(str(report))}</code></td></tr>"
        for report in reports
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>GLM Nsight Compute targeted analysis</title><style>"
        "body{margin:0;background:#f5f7fb;color:#172033;font:15px/1.6 system-ui}"
        "main{max-width:1180px;margin:auto;padding:36px 32px}section{background:#fff;"
        "border:1px solid #dfe5ef;border-radius:12px;padding:22px;margin:20px 0;overflow:auto}"
        "table{border-collapse:collapse;width:100%}th,td{padding:10px;border-bottom:1px solid #e5eaf2;text-align:left}"
        "th{background:#f7f9fc}code{white-space:pre-wrap;word-break:break-all}"
        "</style></head><body><main><h1>GPU 定点算子分析 / Nsight Compute</h1>"
        "<p>这是 NSys 单拓扑分析后的定点重放，不是自然运行时间线，也不能用于判断跨 rank 原始耗时。</p>"
        "<section><h2>采集契约</h2><table><tbody>"
        f"<tr><th>Kernel selector</th><td><code>{html.escape(contract['kernel_name'])}</code></td></tr>"
        f"<tr><th>NVTX range</th><td><code>{html.escape(str(contract.get('nvtx_include') or '-'))}</code></td></tr>"
        f"<tr><th>Section set</th><td>{html.escape(contract['section_set'])}</td></tr>"
        f"<tr><th>Replay mode</th><td>{html.escape(contract['replay_mode'])}</td></tr>"
        f"<tr><th>Launch window</th><td>skip {contract['launch_skip']}, count {contract['launch_count']}</td></tr>"
        f"<tr><th>Runtime log</th><td><code>{html.escape(str(runtime_log))}</code></td></tr>"
        "</tbody></table></section>"
        f"<section><h2>官方 .ncu-rep</h2><table><thead><tr><th>File</th><th>Path</th></tr></thead><tbody>{report_rows}</tbody></table>"
        "<p>用 Nsight Compute UI 打开报告。建议依次检查 SpeedOfLight、MemoryWorkloadAnalysis、SchedulerStats、WarpStateStats；"
        "只有证据指向特定算子后才扩大 section 或 launch count。</p></section>"
        "<section><h2>解释边界</h2><ul><li>NCU replay 会序列化或重复 kernel，时间不能替代 profiler-off 基线。</li>"
        "<li>多进程报告使用 %i 防止覆盖；每个文件仍需结合 rank/process 归属解释。</li>"
        "<li>优化后先回到 profiler-off 重复实验验收，再用 NSys 解释收益来源。</li></ul></section>"
        "</main></body></html>",
        encoding="utf-8",
    )


def run_cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", default="single")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--local-batch-size", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--module", default="glm5")
    parser.add_argument("--config", default="glm5_debugmodel")
    parser.add_argument("--visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    parser.add_argument("--ncu", default=os.environ.get("NCU", "ncu"))
    parser.add_argument("--set", dest="section_set", choices=("basic", "detailed", "full"), default="basic")
    parser.add_argument("--kernel-name", required=True, help="NCU kernel selector, for example regex:.*gemm.*")
    parser.add_argument("--nvtx-include", help="optional NCU NVTX range filter")
    parser.add_argument("--launch-skip", type=int, default=0)
    parser.add_argument("--launch-count", type=int, default=1)
    parser.add_argument("--replay-mode", choices=("kernel", "application", "range"), default="kernel")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.visible_devices:
        parser.error("export CUDA_VISIBLE_DEVICES or pass --visible-devices")
    topology = standard_topologies().get(args.topology)
    if topology is None:
        parser.error(f"unknown topology: {args.topology}")
    visible_count = len([value for value in args.visible_devices.split(",") if value.strip()])
    if topology.world_size > visible_count:
        parser.error(
            f"topology {topology.name} needs {topology.world_size} visible GPUs, "
            f"found {visible_count}"
        )
    version = "dry-run" if args.dry_run else _version(args.ncu)
    contract = _contract(args, topology, version)
    identity = config_name(
        f"cuda-{topology.slug}-bf16-s{args.steps}-seq{args.sequence_length}-{args.section_set}",
        contract,
    )
    card_group = f"{topology.world_size}-card"
    run_dir = _root() / "nvidia_runs" / "performance" / "operator" / card_group / topology.slug / identity
    artifact_dir = _root() / "nvidia_artifacts" / "performance" / "operator" / card_group / topology.slug / identity
    report_path = _root() / "nvidia_reports" / "performance" / "operator" / card_group / topology.slug / f"{identity}.html"
    report_dir = run_dir / "trainer_output" / "profiling" / "ncu"
    manifest_path = artifact_dir / "manifest.json"
    report_pattern = report_dir / "profile-%i"
    if args.dry_run:
        print(
            json.dumps(
                {
                    "contract": contract,
                    "run": display_repository_path(run_dir),
                    "artifact": display_repository_path(artifact_dir),
                    "report_pattern": display_repository_path(report_pattern)
                    + ".ncu-rep",
                    "report": display_repository_path(report_path),
                },
                indent=2,
            )
        )
        return 0
    if not args.force and _completed(manifest_path, report_dir, contract):
        reports = sorted(report_dir.glob("*.ncu-rep"))
        if not report_path.is_file():
            _write_report(
                report_path,
                contract=contract,
                reports=reports,
                runtime_log=run_dir / "ncu_profile.log",
            )
        print_output_path("Skip completed Nsight Compute profile", report_dir)
        print_output_path("Nsight Compute index", report_path)
        return 0
    if run_dir.exists() or artifact_dir.exists():
        reset_output_generation((run_dir, artifact_dir, report_path), active_run_directories=(run_dir,), label="Nsight Compute profile")
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    train_command = [
        "bash", str(_root() / "run_train.sh"),
        f"--dump_folder={run_dir / 'trainer_output'}",
        f"--training.steps={args.steps}", "--training.disable_cuda_graphs",
        *training_command_args(local_batch_size=args.local_batch_size, global_batch_size=args.global_batch_size, sequence_length=args.sequence_length, topology=topology),
        f"--debug.seed={args.seed}", "--metrics.log_freq=1", *topology.command_args(),
    ]
    command = [
        args.ncu, "--target-processes=all", f"--replay-mode={args.replay_mode}",
        f"--set={args.section_set}", f"--kernel-name={args.kernel_name}",
        f"--launch-skip={args.launch_skip}", f"--launch-count={args.launch_count}",
        "--force-overwrite", f"--export={report_pattern}",
    ]
    if args.nvtx_include:
        command.extend(("--nvtx", f"--nvtx-include={args.nvtx_include}"))
    command.extend(train_command)
    env = os.environ.copy()
    env.update({"TORCHTITAN_DEVICE": "gpu", "NGPU": str(topology.world_size), "MODULE": args.module, "CONFIG": args.config, "CUDA_VISIBLE_DEVICES": args.visible_devices})
    write_experiment_overview(run_dir, title="GLM NVIDIA targeted kernel profile", summary={"workflow": "performance/operator", "contract": contract}, entry_command=sys.argv)
    attempt = RunAttempt.start(run_dir, kind="ncu-kernel-profile", context={"kernel_name": args.kernel_name})
    manifest = {"status": "running", "contract": contract, "command": command, "runtime_log": str(run_dir / "ncu_profile.log")}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    try:
        _run(command, run_dir / "ncu_profile.log", env)
    except BaseException:
        manifest["status"] = "failed"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        attempt.update("failed")
        raise
    reports = sorted(report_dir.glob("*.ncu-rep"))
    if not reports:
        attempt.update("failed")
        raise RuntimeError(f"ncu completed without producing reports under {report_dir}; runtime log: {(run_dir / 'ncu_profile.log').resolve()}")
    _write_report(
        report_path,
        contract=contract,
        reports=reports,
        runtime_log=run_dir / "ncu_profile.log",
    )
    manifest.update({"status": "completed", "reports": [str(path) for path in reports], "report": str(report_path)})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    attempt.update("completed")
    for report in reports:
        print_output_path("Nsight Compute report", report)
    print_output_path("Nsight Compute index", report_path)
    return 0
