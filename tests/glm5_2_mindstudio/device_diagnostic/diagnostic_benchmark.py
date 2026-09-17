#!/usr/bin/env python3
"""Orchestrate repeatable Ascend device and HCCL diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import socket
import statistics
import subprocess
import sys
import threading
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.glm5_2_common.cli import (  # noqa: E402
    LoggedProcessError,
    RunAttempt,
    archive_previous_output,
    assert_run_not_active,
    print_output_path,
    print_runtime_log,
    reset_output_generation,
    write_experiment_overview,
)
from tests.glm5_2_mindstudio.artifacts import output_index, write_json  # noqa: E402


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def run_command(
    command: list[str],
    *,
    root: Path,
    log,
    output: Path,
    environment: dict[str, str] | None = None,
) -> None:
    rendered = shlex.join(command)
    print(f"Running device diagnostic stage: {output.name}", flush=True)
    log.write(f"\n$ {rendered}\n")
    log.flush()
    output.parent.mkdir(parents=True, exist_ok=True)
    process_environment = dict(os.environ if environment is None else environment)
    process_environment["PYTHONUNBUFFERED"] = "1"
    with output.open("w", encoding="utf-8") as output_file:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=process_environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )
        assert process.stdout is not None
        assert process.stderr is not None

        def copy_stdout() -> None:
            for line in process.stdout:
                output_file.write(line)
                output_file.flush()
                sys.stdout.write(line)
                sys.stdout.flush()

        def copy_stderr() -> None:
            for line in process.stderr:
                log.write(line)
                log.flush()
                sys.stderr.write(line)
                sys.stderr.flush()

        stdout_thread = threading.Thread(target=copy_stdout)
        stderr_thread = threading.Thread(target=copy_stderr)
        stdout_thread.start()
        stderr_thread.start()
        returncode = process.wait()
        stdout_thread.join()
        stderr_thread.join()
    log.write(f"[exit code: {returncode}]\n")
    log.flush()
    if returncode:
        raise LoggedProcessError(
            returncode,
            command,
            log_path=Path(log.name),
        )


def generation_complete(
    artifact: Path,
    report: Path,
    *,
    experiment_digest: str,
) -> bool:
    complete_path = artifact / "complete.json"
    manifest_path = artifact / "manifest.json"
    official = artifact / "official"
    if not complete_path.is_file() or not manifest_path.is_file():
        return False
    try:
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        complete.get("status") == "completed"
        and complete.get("experiment_digest") == experiment_digest
        and manifest.get("experiment_digest") == experiment_digest
        and official.is_dir()
        and manifest.get("official_files") == output_index(official)
        and (report / "summary.json").is_file()
        and (report / "README.md").is_file()
    )


def prepare_generation(
    run: Path,
    artifact: Path,
    report: Path,
    *,
    experiment_digest: str,
    force: bool,
) -> bool:
    """Return whether the selected generation still needs to run."""

    selected = (run, artifact, report)
    if force:
        reset_output_generation(
            selected,
            active_run_directories=(run,),
            label="device diagnostic",
        )
        return True
    if generation_complete(
        artifact,
        report,
        experiment_digest=experiment_digest,
    ):
        print_output_path("Skip completed device diagnostic", artifact)
        print_runtime_log(run / "runtime.log")
        return False
    if not any(path.exists() for path in selected):
        return True
    assert_run_not_active(run)
    for path in selected:
        archived = archive_previous_output(path)
        if archived is not None:
            print(f"Retry incomplete device diagnostic; archived: {archived}")
    return True


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        object_start = stripped.find("{")
        if object_start < 0:
            continue
        try:
            row, _remainder = json.JSONDecoder().raw_decode(stripped[object_start:])
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    if not rows:
        raise ValueError(f"no JSON measurement rows found in {path}")
    return rows


def make_summary(official: Path) -> dict[str, Any]:
    device_rows = read_json_lines(official / "single_device.jsonl")
    pair_rows = read_json_lines(official / "pairwise_all_reduce.jsonl")
    all_device_rows = read_json_lines(official / "all_device_all_reduce.jsonl")
    ddp_step_path = official / "synthetic_ddp_step.jsonl"
    ddp_step_rows = (
        read_json_lines(ddp_step_path) if ddp_step_path.is_file() else []
    )

    device_samples: dict[str, dict[str, list[float]]] = {}
    for row in device_rows:
        if row.get("benchmark") == "bf16_matmul":
            device = str(row["physical_device"])
            device_samples.setdefault(device, {}).setdefault(
                "matmul_tflops", []
            ).append(float(row["tflops"]))
        elif row.get("benchmark") == "bf16_copy":
            device = str(row["physical_device"])
            device_samples.setdefault(device, {}).setdefault(
                "copy_gib_per_second", []
            ).append(float(row["gib_per_second"]))
        elif row.get("benchmark") == "host_launch":
            device = str(row["physical_device"])
            samples = device_samples.setdefault(device, {})
            samples.setdefault("host_enqueue_us_per_op", []).append(
                float(row["enqueue_us_per_op"])
            )
            samples.setdefault("host_synchronized_us_per_op", []).append(
                float(row["synchronized_us_per_op"])
            )

    device_metrics: dict[str, dict[str, Any]] = {
        device: {
            metric: statistics.median(values)
            for metric, values in samples.items()
        }
        for device, samples in device_samples.items()
    }
    for device, samples in device_samples.items():
        device_metrics[device]["round_values"] = samples

    required_device_metrics = {"matmul_tflops", "copy_gib_per_second"}
    incomplete_devices = {
        device: sorted(required_device_metrics - metrics.keys())
        for device, metrics in device_metrics.items()
        if required_device_metrics - metrics.keys()
    }
    matmul_values = [
        metrics["matmul_tflops"]
        for metrics in device_metrics.values()
        if "matmul_tflops" in metrics
    ]
    copy_values = [
        metrics["copy_gib_per_second"]
        for metrics in device_metrics.values()
        if "copy_gib_per_second" in metrics
    ]
    if not matmul_values or not copy_values:
        raise ValueError(
            "single-device diagnostic produced no usable "
            f"{'matmul' if not matmul_values else 'copy'} measurements; "
            f"missing metrics by device: {incomplete_devices}"
        )
    matmul_median = statistics.median(matmul_values)
    copy_median = statistics.median(copy_values)
    host_enqueue_values = [
        metrics["host_enqueue_us_per_op"]
        for metrics in device_metrics.values()
        if "host_enqueue_us_per_op" in metrics
    ]
    host_synchronized_values = [
        metrics["host_synchronized_us_per_op"]
        for metrics in device_metrics.values()
        if "host_synchronized_us_per_op" in metrics
    ]
    host_enqueue_median = (
        statistics.median(host_enqueue_values) if host_enqueue_values else None
    )
    host_synchronized_median = (
        statistics.median(host_synchronized_values)
        if host_synchronized_values
        else None
    )
    for metrics in device_metrics.values():
        if "matmul_tflops" in metrics:
            metrics["matmul_vs_median"] = (
                metrics["matmul_tflops"] / matmul_median
            )
        if "copy_gib_per_second" in metrics:
            metrics["copy_vs_median"] = (
                metrics["copy_gib_per_second"] / copy_median
            )
        if host_enqueue_median is not None and "host_enqueue_us_per_op" in metrics:
            metrics["host_enqueue_vs_median"] = (
                metrics["host_enqueue_us_per_op"] / host_enqueue_median
            )
        if (
            host_synchronized_median is not None
            and "host_synchronized_us_per_op" in metrics
        ):
            metrics["host_synchronized_vs_median"] = (
                metrics["host_synchronized_us_per_op"]
                / host_synchronized_median
            )

    pair_metrics: dict[str, float] = {}
    for row in pair_rows:
        pair = str(row["visible_devices"])
        pair_metrics[pair] = max(
            pair_metrics.get(pair, 0.0), float(row["median_ms"])
        )
    pair_median = statistics.median(pair_metrics.values())
    pair_ratios = {
        pair: latency / pair_median for pair, latency in pair_metrics.items()
    }

    return {
        "schema": "torchtitan.glm5_2.device_diagnostic_summary",
        "schema_version": 1,
        "device_metrics": device_metrics,
        "incomplete_devices": incomplete_devices,
        "pair_median_ms": pair_median,
        "pair_median_ms_by_devices": pair_metrics,
        "pair_latency_vs_median": pair_ratios,
        "all_device_median_ms_by_rank": {
            str(row["rank"]): float(row["median_ms"])
            for row in all_device_rows
        },
        "synthetic_ddp_step": {
            str(row["visible_devices"]): {
                str(candidate["rank"]): candidate
                for candidate in ddp_step_rows
                if candidate["visible_devices"] == row["visible_devices"]
            }
            for row in ddp_step_rows
        },
        "suspect_compute_devices": sorted(
            device
            for device, metrics in device_metrics.items()
            if metrics.get("matmul_vs_median", 1.0) < 0.9
            or metrics.get("copy_vs_median", 1.0) < 0.9
        ),
        "suspect_pairs": sorted(
            pair for pair, ratio in pair_ratios.items() if ratio > 1.2
        ),
        "suspect_host_devices": sorted(
            device
            for device, metrics in device_metrics.items()
            if metrics.get("host_enqueue_vs_median", 1.0) > 1.2
            or metrics.get("host_synchronized_vs_median", 1.0) > 1.2
        ),
        "thresholds": {
            "device_throughput_vs_median": 0.9,
            "pair_latency_vs_median": 1.2,
            "host_latency_vs_median": 1.2,
        },
    }


def write_report(report: Path, summary: dict[str, Any]) -> None:
    report.mkdir(parents=True, exist_ok=True)
    write_json(report / "summary.json", summary)
    lines = [
        "# Ascend device diagnostic report",
        "",
        "Triage thresholds identify candidates, not hardware pass/fail criteria.",
        "",
        f"- Suspect compute devices: `{summary['suspect_compute_devices']}`",
        f"- Devices with incomplete measurements: `{summary['incomplete_devices']}`",
        f"- Suspect host-path devices: `{summary['suspect_host_devices']}`",
        f"- Suspect HCCL pairs: `{summary['suspect_pairs']}`",
        f"- Median pair latency: `{summary['pair_median_ms']:.3f} ms`",
        "",
        "See `summary.json` for per-device and per-pair measurements.",
        "",
    ]
    (report / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--matrix-size", type=int, default=8192)
    parser.add_argument("--memory-mib", type=int, default=512)
    parser.add_argument("--collective-size-mib", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--single-device-rounds", type=int, default=1)
    parser.add_argument("--launch-batch", type=int, default=100)
    parser.add_argument(
        "--ddp-pairs",
        default="",
        help="Ordered two-device mappings separated by semicolons, for example 0,1;1,0",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    devices = [part.strip() for part in args.devices.split(",") if part.strip()]
    ddp_pairs = [
        [device.strip() for device in pair.split(",")]
        for pair in args.ddp_pairs.split(";")
        if pair.strip()
    ]
    if any(len(pair) != 2 or len(set(pair)) != 2 for pair in ddp_pairs):
        raise ValueError("--ddp-pairs requires ordered unique device pairs")
    if len(devices) < 2 or len(set(devices)) != len(devices):
        raise ValueError("--devices requires at least two unique physical device IDs")
    unknown_ddp_devices = sorted(
        {device for pair in ddp_pairs for device in pair} - set(devices)
    )
    if unknown_ddp_devices:
        raise ValueError(
            "--ddp-pairs devices must also appear in --devices: "
            + ",".join(unknown_ddp_devices)
        )
    if args.repeat < 1:
        raise ValueError("--repeat must be positive")

    root = repository_root()
    script_directory = Path(__file__).resolve().parent
    identity = {
        "devices": devices,
        "matrix_size": args.matrix_size,
        "memory_mib": args.memory_mib,
        "collective_size_mib": args.collective_size_mib,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "single_device_rounds": args.single_device_rounds,
        "launch_batch": args.launch_batch,
        "ddp_pairs": ddp_pairs,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    experiment_id = f"device-diagnostic-{digest[:8]}"
    relative = Path(experiment_id) / f"{len(devices)}-device" / f"candidate-r{args.repeat}"
    run_directory = root / "mindstudio_runs/performance/device_diagnostic" / relative
    artifact_directory = (
        root / "mindstudio_artifacts/performance/device_diagnostic" / relative
    )
    report_directory = (
        root
        / "mindstudio_reports/performance/device_diagnostic"
        / experiment_id
        / f"{len(devices)}-device"
        / f"candidate-r{args.repeat}"
    )
    official = artifact_directory / "official"

    if not prepare_generation(
        run_directory,
        artifact_directory,
        report_directory,
        experiment_digest=digest,
        force=args.force,
    ):
        return

    entry_command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    write_experiment_overview(
        run_directory,
        title="Ascend device and HCCL diagnostic",
        summary={
            "experiment_id": experiment_id,
            "experiment_digest": digest,
            "identity": identity,
            "run_directory": str(run_directory.resolve()),
            "artifact_directory": str(artifact_directory.resolve()),
            "report_directory": str(report_directory.resolve()),
        },
        entry_command=entry_command,
    )
    write_json(run_directory / "resolved_command.json", {"command": entry_command})
    (run_directory / "resolved_launch.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + shlex.join(entry_command)
        + "\n",
        encoding="utf-8",
    )

    attempt = RunAttempt.start(
        run_directory,
        kind="ascend_device_diagnostic",
        context={"experiment_id": experiment_id, "experiment_digest": digest},
    )
    runtime_log = run_directory / "runtime.log"
    try:
        official.mkdir(parents=True, exist_ok=False)
        with runtime_log.open("w", encoding="utf-8") as log:
            print_runtime_log(runtime_log)
            inventory_parts = [f"hostname: {socket.gethostname()}\n"]
            inventory_commands = [
                ["npu-smi", "info"],
                ["npu-smi", "info", "-l"],
                ["npu-smi", "info", "-m"],
                ["npu-smi", "info", "-t", "topo"],
            ]
            for index, command in enumerate(inventory_commands):
                output = official / f".inventory_{index}.txt"
                run_command(command, root=root, log=log, output=output)
                inventory_parts.append(output.read_text(encoding="utf-8"))
                output.unlink()
            (official / "inventory.txt").write_text(
                "\n".join(inventory_parts), encoding="utf-8"
            )

            for device in devices:
                for query in ("hccs", "health"):
                    run_command(
                        ["npu-smi", "info", "-t", query, "-i", device, "-c", "0"],
                        root=root,
                        log=log,
                        output=official / f"{query}_device_{device}.txt",
                    )

            single_output = official / "single_device.jsonl"
            single_parts = []
            for round_index in range(args.single_device_rounds):
                offset = round_index % len(devices)
                round_devices = devices[offset:] + devices[:offset]
                if round_index % 2:
                    round_devices.reverse()
                print(
                    "Single-device round "
                    f"{round_index + 1}/{args.single_device_rounds}: "
                    + ",".join(round_devices),
                    flush=True,
                )
                for device in round_devices:
                    environment = dict(os.environ)
                    environment["ASCEND_RT_VISIBLE_DEVICES"] = device
                    output = (
                        official
                        / f".single_round_{round_index}_device_{device}.jsonl"
                    )
                    run_command(
                        [
                            sys.executable,
                            str(script_directory / "device_benchmark.py"),
                            "--matrix-size", str(args.matrix_size),
                            "--memory-mib", str(args.memory_mib),
                            "--warmup", str(args.warmup),
                            "--iterations", str(args.iterations),
                            "--round-index", str(round_index),
                            "--launch-batch", str(args.launch_batch),
                        ],
                        root=root,
                        log=log,
                        output=output,
                        environment=environment,
                    )
                    single_parts.append(output.read_text(encoding="utf-8"))
                    output.unlink()
            single_output.write_text("".join(single_parts), encoding="utf-8")

            pair_output = official / "pairwise_all_reduce.jsonl"
            pair_parts = []
            for first_index, first in enumerate(devices[:-1]):
                for second in devices[first_index + 1 :]:
                    pair = f"{first},{second}"
                    environment = dict(os.environ)
                    environment["ASCEND_RT_VISIBLE_DEVICES"] = pair
                    output = official / f".pair_{first}_{second}.jsonl"
                    run_command(
                        [
                            sys.executable, "-m", "torch.distributed.run",
                            "--standalone", "--nproc-per-node=2",
                            str(script_directory / "collective_benchmark.py"),
                            "--size-mib", str(args.collective_size_mib),
                            "--warmup", str(args.warmup),
                            "--iterations", str(args.iterations),
                        ],
                        root=root,
                        log=log,
                        output=output,
                        environment=environment,
                    )
                    pair_parts.append(output.read_text(encoding="utf-8"))
                    output.unlink()
            pair_output.write_text("".join(pair_parts), encoding="utf-8")

            environment = dict(os.environ)
            environment["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(devices)
            run_command(
                [
                    sys.executable, "-m", "torch.distributed.run",
                    "--standalone", f"--nproc-per-node={len(devices)}",
                    str(script_directory / "collective_benchmark.py"),
                    "--size-mib", str(args.collective_size_mib),
                    "--warmup", str(args.warmup),
                    "--iterations", str(args.iterations),
                ],
                root=root,
                log=log,
                output=official / "all_device_all_reduce.jsonl",
                environment=environment,
            )

            if ddp_pairs:
                ddp_parts = []
                for pair_devices in ddp_pairs:
                    pair = ",".join(pair_devices)
                    environment = dict(os.environ)
                    environment["ASCEND_RT_VISIBLE_DEVICES"] = pair
                    output = official / f".ddp_step_{'_'.join(pair_devices)}.jsonl"
                    run_command(
                        [
                            sys.executable, "-m", "torch.distributed.run",
                            "--standalone", "--nproc-per-node=2",
                            str(script_directory / "ddp_step_benchmark.py"),
                            "--matrix-size", str(args.matrix_size),
                            "--collective-size-mib", str(args.collective_size_mib),
                            "--warmup", str(args.warmup),
                            "--iterations", str(args.iterations),
                        ],
                        root=root,
                        log=log,
                        output=output,
                        environment=environment,
                    )
                    ddp_parts.append(output.read_text(encoding="utf-8"))
                    output.unlink()
                (official / "synthetic_ddp_step.jsonl").write_text(
                    "".join(ddp_parts), encoding="utf-8"
                )

        summary = make_summary(official)
        write_report(report_directory, summary)
        manifest = {
            "schema": "torchtitan.glm5_2.device_diagnostic_artifact",
            "schema_version": 1,
            "experiment_id": experiment_id,
            "experiment_digest": digest,
            "identity": identity,
            "host": socket.gethostname(),
            "toolchain": {
                "python": sys.version,
                "torch": package_version("torch"),
                "torch_npu": package_version("torch-npu"),
            },
            "official_output": "official",
            "official_files": output_index(official),
            "runtime_log": str(runtime_log.resolve()),
            "report": str(report_directory.resolve()),
        }
        write_json(artifact_directory / "manifest.json", manifest)
        write_json(
            artifact_directory / "complete.json",
            {
                "status": "completed",
                "experiment_digest": digest,
                "attempt_id": attempt.attempt_id,
            },
        )
        attempt.update(
            "completed",
            artifact=str(artifact_directory.resolve()),
            report=str(report_directory.resolve()),
        )
    except BaseException as error:
        attempt.update("failed", error=repr(error))
        print_runtime_log(runtime_log)
        raise

    print_output_path("Run", run_directory)
    print_output_path("Artifact", artifact_directory)
    print_output_path("Report", report_directory)
    print_runtime_log(runtime_log)


if __name__ == "__main__":
    main()
