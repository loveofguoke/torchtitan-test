"""Prepare, capture, and compare distributed TorchTitan training runs."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from distributed.artifacts import (
    FIXTURE_SCHEMA_NAME,
    FIXTURE_SCHEMA_VERSION,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    ArtifactError,
    file_manifest,
    fixture_digest,
    json_digest,
    read_json,
    runtime_metadata,
    validate_capture,
    validate_fixture,
    write_json,
)


PRECISION_TAGS = (
    "loss_metrics/global_avg_loss",
    "loss_metrics/global_max_loss",
    "grad_norm",
)
PERFORMANCE_TAGS = (
    "throughput(tps)",
    "time_metrics/end_to_end(s)",
    "tflops",
    "mfu(%)",
    "memory/max_active(GiB)",
    "memory/max_reserved(GiB)",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_scenario(path: Path) -> tuple[dict[str, Any], str]:
    scenario = read_json(path)
    extends = scenario.pop("extends", None)
    appended_training_args = scenario.pop("training_args_append", [])
    if extends is not None:
        parent_path = Path(extends)
        if not parent_path.is_absolute():
            parent_path = (path.parent / parent_path).resolve()
        parent, _parent_digest = _load_scenario(parent_path)
        parent.update(scenario)
        scenario = parent
    if appended_training_args:
        if not isinstance(appended_training_args, list):
            raise ValueError("training_args_append must be a list")
        scenario["training_args"] = [
            *scenario.get("training_args", []),
            *appended_training_args,
        ]
    required = {
        "name",
        "module",
        "config",
        "world_size",
        "steps",
        "performance_warmup_steps",
        "seed",
        "deterministic",
        "dataset_source",
        "hf_assets_source",
        "seed_args",
        "training_args",
        "tolerances",
    }
    missing = required.difference(scenario)
    if missing:
        raise ValueError(f"scenario is missing fields: {', '.join(sorted(missing))}")
    if int(scenario["world_size"]) < 1:
        raise ValueError("world_size must be positive")
    if int(scenario["steps"]) < 1:
        raise ValueError("steps must be positive")
    if not isinstance(scenario["deterministic"], bool):
        raise ValueError("deterministic must be a boolean")
    warmup = int(scenario["performance_warmup_steps"])
    if warmup < 0 or warmup >= int(scenario["steps"]):
        raise ValueError("performance_warmup_steps must be in [0, steps)")
    for field in ("seed_args", "training_args"):
        if not isinstance(scenario[field], list) or not all(
            isinstance(value, str) for value in scenario[field]
        ):
            raise ValueError(f"{field} must be a list of strings")
    return scenario, json_digest(scenario)


def _resolve_source(scenario_path: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    if not path.is_absolute():
        path = (scenario_path.parent / path).resolve()
    return path


def _launcher(root: Path, backend: str) -> Path:
    launcher = root / f"run_train_{backend}.sh"
    if not launcher.is_file():
        raise FileNotFoundError(f"training launcher not found: {launcher}")
    return launcher


def _run(
    command: list[str],
    *,
    root: Path,
    environment: dict[str, str],
    log: Path,
) -> float:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("command:", " ".join(command))
    print("log:", log)
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    elapsed = time.perf_counter() - started
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return elapsed


def _base_environment(world_size: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "NGPU": str(world_size),
            "LOG_RANK": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"fixture source directory not found: {source}")
    shutil.copytree(source, destination)


def prepare_fixture(
    scenario_path: Path,
    scenario: dict[str, Any],
    scenario_digest: str,
    fixture: Path,
    backend: str,
) -> None:
    if fixture.exists():
        raise FileExistsError(f"fixture already exists: {fixture}")
    root = _repo_root()
    fixture.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{fixture.name}.tmp-", dir=fixture.parent))
    try:
        _copy_tree(
            _resolve_source(scenario_path, str(scenario["dataset_source"])),
            temporary / "data",
        )
        _copy_tree(
            _resolve_source(scenario_path, str(scenario["hf_assets_source"])),
            temporary / "hf_assets",
        )
        seed_dump = temporary / "seed_run"
        command = [
            str(_launcher(root, backend)),
            f"--dump_folder={seed_dump}",
            "--checkpoint.enable",
            "--checkpoint.create_seed_checkpoint",
            f"--dataloader.dataset_path={temporary / 'data'}",
            f"--hf_assets_path={temporary / 'hf_assets'}",
            f"--debug.seed={int(scenario['seed'])}",
            *scenario["seed_args"],
        ]
        environment = _base_environment(1)
        environment["MODULE"] = str(scenario["module"])
        environment["CONFIG"] = str(scenario["config"])
        elapsed = _run(
            command,
            root=root,
            environment=environment,
            log=temporary / "prepare.log",
        )
        checkpoint = seed_dump / "checkpoint" / "step-0"
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"seed checkpoint was not created: {checkpoint}")
        shutil.move(str(checkpoint), temporary / "checkpoint")
        shutil.rmtree(seed_dump)
        manifest: dict[str, Any] = {
            "schema": FIXTURE_SCHEMA_NAME,
            "schema_version": FIXTURE_SCHEMA_VERSION,
            "scenario_digest": scenario_digest,
            "scenario": scenario,
            "prepared_with_backend": backend,
            "prepare_wall_time_seconds": elapsed,
            "environment": runtime_metadata(),
            "files": file_manifest(temporary),
        }
        manifest["fixture_digest"] = fixture_digest(manifest)
        write_json(temporary / "manifest.json", manifest)
        temporary.rename(fixture)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"fixture: {fixture}")
    print(f"fixture_digest: {manifest['fixture_digest']}")


def _tensorboard_scalars(dump_folder: Path) -> dict[str, dict[str, float]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as error:
        raise RuntimeError("tensorboard is required to extract exact metrics") from error

    event_files = sorted((dump_folder / "tb").rglob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"no TensorBoard event file found under {dump_folder / 'tb'}")
    values: dict[str, dict[str, float]] = {}
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file.parent))
        accumulator.Reload()
        for tag in set(PRECISION_TAGS + PERFORMANCE_TAGS):
            if tag not in accumulator.Tags().get("scalars", []):
                continue
            tag_values = values.setdefault(tag, {})
            for scalar in accumulator.Scalars(tag):
                tag_values[str(scalar.step)] = float(scalar.value)
    missing = [tag for tag in PRECISION_TAGS if tag not in values]
    if missing:
        raise ArtifactError(f"capture is missing required metrics: {', '.join(missing)}")
    return values


def capture(
    scenario: dict[str, Any],
    scenario_digest: str,
    fixture: Path,
    backend: str,
    output: Path,
) -> None:
    manifest = validate_fixture(fixture, scenario_digest)
    if output.exists():
        raise FileExistsError(f"capture output already exists: {output}")
    root = _repo_root()
    output.parent.mkdir(parents=True, exist_ok=True)
    run_dir = output.with_suffix("")
    if run_dir.exists():
        raise FileExistsError(f"capture run directory already exists: {run_dir}")
    command = [
        str(_launcher(root, backend)),
        f"--dump_folder={run_dir}",
        f"--training.steps={int(scenario['steps'])}",
        "--metrics.enable_tensorboard",
        "--metrics.log_freq=1",
        "--metrics.disable_color_printing",
        "--metrics.save_tb_folder=tb",
        f"--debug.seed={int(scenario['seed'])}",
        "--checkpoint.enable",
        "--checkpoint.load_only",
        f"--checkpoint.initial_load_path={fixture / 'checkpoint'}",
        f"--dataloader.dataset_path={fixture / 'data'}",
        f"--hf_assets_path={fixture / 'hf_assets'}",
        *scenario["seed_args"],
        *scenario["training_args"],
    ]
    if scenario["deterministic"]:
        command.append("--debug.deterministic")
    environment = _base_environment(int(scenario["world_size"]))
    environment["MODULE"] = str(scenario["module"])
    environment["CONFIG"] = str(scenario["config"])
    log = run_dir / "runtime.log"
    started_at = time.time()
    status = "failed"
    failure = ""
    wall_time = 0.0
    metrics: dict[str, dict[str, float]] = {}
    try:
        wall_time = _run(command, root=root, environment=environment, log=log)
        metrics = _tensorboard_scalars(run_dir)
        status = "success"
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        result = {
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "failure": failure,
            "scenario_digest": scenario_digest,
            "fixture_digest": manifest["fixture_digest"],
            "backend": backend,
            "world_size": int(scenario["world_size"]),
            "started_at_unix": started_at,
            "wall_time_seconds": wall_time,
            "environment": {
                **runtime_metadata(),
                "selected_backend": backend,
                "configured_world_size": environment["NGPU"],
                "module": environment["MODULE"],
                "config": environment["CONFIG"],
            },
            "command": command,
            "metrics": metrics,
            "runtime_log": str(log),
        }
        write_json(output, result)
    print(f"capture: {output}")


def _tolerance(scenario: dict[str, Any], tag: str) -> tuple[float, float]:
    configured = scenario["tolerances"].get(
        tag, scenario["tolerances"].get("default")
    )
    if (
        not isinstance(configured, dict)
        or "atol" not in configured
        or "rtol" not in configured
    ):
        raise ValueError(f"missing atol/rtol tolerance for metric {tag}")
    return float(configured["atol"]), float(configured["rtol"])


def _median_after_warmup(
    metrics: dict[str, dict[str, float]], tag: str, warmup: int
) -> float | None:
    values = [
        value
        for step, value in sorted(
            metrics.get(tag, {}).items(), key=lambda item: int(item[0])
        )
        if int(step) > warmup
    ]
    return statistics.median(values) if values else None


def compare(
    scenario: dict[str, Any],
    scenario_digest: str,
    fixture: Path,
    gpu_path: Path,
    npu_path: Path,
    report_path: Path,
) -> bool:
    manifest = validate_fixture(fixture, scenario_digest)
    gpu = validate_capture(
        gpu_path,
        scenario_digest=scenario_digest,
        expected_fixture_digest=manifest["fixture_digest"],
    )
    npu = validate_capture(
        npu_path,
        scenario_digest=scenario_digest,
        expected_fixture_digest=manifest["fixture_digest"],
    )
    if gpu.get("world_size") != npu.get("world_size"):
        raise ArtifactError("GPU and NPU captures used different world sizes")

    rows: list[dict[str, Any]] = []
    precision_passed = True
    for tag in PRECISION_TAGS:
        gpu_values = gpu["metrics"].get(tag, {})
        npu_values = npu["metrics"].get(tag, {})
        if set(gpu_values) != set(npu_values):
            raise ArtifactError(f"GPU/NPU steps differ for metric {tag}")
        atol, rtol = _tolerance(scenario, tag)
        for step in sorted(gpu_values, key=int):
            expected = float(gpu_values[step])
            actual = float(npu_values[step])
            absolute = abs(actual - expected)
            relative = absolute / max(abs(expected), 1e-30)
            passed = math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol)
            precision_passed &= passed
            rows.append(
                {
                    "metric": tag,
                    "step": int(step),
                    "gpu": expected,
                    "npu": actual,
                    "absolute_error": absolute,
                    "relative_error": relative,
                    "atol": atol,
                    "rtol": rtol,
                    "passed": passed,
                }
            )

    warmup = int(scenario["performance_warmup_steps"])
    performance: dict[str, Any] = {"warmup_steps_excluded": warmup, "metrics": {}}
    for tag in PERFORMANCE_TAGS:
        gpu_median = _median_after_warmup(gpu["metrics"], tag, warmup)
        npu_median = _median_after_warmup(npu["metrics"], tag, warmup)
        if gpu_median is None or npu_median is None:
            continue
        performance["metrics"][tag] = {
            "gpu_median": gpu_median,
            "npu_median": npu_median,
            "npu_over_gpu": npu_median / gpu_median if gpu_median else None,
        }

    report = {
        "scenario": scenario["name"],
        "scenario_digest": scenario_digest,
        "fixture_digest": manifest["fixture_digest"],
        "precision_passed": precision_passed,
        "precision": rows,
        "performance": performance,
        "gpu_capture": str(gpu_path),
        "npu_capture": str(npu_path),
    }
    write_json(report_path, report)
    print(f"precision: {'PASS' if precision_passed else 'FAIL'}")
    print(f"report: {report_path}")
    for tag, values in performance["metrics"].items():
        print(
            f"{tag}: gpu={values['gpu_median']:.6g}, "
            f"npu={values['npu_median']:.6g}, "
            f"npu/gpu={values['npu_over_gpu']:.4f}"
        )
    return precision_passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    stages = parser.add_subparsers(dest="stage", required=True)

    prepare_parser = stages.add_parser("prepare")
    prepare_parser.add_argument("--backend", choices=("gpu", "npu"), default="gpu")

    capture_parser = stages.add_parser("capture")
    capture_parser.add_argument("--backend", choices=("gpu", "npu"), required=True)
    capture_parser.add_argument("--output", type=Path, required=True)

    compare_parser = stages.add_parser("compare")
    compare_parser.add_argument("--gpu", type=Path, required=True)
    compare_parser.add_argument("--npu", type=Path, required=True)
    compare_parser.add_argument("--report", type=Path, required=True)

    arguments = parser.parse_args()
    scenario_path = arguments.scenario.resolve()
    fixture = arguments.fixture.resolve()
    scenario, scenario_digest = _load_scenario(scenario_path)

    if arguments.stage == "prepare":
        prepare_fixture(
            scenario_path,
            scenario,
            scenario_digest,
            fixture,
            arguments.backend,
        )
    elif arguments.stage == "capture":
        capture(
            scenario,
            scenario_digest,
            fixture,
            arguments.backend,
            arguments.output.resolve(),
        )
    elif arguments.stage == "compare":
        passed = compare(
            scenario,
            scenario_digest,
            fixture,
            arguments.gpu.resolve(),
            arguments.npu.resolve(),
            arguments.report.resolve(),
        )
        if not passed:
            sys.exit(1)


if __name__ == "__main__":
    main()
