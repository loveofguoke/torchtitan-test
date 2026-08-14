"""Capture and compare one real four-rank DDP optimizer step."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from distributed.artifacts import (
    file_digest,
    file_manifest,
    read_json,
    validate_fixture,
    write_json,
)
from distributed.fixed_batches import (
    FIXED_BATCHES_ENV,
    TENSOR_KEYS,
    materialize_fixed_batches,
)
from distributed.workflow import _load_scenario


FIXED_BATCH_SCHEMA = "torchtitan.ddp-step-fixed-batches"
FIXED_BATCH_SCHEMA_VERSION = 1
_CAPTURE_INPUT_KEYS = {
    "input": "input.input",
    "positions": "input.positions",
    "labels": "labels",
}


def _run(command: list[str], *, environment: dict[str, str], log: Path) -> float:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
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
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return time.perf_counter() - started


def _fixed_batch_manifest_path(path: Path) -> Path:
    return Path(f"{path}.json")


def prepare_inputs(*, scenario_path: Path, fixture: Path, output: Path) -> None:
    """Materialize the canonical step-1 global batch and bind it to a fixture."""

    _scenario, scenario_digest = _load_scenario(scenario_path)
    fixture_manifest = validate_fixture(fixture, scenario_digest)
    sidecar = _fixed_batch_manifest_path(output)
    for path in (output, sidecar):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite fixed input artifact: {path}")
    materialize_fixed_batches(
        output,
        dataset_path=fixture / "data",
        tokenizer_path=fixture / "hf_assets",
        steps=1,
        global_batch_size=16,
        sequence_length=128,
    )
    payload = {
        "schema": FIXED_BATCH_SCHEMA,
        "schema_version": FIXED_BATCH_SCHEMA_VERSION,
        "scenario_digest": scenario_digest,
        "fixture_digest": fixture_manifest["fixture_digest"],
        "sha256": file_digest(output),
        "size": output.stat().st_size,
        "steps": 1,
        "global_batch_size": 16,
        "local_batch_size": 4,
        "sequence_length": 128,
        "world_size": 4,
    }
    write_json(sidecar, payload)
    print(f"fixed input tensors: {output}")
    print(f"fixed input manifest: {sidecar}")


def validate_fixed_inputs(
    path: Path, *, scenario_digest: str, fixture_digest: str
) -> dict[str, Any]:
    """Validate the fixed input bytes, identity, metadata, keys, and shapes."""

    if not path.is_file():
        raise FileNotFoundError(f"fixed input tensors not found: {path}")
    sidecar = _fixed_batch_manifest_path(path)
    manifest = read_json(sidecar)
    if manifest.get("schema") != FIXED_BATCH_SCHEMA:
        raise ValueError(f"unsupported fixed input schema in {sidecar}")
    if manifest.get("schema_version") != FIXED_BATCH_SCHEMA_VERSION:
        raise ValueError(f"unsupported fixed input schema version in {sidecar}")
    if manifest.get("scenario_digest") != scenario_digest:
        raise ValueError("fixed inputs were generated for a different scenario")
    if manifest.get("fixture_digest") != fixture_digest:
        raise ValueError("fixed inputs were generated from a different fixture")
    digest = file_digest(path)
    if manifest.get("sha256") != digest:
        raise ValueError(f"fixed input checksum mismatch: {path}")
    if manifest.get("size") != path.stat().st_size:
        raise ValueError(f"fixed input size mismatch: {path}")

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        keys = set(handle.keys())
    expected_values = {
        "steps": 1,
        "global_batch_size": 16,
        "local_batch_size": 4,
        "sequence_length": 128,
        "world_size": 4,
    }
    for key, expected in expected_values.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"fixed input manifest has {key}={manifest.get(key)}, expected {expected}"
            )
    for key in ("steps", "global_batch_size", "sequence_length"):
        if metadata.get(key) != str(expected_values[key]):
            raise ValueError(f"fixed input tensor metadata mismatch for {key}")
    if keys != set(TENSOR_KEYS):
        raise ValueError(f"unexpected fixed input tensors: {sorted(keys)}")
    tensors = load_file(str(path), device="cpu")
    expected_shape = (16, 128)
    for key in TENSOR_KEYS:
        tensor = tensors[key]
        if tuple(tensor.shape) != expected_shape or tensor.dtype != torch.int64:
            raise ValueError(
                f"fixed input {key} must be int64{expected_shape}, got "
                f"{tensor.dtype}{tuple(tensor.shape)}"
            )
    return manifest


def verify_captured_inputs(
    *, capture_dir: Path, fixed_inputs: Path, world_size: int = 4
) -> dict[str, Any]:
    """Require every captured rank batch to equal its canonical global slice."""

    fixed = load_file(str(fixed_inputs), device="cpu")
    local_batch_size = 4
    rows: list[dict[str, Any]] = []
    for rank in range(world_size):
        rank_json = read_json(capture_dir / f"rank-{rank:03d}.json")
        actual = load_file(
            str(capture_dir / rank_json["tensor_file"]), device="cpu"
        )
        start = rank * local_batch_size
        stop = start + local_batch_size
        for fixed_key, capture_key in _CAPTURE_INPUT_KEYS.items():
            if capture_key not in actual:
                raise RuntimeError(
                    f"rank {rank} did not capture required input {capture_key}"
                )
            exact = torch.equal(actual[capture_key], fixed[fixed_key][start:stop])
            row = {
                "rank": rank,
                "key": capture_key,
                "global_rows": [start, stop],
                "exact": exact,
            }
            rows.append(row)
            if not exact:
                raise RuntimeError(
                    f"rank {rank} input {capture_key} differs from fixed global "
                    f"rows [{start}:{stop}]"
                )
    return {"passed": True, "rows": rows}


def capture(
    *,
    scenario_path: Path,
    fixture: Path,
    fixed_inputs: Path,
    backend: str,
    output: Path,
) -> None:
    _scenario, scenario_digest = _load_scenario(scenario_path)
    manifest = validate_fixture(fixture, scenario_digest)
    fixed_manifest = validate_fixed_inputs(
        fixed_inputs,
        scenario_digest=scenario_digest,
        fixture_digest=manifest["fixture_digest"],
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(Path(__file__).with_name("run_ddp_step.sh")),
        f"--dump_folder={output / 'run'}",
        "--training.steps=1",
        "--metrics.enable_tensorboard",
        "--metrics.log_freq=1",
        "--metrics.disable_color_printing",
        "--debug.seed=42",
        "--debug.deterministic",
        "--checkpoint.enable",
        "--checkpoint.load_only",
        f"--checkpoint.initial_load_path={fixture / 'checkpoint'}",
        f"--dataloader.dataset_path={fixture / 'data'}",
        f"--hf_assets_path={fixture / 'hf_assets'}",
        "--training.local_batch_size=4",
        "--training.global_batch_size=16",
        "--training.seq_len=128",
        "--parallelism.data_parallel_replicate_degree=4",
        "--parallelism.data_parallel_shard_degree=1",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "NGPU": "4",
            "TORCHTITAN_DEVICE": backend,
            "TORCHTITAN_DIAGNOSTIC_DIR": str(output),
            FIXED_BATCHES_ENV: str(fixed_inputs),
            "PYTHONUNBUFFERED": "1",
        }
    )
    elapsed = _run(command, environment=environment, log=output / "runtime.log")
    rank_files = sorted(output.glob("rank-*.json"))
    if len(rank_files) != 4:
        raise RuntimeError(f"expected four rank captures, found {len(rank_files)}")
    for path in rank_files:
        payload = read_json(path)
        if payload.get("status") != "success":
            raise RuntimeError(f"rank diagnostic failed: {path}")
    input_verification = verify_captured_inputs(
        capture_dir=output, fixed_inputs=fixed_inputs, world_size=4
    )
    metadata = {
        "schema": "torchtitan.ddp-step-diagnostic-capture",
        "schema_version": 1,
        "backend": backend,
        "world_size": 4,
        "scenario_digest": scenario_digest,
        "fixture_digest": manifest["fixture_digest"],
        "fixed_inputs_digest": fixed_manifest["sha256"],
        "fixed_inputs_file": fixed_inputs.name,
        "input_verification": input_verification,
        "wall_time_seconds": elapsed,
        "command": command,
        "files": file_manifest(output),
    }
    write_json(output / "capture.json", metadata)
    print(f"diagnostic capture: {output}")


def _metric(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {"shape_match": False, "passed": False}
    if not reference.is_floating_point() and not candidate.is_floating_point():
        exact = torch.equal(reference, candidate)
        return {"shape_match": True, "exact": exact, "passed": exact}
    expected = reference.double()
    actual = candidate.double()
    difference = (actual - expected).abs()
    max_abs = float(difference.max()) if difference.numel() else 0.0
    mean_abs = float(difference.mean()) if difference.numel() else 0.0
    denominator = expected.abs().clamp_min(1e-12)
    max_rel = float((difference / denominator).max()) if difference.numel() else 0.0
    if expected.numel() and float(expected.norm()) and float(actual.norm()):
        cosine = float(
            torch.nn.functional.cosine_similarity(
                expected.flatten(), actual.flatten(), dim=0
            )
        )
    else:
        cosine = 1.0 if torch.equal(expected, actual) else 0.0
    return {
        "shape_match": True,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_rel": max_rel,
        "cosine": cosine,
    }


def _tolerance(kind: str) -> tuple[float, float]:
    if kind in {"gradient", "parameter_delta"}:
        return 1e-4, 1e-3
    return 1e-5, 1e-5


def compare(*, gpu: Path, npu: Path, report: Path) -> bool:
    gpu_meta = read_json(gpu / "capture.json")
    npu_meta = read_json(npu / "capture.json")
    for field in (
        "world_size",
        "scenario_digest",
        "fixture_digest",
        "fixed_inputs_digest",
    ):
        if not gpu_meta.get(field) or not npu_meta.get(field):
            raise ValueError(f"GPU/NPU diagnostic is missing required field {field}")
        if gpu_meta[field] != npu_meta[field]:
            raise ValueError(f"GPU/NPU diagnostic mismatch for {field}")

    rows: list[dict[str, Any]] = []
    first_failure: dict[str, Any] | None = None
    for rank in range(int(gpu_meta["world_size"])):
        gpu_json = read_json(gpu / f"rank-{rank:03d}.json")
        npu_json = read_json(npu / f"rank-{rank:03d}.json")
        gpu_records = {item["key"]: item for item in gpu_json["records"]}
        npu_records = {item["key"]: item for item in npu_json["records"]}
        gpu_tensors = load_file(gpu / gpu_json["tensor_file"])
        npu_tensors = load_file(npu / npu_json["tensor_file"])
        for item in gpu_json["records"]:
            key = item["key"]
            kind = item["kind"]
            row: dict[str, Any] = {"rank": rank, "key": key, "kind": kind}
            if key not in npu_records or key not in npu_tensors:
                row.update({"passed": False, "reason": "missing on NPU"})
            else:
                row.update(_metric(gpu_tensors[key], npu_tensors[key]))
                if "passed" not in row:
                    atol, rtol = _tolerance(kind)
                    row.update({"atol": atol, "rtol": rtol})
                    scale = float(gpu_tensors[key].double().abs().max())
                    row["passed"] = bool(
                        math.isfinite(row["max_abs"])
                        and row["max_abs"] <= atol + rtol * scale
                    )
            rows.append(row)
            if not row["passed"] and first_failure is None:
                first_failure = row
        for key in sorted(set(npu_records).difference(gpu_records)):
            row = {
                "rank": rank,
                "key": key,
                "kind": npu_records[key]["kind"],
                "passed": False,
                "reason": "missing on GPU",
            }
            rows.append(row)
            if first_failure is None:
                first_failure = row

    passed = first_failure is None
    payload = {
        "schema": "torchtitan.ddp-step-diagnostic-report",
        "schema_version": 1,
        "passed": passed,
        "first_failure": first_failure,
        "rows": rows,
        "gpu": str(gpu),
        "npu": str(npu),
    }
    write_json(report, payload)
    markdown = report.with_suffix(".md")
    lines = [
        "# DDP step-1 GPU/NPU diagnostic",
        "",
        f"Overall: **{'PASS' if passed else 'FAIL'}**",
        "",
    ]
    if first_failure:
        lines.extend(
            [
                "## First failure",
                "",
                f"- rank: `{first_failure['rank']}`",
                f"- key: `{first_failure['key']}`",
                f"- kind: `{first_failure['kind']}`",
                f"- max absolute error: `{first_failure.get('max_abs', 'n/a')}`",
                f"- max relative error: `{first_failure.get('max_rel', 'n/a')}`",
                f"- cosine: `{first_failure.get('cosine', 'n/a')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Failed observations",
            "",
            "| rank | observation | kind | max abs | max rel | cosine |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        if row["passed"]:
            continue
        lines.append(
            f"| {row['rank']} | `{row['key']}` | {row['kind']} | "
            f"{row.get('max_abs', 'n/a')} | {row.get('max_rel', 'n/a')} | "
            f"{row.get('cosine', 'n/a')} |"
        )
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"diagnostic: {'PASS' if passed else 'FAIL'}")
    if first_failure:
        print(
            f"first failure: rank={first_failure['rank']} "
            f"key={first_failure['key']} kind={first_failure['kind']}"
        )
    print(f"report: {report}")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    inputs_parser = stages.add_parser("prepare-inputs")
    inputs_parser.add_argument("--scenario", type=Path, required=True)
    inputs_parser.add_argument("--fixture", type=Path, required=True)
    inputs_parser.add_argument("--output", type=Path, required=True)
    capture_parser = stages.add_parser("capture")
    capture_parser.add_argument("--scenario", type=Path, required=True)
    capture_parser.add_argument("--fixture", type=Path, required=True)
    capture_parser.add_argument("--fixed-inputs", type=Path, required=True)
    capture_parser.add_argument("--backend", choices=("gpu", "npu"), required=True)
    capture_parser.add_argument("--output", type=Path, required=True)
    compare_parser = stages.add_parser("compare")
    compare_parser.add_argument("--gpu", type=Path, required=True)
    compare_parser.add_argument("--npu", type=Path, required=True)
    compare_parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.stage == "prepare-inputs":
        prepare_inputs(
            scenario_path=arguments.scenario.resolve(),
            fixture=arguments.fixture.resolve(),
            output=arguments.output.resolve(),
        )
    elif arguments.stage == "capture":
        capture(
            scenario_path=arguments.scenario.resolve(),
            fixture=arguments.fixture.resolve(),
            fixed_inputs=arguments.fixed_inputs.resolve(),
            backend=arguments.backend,
            output=arguments.output.resolve(),
        )
    elif not compare(
        gpu=arguments.gpu.resolve(),
        npu=arguments.npu.resolve(),
        report=arguments.report.resolve(),
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
