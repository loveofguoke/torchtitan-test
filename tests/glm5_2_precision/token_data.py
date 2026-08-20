# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Portable token plans and runtime input-contract validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable


TOKEN_PLAN_SCHEMA = "torchtitan.glm5_2.fixed_token_plan"
TOKEN_PLAN_VERSION = 1
INPUT_MAPPING = "optimizer-step-global-slot-v1"


class TokenDataError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sample_digest(token_bytes: bytes, position_bytes: bytes) -> str:
    digest = hashlib.sha256(b"glm5-fixed-token-sample-v1\0")
    digest.update(token_bytes)
    digest.update(position_bytes)
    return digest.hexdigest()


def step_digest(sample_digests: Iterable[str]) -> str:
    digest = hashlib.sha256(b"glm5-fixed-token-step-v1\0")
    for slot, value in enumerate(sample_digests):
        digest.update(slot.to_bytes(8, byteorder="little", signed=False))
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def step_series_digest(step_digests: Iterable[str]) -> str:
    digest = hashlib.sha256(b"glm5-fixed-token-step-series-v1\0")
    for step, value in enumerate(step_digests, start=1):
        digest.update(step.to_bytes(8, byteorder="little", signed=False))
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


@dataclass(frozen=True)
class TokenPlan:
    path: Path
    steps: int
    global_batch_size: int
    sequence_length: int
    token_file: Path
    position_file: Path
    sample_sha256: tuple[str, ...]
    step_sha256: tuple[str, ...]

    @property
    def num_samples(self) -> int:
        return self.steps * self.global_batch_size

    def sample_hash(self, *, step: int, slot: int) -> str:
        if not 0 <= step < self.steps:
            raise TokenDataError(f"token-plan step is out of range: {step}")
        if not 0 <= slot < self.global_batch_size:
            raise TokenDataError(f"global sample slot is out of range: {slot}")
        return self.sample_sha256[step * self.global_batch_size + slot]


def load_token_plan(path: str | Path, *, verify_files: bool = True) -> TokenPlan:
    directory = Path(path)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise TokenDataError(f"token-plan manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != TOKEN_PLAN_SCHEMA:
        raise TokenDataError(
            f"unsupported token-plan schema: {manifest.get('schema')!r}"
        )
    if manifest.get("schema_version") != TOKEN_PLAN_VERSION:
        raise TokenDataError(
            "unsupported token-plan schema version: "
            f"{manifest.get('schema_version')!r}"
        )

    steps = int(manifest["steps"])
    global_batch_size = int(manifest["global_batch_size"])
    sequence_length = int(manifest["sequence_length"])
    sample_hashes = tuple(str(value) for value in manifest["sample_sha256"])
    step_hashes = tuple(str(value) for value in manifest["step_sha256"])
    expected_samples = steps * global_batch_size
    if len(sample_hashes) != expected_samples:
        raise TokenDataError(
            f"token plan has {len(sample_hashes)} sample hashes, expected "
            f"{expected_samples}"
        )
    if len(step_hashes) != steps:
        raise TokenDataError(
            f"token plan has {len(step_hashes)} step hashes, expected {steps}"
        )
    for step in range(steps):
        start = step * global_batch_size
        observed = step_digest(sample_hashes[start : start + global_batch_size])
        if observed != step_hashes[step]:
            raise TokenDataError(
                f"token-plan step {step + 1} digest is inconsistent with its samples"
            )

    files = manifest["files"]
    token_file = directory / files["tokens"]["name"]
    position_file = directory / files["positions"]["name"]
    expected_token_bytes = expected_samples * (sequence_length + 1) * 4
    expected_position_bytes = expected_samples * sequence_length * 4
    for label, file_path, expected_bytes in (
        ("tokens", token_file, expected_token_bytes),
        ("positions", position_file, expected_position_bytes),
    ):
        if not file_path.is_file():
            raise TokenDataError(f"token-plan {label} file not found: {file_path}")
        if file_path.stat().st_size != expected_bytes:
            raise TokenDataError(
                f"token-plan {label} size is {file_path.stat().st_size}, expected "
                f"{expected_bytes}: {file_path}"
            )
        if verify_files:
            observed = sha256_file(file_path)
            expected = files[label]["sha256"]
            if observed != expected:
                raise TokenDataError(
                    f"token-plan {label} checksum mismatch: {observed} != {expected}"
                )
    return TokenPlan(
        path=directory,
        steps=steps,
        global_batch_size=global_batch_size,
        sequence_length=sequence_length,
        token_file=token_file,
        position_file=position_file,
        sample_sha256=sample_hashes,
        step_sha256=step_hashes,
    )


def batches_per_optimizer_step(
    *,
    global_batch_size: int,
    training_local_batch_size: int,
    dataloader_batch_size: int,
    dp_world_size: int,
) -> int:
    denominator = training_local_batch_size * dp_world_size
    if global_batch_size % denominator:
        raise TokenDataError(
            "global batch size must be divisible by training local batch size "
            "times DP degree"
        )
    if training_local_batch_size % dataloader_batch_size:
        raise TokenDataError(
            "training local batch size must be divisible by dataloader batch size"
        )
    accumulation_steps = global_batch_size // denominator
    pipeline_microbatches = training_local_batch_size // dataloader_batch_size
    return accumulation_steps * pipeline_microbatches


def global_slots_for_batch(
    *,
    batch_in_step: int,
    dp_rank: int,
    dp_world_size: int,
    global_batch_size: int,
    training_local_batch_size: int,
    dataloader_batch_size: int,
) -> tuple[int, ...]:
    num_batches = batches_per_optimizer_step(
        global_batch_size=global_batch_size,
        training_local_batch_size=training_local_batch_size,
        dataloader_batch_size=dataloader_batch_size,
        dp_world_size=dp_world_size,
    )
    if not 0 <= batch_in_step < num_batches:
        raise TokenDataError(f"batch_in_step is out of range: {batch_in_step}")
    if not 0 <= dp_rank < dp_world_size:
        raise TokenDataError(f"DP rank is out of range: {dp_rank}")
    pipeline_microbatches = training_local_batch_size // dataloader_batch_size
    accumulation = batch_in_step // pipeline_microbatches
    pipeline_microbatch = batch_in_step % pipeline_microbatches
    first_slot = (
        (accumulation * dp_world_size + dp_rank) * training_local_batch_size
        + pipeline_microbatch * dataloader_batch_size
    )
    return tuple(range(first_slot, first_slot + dataloader_batch_size))


def _read_contract_file(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise TokenDataError(
                f"invalid input-contract JSON at {path}:{line_number}: {error}"
            ) from error
    return records


def validate_runtime_input_contract(
    *,
    contract_directory: str | Path,
    plan: TokenPlan,
    steps: int,
    global_batch_size: int,
    training_local_batch_size: int,
    dp_world_size: int,
    context_parallel_degree: int,
    tensor_parallel_degree: int,
    pipeline_parallel_degree: int,
    node_rank: int,
    num_processes_per_node: int,
    wait_seconds: float = 60.0,
) -> dict[str, Any]:
    """Validate every local process against the topology-independent token plan."""

    directory = Path(contract_directory)
    first_global_rank = node_rank * num_processes_per_node
    expected_global_ranks = tuple(
        range(first_global_rank, first_global_rank + num_processes_per_node)
    )
    expected_names = {f"rank-{rank}" for rank in expected_global_ranks}
    deadline = time.monotonic() + wait_seconds
    while True:
        files = {path.stem: path for path in directory.glob("rank-*.jsonl")}
        complete = expected_names.issubset(files) and all(
            len(path.read_text(encoding="utf-8").splitlines()) == steps
            for name, path in files.items()
            if name in expected_names
        )
        if complete or time.monotonic() >= deadline:
            break
        time.sleep(0.25)
    if not expected_names.issubset(files):
        raise TokenDataError(
            "input-contract rank files do not match the local torchrun ranks: "
            f"observed={sorted(files)}, expected={sorted(expected_names)}"
        )
    if plan.steps != steps or plan.global_batch_size != global_batch_size:
        raise TokenDataError("runtime training dimensions do not match the token plan")

    replica_records: dict[tuple[int, int], tuple[tuple[int, ...], tuple[str, ...]]] = {}
    for global_rank in expected_global_ranks:
        records = _read_contract_file(files[f"rank-{global_rank}"])
        if len(records) != steps:
            raise TokenDataError(
                f"rank {global_rank} recorded {len(records)} optimizer steps, "
                f"expected {steps}"
            )
        ranks_per_pipeline_stage = (
            dp_world_size
            * context_parallel_degree
            * tensor_parallel_degree
        )
        expected_pp_rank = global_rank // ranks_per_pipeline_stage
        expected_dp_rank = (
            global_rank // (context_parallel_degree * tensor_parallel_degree)
        ) % dp_world_size
        for expected_step, record in enumerate(records, start=1):
            if int(record["global_rank"]) != global_rank:
                raise TokenDataError(f"rank file {global_rank} contains another rank")
            if int(record["step"]) != expected_step:
                raise TokenDataError(
                    f"rank {global_rank} input step order is not contiguous"
                )
            if int(record["dp_rank"]) != expected_dp_rank:
                raise TokenDataError(
                    f"global rank {global_rank} reported DP rank "
                    f"{record['dp_rank']}, expected {expected_dp_rank}"
                )
            if int(record.get("pp_rank", expected_pp_rank)) != expected_pp_rank:
                raise TokenDataError(
                    f"global rank {global_rank} reported PP rank "
                    f"{record.get('pp_rank')}, expected {expected_pp_rank}"
                )
            slots = tuple(int(value) for value in record["global_slots"])
            hashes = tuple(str(value) for value in record["sample_sha256"])
            if len(slots) != len(hashes):
                raise TokenDataError(
                    f"rank {global_rank} step {expected_step} has mismatched "
                    "slots/hashes"
                )
            expected_slots: list[int] = []
            dataloader_batch_size = int(record["dataloader_batch_size"])
            num_batches = batches_per_optimizer_step(
                global_batch_size=global_batch_size,
                training_local_batch_size=training_local_batch_size,
                dataloader_batch_size=dataloader_batch_size,
                dp_world_size=dp_world_size,
            )
            for batch_in_step in range(num_batches):
                expected_slots.extend(
                    global_slots_for_batch(
                        batch_in_step=batch_in_step,
                        dp_rank=expected_dp_rank,
                        dp_world_size=dp_world_size,
                        global_batch_size=global_batch_size,
                        training_local_batch_size=training_local_batch_size,
                        dataloader_batch_size=dataloader_batch_size,
                    )
                )
            if slots != tuple(expected_slots):
                raise TokenDataError(
                    f"rank {global_rank} step {expected_step} used global slots "
                    f"{slots}, expected {tuple(expected_slots)}"
                )
            expected_hashes = tuple(
                plan.sample_hash(step=expected_step - 1, slot=slot) for slot in slots
            )
            if hashes != expected_hashes:
                raise TokenDataError(
                    f"rank {global_rank} step {expected_step} token hashes do not "
                    "match the shared fixture"
                )
            key = (expected_dp_rank, expected_step)
            previous = replica_records.setdefault(key, (slots, hashes))
            if previous != (slots, hashes):
                raise TokenDataError(
                    f"replicas of DP rank {expected_dp_rank} used different inputs "
                    f"at step {expected_step}"
                )

    summary = {
        "valid": True,
        "mapping": INPUT_MAPPING,
        "steps": steps,
        "global_batch_size": global_batch_size,
        "training_local_batch_size": training_local_batch_size,
        "dp_world_size": dp_world_size,
        "context_parallel_degree": context_parallel_degree,
        "pipeline_parallel_degree": pipeline_parallel_degree,
        "validated_global_ranks": list(expected_global_ranks),
        "token_plan_step_series_sha256": step_series_digest(plan.step_sha256),
    }
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
