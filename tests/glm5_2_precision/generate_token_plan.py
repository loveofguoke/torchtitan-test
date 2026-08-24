#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Materialize TorchTitan's unsharded text stream as a fixed token plan."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import torch

from .token_data import (
    TOKEN_PLAN_SCHEMA,
    TOKEN_PLAN_VERSION,
    sample_digest,
    sha256_file,
    step_digest,
)


OUTPUT_ENV = "GLM5_PRECISION_TOKEN_PLAN_OUTPUT"
STEPS_ENV = "GLM5_PRECISION_TOKEN_PLAN_STEPS"
GLOBAL_BATCH_ENV = "GLM5_PRECISION_TOKEN_PLAN_GLOBAL_BATCH_SIZE"
SEQUENCE_LENGTH_ENV = "GLM5_PRECISION_TOKEN_PLAN_SEQUENCE_LENGTH"


def _build_source_dataloader(
    config: Any,
    *,
    tokenizer: Any,
    global_batch_size: int,
    sequence_length: int,
) -> Any:
    return config.dataloader.build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=tokenizer,
        max_context_length=sequence_length,
        num_tokens_per_batch=global_batch_size * sequence_length,
    )


def _reshape_source_batch(
    input_dict: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    global_batch_size: int,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    expected_num_tokens = global_batch_size * sequence_length
    inputs_T = input_dict["input"]
    positions_T = input_dict["positions"]
    for name, tensor in (
        ("input", inputs_T),
        ("labels", labels),
        ("positions", positions_T),
    ):
        if tensor.numel() != expected_num_tokens:
            raise RuntimeError(
                f"unexpected {name} token count: {tensor.numel()}, "
                f"expected {expected_num_tokens}"
            )
    return (
        inputs_T.reshape(global_batch_size, sequence_length),
        labels.reshape(global_batch_size, sequence_length),
        positions_T.reshape(global_batch_size, sequence_length),
    )


def _write_plan() -> None:
    output = Path(os.environ[OUTPUT_ENV]).resolve()
    steps = int(os.environ[STEPS_ENV])
    global_batch_size = int(os.environ[GLOBAL_BATCH_ENV])
    sequence_length = int(os.environ[SEQUENCE_LENGTH_ENV])

    from torchtitan.config import ConfigManager

    config = ConfigManager().parse_args()
    tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)
    dataloader = _build_source_dataloader(
        config,
        tokenizer=tokenizer,
        global_batch_size=global_batch_size,
        sequence_length=sequence_length,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    token_path = temporary / "tokens.i32"
    position_path = temporary / "positions.i32"
    sample_hashes: list[str] = []
    step_hashes: list[str] = []
    try:
        iterator = iter(dataloader)
        with token_path.open("wb") as token_stream, position_path.open(
            "wb"
        ) as position_stream:
            for step in range(steps):
                input_dict, labels_T = next(iterator)
                inputs_BL, labels_BL, positions_BL = _reshape_source_batch(
                    input_dict,
                    labels_T,
                    global_batch_size=global_batch_size,
                    sequence_length=sequence_length,
                )
                if not torch.equal(inputs_BL[:, 1:], labels_BL[:, :-1]):
                    raise RuntimeError(
                        "fixed token plans require next-token shifted labels"
                    )
                tokens_BQ = torch.cat((inputs_BL, labels_BL[:, -1:]), dim=1).to(
                    dtype=torch.int32
                ).contiguous()
                positions_BL = positions_BL.to(dtype=torch.int32).contiguous()
                token_stream.write(tokens_BQ.numpy().tobytes())
                position_stream.write(positions_BL.numpy().tobytes())
                current_step_hashes: list[str] = []
                for slot in range(global_batch_size):
                    digest = sample_digest(
                        tokens_BQ[slot].numpy().tobytes(),
                        positions_BL[slot].numpy().tobytes(),
                    )
                    sample_hashes.append(digest)
                    current_step_hashes.append(digest)
                step_hashes.append(step_digest(current_step_hashes))

        manifest = {
            "schema": TOKEN_PLAN_SCHEMA,
            "schema_version": TOKEN_PLAN_VERSION,
            "steps": steps,
            "global_batch_size": global_batch_size,
            "sequence_length": sequence_length,
            "sample_sha256": sample_hashes,
            "step_sha256": step_hashes,
            "files": {
                "tokens": {"name": token_path.name, "sha256": sha256_file(token_path)},
                "positions": {
                    "name": position_path.name,
                    "sha256": sha256_file(position_path),
                },
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            shutil.rmtree(output)
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    _write_plan()
