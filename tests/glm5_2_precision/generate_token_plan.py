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


def _write_plan() -> None:
    output = Path(os.environ[OUTPUT_ENV]).resolve()
    steps = int(os.environ[STEPS_ENV])
    global_batch_size = int(os.environ[GLOBAL_BATCH_ENV])
    sequence_length = int(os.environ[SEQUENCE_LENGTH_ENV])

    from torchtitan.config import ConfigManager

    config = ConfigManager().parse_args()
    tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)
    dataloader = config.dataloader.build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=tokenizer,
        seq_len=sequence_length,
        local_batch_size=global_batch_size,
        snapshot_every_n_steps=None,
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
                input_dict, labels = next(iterator)
                inputs_BL = input_dict["input"]
                positions_BL = input_dict["positions"]
                if tuple(inputs_BL.shape) != (global_batch_size, sequence_length):
                    raise RuntimeError(
                        f"unexpected input shape at step {step + 1}: "
                        f"{tuple(inputs_BL.shape)}"
                    )
                if tuple(labels.shape) != tuple(inputs_BL.shape):
                    raise RuntimeError(
                        f"label shape does not match input shape at step {step + 1}"
                    )
                if tuple(positions_BL.shape) != tuple(inputs_BL.shape):
                    raise RuntimeError(
                        f"position shape does not match input shape at step {step + 1}"
                    )
                if not torch.equal(inputs_BL[:, 1:], labels[:, :-1]):
                    raise RuntimeError(
                        "fixed token plans require next-token shifted labels"
                    )
                tokens_BQ = torch.cat((inputs_BL, labels[:, -1:]), dim=1).to(
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
