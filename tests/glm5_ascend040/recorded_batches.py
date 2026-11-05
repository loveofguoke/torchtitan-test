"""Persist the actual CPU microbatches for an exact cross-device replay."""

import hashlib
import json
from dataclasses import fields
from pathlib import Path

import torch
from torch.utils._pytree import tree_leaves, tree_map
from torchtitan.components.data.types import TokenizedTrainingMicrobatch


def record_batches(iterator, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for index, batch in enumerate(iterator, 1):
            if not isinstance(batch, TokenizedTrainingMicrobatch):
                raise TypeError(
                    "Recorded GLM inputs require TokenizedTrainingMicrobatch"
                )
            values = {field.name: getattr(batch, field.name) for field in fields(batch)}
            if any(
                isinstance(value, torch.Tensor) and value.device.type != "cpu"
                for value in tree_leaves(values)
            ):
                raise ValueError("Record inputs before moving them to the accelerator")
            values = tree_map(
                lambda value: (
                    value.detach().clone() if isinstance(value, torch.Tensor) else value
                ),
                values,
            )
            filename = f"microbatch-{index:06d}.pt"
            path = directory / filename
            torch.save(values, path)
            manifest.write(
                json.dumps(
                    {
                        "microbatch": index,
                        "file": filename,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
                + "\n"
            )
            manifest.flush()
            yield batch


def replay_batches(directory):
    directory = Path(directory)
    with (directory / "manifest.jsonl").open(encoding="utf-8") as manifest:
        for index, line in enumerate(manifest, 1):
            row = json.loads(line)
            filename = f"microbatch-{index:06d}.pt"
            if row["microbatch"] != index or row["file"] != filename:
                raise ValueError("Recorded microbatch order is not contiguous")
            path = directory / filename
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"Recorded input checksum mismatch: {filename}")
            yield TokenizedTrainingMicrobatch(
                **torch.load(path, map_location="cpu", weights_only=True)
            )
