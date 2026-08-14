"""Materialize and replay topology-invariant global training batches."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


FIXED_BATCHES_ENV = "GLM5_PRECISION_FIXED_BATCHES_PATH"
TENSOR_KEYS = ("input", "positions", "labels")


def materialize_fixed_batches(
    output: Path,
    *,
    dataset_path: Path,
    tokenizer_path: Path,
    steps: int,
    global_batch_size: int,
    sequence_length: int,
) -> None:
    """Tokenize once at DP degree one and save exact global input tensors."""

    from torchtitan.components.tokenizer import HuggingFaceTokenizer
    from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataset

    tokenizer = HuggingFaceTokenizer.Config().build(tokenizer_path=str(tokenizer_path))
    dataset = HuggingFaceTextDataset(
        dataset_name="c4_test",
        dataset_path=str(dataset_path),
        tokenizer=tokenizer,
        seq_len=sequence_length,
        dp_rank=0,
        dp_world_size=1,
        infinite=True,
    )
    num_rows = steps * global_batch_size
    tensors = {
        key: torch.empty((num_rows, sequence_length), dtype=torch.int64)
        for key in TENSOR_KEYS
    }
    iterator = iter(dataset)
    for row in range(num_rows):
        inputs, labels = next(iterator)
        tensors["input"][row].copy_(inputs["input"])
        tensors["positions"][row].copy_(inputs["positions"])
        tensors["labels"][row].copy_(labels)

    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(output),
        metadata={
            "steps": str(steps),
            "global_batch_size": str(global_batch_size),
            "sequence_length": str(sequence_length),
        },
    )


class FixedGlobalBatchDataLoader:
    """Replay rank slices from precomputed global batches."""

    def __init__(
        self,
        path: Path,
        *,
        dp_world_size: int,
        dp_rank: int,
        local_batch_size: int,
        sequence_length: int,
    ) -> None:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        if metadata is None:
            raise ValueError(f"fixed batch metadata is missing: {path}")
        self.steps = int(metadata["steps"])
        self.global_batch_size = int(metadata["global_batch_size"])
        stored_sequence_length = int(metadata["sequence_length"])
        if stored_sequence_length != sequence_length:
            raise ValueError(
                "fixed batch sequence length does not match training: "
                f"{stored_sequence_length} != {sequence_length}"
            )
        if not 0 <= dp_rank < dp_world_size:
            raise ValueError(f"invalid DP rank {dp_rank} for degree {dp_world_size}")
        rows_per_wave = local_batch_size * dp_world_size
        if self.global_batch_size % rows_per_wave != 0:
            raise ValueError(
                "fixed global batch size must be divisible by local batch size "
                f"times DP degree: {self.global_batch_size} % {rows_per_wave} != 0"
            )
        self.tensors = load_file(str(path), device="cpu")
        if set(self.tensors) != set(TENSOR_KEYS):
            raise ValueError(f"unexpected fixed batch tensors: {sorted(self.tensors)}")
        expected_shape = (self.steps * self.global_batch_size, sequence_length)
        for key, tensor in self.tensors.items():
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"fixed batch tensor {key} has shape {tuple(tensor.shape)}, "
                    f"expected {expected_shape}"
                )
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.local_batch_size = local_batch_size
        self.num_waves = self.global_batch_size // rows_per_wave
        self.cursor = 0

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        total_batches = self.steps * self.num_waves
        while self.cursor < total_batches:
            step, wave = divmod(self.cursor, self.num_waves)
            start = (
                step * self.global_batch_size
                + wave * self.local_batch_size * self.dp_world_size
                + self.dp_rank * self.local_batch_size
            )
            stop = start + self.local_batch_size
            self.cursor += 1
            yield (
                {
                    "input": self.tensors["input"][start:stop],
                    "positions": self.tensors["positions"][start:stop],
                },
                self.tensors["labels"][start:stop],
            )

    def state_dict(self) -> dict[str, int]:
        return {
            "cursor": self.cursor,
            "world_size": self.dp_world_size,
            "rank": self.dp_rank,
        }

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        if not state_dict:
            return
        if state_dict["world_size"] != self.dp_world_size:
            raise ValueError("fixed batch dataloader DP degree changed on resume")
        if state_dict["rank"] != self.dp_rank:
            raise ValueError("fixed batch dataloader DP rank changed on resume")
        self.cursor = state_dict["cursor"]


def install_fixed_batch_dataloader() -> None:
    """Replace the GLM text dataloader when a fixed batch path is configured."""

    configured_path = os.environ.get(FIXED_BATCHES_ENV)
    if not configured_path:
        return

    from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader

    path = Path(configured_path)

    def build_fixed(_config, **kwargs):
        return FixedGlobalBatchDataLoader(
            path,
            dp_world_size=kwargs["dp_world_size"],
            dp_rank=kwargs["dp_rank"],
            local_batch_size=kwargs["local_batch_size"],
            sequence_length=kwargs["seq_len"],
        )

    HuggingFaceTextDataLoader.Config.build = build_fixed
