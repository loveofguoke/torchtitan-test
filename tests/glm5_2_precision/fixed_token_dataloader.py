# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""TorchTitan dataloader backed by a topology-independent fixed token plan."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Iterator

import torch

from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.tokenizer import BaseTokenizer

from .token_data import (
    INPUT_MAPPING,
    batches_per_optimizer_step,
    global_slots_for_batch,
    load_token_plan,
    sample_digest,
)


TOKEN_PLAN_PATH_ENV = "GLM5_PRECISION_TOKEN_PLAN_PATH"
INPUT_CONTRACT_DIR_ENV = "GLM5_PRECISION_INPUT_CONTRACT_DIR"
GLOBAL_BATCH_SIZE_ENV = "GLM5_PRECISION_GLOBAL_BATCH_SIZE"
TRAINING_LOCAL_BATCH_SIZE_ENV = "GLM5_PRECISION_TRAINING_LOCAL_BATCH_SIZE"
TENSOR_PARALLEL_DEGREE_ENV = "GLM5_PRECISION_TENSOR_PARALLEL_DEGREE"
CONTEXT_PARALLEL_DEGREE_ENV = "GLM5_PRECISION_CONTEXT_PARALLEL_DEGREE"


class FixedTokenDataLoader(BaseDataLoader):
    """Map fixed optimizer-step sample slots onto the current DP topology."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        token_plan_path: str = ""
        global_batch_size: int = 0
        training_local_batch_size: int = 0
        input_contract_directory: str = ""

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        max_context_length: int,
        num_tokens_per_batch: int,
        **kwargs: Any,
    ) -> None:
        del tokenizer, kwargs
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        if num_tokens_per_batch % max_context_length:
            raise ValueError(
                "fixed token batches require an integer number of sequences"
            )
        self.dataloader_batch_size = num_tokens_per_batch // max_context_length
        self.global_batch_size = config.global_batch_size
        self.training_local_batch_size = config.training_local_batch_size
        self.plan = load_token_plan(config.token_plan_path)
        if self.plan.sequence_length != max_context_length:
            raise ValueError(
                f"token plan sequence length {self.plan.sequence_length} does not "
                f"match training sequence length {max_context_length}"
            )
        if self.plan.global_batch_size != self.global_batch_size:
            raise ValueError(
                f"token plan global batch {self.plan.global_batch_size} does not "
                f"match training global batch {self.global_batch_size}"
            )
        self.num_batches_per_step = batches_per_optimizer_step(
            global_batch_size=self.global_batch_size,
            training_local_batch_size=self.training_local_batch_size,
            dataloader_batch_size=self.dataloader_batch_size,
            dp_world_size=self.dp_world_size,
        )
        self._next_batch = 0
        self._tokens_IQ = torch.from_file(
            str(self.plan.token_file),
            shared=False,
            size=self.plan.num_samples * (self.plan.sequence_length + 1),
            dtype=torch.int32,
        ).view(self.plan.num_samples, self.plan.sequence_length + 1)
        self._positions_IL = torch.from_file(
            str(self.plan.position_file),
            shared=False,
            size=self.plan.num_samples * self.plan.sequence_length,
            dtype=torch.int32,
        ).view(self.plan.num_samples, self.plan.sequence_length)
        contract_directory = Path(config.input_contract_directory)
        contract_directory.mkdir(parents=True, exist_ok=True)
        self.global_rank = int(os.environ.get("RANK", "0"))
        tensor_parallel_degree = int(os.environ[TENSOR_PARALLEL_DEGREE_ENV])
        context_parallel_degree = int(os.environ[CONTEXT_PARALLEL_DEGREE_ENV])
        self.pp_rank = self.global_rank // (
            self.dp_world_size
            * context_parallel_degree
            * tensor_parallel_degree
        )
        self.contract_path = contract_directory / f"rank-{self.global_rank}.jsonl"
        self._contract_stream = self.contract_path.open(
            "w", encoding="utf-8", buffering=1
        )
        self._step_slots: list[int] = []
        self._step_hashes: list[str] = []

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        while self._next_batch < self.plan.steps * self.num_batches_per_step:
            optimizer_step = self._next_batch // self.num_batches_per_step
            batch_in_step = self._next_batch % self.num_batches_per_step
            slots = global_slots_for_batch(
                batch_in_step=batch_in_step,
                dp_rank=self.dp_rank,
                dp_world_size=self.dp_world_size,
                global_batch_size=self.global_batch_size,
                training_local_batch_size=self.training_local_batch_size,
                dataloader_batch_size=self.dataloader_batch_size,
            )
            indices = [
                optimizer_step * self.global_batch_size + slot for slot in slots
            ]
            tokens_BQ = self._tokens_IQ[indices]
            positions_BL = self._positions_IL[indices]
            observed_hashes: list[str] = []
            for index, slot in zip(indices, slots, strict=True):
                observed = sample_digest(
                    self._tokens_IQ[index].numpy().tobytes(),
                    self._positions_IL[index].numpy().tobytes(),
                )
                expected = self.plan.sample_hash(step=optimizer_step, slot=slot)
                if observed != expected:
                    raise RuntimeError(
                        f"fixed token sample checksum mismatch at optimizer step "
                        f"{optimizer_step + 1}, global slot {slot}"
                    )
                observed_hashes.append(observed)
            self._step_slots.extend(slots)
            self._step_hashes.extend(observed_hashes)
            self._next_batch += 1
            if batch_in_step + 1 == self.num_batches_per_step:
                self._write_step_contract(optimizer_step)

            input_BL = tokens_BQ[:, :-1].to(torch.long)
            labels_BL = tokens_BQ[:, 1:].to(torch.long)
            yield {
                "input": input_BL.reshape(-1),
                "positions": positions_BL.to(torch.long).reshape(-1),
            }, labels_BL.reshape(-1)

    def _write_step_contract(self, optimizer_step: int) -> None:
        record = {
            "schema_version": 1,
            "mapping": INPUT_MAPPING,
            "global_rank": self.global_rank,
            "dp_rank": self.dp_rank,
            "pp_rank": self.pp_rank,
            "dp_world_size": self.dp_world_size,
            "step": optimizer_step + 1,
            "dataloader_batch_size": self.dataloader_batch_size,
            "global_slots": self._step_slots,
            "sample_sha256": self._step_hashes,
        }
        self._contract_stream.write(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        )
        self._step_slots = []
        self._step_hashes = []

    def state_dict(self) -> dict[str, Any]:
        return {
            f"dp_rank_{self.dp_rank}": json.dumps(
                {"next_batch": self._next_batch},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
            "world_size": self.dp_world_size,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return
        if int(state_dict["world_size"]) != self.dp_world_size:
            raise ValueError(
                "fixed token dataloader resharding is not supported while resuming"
            )
        rank_state = state_dict.get(f"dp_rank_{self.dp_rank}")
        if rank_state is not None:
            if isinstance(rank_state, torch.Tensor):
                rank_state = bytes(
                    rank_state.detach().cpu().to(torch.uint8).flatten().tolist()
                )
            if not isinstance(rank_state, (bytes, bytearray)):
                raise ValueError("invalid fixed token dataloader checkpoint state")
            self._next_batch = int(json.loads(rank_state)["next_batch"])


def install_fixed_token_dataloader() -> None:
    """Replace the parsed training dataloader inside this test process only."""

    plan_path = os.environ.get(TOKEN_PLAN_PATH_ENV)
    if not plan_path:
        raise RuntimeError(f"{TOKEN_PLAN_PATH_ENV} must name the shared token plan")
    contract_directory = os.environ.get(INPUT_CONTRACT_DIR_ENV)
    if not contract_directory:
        raise RuntimeError(
            f"{INPUT_CONTRACT_DIR_ENV} must name the input-contract directory"
        )
    global_batch_size = int(os.environ[GLOBAL_BATCH_SIZE_ENV])
    training_local_batch_size = int(os.environ[TRAINING_LOCAL_BATCH_SIZE_ENV])

    from torchtitan.config import ConfigManager

    original_parse_args = ConfigManager.parse_args

    def parse_args_with_fixed_tokens(
        manager: ConfigManager,
        args: list[str] | None = None,
    ):
        config = original_parse_args(manager, args)
        config.dataloader = FixedTokenDataLoader.Config(
            token_plan_path=plan_path,
            global_batch_size=global_batch_size,
            training_local_batch_size=training_local_batch_size,
            input_contract_directory=contract_directory,
        )
        # Loading the shared seed checkpoint is intentionally model-only. The
        # fixed dataloader starts from token-plan step zero for every endpoint.
        config.checkpoint.initial_load_model_only = True
        return config

    ConfigManager.parse_args = parse_args_with_fixed_tokens
