# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Tests for GLM-5 distributed HuggingFace checkpoint conversion."""

from pathlib import Path

import pytest
import torch

from tests.glm5_2_checkpoint.distributed_hf_roundtrip import (
    run_pp_tp_hf_roundtrip,
    run_tp_hf_roundtrip,
)


@pytest.mark.parametrize(
    ("world_size", "expert_parallel"),
    ((2, False), (4, True)),
    ids=("tp2", "ep2-tp2"),
)
def test_distributed_hf_safetensors_roundtrip(
    tmp_path: Path,
    world_size: int,
    expert_parallel: bool,
) -> None:
    rendezvous_uri = (tmp_path / "rendezvous").as_uri()
    checkpoint_path = str(tmp_path / "hf_checkpoint")
    torch.multiprocessing.spawn(
        run_tp_hf_roundtrip,
        args=(world_size, rendezvous_uri, checkpoint_path, expert_parallel),
        nprocs=world_size,
        join=True,
    )


def test_pp2_tp2_hf_safetensors_roundtrip(tmp_path: Path) -> None:
    world_size = 4
    rendezvous_uri = (tmp_path / "rendezvous").as_uri()
    checkpoint_path = str(tmp_path / "hf_checkpoint")
    torch.multiprocessing.spawn(
        run_pp_tp_hf_roundtrip,
        args=(world_size, rendezvous_uri, checkpoint_path),
        nprocs=world_size,
        join=True,
    )
