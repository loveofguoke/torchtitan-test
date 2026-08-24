# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Distributed validation for GLM-5 HuggingFace checkpoint conversion."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file
from torch.distributed.checkpoint import (
    HuggingFaceStorageReader,
    HuggingFaceStorageWriter,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Shard

from torchtitan.models.glm5 import glm5_configs
from torchtitan.models.glm5.state_dict_adapter import Glm5StateDictAdapter


_GATE_KEY = "layers.1.moe.routed_experts.inner_experts.w1_EFD"
_UP_KEY = "layers.1.moe.routed_experts.inner_experts.w3_EFD"
_DOWN_KEY = "layers.1.moe.routed_experts.inner_experts.w2_EDF"
_HF_FUSED_KEY = "model.layers.1.mlp.experts.gate_up_proj"
_HF_DOWN_KEY = "model.layers.1.mlp.experts.down_proj"


def _keys(layer_index: int) -> tuple[str, str, str, str, str]:
    return (
        f"layers.{layer_index}.moe.routed_experts.inner_experts.w1_EFD",
        f"layers.{layer_index}.moe.routed_experts.inner_experts.w3_EFD",
        f"layers.{layer_index}.moe.routed_experts.inner_experts.w2_EDF",
        f"model.layers.{layer_index}.mlp.experts.gate_up_proj",
        f"model.layers.{layer_index}.mlp.experts.down_proj",
    )


def _source_weights(
    *,
    base_offset: int = 0,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_experts, hidden_dim, dim = 8, 256, 256
    gate = torch.arange(
        num_experts * hidden_dim * dim,
        dtype=torch.float32,
        device=device,
    ).reshape(num_experts, hidden_dim, dim) + base_offset
    up = gate + gate.numel()
    down = torch.arange(
        num_experts * dim * hidden_dim,
        dtype=torch.float32,
        device=device,
    ).reshape(num_experts, dim, hidden_dim)
    return gate, up, down


def _distributed_titan_state(
    mesh,
    *,
    zero: bool = False,
    expert_parallel: bool = False,
    layer_index: int = 1,
    base_offset: int = 0,
    device: torch.device | str = "cpu",
) -> dict[str, DTensor]:
    gate_key, up_key, down_key, _, _ = _keys(layer_index)
    gate, up, down = _source_weights(base_offset=base_offset, device=device)
    if zero:
        gate = torch.zeros_like(gate)
        up = torch.zeros_like(up)
        down = torch.zeros_like(down)
    if expert_parallel:
        gate_up_placements = (Shard(0), Shard(1))
        down_placements = (Shard(0), Shard(2))
    else:
        gate_up_placements = (Shard(1),)
        down_placements = (Shard(2),)
    return {
        gate_key: distribute_tensor(gate, mesh, gate_up_placements),
        up_key: distribute_tensor(up, mesh, gate_up_placements),
        down_key: distribute_tensor(down, mesh, down_placements),
    }


def run_tp_hf_roundtrip(
    rank: int,
    world_size: int,
    rendezvous_uri: str,
    checkpoint_path: str,
    expert_parallel: bool = False,
    backend: str = "gloo",
    device_type: str = "cpu",
) -> None:
    if device_type == "npu":
        torch.npu.set_device(rank)
    dist.init_process_group(
        backend=backend,
        init_method=rendezvous_uri,
        rank=rank,
        world_size=world_size,
    )
    try:
        if expert_parallel:
            mesh = init_device_mesh(
                device_type,
                (2, 2),
                mesh_dim_names=("ep", "tp"),
            )
        else:
            mesh = init_device_mesh(
                device_type,
                (world_size,),
                mesh_dim_names=("tp",),
            )
        config = glm5_configs["debugmodel"]()
        adapter = Glm5StateDictAdapter(config, hf_assets_path=None)
        source_state = _distributed_titan_state(
            mesh,
            expert_parallel=expert_parallel,
            device=device_type,
        )

        hf_state = adapter.to_hf(source_state)
        fused = hf_state[_HF_FUSED_KEY]
        if not isinstance(fused, DTensor):
            raise AssertionError(f"expected DTensor, got {type(fused)}")
        if fused._local_tensor.numel() * world_size != fused.numel():
            raise AssertionError("fused checkpoint conversion materialized a full TP tensor")
        gate, up, _ = _source_weights(device=device_type)
        tp_rank = mesh.get_local_rank("tp")
        if expert_parallel:
            ep_rank = mesh.get_local_rank("ep")
            num_local_experts = gate.shape[0] // 2
            expert_slice = slice(
                ep_rank * num_local_experts,
                (ep_rank + 1) * num_local_experts,
            )
            gate = gate[expert_slice]
            up = up[expert_slice]
        expected_local_fused = gate if tp_rank == 0 else up
        torch.testing.assert_close(fused._local_tensor, expected_local_fused)

        writer = HuggingFaceStorageWriter(
            path=checkpoint_path,
            save_distributed=True,
            enable_consolidation=True,
        )
        dcp.save(hf_state, storage_writer=writer)
        dist.barrier()

        if rank == 0:
            safetensor_files = sorted(Path(checkpoint_path).glob("*.safetensors"))
            if len(safetensor_files) != 1:
                raise AssertionError(
                    f"expected one consolidated safetensors file, got {safetensor_files}"
                )
            saved = load_file(safetensor_files[0])
            gate, up, down = _source_weights()
            torch.testing.assert_close(saved[_HF_FUSED_KEY], torch.cat((gate, up), dim=1))
            torch.testing.assert_close(saved[_HF_DOWN_KEY], down)

        dist.barrier()
        target_adapter = Glm5StateDictAdapter(config, hf_assets_path=None)
        target_state = _distributed_titan_state(
            mesh,
            zero=True,
            expert_parallel=expert_parallel,
            device=device_type,
        )
        target_hf_state = target_adapter.to_hf(target_state)
        dcp.load(
            target_hf_state,
            storage_reader=HuggingFaceStorageReader(checkpoint_path),
        )
        restored = target_adapter.from_hf(target_hf_state)
        gate, up, down = _source_weights(device=device_type)
        torch.testing.assert_close(restored[_GATE_KEY].full_tensor(), gate)
        torch.testing.assert_close(restored[_UP_KEY].full_tensor(), up)
        torch.testing.assert_close(restored[_DOWN_KEY].full_tensor(), down)
    finally:
        dist.destroy_process_group()


def run_pp_tp_hf_roundtrip(
    rank: int,
    world_size: int,
    rendezvous_uri: str,
    checkpoint_path: str,
    backend: str = "gloo",
    device_type: str = "cpu",
) -> None:
    if device_type == "npu":
        torch.npu.set_device(rank)
    dist.init_process_group(
        backend=backend,
        init_method=rendezvous_uri,
        rank=rank,
        world_size=world_size,
    )
    try:
        world_mesh = init_device_mesh(
            device_type,
            (2, 2),
            mesh_dim_names=("pp", "tp"),
        )
        pp_rank = world_mesh.get_local_rank("pp")
        tp_rank = world_mesh.get_local_rank("tp")
        tp_mesh = world_mesh["tp"]
        layer_index = 1 + pp_rank
        base_offset = pp_rank * 2_000_000
        gate_key, up_key, down_key, hf_fused_key, hf_down_key = _keys(layer_index)

        config = glm5_configs["debugmodel"]()
        adapter = Glm5StateDictAdapter(config, hf_assets_path=None)
        source_state = _distributed_titan_state(
            tp_mesh,
            layer_index=layer_index,
            base_offset=base_offset,
            device=device_type,
        )
        hf_state = adapter.to_hf(source_state)
        gate, up, _ = _source_weights(
            base_offset=base_offset,
            device=device_type,
        )
        expected_local_fused = gate if tp_rank == 0 else up
        torch.testing.assert_close(
            hf_state[hf_fused_key]._local_tensor,
            expected_local_fused,
        )

        writer = HuggingFaceStorageWriter(
            path=checkpoint_path,
            save_distributed=True,
            enable_consolidation=True,
        )
        dcp.save(hf_state, storage_writer=writer)
        dist.barrier()

        if rank == 0:
            safetensor_files = sorted(Path(checkpoint_path).glob("*.safetensors"))
            if len(safetensor_files) != 1:
                raise AssertionError(
                    f"expected one consolidated safetensors file, got {safetensor_files}"
                )
            saved = load_file(safetensor_files[0])
            for saved_pp_rank in range(2):
                saved_layer = 1 + saved_pp_rank
                _, _, _, saved_fused_key, saved_down_key = _keys(saved_layer)
                saved_gate, saved_up, saved_down = _source_weights(
                    base_offset=saved_pp_rank * 2_000_000
                )
                torch.testing.assert_close(
                    saved[saved_fused_key],
                    torch.cat((saved_gate, saved_up), dim=1),
                )
                torch.testing.assert_close(saved[saved_down_key], saved_down)

        dist.barrier()
        target_adapter = Glm5StateDictAdapter(config, hf_assets_path=None)
        target_state = _distributed_titan_state(
            tp_mesh,
            zero=True,
            layer_index=layer_index,
            base_offset=base_offset,
            device=device_type,
        )
        target_hf_state = target_adapter.to_hf(target_state)
        dcp.load(
            target_hf_state,
            storage_reader=HuggingFaceStorageReader(checkpoint_path),
        )
        restored = target_adapter.from_hf(target_hf_state)
        gate, up, down = _source_weights(
            base_offset=base_offset,
            device=device_type,
        )
        torch.testing.assert_close(restored[gate_key].full_tensor(), gate)
        torch.testing.assert_close(restored[up_key].full_tensor(), up)
        torch.testing.assert_close(restored[down_key].full_tensor(), down)
    finally:
        dist.destroy_process_group()
