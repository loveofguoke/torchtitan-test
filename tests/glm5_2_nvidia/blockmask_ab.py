# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the BSD-style license in the TorchTitan repository.
"""Replay the historical block-mask algorithm in the current matched training harness."""
from unittest.mock import patch

import torch
from torch.nn.attention.flex_attention import BlockMask
from torchtitan.models.common.attention import FlexInnerAttention, MLAAttention
from torchtitan.models.glm5.dsa import DsaMetadata, Glm5MLAAttention
from tests.glm5_2_nvidia import attention_ab

# Verbatim from pytorch/torchtitan PR 5578 at cf7f91e3, including the per-score scan.
def build_dsa_block_mask(
    indices_QS: torch.Tensor,
    metadata: DsaMetadata,
    block_size: int = 128,
) -> BlockMask:
    """Build conservative KV block lists and an exact token-level predicate."""
    Q, S = indices_QS.shape
    K = metadata.positions_T.numel()
    B = block_size
    valid_QS = (indices_QS >= 0) & (indices_QS < K)
    safe_QS = indices_QS.clamp(0, K - 1).long()
    queries_Q = metadata.query_indices_Q
    valid_QS = valid_QS & metadata.valid_T[queries_Q, None] & metadata.valid_T[safe_QS]
    valid_QS = valid_QS & (
        metadata.documents_T[queries_Q, None] == metadata.documents_T[safe_QS]
    )
    valid_QS = valid_QS & (
        metadata.positions_T[queries_Q, None] >= metadata.positions_T[safe_QS]
    )
    num_q_blocks, num_k_blocks = (Q + B - 1) // B, (K + B - 1) // B
    block_ids_QS = (
        torch.arange(Q, device=indices_QS.device)[:, None] // B
    ) * num_k_blocks + safe_QS // B
    occupancy = torch.zeros(
        num_q_blocks * num_k_blocks, device=indices_QS.device, dtype=torch.int32
    )
    occupancy.scatter_add_(
        0, block_ids_QS.flatten(), valid_QS.flatten().to(torch.int32)
    )
    occupied = occupancy.view(1, 1, num_q_blocks, num_k_blocks) > 0
    counts = occupied.sum(-1).to(torch.int32)
    blocks = occupied.to(torch.float32).argsort(dim=-1, descending=True, stable=True)

    def mask_mod(b, h, q, k):
        row = q.clamp(max=Q - 1)
        return (
            (q < Q)
            & (k < K)
            & ((indices_QS[row] == k[..., None]) & valid_QS[row]).any(dim=-1)
        )

    return BlockMask.from_kv_blocks(
        counts,
        blocks.to(torch.int32),
        BLOCK_SIZE=B,
        mask_mod=mask_mod,
        seq_lengths=(Q, K),
    )


class HistoricalFlex(FlexInnerAttention):
    # The current GLM block consults this flag before dispatching the frozen path.
    warmup_context_length = None
    indexer_loss = None

    def forward(self, q, k, v, *, attention_metadata, topk_indices_QS, scale, **unused):
        mask = build_dsa_block_mask(topk_indices_QS, attention_metadata)
        if not getattr(self, "_reported", False):
            self._reported = True
            blocks = (q.shape[0] + 127) // 128
            causal = blocks * (blocks + 1) // 2
            visited = mask.kv_num_blocks.sum().item()
            print(f"BLOCKMASK first-use layer: visited={visited} causal={causal} "
                  f"skip_vs_causal={1-visited/causal:.4%} full_blocks=0", flush=True)
        return super().forward(q, k, v, attention_metadata=mask, scale=scale)


def main():
    init = Glm5MLAAttention.__init__

    def expanded_init(self, config):
        init(self, config)
        self.inner_attention = HistoricalFlex(FlexInnerAttention.Config())

    with patch.object(Glm5MLAAttention, "__init__", expanded_init), patch.object(
        Glm5MLAAttention, "forward", MLAAttention.forward
    ):
        attention_ab.main()
