#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Compare an optional GLM-5 DSA operator with its same-device reference."""

import argparse
import json
from pathlib import Path
import platform
import sys

import torch

from torchtitan.models.glm5.model import DSAIndexerTopK, SparseMLA


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("gpu", "npu"), required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Result directory (default: smoke_runs/full-dsa-operator/<device>).",
    )
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--queries", type=int, default=16)
    parser.add_argument("--keys", type=int, default=128)
    parser.add_argument("--output-atol", type=float, default=0.05)
    parser.add_argument("--grad-atol", type=float, default=0.05)
    parser.add_argument("--index-overlap", type=float, default=0.99)
    return parser.parse_args()


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return (left.float() - right.float()).abs().max().item()


def _sparse_mla_probe(
    *,
    device: torch.device,
    device_kind: str,
    queries: int,
    keys: int,
) -> dict[str, float | bool]:
    num_heads = 64
    latent_dim = 512
    rope_dim = 64
    topk = 64
    if keys < topk:
        raise ValueError("--keys must be at least 64 for the operator probe.")

    q_QNH = torch.randn(
        queries,
        num_heads,
        latent_dim + rope_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    kv_K1H = torch.randn(
        keys,
        1,
        latent_dim + rope_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    topk_indices_QS = torch.arange(
        topk, device=device, dtype=torch.int32
    ).repeat(queries, 1)
    attention_masks_1QK = torch.zeros(
        1, queries, keys, device=device, dtype=torch.float32
    )
    scale = (latent_dim + rope_dim) ** -0.5

    reference = SparseMLA.Config(attention_dropout=0.0).build().to(device)
    if device_kind == "gpu":
        from torchtitan.models.glm5.ops.tilelang import TileLangSparseMLA

        candidate = TileLangSparseMLA.Config(attention_dropout=0.0).build().to(
            device
        )
    else:
        from torchtitanturbo.models.glm5.ops.sparse_mla import NpuSparseMLA

        candidate = NpuSparseMLA.Config(attention_dropout=0.0).build().to(device)

    reference.train()
    candidate.train()
    ref_q = q_QNH.detach().clone().requires_grad_(True)
    ref_kv = kv_K1H.detach().clone().requires_grad_(True)
    candidate_q = q_QNH.detach().clone().requires_grad_(True)
    candidate_kv = kv_K1H.detach().clone().requires_grad_(True)

    reference_output = reference(
        ref_q,
        ref_kv,
        attention_masks_1QK,
        topk_indices_QS,
        scale=scale,
        latent_dim=latent_dim,
    )
    candidate_output = candidate(
        candidate_q,
        candidate_kv,
        attention_masks_1QK,
        topk_indices_QS,
        scale=scale,
        latent_dim=latent_dim,
    )
    reference_output.float().square().mean().backward()
    candidate_output.float().square().mean().backward()
    assert ref_q.grad is not None and ref_kv.grad is not None
    assert candidate_q.grad is not None and candidate_kv.grad is not None

    return {
        "output_max_abs": _max_abs(candidate_output, reference_output),
        "q_grad_max_abs": _max_abs(candidate_q.grad, ref_q.grad),
        "kv_grad_max_abs": _max_abs(candidate_kv.grad, ref_kv.grad),
    }


@torch.no_grad()
def _gpu_indexer_probe(
    *,
    device: torch.device,
    queries: int,
    keys: int,
) -> dict[str, float]:
    from torchtitan.models.glm5.ops.tilelang import TileLangDSAIndexerTopK

    num_heads = 32
    head_dim = 128
    topk = min(64, keys)
    scale = head_dim**-0.5
    q_QNH = torch.randn(
        queries,
        num_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    k_KH = torch.randn(keys, head_dim, device=device, dtype=torch.bfloat16)
    weights_QN = torch.randn(
        queries, num_heads, device=device, dtype=torch.float32
    )
    attention_mask_QK = torch.zeros(
        queries, keys, device=device, dtype=torch.float32
    )

    reference = DSAIndexerTopK.Config(
        index_topk=topk,
        softmax_scale=scale,
    ).build()
    candidate = TileLangDSAIndexerTopK.Config(
        index_topk=topk,
        softmax_scale=scale,
        query_block_size=8,
    ).build()
    reference_indices_QS = reference(
        q_QNH, k_KH, weights_QN, attention_mask_QK
    )
    candidate_indices_QS = candidate(
        q_QNH, k_KH, weights_QN, attention_mask_QK
    )
    exact_rows = (candidate_indices_QS == reference_indices_QS).all(dim=-1)
    overlap = []
    for candidate_row, reference_row in zip(
        candidate_indices_QS.tolist(), reference_indices_QS.tolist()
    ):
        overlap.append(len(set(candidate_row).intersection(reference_row)) / topk)
    return {
        "exact_row_fraction": exact_rows.float().mean().item(),
        "mean_topk_overlap": sum(overlap) / len(overlap),
    }


def main() -> int:
    args = _parse_args()
    if args.device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        device = torch.device("cuda")
    else:
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            raise RuntimeError("NPU is not available.")
        device = torch.device("npu")

    torch.manual_seed(args.seed)
    output_dir = (
        args.output_dir
        or Path("smoke_runs") / "full-dsa-operator" / args.device
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    sparse = _sparse_mla_probe(
        device=device,
        device_kind=args.device,
        queries=args.queries,
        keys=args.keys,
    )
    indexer = (
        _gpu_indexer_probe(device=device, queries=args.queries, keys=args.keys)
        if args.device == "gpu"
        else {"status": "reference_only"}
    )
    passed = (
        sparse["output_max_abs"] <= args.output_atol
        and sparse["q_grad_max_abs"] <= args.grad_atol
        and sparse["kv_grad_max_abs"] <= args.grad_atol
        and (
            args.device != "gpu"
            or indexer["mean_topk_overlap"] >= args.index_overlap
        )
    )
    result = {
        "passed": passed,
        "device": args.device,
        "seed": args.seed,
        "queries": args.queries,
        "keys": args.keys,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "argv": sys.argv,
        "thresholds": {
            "output_atol": args.output_atol,
            "grad_atol": args.grad_atol,
            "index_overlap": args.index_overlap,
        },
        "sparse_mla": sparse,
        "indexer": indexer,
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Result: {result_path.resolve()}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
