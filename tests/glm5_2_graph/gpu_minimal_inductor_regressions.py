#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Minimal CUDA Inductor checks for rounding and materialization boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


@torch.library.custom_op(
    "torchtitan_test::gpu_diagnostic_materialize", mutates_args=()
)
def _materialize(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@_materialize.register_fake
def _materialize_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def _silu_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.silu(a) * b


def _silu_materialized_multiply(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    return torch.ops.torchtitan_test.gpu_diagnostic_materialize(F.silu(a)) * b


def _linear_residual(
    x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    return residual + F.linear(x, weight)


def _linear_materialized_residual(
    x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    projected = torch.ops.torchtitan_test.gpu_diagnostic_materialize(
        F.linear(x, weight)
    )
    return residual + projected


def _sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _compare(name: str, eager_fn, compiled_fn, arguments) -> dict[str, object]:
    eager = eager_fn(*arguments)
    compiled = torch.compile(compiled_fn, backend="inductor", fullgraph=True)(
        *arguments
    )
    torch.cuda.synchronize()
    difference = compiled.float() - eager.float()
    return {
        "name": name,
        "shape": list(eager.shape),
        "dtype": str(eager.dtype),
        "eager_sha256": _sha256(eager),
        "inductor_sha256": _sha256(compiled),
        "exact": bool(torch.equal(eager, compiled)),
        "mismatch_count": int(torch.count_nonzero(eager != compiled).item()),
        "mean_abs_diff": float(difference.abs().mean().item()),
        "max_abs_diff": float(difference.abs().max().item()),
    }


def _dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16}[name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("fp32", "bf16"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.use_deterministic_algorithms(True, warn_only=False)
    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    generator = torch.Generator(device="cpu").manual_seed(61)

    silu_a = torch.randn(64, 64, generator=generator).to(dtype).to(device)
    silu_b = torch.randn(64, 64, generator=generator).to(dtype).to(device)
    linear_x = torch.randn(32, 128, generator=generator).to(dtype).to(device)
    linear_weight = torch.randn(256, 128, generator=generator).to(dtype).to(device)
    residual = torch.randn(32, 256, generator=generator).to(dtype).to(device)

    rows = [
        _compare("silu_multiply", _silu_multiply, _silu_multiply, (silu_a, silu_b)),
        _compare(
            "silu_materialized_multiply",
            _silu_multiply,
            _silu_materialized_multiply,
            (silu_a, silu_b),
        ),
        _compare(
            "linear_residual",
            _linear_residual,
            _linear_residual,
            (linear_x, linear_weight, residual),
        ),
        _compare(
            "linear_materialized_residual",
            _linear_residual,
            _linear_materialized_residual,
            (linear_x, linear_weight, residual),
        ),
    ]
    metadata = {
        "type": "metadata",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "dtype": args.dtype,
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "allow_bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "allow_fp16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(
            json.dumps(row, sort_keys=True) for row in (metadata, *rows)
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, sort_keys=True))
    for row in rows:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
