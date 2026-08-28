"""Test-only dense-SwiGLU SiLU materialization for CUDA training A/B runs."""

from __future__ import annotations

import os


MATERIALIZATION_POINTS = frozenset(("w1", "w3", "silu", "product", "down"))


def install_gpu_silu_materialization() -> None:
    import torch
    import torch.nn.functional as F
    from torchtitan.models.common.feed_forward import FeedForward

    points = frozenset(
        point.strip()
        for point in os.environ.get(
            "GLM5_GPU_DENSE_SILU_MATERIALIZATION_POINTS", "silu"
        ).split(",")
        if point.strip()
    )
    unknown = points - MATERIALIZATION_POINTS
    if unknown or not points:
        raise ValueError(
            "invalid dense-SwiGLU materialization points: "
            f"{sorted(unknown or points)}"
        )

    @torch.library.custom_op(
        "torchtitan_test::gpu_training_materialize",
        mutates_args=(),
        schema="(Tensor x) -> Tensor",
    )
    def materialize(x: torch.Tensor) -> torch.Tensor:
        return x.clone()

    @materialize.register_fake
    def materialize_fake(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    def materialize_backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        del ctx
        return grad

    torch.library.register_autograd(
        "torchtitan_test::gpu_training_materialize",
        materialize_backward,
    )

    def apply_boundary(point: str, tensor: torch.Tensor) -> torch.Tensor:
        if point not in points:
            return tensor
        return torch.ops.torchtitan_test.gpu_training_materialize(tensor)

    def materialized_forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_linear = apply_boundary("w1", self.w1(x))
        gate = apply_boundary("silu", F.silu(gate_linear))
        up = apply_boundary("w3", self.w3(x))
        product = apply_boundary("product", gate * up)
        return apply_boundary("down", self.w2(product))

    FeedForward.forward = materialized_forward
    print(
        "Installed CUDA dense-SwiGLU materialization boundaries: "
        + ",".join(sorted(points)),
        flush=True,
    )


__all__ = ["MATERIALIZATION_POINTS", "install_gpu_silu_materialization"]
