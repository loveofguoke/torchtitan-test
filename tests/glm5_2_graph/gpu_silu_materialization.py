"""Test-only dense-SwiGLU SiLU materialization for CUDA training A/B runs."""

from __future__ import annotations


def install_gpu_silu_materialization() -> None:
    import torch
    import torch.nn.functional as F
    from torchtitan.models.common.feed_forward import FeedForward

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

    def materialized_forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.ops.torchtitan_test.gpu_training_materialize(
            F.silu(self.w1(x))
        )
        return self.w2(gate * self.w3(x))

    FeedForward.forward = materialized_forward
    print(
        "Installed CUDA dense-SwiGLU SiLU materialization boundary",
        flush=True,
    )


__all__ = ["install_gpu_silu_materialization"]
