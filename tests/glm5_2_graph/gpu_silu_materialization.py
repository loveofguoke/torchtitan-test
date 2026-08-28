"""Test-only dense-SwiGLU SiLU materialization for CUDA training A/B runs."""

from __future__ import annotations

import os


MATERIALIZATION_POINTS = frozenset(
    (
        "attention_output",
        "attention_residual",
        "ffn_input",
        "w1",
        "w3",
        "silu",
        "product",
        "down",
    )
)


def install_gpu_silu_materialization() -> None:
    import torch
    import torch.nn.functional as F
    from torchtitan.models.common.feed_forward import FeedForward
    from torchtitan.models.glm5.model import Glm5TransformerBlock

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
    block_points = points & {
        "attention_output",
        "attention_residual",
        "ffn_input",
    }
    if block_points:

        def materialized_block_forward(
            self,
            x_TD: torch.Tensor,
            attention_masks: torch.Tensor,
            positions=None,
            *attention_args,
            **attention_kwargs,
        ):
            attention_result = self.attention(
                self.attention_norm(x_TD),
                attention_masks,
                positions,
                *attention_args,
                **attention_kwargs,
            )
            if isinstance(attention_result, tuple):
                attention_output_TD = attention_result[0]
                auxiliary_outputs = attention_result[1:]
            else:
                attention_output_TD = attention_result
                auxiliary_outputs = ()
            if not self.moe_enabled:
                attention_output_TD = apply_boundary(
                    "attention_output", attention_output_TD
                )
            attention_residual_TD = x_TD + attention_output_TD
            if not self.moe_enabled:
                attention_residual_TD = apply_boundary(
                    "attention_residual", attention_residual_TD
                )
            ffn_input_TD = self.ffn_norm(attention_residual_TD)
            if not self.moe_enabled:
                ffn_input_TD = apply_boundary("ffn_input", ffn_input_TD)
            ffn_output_TD = (
                self.moe(ffn_input_TD)
                if self.moe_enabled
                else self.feed_forward(ffn_input_TD)
            )
            block_output_TD = attention_residual_TD + ffn_output_TD
            return (
                (block_output_TD, *auxiliary_outputs)
                if auxiliary_outputs
                else block_output_TD
            )

        Glm5TransformerBlock.forward = materialized_block_forward
    print(
        "Installed CUDA dense-SwiGLU materialization boundaries: "
        + ",".join(sorted(points)),
        flush=True,
    )


__all__ = ["MATERIALIZATION_POINTS", "install_gpu_silu_materialization"]
