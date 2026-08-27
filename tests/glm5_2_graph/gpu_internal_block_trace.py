"""Test-only CUDA GLM-5 Block 0 internal forward-stage tracing."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any


STAGE_NAMES = (
    "block_input",
    "attention_norm",
    "attention_output",
    "attention_residual",
    "ffn_norm",
    "ffn_gate_linear",
    "ffn_gate_silu",
    "ffn_up_linear",
    "ffn_gated_product",
    "ffn_down_projection",
    "block_output",
)


def install_gpu_internal_block_trace() -> None:
    output_path = os.environ.get("GLM5_GPU_INTERNAL_TRACE_PATH")
    if not output_path:
        return

    import torch
    import torch.nn.functional as F
    from torch.distributed.tensor import DTensor
    from torchtitan.trainer import Trainer

    selected_steps = {
        int(value)
        for value in os.environ.get(
            "GLM5_GPU_INTERNAL_TRACE_STEPS", "1,2,8,9,10"
        ).split(",")
        if value.strip()
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    lock = threading.Lock()

    def first_tensor(value: Any) -> torch.Tensor | None:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                tensor = first_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    def record(*, step: int, name: str, tensor: torch.Tensor) -> None:
        local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
        cpu = local.detach().contiguous().cpu()
        raw = cpu.view(torch.uint8).numpy().tobytes()
        fp32 = cpu.float()
        payload = {
            "step": step,
            "name": name,
            "kind": "forward",
            "shape": list(cpu.shape),
            "dtype": str(cpu.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "mean": float(fp32.mean().item()),
            "max_abs": float(fp32.abs().max().item()),
            "l2": float(torch.linalg.vector_norm(fp32).item()),
        }
        with lock, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    original_init = Trainer.__init__

    def traced_init(self, config):
        original_init(self, config)
        if len(self.model_parts) != 1:
            raise RuntimeError("GPU internal trace requires one model part")
        layers = getattr(self.model_parts[0], "layers", None)
        if layers is None or "0" not in layers:
            raise RuntimeError("GPU internal trace could not find model.layers['0']")
        layer = layers["0"]
        if layer.moe_enabled:
            raise RuntimeError("GPU internal trace requires dense Block 0")

        def diagnostic_forward(
            x_TD: torch.Tensor,
            attention_masks: torch.Tensor,
            positions: torch.Tensor | None = None,
        ):
            attention_norm_TD = layer.attention_norm(x_TD)
            attention_output_TD = layer.attention(
                attention_norm_TD, attention_masks, positions
            )
            attention_residual_TD = x_TD + attention_output_TD
            ffn_norm_TD = layer.ffn_norm(attention_residual_TD)
            feed_forward = layer.feed_forward
            ffn_gate_linear_TF = feed_forward.w1(ffn_norm_TD)
            ffn_gate_silu_TF = F.silu(ffn_gate_linear_TF)
            ffn_up_linear_TF = feed_forward.w3(ffn_norm_TD)
            ffn_gated_product_TF = ffn_gate_silu_TF * ffn_up_linear_TF
            ffn_down_projection_TD = feed_forward.w2(ffn_gated_product_TF)
            block_output_TD = attention_residual_TD + ffn_down_projection_TD
            return (
                x_TD,
                attention_norm_TD,
                attention_output_TD,
                attention_residual_TD,
                ffn_norm_TD,
                ffn_gate_linear_TF,
                ffn_gate_silu_TF,
                ffn_up_linear_TF,
                ffn_gated_product_TF,
                ffn_down_projection_TD,
                block_output_TD,
            )

        def unwrap_and_record(outputs):
            step = int(self.step)
            if step in selected_steps:
                for name, value in zip(STAGE_NAMES, outputs, strict=True):
                    tensor = first_tensor(value)
                    if tensor is not None:
                        record(step=step, name=name, tensor=tensor)
            return outputs[-1]

        was_compiled = getattr(layer, "_compiled_call_impl", None) is not None
        layer.forward = diagnostic_forward
        if was_compiled:
            layer.compile(backend=config.compile.backend, fullgraph=True)
            compiled_call = layer._compiled_call_impl
            if compiled_call is None:
                raise RuntimeError("GPU internal trace failed to compile Block 0")

            def traced_compiled_call(*args, **kwargs):
                return unwrap_and_record(compiled_call(*args, **kwargs))

            layer._compiled_call_impl = traced_compiled_call
        else:

            def traced_forward(*args, **kwargs):
                return unwrap_and_record(diagnostic_forward(*args, **kwargs))

            layer.forward = traced_forward

    Trainer.__init__ = traced_init


__all__ = ["STAGE_NAMES", "install_gpu_internal_block_trace"]
