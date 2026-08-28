"""Test-only forward-stage trace for the first GLM-5 MoE layer."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any


MOE_STAGE_NAMES = (
    "layer_input",
    "attention_norm",
    "attention_output",
    "attention_residual",
    "ffn_norm",
    "router_scores",
    "router_topk_scores",
    "router_topk_indices",
    "router_token_counts",
    "routed_experts_output",
    "shared_expert_output",
    "moe_output",
    "layer_output",
)


def _split_primary_and_auxiliary(value):
    if isinstance(value, tuple):
        return value[0], value[1:]
    return value, ()


def install_gpu_moe_layer_trace() -> None:
    output_path = os.environ.get("GLM5_GPU_MOE_LAYER_TRACE_PATH")
    if not output_path:
        return

    import torch
    from torch.distributed.tensor import DTensor
    from torchtitan.trainer import Trainer

    selected_layer = str(os.environ.get("GLM5_GPU_MOE_LAYER", "1"))
    selected_step = int(os.environ.get("GLM5_GPU_MOE_LAYER_TRACE_STEP", "1"))
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

    def record(name: str, tensor: torch.Tensor) -> None:
        local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
        cpu = local.detach().contiguous().cpu()
        raw = cpu.view(torch.uint8).numpy().tobytes()
        fp32 = cpu.float()
        payload = {
            "step": selected_step,
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
            raise RuntimeError("GPU MoE trace requires one model part")
        layers = getattr(self.model_parts[0], "layers", None)
        if layers is None or selected_layer not in layers:
            raise RuntimeError(
                f"GPU MoE trace could not find model.layers[{selected_layer!r}]"
            )
        layer = layers[selected_layer]
        if not layer.moe_enabled:
            raise RuntimeError(
                f"GPU MoE trace requires an MoE layer, got layer {selected_layer}"
            )

        def diagnostic_forward(
            x_TD: torch.Tensor,
            attention_masks: torch.Tensor,
            positions=None,
            *attention_args,
            **attention_kwargs,
        ):
            attention_norm_TD = layer.attention_norm(x_TD)
            attention_result = layer.attention(
                attention_norm_TD,
                attention_masks,
                positions,
                *attention_args,
                **attention_kwargs,
            )
            attention_output_TD, auxiliary_outputs = (
                _split_primary_and_auxiliary(attention_result)
            )
            attention_residual_TD = x_TD + attention_output_TD
            ffn_norm_TD = layer.ffn_norm(attention_residual_TD)

            moe = layer.moe
            (
                router_topk_scores_TK,
                router_topk_indices_TK,
                router_scores_TE,
            ) = moe.router(ffn_norm_TD, moe.expert_bias_E)
            routing_map_TE = torch.zeros_like(
                router_scores_TE, dtype=torch.bool
            ).scatter_(-1, router_topk_indices_TK, True)
            router_token_counts_E = routing_map_TE.sum(dim=0)
            if moe.training:
                with torch.no_grad():
                    moe.tokens_per_expert_E.add_(router_token_counts_E)

            routed_experts_output_TD = moe.routed_experts(
                ffn_norm_TD,
                router_topk_scores_TK,
                router_topk_indices_TK,
                router_token_counts_E,
            )
            if moe.shared_experts is None:
                shared_expert_output_TD = torch.zeros_like(
                    routed_experts_output_TD
                )
                moe_output_TD = routed_experts_output_TD
            else:
                shared_expert_output_TD = moe.shared_experts(ffn_norm_TD)
                moe_output_TD = (
                    routed_experts_output_TD + shared_expert_output_TD
                )
            layer_output_TD = attention_residual_TD + moe_output_TD
            model_output = (
                (layer_output_TD, *auxiliary_outputs)
                if auxiliary_outputs
                else layer_output_TD
            )
            stage_outputs = (
                x_TD,
                attention_norm_TD,
                attention_output_TD,
                attention_residual_TD,
                ffn_norm_TD,
                router_scores_TE,
                router_topk_scores_TK,
                router_topk_indices_TK,
                router_token_counts_E,
                routed_experts_output_TD,
                shared_expert_output_TD,
                moe_output_TD,
                layer_output_TD,
            )
            return model_output, stage_outputs

        def unwrap_and_record(outputs):
            model_output, stage_outputs = outputs
            if int(self.step) == selected_step:
                for name, value in zip(
                    MOE_STAGE_NAMES, stage_outputs, strict=True
                ):
                    tensor = first_tensor(value)
                    if tensor is not None:
                        record(name, tensor)
            return model_output

        was_compiled = getattr(layer, "_compiled_call_impl", None) is not None
        layer.forward = diagnostic_forward
        if was_compiled:
            layer.compile(backend=config.compile.backend, fullgraph=True)
            compiled_call = layer._compiled_call_impl
            if compiled_call is None:
                raise RuntimeError("GPU MoE trace failed to compile selected layer")

            def traced_compiled_call(*args, **kwargs):
                return unwrap_and_record(compiled_call(*args, **kwargs))

            layer._compiled_call_impl = traced_compiled_call
        else:

            def traced_forward(*args, **kwargs):
                return unwrap_and_record(diagnostic_forward(*args, **kwargs))

            layer.forward = traced_forward

    Trainer.__init__ = traced_init


__all__ = [
    "MOE_STAGE_NAMES",
    "_split_primary_and_auxiliary",
    "install_gpu_moe_layer_trace",
]
