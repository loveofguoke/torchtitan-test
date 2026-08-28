"""Test-only CUDA GLM-5 Block 0 absorbed-attention forward tracing."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading


ATTENTION_STAGE_NAMES = (
    "block_input",
    "attention_norm",
    "wq_a",
    "q_norm",
    "wq_b",
    "wkv_a",
    "kv_pre_norm",
    "kv_norm",
    "q_rope",
    "k_rope",
    "q_nope_absorbed",
    "q_sparse",
    "kv_sparse",
    "topk_indices",
    "inner_attention",
    "value_unabsorbed",
    "attention_output",
)


def install_gpu_attention_internal_trace() -> None:
    output_path = os.environ.get("GLM5_GPU_ATTENTION_TRACE_PATH")
    if not output_path:
        return

    import torch
    from torch.distributed.tensor import DTensor
    from torchtitan.trainer import Trainer

    selected_step = int(os.environ.get("GLM5_GPU_ATTENTION_TRACE_STEP", "1"))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    lock = threading.Lock()

    def record(*, name: str, tensor: torch.Tensor) -> None:
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
            raise RuntimeError("GPU attention trace requires one model part")
        layers = getattr(self.model_parts[0], "layers", None)
        if layers is None or "0" not in layers:
            raise RuntimeError("GPU attention trace could not find model.layers['0']")
        layer = layers["0"]
        attention = layer.attention
        if not hasattr(attention, "layer_id"):
            raise RuntimeError(
                "GPU attention trace requires shared-index absorbed GLM-5 attention"
            )

        def diagnostic_forward(
            x_TD: torch.Tensor,
            attention_masks: torch.Tensor,
            positions_T: torch.Tensor | None = None,
            topk_indices_QS: torch.Tensor | None = None,
        ):
            T = x_TD.shape[0]
            if positions_T is None:
                positions_T = torch.arange(T, device=x_TD.device)

            attention_norm_TD = layer.attention_norm(x_TD)
            wq_a_TR = attention.wq_a(attention_norm_TD)
            q_norm_TR = attention.q_norm(wq_a_TR)
            wq_b_TNH = attention.wq_b(q_norm_TR).view(
                T, attention.n_heads, attention.qk_head_dim
            )
            q_nope_TNP, q_rope_input_TNR = torch.split(
                wq_b_TNH,
                [attention.qk_nope_head_dim, attention.qk_rope_head_dim],
                dim=-1,
            )
            wkv_a_TC = attention.wkv_a(attention_norm_TD)
            kv_pre_norm_TR, k_rope_input_TR = torch.split(
                wkv_a_TC,
                [attention.kv_lora_rank, attention.qk_rope_head_dim],
                dim=-1,
            )
            kv_norm_TR = attention.kv_norm(kv_pre_norm_TR)
            q_rope_TNR, k_rope_T1R = attention.rope(
                q_rope_input_TNR,
                k_rope_input_TR.unsqueeze(1),
                positions_T,
            )
            wkv_b_NXC = attention.wkv_b.weight.view(
                attention.n_heads,
                attention.qk_nope_head_dim + attention.v_head_dim,
                attention.kv_lora_rank,
            )
            w_k_NPC, w_v_NVC = torch.split(
                wkv_b_NXC,
                [attention.qk_nope_head_dim, attention.v_head_dim],
                dim=1,
            )
            q_nope_absorbed_TNC = torch.einsum(
                "tnp,npc->tnc", q_nope_TNP, w_k_NPC
            )
            q_sparse_TNH = torch.cat(
                (q_nope_absorbed_TNC, q_rope_TNR), dim=-1
            )
            kv_sparse_T1H = torch.cat(
                (kv_norm_TR.unsqueeze(1), k_rope_T1R), dim=-1
            )
            if attention.indexer is not None:
                topk_indices_QS = attention.indexer(
                    attention_norm_TD,
                    q_norm_TR,
                    positions_T,
                    attention_masks[0],
                )
            elif topk_indices_QS is None:
                raise RuntimeError("Block 0 attention trace has no top-k indices")
            latent_output_TNC = attention.inner_attention(
                q_sparse_TNH,
                kv_sparse_T1H,
                attention_masks,
                topk_indices_QS,
                scale=attention.softmax_scale,
                latent_dim=attention.kv_lora_rank,
            )
            value_unabsorbed_TNV = torch.einsum(
                "tnc,nvc->tnv", latent_output_TNC, w_v_NVC
            )
            attention_output_TD = attention.wo(
                value_unabsorbed_TNV.contiguous().view(T, -1)
            )
            attention_residual_TD = x_TD + attention_output_TD
            ffn_norm_TD = layer.ffn_norm(attention_residual_TD)
            if layer.moe_enabled:
                ffn_output_TD = layer.moe(ffn_norm_TD)
            else:
                ffn_output_TD = layer.feed_forward(ffn_norm_TD)
            block_output_TD = attention_residual_TD + ffn_output_TD
            stage_outputs = (
                x_TD,
                attention_norm_TD,
                wq_a_TR,
                q_norm_TR,
                wq_b_TNH,
                wkv_a_TC,
                kv_pre_norm_TR,
                kv_norm_TR,
                q_rope_TNR,
                k_rope_T1R,
                q_nope_absorbed_TNC,
                q_sparse_TNH,
                kv_sparse_T1H,
                topk_indices_QS,
                latent_output_TNC,
                value_unabsorbed_TNV,
                attention_output_TD,
            )
            return (block_output_TD, topk_indices_QS), stage_outputs

        def unwrap_and_record(outputs):
            model_output, stage_outputs = outputs
            if int(self.step) == selected_step:
                for name, tensor in zip(
                    ATTENTION_STAGE_NAMES, stage_outputs, strict=True
                ):
                    record(name=name, tensor=tensor)
            return model_output

        was_compiled = getattr(layer, "_compiled_call_impl", None) is not None
        layer.forward = diagnostic_forward
        if was_compiled:
            layer.compile(backend=config.compile.backend, fullgraph=True)
            compiled_call = layer._compiled_call_impl
            if compiled_call is None:
                raise RuntimeError("GPU attention trace failed to compile Block 0")

            def traced_compiled_call(*args, **kwargs):
                return unwrap_and_record(compiled_call(*args, **kwargs))

            layer._compiled_call_impl = traced_compiled_call
        else:

            def traced_forward(*args, **kwargs):
                return unwrap_and_record(diagnostic_forward(*args, **kwargs))

            layer.forward = traced_forward

    Trainer.__init__ = traced_init


__all__ = ["ATTENTION_STAGE_NAMES", "install_gpu_attention_internal_trace"]
