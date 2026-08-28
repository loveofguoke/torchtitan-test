"""Test-only absorbed-attention trace for a shared-index GLM-5 layer."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading


SHARED_ATTENTION_STAGE_NAMES = (
    "layer_input",
    "incoming_topk_indices",
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
    "indexer_q_projection",
    "indexer_k_projection",
    "indexer_k_norm",
    "indexer_q_rope",
    "indexer_k_rope",
    "indexer_q",
    "indexer_k",
    "indexer_weights",
    "indexer_qk_scores",
    "indexer_relu_scores",
    "indexer_weighted_scores",
    "indexer_masked_scores",
    "indexer_topk_values",
    "indexer_topk_boundary_values",
    "indexer_topk_margin",
    "effective_topk_indices",
    "inner_attention",
    "value_unabsorbed",
    "attention_output",
)


def install_gpu_shared_attention_layer_trace() -> None:
    output_path = os.environ.get("GLM5_GPU_SHARED_ATTENTION_TRACE_PATH")
    if not output_path:
        return

    import torch
    from torch.distributed.tensor import DTensor
    from torchtitan.trainer import Trainer

    selected_layer = str(os.environ.get("GLM5_GPU_SHARED_ATTENTION_LAYER", "1"))
    selected_step = int(
        os.environ.get("GLM5_GPU_SHARED_ATTENTION_TRACE_STEP", "1")
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    lock = threading.Lock()

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
            "min": float(fp32.min().item()),
            "max": float(fp32.max().item()),
            "max_abs": float(fp32.abs().max().item()),
            "l2": float(torch.linalg.vector_norm(fp32).item()),
        }
        if not cpu.is_floating_point() or name in {
            "indexer_topk_boundary_values",
            "indexer_topk_margin",
        }:
            payload["values"] = fp32.reshape(-1).tolist()
        with lock, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    original_init = Trainer.__init__

    def traced_init(self, config):
        original_init(self, config)
        if len(self.model_parts) != 1:
            raise RuntimeError("GPU shared-attention trace requires one model part")
        layers = getattr(self.model_parts[0], "layers", None)
        if layers is None or selected_layer not in layers:
            raise RuntimeError(
                "GPU shared-attention trace could not find "
                f"model.layers[{selected_layer!r}]"
            )
        layer = layers[selected_layer]
        attention = layer.attention
        if not hasattr(attention, "layer_id"):
            raise RuntimeError(
                "GPU shared-attention trace requires shared-index attention"
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
            incoming_topk_indices_QS = topk_indices_QS
            if incoming_topk_indices_QS is None:
                raise RuntimeError(
                    f"Layer {selected_layer} did not receive shared DSA top-k indices"
                )

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
            indexer = attention.indexer
            if indexer is None:
                raise RuntimeError(
                    f"Layer {selected_layer} has no DSA indexer to trace"
                )
            with torch.no_grad():
                indexer_q_projection_TNH = indexer.wq_b(q_norm_TR).view(
                    T, indexer.n_heads, indexer.head_dim
                )
                indexer_q_rot_TNR, indexer_q_pass_TNP = torch.split(
                    indexer_q_projection_TNH,
                    [
                        indexer.qk_rope_head_dim,
                        indexer.head_dim - indexer.qk_rope_head_dim,
                    ],
                    dim=-1,
                )
                indexer_k_projection_TH = indexer.wk(attention_norm_TD)
                indexer_k_norm_T1H = indexer.k_norm(
                    indexer_k_projection_TH
                ).unsqueeze(1)
                indexer_k_rot_T1R, indexer_k_pass_T1P = torch.split(
                    indexer_k_norm_T1H,
                    [
                        indexer.qk_rope_head_dim,
                        indexer.head_dim - indexer.qk_rope_head_dim,
                    ],
                    dim=-1,
                )
                indexer_q_rot_TNR, indexer_k_rot_T1R = indexer.rope(
                    indexer_q_rot_TNR,
                    indexer_k_rot_T1R,
                    positions_T,
                )
                indexer_q_TNH = torch.cat(
                    (indexer_q_rot_TNR, indexer_q_pass_TNP), dim=-1
                )
                indexer_k_TH = torch.cat(
                    (indexer_k_rot_T1R, indexer_k_pass_T1P), dim=-1
                ).squeeze(1)
                indexer_weights_TN = indexer.weights_proj(
                    attention_norm_TD.to(indexer.weights_proj.weight.dtype)
                ).float() * (indexer.n_heads**-0.5)
                indexer_qk_scores_NQK = torch.matmul(
                    indexer_q_TNH.float().transpose(0, 1),
                    indexer_k_TH.float().transpose(0, 1),
                ) * indexer.topk.softmax_scale
                indexer_relu_scores_NQK = torch.relu(indexer_qk_scores_NQK)
                indexer_weighted_scores_QK = torch.matmul(
                    indexer_weights_TN.unsqueeze(1),
                    indexer_relu_scores_NQK.transpose(0, 1),
                ).squeeze(1)
                indexer_masked_scores_QK = (
                    indexer_weighted_scores_QK + attention_masks[0].float()
                )
                topk = min(
                    indexer.topk.index_topk,
                    indexer_masked_scores_QK.shape[-1],
                )
                (
                    indexer_topk_values_QS,
                    effective_topk_indices_QS,
                ) = indexer_masked_scores_QK.topk(topk, dim=-1)
                boundary_k = min(
                    topk + 1, indexer_masked_scores_QK.shape[-1]
                )
                indexer_topk_boundary_values_QB = (
                    indexer_masked_scores_QK.topk(
                        boundary_k, dim=-1
                    ).values
                )
                if boundary_k > topk:
                    indexer_topk_margin_Q = (
                        indexer_topk_boundary_values_QB[..., topk - 1]
                        - indexer_topk_boundary_values_QB[..., topk]
                    )
                else:
                    indexer_topk_margin_Q = torch.zeros(
                        (T,), dtype=torch.float32, device=x_TD.device
                    )
            latent_output_TNC = attention.inner_attention(
                q_sparse_TNH,
                kv_sparse_T1H,
                attention_masks,
                effective_topk_indices_QS,
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
            ffn_output_TD = (
                layer.moe(ffn_norm_TD)
                if layer.moe_enabled
                else layer.feed_forward(ffn_norm_TD)
            )
            layer_output_TD = attention_residual_TD + ffn_output_TD
            model_output = (
                (layer_output_TD, effective_topk_indices_QS)
                if attention.indexer is not None
                else layer_output_TD
            )
            stage_outputs = (
                x_TD,
                incoming_topk_indices_QS,
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
                indexer_q_projection_TNH,
                indexer_k_projection_TH,
                indexer_k_norm_T1H,
                indexer_q_rot_TNR,
                indexer_k_rot_T1R,
                indexer_q_TNH,
                indexer_k_TH,
                indexer_weights_TN,
                indexer_qk_scores_NQK,
                indexer_relu_scores_NQK,
                indexer_weighted_scores_QK,
                indexer_masked_scores_QK,
                indexer_topk_values_QS,
                indexer_topk_boundary_values_QB,
                indexer_topk_margin_Q,
                effective_topk_indices_QS,
                latent_output_TNC,
                value_unabsorbed_TNV,
                attention_output_TD,
            )
            return model_output, stage_outputs

        def unwrap_and_record(outputs):
            model_output, stage_outputs = outputs
            if int(self.step) == selected_step:
                for name, tensor in zip(
                    SHARED_ATTENTION_STAGE_NAMES, stage_outputs, strict=True
                ):
                    record(name, tensor)
            return model_output

        was_compiled = getattr(layer, "_compiled_call_impl", None) is not None
        layer.forward = diagnostic_forward
        if was_compiled:
            layer.compile(backend=config.compile.backend, fullgraph=True)
            compiled_call = layer._compiled_call_impl
            if compiled_call is None:
                raise RuntimeError(
                    "GPU shared-attention trace failed to compile selected layer"
                )

            def traced_compiled_call(*args, **kwargs):
                return unwrap_and_record(compiled_call(*args, **kwargs))

            layer._compiled_call_impl = traced_compiled_call
        else:

            def traced_forward(*args, **kwargs):
                return unwrap_and_record(diagnostic_forward(*args, **kwargs))

            layer.forward = traced_forward

    Trainer.__init__ = traced_init


__all__ = [
    "SHARED_ATTENTION_STAGE_NAMES",
    "install_gpu_shared_attention_layer_trace",
]
