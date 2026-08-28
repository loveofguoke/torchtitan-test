"""Test-only controlled materialization inside one GLM-5 DSA indexer."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading


INDEXER_CONTROL_VARIANTS = (
    "original",
    "decomposed-none",
    "qk",
    "relu",
    "weighted",
    "masked",
    "q-proj",
    "k-proj",
    "k-norm",
    "q-rope",
    "k-rope",
    "final-q",
    "final-k",
    "weights",
    "cum-qk-proj",
    "cum-k-norm",
    "cum-rope",
    "cum-final-qk",
    "cum-weights",
    "q-rope-k-norm",
)

_CUMULATIVE_POINTS = {
    "cum-qk-proj": {"q-proj", "k-proj"},
    "cum-k-norm": {"q-proj", "k-proj", "k-norm"},
    "cum-rope": {"q-proj", "k-proj", "k-norm", "q-rope", "k-rope"},
    "cum-final-qk": {
        "q-proj",
        "k-proj",
        "k-norm",
        "q-rope",
        "k-rope",
        "final-q",
        "final-k",
    },
    "cum-weights": {
        "q-proj",
        "k-proj",
        "k-norm",
        "q-rope",
        "k-rope",
        "final-q",
        "final-k",
        "weights",
    },
    "q-rope-k-norm": {"q-rope", "k-norm"},
}


def indexer_control_points(variant: str) -> frozenset[str]:
    if variant not in INDEXER_CONTROL_VARIANTS:
        raise ValueError(f"unknown GPU indexer control variant: {variant}")
    return frozenset(_CUMULATIVE_POINTS.get(variant, {variant}))


def install_gpu_indexer_boundary_control() -> None:
    trace_path_value = os.environ.get("GLM5_GPU_INDEXER_CONTROL_TRACE_PATH")
    if not trace_path_value:
        return

    import torch
    from torch.distributed.tensor import DTensor
    from torchtitan.trainer import Trainer

    variant = os.environ.get("GLM5_GPU_INDEXER_CONTROL_VARIANT", "original")
    active_points = indexer_control_points(variant)
    selected_layer = str(os.environ.get("GLM5_GPU_INDEXER_CONTROL_LAYER", "1"))
    selected_step = int(os.environ.get("GLM5_GPU_INDEXER_CONTROL_STEP", "1"))
    path = Path(trace_path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    lock = threading.Lock()

    def record_indices(indices: torch.Tensor) -> None:
        local = indices.to_local() if isinstance(indices, DTensor) else indices
        cpu = local.detach().contiguous().cpu()
        raw = cpu.view(torch.uint8).numpy().tobytes()
        fp32 = cpu.float()
        payload = {
            "step": selected_step,
            "name": "effective_topk_indices",
            "kind": "forward",
            "variant": variant,
            "shape": list(cpu.shape),
            "dtype": str(cpu.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "mean": float(fp32.mean().item()),
            "max_abs": float(fp32.abs().max().item()),
            "l2": float(torch.linalg.vector_norm(fp32).item()),
            "values": fp32.reshape(-1).tolist(),
        }
        with lock, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    @torch.library.custom_op(
        "torchtitan_test::record_indexer_control",
        mutates_args=(),
        schema="(Tensor indices) -> Tensor",
    )
    def record_indexer_control(indices: torch.Tensor) -> torch.Tensor:
        record_indices(indices)
        return indices.clone()

    @record_indexer_control.register_fake
    def record_indexer_control_fake(indices: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(indices)

    def observe(indices: torch.Tensor) -> torch.Tensor:
        return torch.ops.torchtitan_test.record_indexer_control(indices)

    def boundary(point: str, tensor: torch.Tensor) -> torch.Tensor:
        if point not in active_points:
            return tensor
        return torch.ops.torchtitan_test.gpu_training_materialize(tensor)

    original_init = Trainer.__init__

    def traced_init(self, config):
        original_init(self, config)
        if len(self.model_parts) != 1:
            raise RuntimeError("GPU indexer control requires one model part")
        layers = getattr(self.model_parts[0], "layers", None)
        if layers is None or selected_layer not in layers:
            raise RuntimeError(
                f"GPU indexer control could not find layer {selected_layer}"
            )
        layer = layers[selected_layer]
        indexer = layer.attention.indexer
        if indexer is None:
            raise RuntimeError(
                f"GPU indexer control layer {selected_layer} has no indexer"
            )
        original_forward = indexer.forward
        was_compiled = getattr(layer, "_compiled_call_impl", None) is not None

        if variant == "original":

            def controlled_indexer_forward(*args, **kwargs):
                return observe(original_forward(*args, **kwargs))

        else:

            @torch.no_grad()
            def controlled_indexer_forward(
                hidden_states_TD: torch.Tensor,
                q_resid_TR: torch.Tensor,
                positions_T: torch.Tensor,
                attention_mask_TK: torch.Tensor | None,
            ) -> torch.Tensor:
                T = hidden_states_TD.shape[0]
                q_TNH = boundary(
                    "q-proj",
                    indexer.wq_b(q_resid_TR).view(
                        T, indexer.n_heads, indexer.head_dim
                    ),
                )
                q_rot_TNR, q_pass_TNP = torch.split(
                    q_TNH,
                    [
                        indexer.qk_rope_head_dim,
                        indexer.head_dim - indexer.qk_rope_head_dim,
                    ],
                    dim=-1,
                )
                k_projection_TH = boundary(
                    "k-proj", indexer.wk(hidden_states_TD)
                )
                k_T1H = boundary(
                    "k-norm", indexer.k_norm(k_projection_TH)
                ).unsqueeze(1)
                k_rot_T1R, k_pass_T1P = torch.split(
                    k_T1H,
                    [
                        indexer.qk_rope_head_dim,
                        indexer.head_dim - indexer.qk_rope_head_dim,
                    ],
                    dim=-1,
                )
                q_rot_TNR, k_rot_T1R = indexer.rope(
                    q_rot_TNR, k_rot_T1R, positions_T
                )
                q_rot_TNR = boundary("q-rope", q_rot_TNR)
                k_rot_T1R = boundary("k-rope", k_rot_T1R)
                q_TNH = boundary(
                    "final-q",
                    torch.cat((q_rot_TNR, q_pass_TNP), dim=-1),
                )
                k_TH = boundary(
                    "final-k",
                    torch.cat((k_rot_T1R, k_pass_T1P), dim=-1).squeeze(1),
                )
                weights_TN = boundary(
                    "weights",
                    indexer.weights_proj(
                        hidden_states_TD.to(indexer.weights_proj.weight.dtype)
                    ).float()
                    * (indexer.n_heads**-0.5),
                )
                qk_scores_NQK = boundary(
                    "qk",
                    torch.matmul(
                        q_TNH.float().transpose(0, 1),
                        k_TH.float().transpose(0, 1),
                    )
                    * indexer.topk.softmax_scale,
                )
                relu_scores_NQK = boundary("relu", torch.relu(qk_scores_NQK))
                weighted_scores_QK = boundary(
                    "weighted",
                    torch.matmul(
                        weights_TN.unsqueeze(1),
                        relu_scores_NQK.transpose(0, 1),
                    ).squeeze(1),
                )
                if attention_mask_TK is None:
                    token_indices_T = torch.arange(
                        T, device=positions_T.device
                    )
                    document_ids_T = (positions_T == 0).cumsum(0)
                    allowed_TT = (
                        token_indices_T[:, None] >= token_indices_T[None, :]
                    ) & (
                        document_ids_T[:, None] == document_ids_T[None, :]
                    )
                    attention_mask_TK = torch.zeros(
                        T,
                        T,
                        dtype=torch.float32,
                        device=positions_T.device,
                    ).masked_fill(~allowed_TT, torch.finfo(torch.float32).min)
                masked_scores_QK = boundary(
                    "masked",
                    weighted_scores_QK + attention_mask_TK.float(),
                )
                topk = min(
                    indexer.topk.index_topk, masked_scores_QK.shape[-1]
                )
                indices_QS = masked_scores_QK.topk(topk, dim=-1).indices.to(
                    torch.int32
                )
                return observe(indices_QS)

        indexer.forward = controlled_indexer_forward
        if was_compiled:
            layer.compile(backend=config.compile.backend, fullgraph=True)

    Trainer.__init__ = traced_init


__all__ = [
    "INDEXER_CONTROL_VARIANTS",
    "indexer_control_points",
    "install_gpu_indexer_boundary_control",
]
