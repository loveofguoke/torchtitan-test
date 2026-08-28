"""Test-only CUDA tracing at every GLM-5 Transformer-layer boundary."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any


def install_gpu_layer_boundary_trace() -> None:
    output_path = os.environ.get("GLM5_GPU_LAYER_BOUNDARY_TRACE_PATH")
    if not output_path:
        return

    import torch
    from torch.distributed.tensor import DTensor
    from torchtitan.trainer import Trainer

    selected_steps = {
        int(value)
        for value in os.environ.get(
            "GLM5_GPU_LAYER_BOUNDARY_TRACE_STEPS", "1"
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
        if isinstance(value, dict):
            for item in value.values():
                tensor = first_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    def record(
        *, step: int, name: str, kind: str, tensor: torch.Tensor
    ) -> None:
        local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
        cpu = local.detach().contiguous().cpu()
        raw = cpu.view(torch.uint8).numpy().tobytes()
        fp32 = cpu.float()
        payload = {
            "step": step,
            "name": name,
            "kind": kind,
            "shape": list(cpu.shape),
            "dtype": str(cpu.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "mean": float(fp32.mean().item()),
            "max_abs": float(fp32.abs().max().item()),
            "l2": float(torch.linalg.vector_norm(fp32).item()),
        }
        with lock, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def gradient_hook(*, step: int, name: str):
        def hook(gradient: torch.Tensor | None):
            if gradient is not None:
                record(
                    step=step,
                    name=name,
                    kind="backward_grad",
                    tensor=gradient,
                )
            return gradient

        return hook

    original_init = Trainer.__init__

    def traced_init(self, config):
        original_init(self, config)
        if len(self.model_parts) != 1:
            raise RuntimeError("GPU layer trace requires one model part")
        layers = getattr(self.model_parts[0], "layers", None)
        if layers is None or not layers:
            raise RuntimeError("GPU layer trace could not find model.layers")

        def trace_call(layer_id: str, call, *args, **kwargs):
            step = int(self.step)
            layer_input = first_tensor(args)
            output = call(*args, **kwargs)
            if step not in selected_steps:
                return output
            layer_output = first_tensor(output)
            for boundary, tensor in (
                ("input", layer_input),
                ("output", layer_output),
            ):
                if tensor is None:
                    continue
                name = f"layer.{layer_id}.{boundary}"
                record(step=step, name=name, kind="forward", tensor=tensor)
                if tensor.requires_grad:
                    tensor.register_hook(gradient_hook(step=step, name=name))
            return output

        for layer_id, layer in layers.items():
            compiled_call = getattr(layer, "_compiled_call_impl", None)
            if compiled_call is not None:

                def traced_compiled_call(
                    *args,
                    _layer_id=str(layer_id),
                    _call=compiled_call,
                    **kwargs,
                ):
                    return trace_call(_layer_id, _call, *args, **kwargs)

                layer._compiled_call_impl = traced_compiled_call
            else:
                original_forward = layer.forward

                def traced_forward(
                    *args,
                    _layer_id=str(layer_id),
                    _call=original_forward,
                    **kwargs,
                ):
                    return trace_call(_layer_id, _call, *args, **kwargs)

                layer.forward = traced_forward

    Trainer.__init__ = traced_init


__all__ = ["install_gpu_layer_boundary_trace"]
