"""Test-only CUDA GLM-5 Block 0 activation and gradient tracing."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any


def install_gpu_block_trace() -> None:
    output_path = os.environ.get("GLM5_GPU_BLOCK_TRACE_PATH")
    if not output_path:
        return

    import torch
    from torch.distributed.tensor import DTensor
    from torchtitan.trainer import Trainer
    import torchtitan.trainer as trainer_module

    selected_steps = {
        int(value)
        for value in os.environ.get(
            "GLM5_GPU_BLOCK_TRACE_STEPS", "1,2,8,9,10"
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
                result = first_tensor(item)
                if result is not None:
                    return result
        if isinstance(value, dict):
            for item in value.values():
                result = first_tensor(item)
                if result is not None:
                    return result
        return None

    def local_tensor(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor

    def record(*, step: int, name: str, kind: str, tensor: torch.Tensor) -> None:
        cpu = local_tensor(tensor).detach().contiguous().cpu()
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

    original_init = Trainer.__init__

    def traced_init(self, config):
        original_init(self, config)
        if len(self.model_parts) != 1:
            raise RuntimeError("GPU Block trace requires one model part")
        layers = getattr(self.model_parts[0], "layers", None)
        if layers is None or "0" not in layers:
            raise RuntimeError("GPU Block trace could not find model.layers['0']")
        layer = layers["0"]

        def trace_call(call, *args, **kwargs):
            step = int(self.step)
            block_input = first_tensor(args)
            output = call(*args, **kwargs)
            if step not in selected_steps:
                return output
            block_output = first_tensor(output)
            if block_input is not None:
                record(
                    step=step,
                    name="block_input",
                    kind="forward",
                    tensor=block_input,
                )
                if block_input.requires_grad:
                    block_input.register_hook(
                        lambda grad, step=step: (
                            record(
                                step=step,
                                name="block_input",
                                kind="backward_grad",
                                tensor=grad,
                            )
                            or grad
                        )
                    )
            if block_output is not None:
                record(
                    step=step,
                    name="block_output",
                    kind="forward",
                    tensor=block_output,
                )
                if block_output.requires_grad:
                    block_output.register_hook(
                        lambda grad, step=step: (
                            record(
                                step=step,
                                name="block_output",
                                kind="backward_grad",
                                tensor=grad,
                            )
                            or grad
                        )
                    )
            return output

        compiled_call = getattr(layer, "_compiled_call_impl", None)
        if compiled_call is not None:

            def traced_compiled_call(*args, **kwargs):
                return trace_call(compiled_call, *args, **kwargs)

            layer._compiled_call_impl = traced_compiled_call
        else:
            original_forward = layer.forward

            def traced_forward(*args, **kwargs):
                return trace_call(original_forward, *args, **kwargs)

            layer.forward = traced_forward

        parameter_names = {
            id(parameter): name for name, parameter in layer.named_parameters()
        }
        original_clip_grad_norm = trainer_module.dist_utils.clip_grad_norm_

        def traced_clip_grad_norm(parameters, *args, **kwargs):
            parameter_list = list(parameters)
            step = int(self.step)
            if step in selected_steps:
                for parameter in parameter_list:
                    name = parameter_names.get(id(parameter))
                    if name is not None and parameter.grad is not None:
                        record(
                            step=step,
                            name=f"parameter.{name}",
                            kind="gradient_pre_clip",
                            tensor=parameter.grad,
                        )
            return original_clip_grad_norm(parameter_list, *args, **kwargs)

        trainer_module.dist_utils.clip_grad_norm_ = traced_clip_grad_norm

    Trainer.__init__ = traced_init


__all__ = ["install_gpu_block_trace"]
