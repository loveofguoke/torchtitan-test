#!/usr/bin/env python3
"""Apply TorchTitanTurbo's NPU integration before starting TorchTitan."""

import os
import sys

import torchtitanturbo  # noqa: F401


def _install_nonfinite_gradient_diagnostics() -> None:
    """Log the first non-finite gradient before TorchTitan aborts a step."""
    if os.environ.get("TORCHTITAN_DIAGNOSE_NONFINITE") != "1":
        return

    import torch
    import torch.distributed as dist

    from torchtitan.distributed import utils as dist_utils
    from torchtitan.models.glm5 import parallelize as glm5_parallelize
    from torchtitan.models.glm5.dsa import Glm5FlexAttention
    from torchtitan.trainer import Trainer

    original_clip_grad_norm = dist_utils.clip_grad_norm_
    original_trainer_init = Trainer.__init__
    original_flex_cp_allgather = glm5_parallelize.flex_cp_allgather
    parameter_names: dict[int, str] = {}
    cp_gather_index = 0

    def register_tensor_diagnostic(tensor, tag):
        if not tensor.requires_grad:
            return

        def report_gradient(gradient):
            local_gradient = (
                gradient.to_local() if hasattr(gradient, "to_local") else gradient
            )
            finite = torch.isfinite(local_gradient)
            rank = dist.get_rank() if dist.is_initialized() else 0
            finite_count = int(finite.sum().item())
            numel = local_gradient.numel()
            max_abs = (
                float(local_gradient.abs().max().item())
                if numel and finite_count == numel
                else None
            )
            print(
                f"[cp-gradient-diagnostic] rank={rank} tag={tag} "
                f"shape={tuple(local_gradient.shape)} dtype={local_gradient.dtype} "
                f"finite={finite_count}/{numel} "
                f"nan={int(torch.isnan(local_gradient).sum().item())} "
                f"posinf={int(torch.isposinf(local_gradient).sum().item())} "
                f"neginf={int(torch.isneginf(local_gradient).sum().item())} "
                f"max_abs={max_abs}",
                flush=True,
            )
            return gradient

        tensor.register_hook(report_gradient)

    def flex_cp_allgather_with_diagnostics(k_local, v_local, *args, **kwargs):
        nonlocal cp_gather_index
        gather_index = cp_gather_index
        cp_gather_index += 1
        register_tensor_diagnostic(k_local, f"gather.{gather_index}.local_k")
        register_tensor_diagnostic(v_local, f"gather.{gather_index}.local_v")
        k_global, v_global = original_flex_cp_allgather(
            k_local, v_local, *args, **kwargs
        )
        register_tensor_diagnostic(k_global, f"gather.{gather_index}.global_k")
        register_tensor_diagnostic(v_global, f"gather.{gather_index}.global_v")
        return k_global, v_global

    glm5_parallelize.flex_cp_allgather = flex_cp_allgather_with_diagnostics

    def trainer_init_with_diagnostics(self, *args, **kwargs):
        original_trainer_init(self, *args, **kwargs)
        for part_index, model_part in enumerate(self.model_parts):
            for name, parameter in model_part.named_parameters():
                parameter_names[id(parameter)] = f"model_parts.{part_index}.{name}"
            for name, module in model_part.named_modules():
                if not isinstance(module, Glm5FlexAttention):
                    continue
                module_fqn = f"model_parts.{part_index}.{name}"

                def register_attention_inputs(
                    _module,
                    args,
                    _kwargs,
                    *,
                    fqn=module_fqn,
                ):
                    register_tensor_diagnostic(args[0], f"{fqn}.q")
                    register_tensor_diagnostic(args[1], f"{fqn}.global_k")
                    register_tensor_diagnostic(args[2], f"{fqn}.global_v")

                module.register_forward_pre_hook(
                    register_attention_inputs,
                    with_kwargs=True,
                )

    Trainer.__init__ = trainer_init_with_diagnostics

    def clip_grad_norm_with_diagnostics(parameters, *args, **kwargs):
        parameters = list(parameters)
        grad_norm = original_clip_grad_norm(parameters, *args, **kwargs)
        rank = dist.get_rank() if dist.is_initialized() else 0
        first_nonfinite = None
        for index, parameter in enumerate(parameters):
            gradient = parameter.grad
            if gradient is None:
                continue
            local_gradient = (
                gradient.to_local() if hasattr(gradient, "to_local") else gradient
            )
            finite = torch.isfinite(local_gradient)
            if not bool(finite.all().item()):
                first_nonfinite = {
                    "parameter_index": index,
                    "fqn": parameter_names.get(id(parameter), "unknown"),
                    "shape": tuple(parameter.shape),
                    "dtype": str(local_gradient.dtype),
                    "nan_count": int(torch.isnan(local_gradient).sum().item()),
                    "posinf_count": int(torch.isposinf(local_gradient).sum().item()),
                    "neginf_count": int(torch.isneginf(local_gradient).sum().item()),
                }
                break
        print(
            f"[nonfinite-diagnostic] rank={rank} grad_norm={grad_norm} "
            f"grad_norm_finite={bool(torch.isfinite(grad_norm).all().item())} "
            f"first_nonfinite_gradient={first_nonfinite}",
            flush=True,
        )
        return grad_norm

    dist_utils.clip_grad_norm_ = clip_grad_norm_with_diagnostics


_install_nonfinite_gradient_diagnostics()


# Cold graph compilation can delay a pipeline peer beyond the ordinary
# process-group timeout. This remains a launcher concern: append the isolated
# experiment override only when the user did not provide an explicit value.
comm_timeout = os.environ.get("TORCHTITAN_COMM_INIT_TIMEOUT_SECONDS")
if comm_timeout is not None:
    try:
        timeout_value = int(comm_timeout)
    except ValueError as error:
        raise ValueError(
            "TORCHTITAN_COMM_INIT_TIMEOUT_SECONDS must be a positive integer; "
            f"got {comm_timeout!r}"
        ) from error
    if timeout_value <= 0:
        raise ValueError(
            "TORCHTITAN_COMM_INIT_TIMEOUT_SECONDS must be a positive integer; "
            f"got {comm_timeout!r}"
        )
    if not any(
        argument.startswith("--comm.init_timeout_seconds")
        for argument in sys.argv
    ):
        sys.argv.append(f"--comm.init_timeout_seconds={timeout_value}")

from torchtitan.train import main  # noqa: E402


if __name__ == "__main__":
    main()
