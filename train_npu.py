#!/usr/bin/env python3
"""Apply TorchTitanTurbo's NPU integration before starting TorchTitan."""

import json
import os
import sys


def _configure_nonfinite_compiler_diagnostics() -> None:
    """Route compiler evidence to rank-specific directories before imports."""
    if os.environ.get("TORCHTITAN_DIAGNOSE_NONFINITE") != "1":
        return
    configured_root = os.environ.get("TORCHTITAN_NONFINITE_DUMP_DIR")
    if not configured_root:
        return
    from pathlib import Path

    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    compiler_root = Path(configured_root) / "compiler" / f"rank{rank}"
    trace_directory = compiler_root / "trace"
    debug_directory = compiler_root / "debug"
    trace_directory.mkdir(parents=True, exist_ok=True)
    debug_directory.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_TRACE"] = str(trace_directory)
    os.environ["TORCH_COMPILE_DEBUG_DIR"] = str(debug_directory)
    os.environ["TORCH_COMPILE_DEBUG"] = "1"


_configure_nonfinite_compiler_diagnostics()

import torchtitanturbo  # noqa: F401


def _install_nonfinite_gradient_diagnostics() -> None:
    """Log the first non-finite gradient before TorchTitan aborts a step."""
    if os.environ.get("TORCHTITAN_DIAGNOSE_NONFINITE") != "1":
        return

    import torch
    import torch.distributed as dist
    from pathlib import Path

    from torchtitan.distributed import utils as dist_utils
    from torchtitan.models.glm5 import parallelize as glm5_parallelize
    from torchtitan.models.glm5.dsa import Glm5FlexAttention
    from torchtitan.trainer import Trainer

    original_clip_grad_norm = dist_utils.clip_grad_norm_
    original_trainer_init = Trainer.__init__
    original_flex_cp_allgather = glm5_parallelize.flex_cp_allgather
    original_glm5_flex_forward = Glm5FlexAttention.forward
    parameter_names: dict[int, str] = {}
    cp_gather_index = 0
    flex_capture_index = 0

    def cpu_clone(tensor):
        return tensor.detach().to(device="cpu", copy=True)

    def tensor_statistics(tensor):
        finite = torch.isfinite(tensor)
        finite_count = int(finite.sum().item())
        numel = tensor.numel()
        return {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "finite_count": finite_count,
            "numel": numel,
            "nan_count": int(torch.isnan(tensor).sum().item()),
            "posinf_count": int(torch.isposinf(tensor).sum().item()),
            "neginf_count": int(torch.isneginf(tensor).sum().item()),
            "max_abs": (
                float(tensor.abs().max().item())
                if numel and finite_count == numel
                else None
            ),
        }

    def tensor_runtime_metadata(tensor):
        local_tensor = tensor.to_local() if hasattr(tensor, "to_local") else tensor
        metadata = {
            "shape": tuple(local_tensor.shape),
            "stride": tuple(local_tensor.stride()),
            "storage_offset": local_tensor.storage_offset(),
            "data_ptr": local_tensor.data_ptr(),
            "version": local_tensor._version,
            "is_contiguous": local_tensor.is_contiguous(),
            "dtype": str(local_tensor.dtype),
            "device": str(local_tensor.device),
        }
        if local_tensor.device.type == "npu":
            metadata["stream"] = str(torch.npu.current_stream(local_tensor.device))
        return metadata

    def allocator_statistics(device):
        if device.type != "npu":
            return None
        statistics = torch.npu.memory_stats(device)
        keys = (
            "allocated_bytes.all.current",
            "allocated_bytes.all.peak",
            "active_bytes.all.current",
            "active_bytes.all.peak",
            "reserved_bytes.all.current",
            "reserved_bytes.all.peak",
            "num_alloc_retries",
            "num_ooms",
        )
        return {key: int(statistics.get(key, 0)) for key in keys}

    def glm5_flex_forward_with_capture(
        self,
        q_QNH,
        k_KNH,
        v_KNV,
        attention_masks,
        topk_indices_QS,
        *,
        scale=None,
    ):
        nonlocal flex_capture_index
        output_QNV = original_glm5_flex_forward(
            self,
            q_QNH,
            k_KNH,
            v_KNV,
            attention_masks,
            topk_indices_QS,
            scale=scale,
        )
        rank = dist.get_rank() if dist.is_initialized() else 0
        target_rank_value = os.environ.get(
            "TORCHTITAN_NONFINITE_CAPTURE_RANK", "6"
        )
        target_rank = (
            None if target_rank_value == "all" else int(target_rank_value)
        )
        target_layer = os.environ.get(
            "TORCHTITAN_NONFINITE_CAPTURE_LAYER",
            "layers.6.attention.inner_attention",
        )
        module_fqn = getattr(self, "_nonfinite_diagnostic_fqn", "unknown")
        if (
            target_rank is not None and rank != target_rank
        ) or target_layer not in module_fqn:
            return output_QNV

        capture_index = flex_capture_index
        flex_capture_index += 1
        configured_root = os.environ.get("TORCHTITAN_NONFINITE_DUMP_DIR")
        if configured_root:
            dump_root = Path(configured_root)
        else:
            runtime_log = Path(os.environ["TORCHTITAN_RUN_LOG"])
            dump_root = runtime_log.parent / "nonfinite_replay"
        capture_directory = dump_root / f"rank{rank}" / f"call{capture_index:03d}"
        capture_directory.mkdir(parents=True, exist_ok=True)
        output_snapshot_QNV = cpu_clone(output_QNV)
        torch.save(
            {
                "schema_version": 4,
                "rank": rank,
                "module_fqn": module_fqn,
                "call_index": capture_index,
                "q_QNH": cpu_clone(q_QNH),
                "k_KNH": cpu_clone(k_KNH),
                "v_KNV": cpu_clone(v_KNV),
                "attention_masks": cpu_clone(attention_masks),
                "topk_indices_QS": cpu_clone(topk_indices_QS),
                "output_QNV": output_snapshot_QNV,
                "scale": scale,
                "block_size": self.block_size,
                "lse": None,
                "lse_note": (
                    "The GLM FlexAttention API does not expose its internal backward "
                    "LSE. Recompute it from the captured inputs during standalone replay."
                ),
                "runtime_metadata": {
                    "q": tensor_runtime_metadata(q_QNH),
                    "k": tensor_runtime_metadata(k_KNH),
                    "v": tensor_runtime_metadata(v_KNV),
                    "output": tensor_runtime_metadata(output_QNV),
                    "attention_masks": tensor_runtime_metadata(attention_masks),
                    "topk_indices": tensor_runtime_metadata(topk_indices_QS),
                },
                "compiler_environment": {
                    name: os.environ.get(name)
                    for name in (
                        "TORCHINDUCTOR_NPU_BACKEND",
                        "TORCHINDUCTOR_FLEXATTENTION_MASKOUT",
                        "TORCHINDUCTOR_MAX_AUTOTUNE",
                        "INDUCTOR_ASCEND_AGGRESSIVE_AUTOTUNE",
                        "TORCHINDUCTOR_COMPILE_THREADS",
                        "TASK_QUEUE_ENABLE",
                        "ASCEND_LAUNCH_BLOCKING",
                        "TORCH_TRACE",
                        "TORCH_COMPILE_DEBUG",
                        "TORCH_COMPILE_DEBUG_DIR",
                        "TORCHINDUCTOR_CACHE_DIR",
                        "TRITON_CACHE_DIR",
                    )
                },
                "allocator_at_forward": allocator_statistics(q_QNH.device),
            },
            capture_directory / "forward.pt",
        )

        actual_gradients = {}

        def save_actual_gradient(name):
            def hook(gradient):
                actual_gradients[name] = cpu_clone(gradient)
                if len(actual_gradients) == 3:
                    payload = {
                        "dq_QNH": actual_gradients["dq_QNH"],
                        "dk_KNH": actual_gradients["dk_KNH"],
                        "dv_KNV": actual_gradients["dv_KNV"],
                        "statistics": {
                            key: tensor_statistics(value)
                            for key, value in actual_gradients.items()
                        },
                        "runtime_metadata": {
                            key: tensor_runtime_metadata(value)
                            for key, value in actual_gradients.items()
                        },
                        "allocator_after_flex_backward": allocator_statistics(
                            gradient.device
                        ),
                    }
                    torch.save(payload, capture_directory / "actual_gradients.pt")
                    (capture_directory / "actual_gradients.json").write_text(
                        json.dumps(payload["statistics"], indent=2) + "\n",
                        encoding="utf-8",
                    )
                    print(
                        f"[nonfinite-actual-gradients] rank={rank} "
                        f"fqn={module_fqn} call={capture_index} "
                        f"statistics={payload['statistics']} "
                        f"path={capture_directory / 'actual_gradients.pt'}",
                        flush=True,
                    )
                return gradient

            return hook

        if q_QNH.requires_grad:
            q_QNH.register_hook(save_actual_gradient("dq_QNH"))
        if k_KNH.requires_grad:
            k_KNH.register_hook(save_actual_gradient("dk_KNH"))
        if v_KNV.requires_grad:
            v_KNV.register_hook(save_actual_gradient("dv_KNV"))

        def save_grad_output(gradient_QNV):
            # This hook runs before FlexAttention consumes grad_output. Reading
            # output_QNV here tests whether the forward result retained by the
            # autograd graph was overwritten during the intervening layers.
            output_at_backward_QNV = cpu_clone(output_QNV)
            gradient_at_backward_QNV = cpu_clone(gradient_QNV)
            output_max_abs_diff = float(
                (output_at_backward_QNV - output_snapshot_QNV).abs().max().item()
            )
            # FlexAttention backward forms DELTA = sum(out * grad_out, -1).
            # Compute the reference in FP32 before entering the compiled
            # backward so it is independent of the NPU lowering and kernel.
            delta_ref_QN = (
                output_snapshot_QNV.float() * gradient_at_backward_QNV.float()
            ).sum(dim=-1)
            torch.save(
                {
                    "grad_output_QNV": gradient_at_backward_QNV,
                    "output_at_backward_QNV": output_at_backward_QNV,
                    "output_max_abs_diff": output_max_abs_diff,
                    "delta_ref_QN": delta_ref_QN,
                    "delta_ref_statistics": tensor_statistics(delta_ref_QN),
                    "runtime_metadata": {
                        "output": tensor_runtime_metadata(output_QNV),
                        "grad_output": tensor_runtime_metadata(gradient_QNV),
                    },
                    "allocator_at_backward_entry": allocator_statistics(
                        gradient_QNV.device
                    ),
                },
                capture_directory / "backward.pt",
            )
            print(
                f"[nonfinite-replay-capture] rank={rank} fqn={module_fqn} "
                f"call={capture_index} output_max_abs_diff={output_max_abs_diff} "
                f"delta_ref={tensor_statistics(delta_ref_QN)} "
                f"path={capture_directory}",
                flush=True,
            )
            return gradient_QNV

        if output_QNV.requires_grad:
            output_QNV.register_hook(save_grad_output)
        return output_QNV

    Glm5FlexAttention.forward = glm5_flex_forward_with_capture

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
                module._nonfinite_diagnostic_fqn = module_fqn

                def register_attention_inputs(
                    _module,
                    args,
                    _kwargs,
                    *,
                    fqn=module_fqn,
                ):
                    register_tensor_diagnostic(args[0], f"{fqn}.q")
                    register_tensor_diagnostic(args[1], f"{fqn}.input_local_k")
                    register_tensor_diagnostic(args[2], f"{fqn}.input_local_v")

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
