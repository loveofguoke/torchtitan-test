#!/usr/bin/env python3
"""Apply TorchTitanTurbo's NPU integration before starting TorchTitan."""

import os
import sys


_task_queue_enable = os.environ.get("TORCHTITAN_TASK_QUEUE_ENABLE")
if _task_queue_enable not in (None, "0", "1", "2"):
    raise ValueError(
        "TORCHTITAN_TASK_QUEUE_ENABLE must be 0, 1, or 2; "
        f"got {_task_queue_enable!r}"
    )
_npugraph_capture_mode = os.environ.get(
    "TORCHTITAN_NPUGRAPH_CAPTURE_ERROR_MODE"
) or None
if _npugraph_capture_mode not in (None, "global", "thread_local", "relaxed"):
    raise ValueError(
        "TORCHTITAN_NPUGRAPH_CAPTURE_ERROR_MODE must be global, "
        f"thread_local, or relaxed; got {_npugraph_capture_mode!r}"
    )
_npugraph_unsafe_ops = tuple(
    value.strip()
    for value in os.environ.get("TORCHTITAN_NPUGRAPH_UNSAFE_OPS", "").split(",")
    if value.strip()
)
_npugraph_skip_all = os.environ.get("TORCHTITAN_NPUGRAPH_SKIP_ALL")
if _npugraph_skip_all not in (None, "0", "1"):
    raise ValueError(
        "TORCHTITAN_NPUGRAPH_SKIP_ALL must be 0 or 1; "
        f"got {_npugraph_skip_all!r}"
    )
_register_complex_dtensor_strategy = os.environ.get(
    "TORCHTITAN_REGISTER_COMPLEX_DTENSOR_STRATEGY"
)
if _register_complex_dtensor_strategy not in (None, "0", "1"):
    raise ValueError(
        "TORCHTITAN_REGISTER_COMPLEX_DTENSOR_STRATEGY must be 0 or 1; "
        f"got {_register_complex_dtensor_strategy!r}"
    )
_pipeline_meta_use_batch = os.environ.get(
    "TORCHTITAN_PIPELINE_META_USE_BATCH"
)
if _pipeline_meta_use_batch not in (None, "0", "1"):
    raise ValueError(
        "TORCHTITAN_PIPELINE_META_USE_BATCH must be 0 or 1; "
        f"got {_pipeline_meta_use_batch!r}"
    )
_safe_empty_grouped_mm = os.environ.get("TORCHTITAN_SAFE_EMPTY_GROUPED_MM")
if _safe_empty_grouped_mm not in (None, "0", "1"):
    raise ValueError(
        "TORCHTITAN_SAFE_EMPTY_GROUPED_MM must be 0 or 1; "
        f"got {_safe_empty_grouped_mm!r}"
    )
_safe_zero_numel_triton = os.environ.get("TORCHTITAN_SAFE_ZERO_NUMEL_TRITON")
if _safe_zero_numel_triton not in (None, "0", "1"):
    raise ValueError(
        "TORCHTITAN_SAFE_ZERO_NUMEL_TRITON must be 0 or 1; "
        f"got {_safe_zero_numel_triton!r}"
    )
_comm_init_timeout_seconds = os.environ.get(
    "TORCHTITAN_COMM_INIT_TIMEOUT_SECONDS"
)
if _comm_init_timeout_seconds is not None:
    try:
        _comm_init_timeout_value = int(_comm_init_timeout_seconds)
    except ValueError as error:
        raise ValueError(
            "TORCHTITAN_COMM_INIT_TIMEOUT_SECONDS must be a positive integer; "
            f"got {_comm_init_timeout_seconds!r}"
        ) from error
    if _comm_init_timeout_value <= 0:
        raise ValueError(
            "TORCHTITAN_COMM_INIT_TIMEOUT_SECONDS must be a positive integer; "
            f"got {_comm_init_timeout_seconds!r}"
        )

import torchtitanturbo  # noqa: F401

# Cold graph compilation can delay a pipeline peer well beyond TorchTitan's
# normal five-minute process-group timeout. Keep the override launcher-only and
# avoid replacing an explicit CLI choice.
if _comm_init_timeout_seconds is not None and not any(
    argument.startswith("--comm.init_timeout_seconds") for argument in sys.argv
):
    sys.argv.append(
        f"--comm.init_timeout_seconds={_comm_init_timeout_seconds}"
    )

# TorchTitanTurbo selects mode 2 for normal training. NPU graph capture only
# supports mode 0/1, so allow an isolated launcher to restore an explicit value
# after Turbo has applied its other NPU patches.
if _task_queue_enable is not None:
    os.environ["TASK_QUEUE_ENABLE"] = _task_queue_enable

# torch_npu's Triton launcher currently forwards a zero dynamic grid to CANN,
# which rejects coreDim=0. Empty pointwise outputs require no kernel work, so
# stop before autotuning/launching when any non-reduction numel is zero. This
# is launcher-only and leaves reductions with an empty reduction axis alone.
if _safe_zero_numel_triton == "1":
    import operator

    import torch_npu._inductor.runtime.triton_heuristics as _npu_triton_runtime

    def _npu_triton_has_zero_numel(self, args):
        for argument_name, argument_value in zip(self.fn.arg_names, args):
            # Inductor uses both ``xnumel`` and multi-axis names such as
            # ``x0_numel``.  Match the semantic suffix instead of requiring an
            # underscore, otherwise ordinary pointwise kernels can still
            # submit a zero grid to CANN.
            if not argument_name.endswith("numel") or argument_name.startswith("r"):
                continue
            try:
                is_zero = operator.index(argument_value) == 0
            except TypeError:
                is_zero = False
            if is_zero:
                print(
                    "[torchtitan-npu] skipped zero-numel Triton kernel: "
                    f"{self.get_fn_name()} ({argument_name}=0)",
                    flush=True,
                )
                return True
        return False

    def _install_zero_numel_guard(autotuner_class):
        original_run = autotuner_class.run

        def run_without_zero_grid(
            self,
            *args,
            stream,
            benchmark_run=False,
            **kwargs,
        ):
            if _npu_triton_has_zero_numel(self, args):
                return None
            return original_run(
                self,
                *args,
                stream=stream,
                benchmark_run=benchmark_run,
                **kwargs,
            )

        autotuner_class.run = run_without_zero_grid

    # The symbolic grouped autotuner overrides run(), so both dispatch paths
    # require the guard. Patching the base class alone does not affect it.
    _install_zero_numel_guard(_npu_triton_runtime.NPUCachingAutotuner)
    _install_zero_numel_guard(_npu_triton_runtime.NPUSymbolicGroupedAutotuner)

# With EP, a rank can legitimately receive zero routed tokens for a training
# step. CANN 9.1's grouped-mm kernel attempts a zero-core launch for a 0xD
# input and reports EE1003 (or segfaults in asynchronous queue mode). Bypass
# the CANN kernel for an empty input and use an explicit zero-gradient backward.
# Keep this graph-launcher-only until the backend handles empty grouped GEMMs.
if _safe_empty_grouped_mm == "1":
    import torch
    from torchtitan.models.common.moe import GroupedExperts

    @torch.library.custom_op(
        "torchtitan_graph_debug::safe_grouped_mm",
        mutates_args=(),
    )
    def _safe_grouped_mm(
        A: torch.Tensor,
        B_t: torch.Tensor,
        offs: torch.Tensor,
    ) -> torch.Tensor:
        group_sizes = torch.diff(
            torch.cat((offs.new_zeros((1,)), offs)),
        )
        if bool(torch.all(group_sizes > 0).item()):
            return torch._grouped_mm(A, B_t, offs=offs)
        print(
            "[torchtitan-npu] padding empty grouped-mm experts: "
            f"rows={A.shape[0]}, group_sizes={group_sizes.cpu().tolist()}",
            flush=True,
        )
        if A.shape[0] == 0:
            return A.new_empty((0, B_t.shape[-1]))

        empty_groups = group_sizes == 0
        padded_group_sizes = torch.clamp_min(group_sizes, 1)
        # torch.cumsum promotes integral inputs to int64 unless dtype is
        # explicit, while CANN grouped-mm requires int32 offsets.
        padded_offs = torch.cumsum(
            padded_group_sizes,
            dim=0,
            dtype=torch.int32,
        )
        expert_ids = torch.repeat_interleave(
            torch.arange(B_t.shape[0], device=A.device),
            group_sizes,
        )
        empty_groups_before = torch.cumsum(
            empty_groups.to(torch.int64),
            dim=0,
        ) - empty_groups.to(torch.int64)
        original_positions = torch.arange(
            A.shape[0],
            device=A.device,
        ) + empty_groups_before.index_select(0, expert_ids)
        padded_A = A.new_zeros((padded_offs[-1].item(), A.shape[-1]))
        padded_A.index_copy_(0, original_positions, A)
        padded_output = torch._grouped_mm(
            padded_A,
            B_t,
            offs=padded_offs,
        )
        return padded_output.index_select(0, original_positions)

    @_safe_grouped_mm.register_fake
    def _safe_grouped_mm_fake(
        A: torch.Tensor,
        B_t: torch.Tensor,
        offs: torch.Tensor,
    ) -> torch.Tensor:
        del offs
        return A.new_empty((A.shape[0], B_t.shape[-1]))

    @torch.library.custom_op(
        "torchtitan_graph_debug::safe_grouped_mm_backward",
        mutates_args=(),
    )
    def _safe_grouped_mm_backward(
        grad_output: torch.Tensor,
        A: torch.Tensor,
        B_t: torch.Tensor,
        offs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if A.shape[0] == 0:
            return torch.zeros_like(A), B_t.new_zeros(B_t.shape)
        group_sizes = torch.diff(
            torch.cat((offs.new_zeros((1,)), offs)),
        )
        expert_ids = torch.repeat_interleave(
            torch.arange(B_t.shape[0], device=A.device),
            group_sizes,
        )
        row_weights = B_t.index_select(0, expert_ids)
        grad_A = torch.bmm(
            grad_output.unsqueeze(1),
            row_weights.transpose(-2, -1),
        ).squeeze(1)
        per_row_grad_B_t = A.unsqueeze(-1) * grad_output.unsqueeze(-2)
        grad_B_t = torch.index_add(
            B_t.new_zeros(B_t.shape),
            0,
            expert_ids,
            per_row_grad_B_t,
        )
        return grad_A, grad_B_t

    @_safe_grouped_mm_backward.register_fake
    def _safe_grouped_mm_backward_fake(
        grad_output: torch.Tensor,
        A: torch.Tensor,
        B_t: torch.Tensor,
        offs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del grad_output, offs
        return torch.empty_like(A), B_t.new_empty(B_t.shape)

    def _safe_grouped_mm_setup_context(ctx, inputs, output):
        del output
        ctx.save_for_backward(*inputs)

    def _safe_grouped_mm_autograd_backward(ctx, grad_output):
        A, B_t, offs = ctx.saved_tensors
        grad_A, grad_B_t = _safe_grouped_mm_backward(
            grad_output,
            A,
            B_t,
            offs,
        )
        return grad_A, grad_B_t, None

    _safe_grouped_mm.register_autograd(
        _safe_grouped_mm_autograd_backward,
        setup_context=_safe_grouped_mm_setup_context,
    )

    def _grouped_mm_with_empty_input(self, *, A, B_t, offs):
        del self
        return _safe_grouped_mm(A, B_t, offs)

    GroupedExperts._grouped_mm = _grouped_mm_with_empty_input

# PyTorch 2.14 does not discover aten.complex as a pointwise DTensor op. NPU
# RoPE constructs a complex tensor from real/imaginary DTensors, so TP compile
# needs the same broadcast-aware sharding strategy as other binary pointwise
# operators.
if _register_complex_dtensor_strategy == "1":
    import torch
    from torch.distributed.tensor._ops._pointwise_ops import (
        _register_single_dim_pointwise,
    )

    _register_single_dim_pointwise(torch.ops.aten.complex.default)

# PyTorch 2.14's manual PipelineStage exchanges inferred tensor metadata with
# batched object P2P. On the tested HCCL stack, concurrent stage initialization
# can corrupt the serialized-size message (the receiver then tries to allocate
# more than 1 EiB). Let the isolated graph launcher select regular P2P for this
# small, one-time metadata exchange. This does not affect activation traffic.
if _pipeline_meta_use_batch == "0":
    import torch.distributed as dist
    from torch.distributed.pipelining.stage import PipelineStage

    def _recv_pipeline_meta_without_batch(self, src_stage):
        objects = [None]
        dist.recv_object_list(
            objects,
            src=self._resolve_peer_global_rank(src_stage),
            group=self.group,
            device=self.device,
            use_batch=False,
        )
        return objects[0]

    def _send_pipeline_meta_without_batch(self, meta, dst_stage):
        dist.send_object_list(
            [meta],
            dst=self._resolve_peer_global_rank(dst_stage),
            group=self.group,
            device=self.device,
            use_batch=False,
        )

    PipelineStage._recv_meta = _recv_pipeline_meta_without_batch
    PipelineStage._send_meta = _send_pipeline_meta_without_batch

# NPUGraphs can safely run an AOT graph without graph replay when it contains an
# incompatible operator. CANN 9.1's aten._grouped_mm path synchronizes a device
# offset tensor during capture, but torch 2.14 does not yet tag the operator as
# cudagraph_unsafe. Keep this extra check opt-in so normal training and future
# fixed toolchains are unaffected.
if _npugraph_unsafe_ops or _npugraph_skip_all == "1":
    import torch
    import torch_npu.utils._graph_tree as _npugraph_backend

    _original_check_for_npugraph_skip = _npugraph_backend.check_for_skip
    _npugraph_unsafe_op_set = set(_npugraph_unsafe_ops)

    def _check_for_npugraph_skip(graph_module, num_fixed):
        if _npugraph_skip_all == "1":
            print(
                "[torchtitan-npu] NPUGraph replay disabled by compatibility profile",
                flush=True,
            )
            from torch._inductor.cudagraph_utils import format_default_skip_message

            return format_default_skip_message(
                "NPUGraph replay disabled by compatibility profile"
            )
        graph_modules = [
            module
            for module in graph_module.modules()
            if isinstance(module, torch.fx.GraphModule)
        ]
        targets = {
            str(node.target)
            for module in graph_modules
            for node in module.graph.nodes
            if node.op == "call_function"
        }
        matched = sorted(targets & _npugraph_unsafe_op_set)
        if matched:
            print(
                "[torchtitan-npu] NPUGraph capture skipped for unsafe ops: "
                + ",".join(matched),
                flush=True,
            )
            from torch._inductor.cudagraph_utils import format_default_skip_message

            return format_default_skip_message(
                f"configured incompatible op ({matched[0]})"
            )
        return _original_check_for_npugraph_skip(graph_module, num_fixed)

    _npugraph_backend.check_for_skip = _check_for_npugraph_skip

if _npugraph_capture_mode is not None:
    import torch
    import torch_npu

    _original_npu_graph = torch.npu.graph

    def _npu_graph_with_requested_capture_mode(*args, **kwargs):
        kwargs["capture_error_mode"] = _npugraph_capture_mode
        return _original_npu_graph(*args, **kwargs)

    torch.npu.graph = _npu_graph_with_requested_capture_mode
    torch_npu.npu.graph = _npu_graph_with_requested_capture_mode

from torchtitan.train import main


if __name__ == "__main__":
    main()
