#!/usr/bin/env python3
"""MindStudio Probe capture and TensorBoard visualization helpers.

The formal precision artifacts intentionally remain independent from this
instrumented diagnostic path.  msProbe hooks may introduce synchronization,
so their metrics must not be used as authoritative precision or performance
evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Literal, Sequence


MSPROBE_CONFIG_PATH_ENV = "GLM5_MSPROBE_CONFIG_PATH"
MSPROBE_BLOCK_BOUNDARIES_ENV = "GLM5_MSPROBE_BLOCK_BOUNDARIES"
MSPROBE_BLOCK_GLOBAL_STEP_ENV = "GLM5_MSPROBE_BLOCK_GLOBAL_STEP"
MSPROBE_BLOCK_BACKWARD_ENV = "GLM5_MSPROBE_BLOCK_BACKWARD"
MSPROBE_PARAMETER_STATE_ENV = "GLM5_MSPROBE_PARAMETER_STATE"
MSPROBE_ROUTER_STATE_ENV = "GLM5_MSPROBE_ROUTER_STATE"
MSPROBE_OPTIMIZER_STATE_ENV = "GLM5_MSPROBE_OPTIMIZER_STATE"
MSPROBE_FINAL_NORM_STATE_ENV = "GLM5_MSPROBE_FINAL_NORM_STATE"
MSPROBE_FINAL_NORM_REDUCE_TRANSITION_ENV = (
    "GLM5_MSPROBE_FINAL_NORM_REDUCE_TRANSITION"
)
MSPROBE_FINAL_NORM_PRE_REDUCE_SYNC_ENV = (
    "GLM5_MSPROBE_FINAL_NORM_PRE_REDUCE_SYNC"
)
MSPROBE_FINAL_NORM_SHARDED_GRAD_ALL_REDUCE_ENV = (
    "GLM5_MSPROBE_FINAL_NORM_SHARDED_GRAD_ALL_REDUCE"
)
MSPROBE_FINAL_NORM_NATIVE_LAST_BACKWARD_SYNC_ENV = (
    "GLM5_MSPROBE_FINAL_NORM_NATIVE_LAST_BACKWARD_SYNC"
)
SCHEMA = "torchtitan.glm5_2.msprobe_tensorboard"
SCHEMA_VERSION = 2

OPTIMIZER_DIAGNOSTIC_PARAMETERS = (
    "layers.3.attention.kv_norm.weight",
    "layers.6.attention.kv_norm.weight",
    "layers.4.moe.router.gate.weight",
)

FSDP_LIFECYCLE_EVENT_CODES = {
    "pre_forward": 1,
    "unshard": 2,
    "wait_for_unshard": 3,
    "post_forward": 4,
    "pre_backward": 5,
    "post_backward": 6,
}


def adamw_update_components(
    initial_parameter: Any,
    exp_avg: Any,
    exp_avg_sq: Any,
    *,
    step: float,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
) -> tuple[Any, Any, Any]:
    """Reconstruct AdamW's FP32 adaptive, decay, and combined deltas."""

    if step <= 0:
        raise ValueError("AdamW step must be positive")
    beta1, beta2 = betas
    bias_correction1 = 1.0 - beta1**step
    bias_correction2_sqrt = (1.0 - beta2**step) ** 0.5
    denominator = exp_avg_sq.float().sqrt() / bias_correction2_sqrt + eps
    adaptive = -(lr / bias_correction1) * exp_avg.float() / denominator
    decay = -lr * weight_decay * initial_parameter.float()
    return adaptive, decay, adaptive + decay


def reconstruct_rmsnorm_weight_gradient(
    norm_output: Any,
    grad_output: Any,
    weight: Any,
) -> Any:
    """Reconstruct an RMSNorm weight gradient from its output boundary."""

    if norm_output.shape != grad_output.shape:
        raise ValueError(
            "RMSNorm output and output gradient must have identical shapes"
        )
    if norm_output.ndim < 1 or tuple(weight.shape) != (norm_output.shape[-1],):
        raise ValueError("RMSNorm weight must match the output feature dimension")
    normalized = norm_output.float() / weight.float()
    reduction_dims = tuple(range(norm_output.ndim - 1))
    return (normalized * grad_output.float()).sum(dim=reduction_dims)


@dataclass(frozen=True)
class MsprobeParallelSpec:
    """Parallel metadata consumed by msProbe's graph-merging interface."""

    rank_size: int
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    data_parallel: int = 1
    expert_parallel: int = 1
    virtual_pipeline_parallel: int = 1

    def __post_init__(self) -> None:
        values = (
            self.rank_size,
            self.tensor_parallel,
            self.pipeline_parallel,
            self.data_parallel,
            self.expert_parallel,
            self.virtual_pipeline_parallel,
        )
        if any(value < 1 for value in values):
            raise ValueError("msProbe parallel degrees must be positive")
        expected = (
            self.tensor_parallel
            * self.pipeline_parallel
            * self.data_parallel
        )
        if expected != self.rank_size:
            raise ValueError(
                "msProbe rank_size must equal tp * pp * dp when CP is disabled: "
                f"{self.rank_size} != {expected}"
            )


def validate_parallel_merge_pair(
    reference: MsprobeParallelSpec,
    candidate: MsprobeParallelSpec,
) -> None:
    """Enforce the cross-partition support documented by msProbe 26.1."""

    if reference.expert_parallel != 1 or candidate.expert_parallel != 1:
        raise ValueError(
            "msProbe 26.1 graph merging does not support Expert Parallelism"
        )
    if reference.data_parallel != candidate.data_parallel:
        raise ValueError(
            "msProbe 26.1 graph merging requires identical Data Parallelism; "
            f"reference dp={reference.data_parallel}, "
            f"candidate dp={candidate.data_parallel}"
        )


@dataclass(frozen=True)
class MsprobeCaptureConfig:
    """Small, visualization-oriented msProbe capture configuration."""

    steps: tuple[int, ...] = (0,)
    ranks: tuple[int, ...] = ()
    task: Literal["statistics", "tensor"] = "statistics"
    level: Literal["L0", "mix", "debug"] = "mix"
    block_boundaries: bool = False
    block_global_step: bool = False
    block_backward: bool = False
    parameter_state: bool = False
    router_state: bool = False
    optimizer_state: bool = False
    final_norm_state: bool = False
    final_norm_reduce_transition: bool = False
    final_norm_pre_reduce_sync: bool = False
    final_norm_sharded_grad_all_reduce: bool = False
    final_norm_native_last_backward_sync: bool = False

    def __post_init__(self) -> None:
        if not self.steps or any(step < 0 for step in self.steps):
            raise ValueError("msProbe steps must contain non-negative step indexes")
        if tuple(sorted(set(self.steps))) != self.steps:
            raise ValueError("msProbe steps must be sorted and unique")
        if any(rank < 0 for rank in self.ranks):
            raise ValueError("msProbe ranks must be non-negative")
        if tuple(sorted(set(self.ranks))) != self.ranks:
            raise ValueError("msProbe ranks must be sorted and unique")
        if self.block_boundaries and (
            self.task != "tensor" or self.level != "debug"
        ):
            raise ValueError(
                "block-boundary capture requires msProbe task=tensor and level=debug"
            )
        if self.block_global_step and not self.block_boundaries:
            raise ValueError(
                "global-step block capture requires block-boundary capture"
            )
        if self.block_backward and not self.block_global_step:
            raise ValueError(
                "block backward capture requires global-step block capture"
            )
        if self.parameter_state and (
            self.task != "tensor" or self.level != "debug"
        ):
            raise ValueError(
                "parameter-state capture requires msProbe task=tensor and level=debug"
            )
        if self.router_state and (
            self.task != "tensor" or self.level != "debug"
        ):
            raise ValueError(
                "router-state capture requires msProbe task=tensor and level=debug"
            )
        if self.optimizer_state and not self.parameter_state:
            raise ValueError(
                "optimizer-state capture requires parameter-state capture"
            )
        if self.final_norm_state and (
            self.task != "tensor" or self.level != "debug"
        ):
            raise ValueError(
                "final-norm-state capture requires msProbe task=tensor and level=debug"
            )
        if self.final_norm_reduce_transition and not self.final_norm_state:
            raise ValueError(
                "final-norm reduce-transition capture requires final-norm capture"
            )
        if self.final_norm_pre_reduce_sync and not self.final_norm_reduce_transition:
            raise ValueError(
                "final-norm pre-reduce sync requires reduce-transition capture"
            )
        if (
            self.final_norm_sharded_grad_all_reduce
            and not self.final_norm_reduce_transition
        ):
            raise ValueError(
                "final-norm sharded-grad all-reduce requires reduce-transition capture"
            )
        if (
            self.final_norm_native_last_backward_sync
            and not self.final_norm_reduce_transition
        ):
            raise ValueError(
                "final-norm native last-backward sync requires "
                "reduce-transition capture"
            )

    def payload(self, dump_path: str | Path) -> dict[str, Any]:
        task_options: dict[str, Any] = {
            "scope": [],
            "list": [],
            "data_mode": ["all"],
            "summary_mode": "statistics",
        }
        return {
            "task": self.task,
            "dump_path": str(Path(dump_path).resolve()),
            "rank": list(self.ranks),
            "step": list(self.steps),
            "level": self.level,
            "async_dump": False,
            self.task: task_options,
        }


def write_capture_config(
    path: str | Path,
    *,
    dump_path: str | Path,
    config: MsprobeCaptureConfig,
) -> Path:
    """Write the official ``PrecisionDebugger`` JSON configuration."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(config.payload(dump_path), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def install_trainer_capture(config_path: str | Path | None = None) -> Any:
    """Wrap one TorchTitan process' train steps with ``PrecisionDebugger``."""

    value = config_path or os.environ.get(MSPROBE_CONFIG_PATH_ENV)
    if not value:
        raise RuntimeError(f"{MSPROBE_CONFIG_PATH_ENV} must name an msProbe config")
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        from msprobe.pytorch import PrecisionDebugger
    except ImportError as error:
        raise RuntimeError(
            "mindstudio-probe with PyTorch support is required for msProbe capture"
        ) from error

    from torchtitan.trainer import Trainer

    marker = "_glm5_msprobe_original_train_step"
    if hasattr(Trainer, marker):
        return getattr(Trainer, "_glm5_msprobe_debugger")

    debugger = PrecisionDebugger(config_path=str(path))
    original_train_step = Trainer.train_step

    def reorder_rank_major_batches(
        gathered: Any,
        *,
        data_parallel_size: int,
        waves: int,
    ) -> Any:
        """Restore fixture order from rank-major all-gather output."""

        if data_parallel_size < 1 or waves < 1:
            raise ValueError("data-parallel size and waves must be positive")
        groups = data_parallel_size * waves
        if gathered.shape[0] % groups:
            raise ValueError(
                f"gathered batch {gathered.shape[0]} is not divisible by {groups}"
            )
        local_batch_size = gathered.shape[0] // groups
        trailing_shape = tuple(gathered.shape[1:])
        rank_major = gathered.reshape(
            data_parallel_size,
            waves,
            local_batch_size,
            *trailing_shape,
        )
        fixture_order = rank_major.permute(
            1,
            0,
            2,
            *range(3, rank_major.ndim),
        )
        return fixture_order.reshape(-1, *trailing_shape).contiguous()

    def logical_tensor(value: Any) -> Any:
        """Return the global logical value without attaching an observer graph."""

        import torch

        def wait_if_async(tensor: Any) -> Any:
            if type(tensor).__name__ != "AsyncCollectiveTensor":
                return tensor
            from torch.distributed._functional_collectives import wait_tensor

            return wait_tensor(tensor)

        if not isinstance(value, torch.Tensor):
            return value
        value = value.detach()
        try:
            from torch.distributed.tensor import DTensor, Replicate
        except ImportError:  # pragma: no cover - older PyTorch compatibility
            return value
        if not isinstance(value, DTensor):
            return wait_if_async(value)
        if all(isinstance(placement, Replicate) for placement in value.placements):
            return wait_if_async(value.to_local())
        # Every TP rank must execute this collective before rank 0 saves it.
        return wait_if_async(value.full_tensor())

    def install_block_boundary_hooks(trainer: Any) -> None:
        instance_marker = "_glm5_msprobe_block_boundary_handles"
        if hasattr(trainer, instance_marker):
            return

        import torch.distributed as dist

        handles = []
        discovered: set[int] = set()
        buffers: dict[tuple[int, str], list[Any]] = {}
        gradient_buffers: dict[tuple[int, str], dict[int, Any]] = {}
        setattr(trainer, "_glm5_msprobe_block_boundary_buffers", buffers)
        setattr(trainer, "_glm5_msprobe_block_gradient_buffers", gradient_buffers)
        for model_part in trainer.model_parts:
            for module_name, module in model_part.named_modules():
                match = re.search(r"(?:^|\.)layers\.(\d+)$", module_name)
                if match is None:
                    continue
                block_index = int(match.group(1))
                if block_index in discovered:
                    continue
                discovered.add(block_index)

                def save_boundary(
                    _module: Any,
                    args: tuple[Any, ...],
                    output: Any,
                    *,
                    index: int = block_index,
                ) -> None:
                    if not args:
                        raise RuntimeError(f"GLM5 block {index} has no positional input")
                    block_input = logical_tensor(args[0])
                    block_output = logical_tensor(output)
                    if os.environ.get(MSPROBE_BLOCK_GLOBAL_STEP_ENV) == "1":
                        input_chunks = buffers.setdefault((index, "input"), [])
                        invocation = len(input_chunks)
                        input_chunks.append(block_input.clone())
                        buffers.setdefault((index, "output"), []).append(
                            block_output.clone()
                        )
                        if os.environ.get(MSPROBE_BLOCK_BACKWARD_ENV) == "1":
                            import torch

                            def register_gradient(
                                value: Any,
                                boundary: str,
                            ) -> None:
                                if not isinstance(value, torch.Tensor):
                                    raise RuntimeError(
                                        f"GLM5 block {index} {boundary} is not a tensor"
                                    )
                                if not value.requires_grad:
                                    raise RuntimeError(
                                        f"GLM5 block {index} {boundary} does not require grad"
                                    )

                                def capture_gradient(
                                    gradient: Any,
                                    *,
                                    block: int = index,
                                    edge: str = boundary,
                                    call: int = invocation,
                                ) -> Any:
                                    logical_gradient = logical_tensor(gradient)
                                    gradient_buffers.setdefault(
                                        (block, edge), {}
                                    )[call] = logical_gradient.clone()
                                    return gradient

                                value.register_hook(capture_gradient)

                            register_gradient(args[0], "grad_input")
                            register_gradient(output, "grad_output")
                    elif not dist.is_initialized() or dist.get_rank() == 0:
                        debugger.save(
                            block_input,
                            f"block_{index:02d}_input",
                            save_backward=False,
                        )
                        debugger.save(
                            block_output,
                            f"block_{index:02d}_output",
                            save_backward=False,
                        )

                handles.append(module.register_forward_hook(save_boundary))

        if not discovered:
            raise RuntimeError("no GLM5 transformer blocks found for msProbe capture")
        setattr(trainer, instance_marker, tuple(handles))

    def clear_block_boundary_buffers(trainer: Any) -> None:
        buffers = getattr(trainer, "_glm5_msprobe_block_boundary_buffers")
        buffers.clear()
        gradient_buffers = getattr(
            trainer, "_glm5_msprobe_block_gradient_buffers"
        )
        gradient_buffers.clear()

    def stage_dump_owner(trainer: Any) -> bool:
        import torch.distributed as dist

        parallel_dims = trainer.parallel_dims
        data_parallel_size = (
            parallel_dims.dp_replicate * parallel_dims.dp_shard
        )
        stage_width = data_parallel_size * parallel_dims.cp * parallel_dims.tp
        return not dist.is_initialized() or dist.get_rank() % stage_width == 0

    def expected_local_step_rows(trainer: Any) -> int:
        """Return leading rows in one logical step across config versions."""

        data_parallel_size = (
            trainer.parallel_dims.dp_replicate * trainer.parallel_dims.dp_shard
        )
        training = trainer.config.training
        token_value = getattr(training, "num_tokens_per_train_step", None)
        if token_value is not None:
            tokens = int(token_value)
            if tokens <= 0 or tokens % data_parallel_size:
                raise RuntimeError(
                    "invalid token-based global step for boundary capture: "
                    f"num_tokens_per_train_step={tokens}, "
                    f"data_parallel_size={data_parallel_size}"
                )
            return tokens // data_parallel_size
        legacy_value = getattr(training, "global_batch_size", None)
        if legacy_value is not None:
            samples = int(legacy_value)
            if samples <= 0 or samples % data_parallel_size:
                raise RuntimeError(
                    "invalid sample-based global step for boundary capture: "
                    f"global_batch_size={samples}, "
                    f"data_parallel_size={data_parallel_size}"
                )
            return samples // data_parallel_size
        raise RuntimeError("training config has no global-step size")

    def global_step_tensor(
        trainer: Any,
        chunks: list[Any],
        *,
        allow_leading_extra: bool,
    ) -> Any:
        import torch
        import torch.distributed as dist

        parallel_dims = trainer.parallel_dims
        data_parallel_size = (
            parallel_dims.dp_replicate * parallel_dims.dp_shard
        )
        first_shape = tuple(chunks[0].shape)
        if any(tuple(chunk.shape) != first_shape for chunk in chunks):
            raise RuntimeError("block boundary has inconsistent chunks")
        local_step = torch.cat(chunks, dim=0).contiguous()
        expected_local_rows = expected_local_step_rows(trainer)
        if local_step.shape[0] < expected_local_rows:
            raise RuntimeError(
                f"captured {local_step.shape[0]} rows, expected {expected_local_rows}"
            )
        extra_rows = local_step.shape[0] - expected_local_rows
        chunk_rows = first_shape[0]
        if extra_rows % chunk_rows:
            raise RuntimeError("block boundary has a partial extra chunk")
        if extra_rows and not allow_leading_extra:
            raise RuntimeError(f"block gradient has {extra_rows} unexpected rows")
        if extra_rows:
            local_step = local_step[extra_rows:].contiguous()
        waves = expected_local_rows // chunk_rows
        if data_parallel_size > 1:
            batch_mesh = parallel_dims.get_mesh("batch")
            gathered = torch.empty(
                (data_parallel_size * local_step.shape[0], *local_step.shape[1:]),
                dtype=local_step.dtype,
                device=local_step.device,
            )
            dist.all_gather_into_tensor(
                gathered,
                local_step,
                group=batch_mesh.get_group(),
            )
            return reorder_rank_major_batches(
                gathered,
                data_parallel_size=data_parallel_size,
                waves=waves,
            )
        return local_step

    def save_global_step_boundaries(trainer: Any) -> None:
        buffers = getattr(trainer, "_glm5_msprobe_block_boundary_buffers")
        owns_stage_dump = stage_dump_owner(trainer)

        for (block_index, boundary), chunks in sorted(buffers.items()):
            if not chunks:
                raise RuntimeError(
                    f"GLM5 block {block_index} {boundary} captured no tensors"
                )
            # Pipeline schedules perform one leading shape-propagation call when
            # they are first stepped.  It is not a training microbatch (and on
            # non-first stages may contain uninitialized placeholder values).
            # Retain exactly the trailing rows belonging to this optimizer step.
            global_step = global_step_tensor(
                trainer,
                chunks,
                allow_leading_extra=True,
            )
            if owns_stage_dump:
                debugger.save(
                    global_step,
                    f"block_{block_index:02d}_{boundary}",
                    save_backward=False,
                )

    def save_global_step_gradients(trainer: Any) -> None:
        buffers = getattr(trainer, "_glm5_msprobe_block_boundary_buffers")
        gradient_buffers = getattr(
            trainer, "_glm5_msprobe_block_gradient_buffers"
        )
        owns_stage_dump = stage_dump_owner(trainer)
        data_parallel_size = (
            trainer.parallel_dims.dp_replicate * trainer.parallel_dims.dp_shard
        )
        expected_local_rows = expected_local_step_rows(trainer)

        for (block_index, boundary), captured in sorted(gradient_buffers.items()):
            forward_boundary = "input" if boundary == "grad_input" else "output"
            forward_chunks = buffers[(block_index, forward_boundary)]
            chunk_rows = forward_chunks[0].shape[0]
            expected_calls = expected_local_rows // chunk_rows
            extra_calls = len(forward_chunks) - expected_calls
            expected_indexes = range(extra_calls, len(forward_chunks))
            missing = [index for index in expected_indexes if index not in captured]
            if missing:
                raise RuntimeError(
                    f"GLM5 block {block_index} {boundary} is missing calls {missing}"
                )
            ordered_chunks = [captured[index] for index in expected_indexes]
            global_step = global_step_tensor(
                trainer,
                ordered_chunks,
                allow_leading_extra=False,
            )
            if owns_stage_dump:
                debugger.save(
                    global_step,
                    f"block_{block_index:02d}_{boundary}",
                    save_backward=False,
                )

    def install_final_norm_state_hooks(trainer: Any) -> None:
        instance_marker = "_glm5_msprobe_final_norm_state_handles"
        if hasattr(trainer, instance_marker):
            return

        import torch

        handles = []
        discovered = 0
        buffers: dict[str, list[Any]] = {
            "input": [],
            "output": [],
            "weight_local_numel": [],
            "is_backward_recompute": [],
        }
        gradient_buffers: dict[int, Any] = {}
        setattr(trainer, "_glm5_msprobe_final_norm_state_buffers", buffers)
        setattr(
            trainer,
            "_glm5_msprobe_final_norm_state_gradient_buffers",
            gradient_buffers,
        )

        for model_part in trainer.model_parts:
            for module_name, module in model_part.named_modules():
                if re.search(r"(?:^|\.)norm$", module_name) is None:
                    continue
                if not hasattr(module, "weight"):
                    continue
                discovered += 1

                def capture_final_norm(
                    _module: Any,
                    args: tuple[Any, ...],
                    output: Any,
                ) -> None:
                    if not args:
                        raise RuntimeError("GLM5 final norm has no positional input")
                    if not isinstance(output, torch.Tensor) or not output.requires_grad:
                        raise RuntimeError(
                            "GLM5 final norm output must be a differentiable tensor"
                        )
                    invocation = len(buffers["output"])
                    buffers["input"].append(logical_tensor(args[0]).clone())
                    buffers["output"].append(logical_tensor(output).clone())
                    weight = _module.weight
                    if type(weight).__name__ == "DTensor":
                        weight = weight.to_local()
                    buffers["weight_local_numel"].append(weight.numel())
                    buffers["is_backward_recompute"].append(
                        float(torch._C._current_graph_task_id() != -1)
                    )

                    def capture_gradient(gradient: Any, *, call: int = invocation) -> Any:
                        gradient_buffers[call] = logical_tensor(gradient).clone()
                        return gradient

                    output.register_hook(capture_gradient)

                handles.append(module.register_forward_hook(capture_final_norm))

        if discovered > 1:
            raise RuntimeError(f"found {discovered} GLM5 final norms on one stage")
        if discovered == 0 and getattr(trainer.parallel_dims, "pp", 1) == 1:
            raise RuntimeError("no GLM5 final norm found for msProbe capture")
        setattr(trainer, instance_marker, tuple(handles))

    def clear_final_norm_state_buffers(trainer: Any) -> None:
        getattr(trainer, "_glm5_msprobe_final_norm_state_buffers")["input"].clear()
        getattr(trainer, "_glm5_msprobe_final_norm_state_buffers")["output"].clear()
        getattr(trainer, "_glm5_msprobe_final_norm_state_buffers")[
            "weight_local_numel"
        ].clear()
        getattr(trainer, "_glm5_msprobe_final_norm_state_buffers")[
            "is_backward_recompute"
        ].clear()
        getattr(
            trainer,
            "_glm5_msprobe_final_norm_state_gradient_buffers",
        ).clear()

    def final_norm_target(trainer: Any) -> dict[str, Any]:
        matches = {
            name: parameter
            for name, parameter in named_trainable_parameters(trainer).items()
            if name == "norm.weight" or name.endswith(".norm.weight")
        }
        if len(matches) > 1:
            raise RuntimeError(
                f"found multiple final norm parameters: {sorted(matches)}"
            )
        if not matches and getattr(trainer.parallel_dims, "pp", 1) == 1:
            raise RuntimeError("no final norm parameter found for msProbe capture")
        return matches

    def save_final_norm_state(trainer: Any, capture: dict[str, Any] | None) -> None:
        import torch

        if capture is None:
            return
        if not capture.get("clip_called"):
            raise RuntimeError("final norm diagnostic did not observe gradient clipping")

        buffers = getattr(trainer, "_glm5_msprobe_final_norm_state_buffers")
        gradient_buffers = getattr(
            trainer,
            "_glm5_msprobe_final_norm_state_gradient_buffers",
        )
        output_chunks = buffers["output"]
        if not output_chunks:
            raise RuntimeError("GLM5 final norm captured no tensors")

        data_parallel_size = (
            trainer.parallel_dims.dp_replicate * trainer.parallel_dims.dp_shard
        )
        expected_local_rows = expected_local_step_rows(trainer)
        chunk_rows = output_chunks[0].shape[0]
        expected_calls = expected_local_rows // chunk_rows
        extra_calls = len(output_chunks) - expected_calls
        expected_indexes = range(extra_calls, len(output_chunks))
        missing = [index for index in expected_indexes if index not in gradient_buffers]
        if missing:
            raise RuntimeError(
                f"GLM5 final norm output gradient is missing calls {missing}"
            )

        final_input = global_step_tensor(
            trainer,
            buffers["input"],
            allow_leading_extra=True,
        )
        final_output = global_step_tensor(
            trainer,
            output_chunks,
            allow_leading_extra=True,
        )
        final_grad_output = global_step_tensor(
            trainer,
            [gradient_buffers[index] for index in expected_indexes],
            allow_leading_extra=False,
        )
        initial_weight = capture["initial_weight"]
        reconstructed = reconstruct_rmsnorm_weight_gradient(
            final_output,
            final_grad_output,
            initial_weight,
        )
        preclip = next(iter(capture["preclip_grad"].values())).float()
        postclip = next(iter(capture["postclip_grad"].values())).float()

        if capture.get("reduce_transition"):
            for index, gradient in enumerate(capture["after_backward"]):
                if gradient is not None:
                    debugger.save(
                        gradient.float(),
                        f"final_norm_rank_local_grad_after_backward_{index}",
                        save_backward=False,
                    )
            for boundary in ("before_reduce", "after_reduce"):
                gradient = capture.get(boundary)
                if gradient is not None:
                    debugger.save(
                        gradient.float(),
                        f"final_norm_rank_local_grad_{boundary}",
                        save_backward=False,
                    )
            presence = torch.tensor(
                [
                    float(gradient is not None)
                    for gradient in (
                        *capture["after_backward"],
                        capture.get("before_reduce"),
                        capture.get("after_reduce"),
                    )
                ],
                dtype=torch.float32,
                device=final_output.device,
            )
            debugger.save(
                presence,
                "final_norm_rank_local_grad_presence",
                save_backward=False,
            )
            debugger.save(
                torch.tensor(
                    capture["backward_last_flags"],
                    dtype=torch.float32,
                    device=final_output.device,
                ),
                "final_norm_pipeline_last_backward_flags",
                save_backward=False,
            )
            debugger.save(
                torch.tensor(
                    [
                        *capture["reduce_enabled_after_backward"],
                        capture["reduce_enabled_before_reduce"],
                        capture["reduce_enabled_after_reduce"],
                    ],
                    dtype=torch.float32,
                    device=final_output.device,
                ),
                "final_norm_fsdp_reduce_enabled",
                save_backward=False,
            )
            debugger.save(
                torch.tensor(
                    [float(capture["forced_pre_reduce_sync"])],
                    dtype=torch.float32,
                    device=final_output.device,
                ),
                "final_norm_forced_pre_reduce_sync",
                save_backward=False,
            )
            debugger.save(
                torch.tensor(
                    [float(capture["forced_sharded_grad_all_reduce"])],
                    dtype=torch.float32,
                    device=final_output.device,
                ),
                "final_norm_forced_sharded_grad_all_reduce",
                save_backward=False,
            )
            debugger.save(
                torch.tensor(
                    [float(capture["forced_native_last_backward_sync"])],
                    dtype=torch.float32,
                    device=final_output.device,
                ),
                "final_norm_forced_native_last_backward_sync",
                save_backward=False,
            )
            debugger.save(
                torch.tensor(
                    buffers["weight_local_numel"],
                    dtype=torch.float32,
                    device=final_output.device,
                ),
                "final_norm_forward_weight_local_numel",
                save_backward=False,
            )
            debugger.save(
                torch.tensor(
                    buffers["is_backward_recompute"],
                    dtype=torch.float32,
                    device=final_output.device,
                ),
                "final_norm_forward_is_backward_recompute",
                save_backward=False,
            )
            internal_boundaries = (
                *capture["fsdp_internal_after_backward"],
                capture["fsdp_internal_before_reduce"],
                capture["fsdp_internal_after_reduce"],
            )
            for source in ("unsharded_accumulated", "unsharded_direct"):
                source_values = [boundary[source] for boundary in internal_boundaries]
                debugger.save(
                    torch.tensor(
                        [float(value is not None) for value in source_values],
                        dtype=torch.float32,
                        device=final_output.device,
                    ),
                    f"final_norm_fsdp_{source}_presence",
                    save_backward=False,
                )
                for index, value in enumerate(source_values):
                    if value is not None:
                        debugger.save(
                            value.float(),
                            f"final_norm_fsdp_{source}_{index}",
                            save_backward=False,
                        )
            debugger.save(
                torch.tensor(
                    capture["fsdp_lifecycle_events"],
                    dtype=torch.float32,
                    device=final_output.device,
                ),
                "final_norm_fsdp_lifecycle_events",
                save_backward=False,
            )
            for microbatch, index in enumerate(expected_indexes):
                reconstructed_microbatch = reconstruct_rmsnorm_weight_gradient(
                    output_chunks[index],
                    gradient_buffers[index],
                    initial_weight,
                )
                debugger.save(
                    reconstructed_microbatch,
                    f"final_norm_boundary_reconstructed_grad_fp32_microbatch_{microbatch}",
                    save_backward=False,
                )

        if stage_dump_owner(trainer):
            values = {
                "final_norm_input": final_input,
                "final_norm_output": final_output,
                "final_norm_grad_output": final_grad_output,
                "final_norm_boundary_reconstructed_grad_fp32": reconstructed,
                "final_norm_parameter_preclip_grad": preclip,
                "final_norm_parameter_postclip_grad": postclip,
                "final_norm_preclip_minus_boundary_reconstructed_fp32": (
                    preclip - reconstructed
                ),
            }
            for name, value in values.items():
                debugger.save(value, name, save_backward=False)

    def install_final_norm_reduce_transition_hooks(
        trainer: Any,
        capture: dict[str, Any] | None,
    ) -> Any:
        """Snapshot the rank-local final-norm gradient around PP reduction."""

        if capture is None:
            return None
        schedule = trainer.pp_schedule
        stages = getattr(schedule, "_stages", None)
        if stages is None:
            stage = getattr(schedule, "_stage", None)
            stages = [] if stage is None else [stage]
        if not stages:
            raise RuntimeError("pipeline schedule exposes no stage for transition probe")

        parameter = next(iter(capture["targets"].values()))
        capture.update(
            {
                "reduce_transition": True,
                "after_backward": [],
                "before_reduce": None,
                "after_reduce": None,
                "reduce_called": False,
                "backward_last_flags": [],
                "reduce_enabled_after_backward": [],
                "reduce_enabled_before_reduce": None,
                "reduce_enabled_after_reduce": None,
                "forced_pre_reduce_sync": (
                    os.environ.get(MSPROBE_FINAL_NORM_PRE_REDUCE_SYNC_ENV) == "1"
                ),
                "forced_sharded_grad_all_reduce": (
                    os.environ.get(MSPROBE_FINAL_NORM_SHARDED_GRAD_ALL_REDUCE_ENV)
                    == "1"
                ),
                "forced_native_last_backward_sync": (
                    os.environ.get(MSPROBE_FINAL_NORM_NATIVE_LAST_BACKWARD_SYNC_ENV)
                    == "1"
                ),
                "fsdp_internal_after_backward": [],
                "fsdp_internal_before_reduce": None,
                "fsdp_internal_after_reduce": None,
                "fsdp_lifecycle_events": [],
            }
        )
        originals = []

        def snapshot_tensor(value: Any | None) -> Any | None:
            import torch

            if value is None:
                return None
            try:
                from torch.distributed.tensor import DTensor
            except ImportError:  # pragma: no cover - older PyTorch compatibility
                DTensor = ()
            if isinstance(value, DTensor):
                value = value.to_local()
            if type(value).__name__ == "AsyncCollectiveTensor":
                from torch.distributed._functional_collectives import wait_tensor

                value = wait_tensor(value)
            if not isinstance(value, torch.Tensor):
                raise RuntimeError("final norm rank-local gradient is not a tensor")
            return value.detach().clone()

        def snapshot_local_gradient() -> Any | None:
            return snapshot_tensor(parameter.grad)

        def fsdp_reduce_enabled(stage: Any) -> bool:
            from torch.distributed.fsdp import fully_shard

            distributed_state = fully_shard.state(stage.submod)
            values = {
                bool(group.reduce_grads)
                for state in distributed_state._state_ctx.all_states
                for group in state._fsdp_param_groups
            }
            if len(values) != 1:
                raise RuntimeError(
                    "FSDP parameter groups disagree on reduce_grads: "
                    f"{sorted(values)}"
                )
            return values.pop()

        def find_target_fsdp_param(stage: Any) -> Any:
            from torch.distributed.fsdp import fully_shard

            distributed_state = fully_shard.state(stage.submod)
            fsdp_params = [
                fsdp_param
                for state in distributed_state._state_ctx.all_states
                for group in state._fsdp_param_groups
                for fsdp_param in group.fsdp_params
            ]
            identity_matches = [
                fsdp_param
                for fsdp_param in fsdp_params
                if fsdp_param.sharded_param is parameter
            ]
            if len(identity_matches) == 1:
                return identity_matches[0]
            target_name = next(iter(capture["targets"]))
            name_matches = [
                fsdp_param
                for fsdp_param in fsdp_params
                if fsdp_param._param_fqn == target_name
                or (
                    fsdp_param._param_fqn is not None
                    and target_name.endswith(fsdp_param._param_fqn)
                )
            ]
            if len(name_matches) != 1:
                raise RuntimeError(
                    "could not uniquely resolve final norm FSDP parameter: "
                    f"identity={len(identity_matches)}, name={len(name_matches)}"
                )
            return name_matches[0]

        def find_target_fsdp_group(stage: Any) -> tuple[Any, Any]:
            from torch.distributed.fsdp import fully_shard

            fsdp_param = find_target_fsdp_param(stage)
            distributed_state = fully_shard.state(stage.submod)
            matches = [
                group
                for state in distributed_state._state_ctx.all_states
                for group in state._fsdp_param_groups
                if fsdp_param in group.fsdp_params
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "could not uniquely resolve final norm FSDP parameter group: "
                    f"{len(matches)}"
                )
            return matches[0], fsdp_param

        def sharded_state_code(fsdp_param: Any) -> float:
            return float(fsdp_param.sharded_state.value)

        def registered_parameter_local_numel(fsdp_param: Any) -> float:
            current = getattr(
                fsdp_param._module_info.module,
                fsdp_param._module_info.param_name,
            )
            if type(current).__name__ == "DTensor":
                current = current.to_local()
            return float(current.numel())

        def snapshot_fsdp_internal(stage: Any) -> dict[str, Any | None]:
            fsdp_param = find_target_fsdp_param(stage)
            unsharded_param = getattr(fsdp_param, "_unsharded_param", None)
            return {
                "unsharded_accumulated": snapshot_tensor(
                    fsdp_param.unsharded_accumulated_grad
                ),
                "unsharded_direct": snapshot_tensor(
                    None if unsharded_param is None else unsharded_param.grad
                ),
            }

        def all_reduce_sharded_gradient(stage: Any) -> None:
            import torch
            import torch.distributed as dist

            gradient = parameter.grad
            if gradient is None:
                raise RuntimeError("no sharded final norm gradient to all-reduce")
            try:
                from torch.distributed.tensor import DTensor
            except ImportError:  # pragma: no cover - older PyTorch compatibility
                DTensor = ()
            local_gradient = gradient.to_local() if isinstance(gradient, DTensor) else gradient
            if type(local_gradient).__name__ == "AsyncCollectiveTensor":
                from torch.distributed._functional_collectives import wait_tensor

                local_gradient = wait_tensor(local_gradient)
            if not isinstance(local_gradient, torch.Tensor):
                raise RuntimeError("sharded final norm gradient is not a tensor")
            fsdp_param = find_target_fsdp_param(stage)
            dist.all_reduce(
                local_gradient,
                op=dist.ReduceOp.SUM,
                group=fsdp_param.mesh_info.shard_process_group,
            )

        def run_backward_with_native_last_sync(
            stage: Any,
            original: Any,
            backward_args: tuple[Any, ...],
            backward_kwargs: dict[str, Any],
        ) -> Any:
            method_names = (
                "set_is_last_backward",
                "set_reshard_after_backward",
                "set_requires_gradient_sync",
            )
            originals = {
                name: getattr(stage.submod, name)
                for name in method_names
            }
            try:
                for name, setter in originals.items():
                    def force_true(
                        *unused_args: Any,
                        _setter: Any = setter,
                        **setter_kwargs: Any,
                    ) -> Any:
                        return _setter(True, **setter_kwargs)

                    setattr(stage.submod, name, force_true)
                return original(*backward_args, **backward_kwargs)
            finally:
                for name, setter in originals.items():
                    setattr(stage.submod, name, setter)

        for stage in stages:
            original_forward = stage.forward_maybe_with_nosync
            original_backward = stage.backward_maybe_with_nosync
            original_reduce = stage.perform_reduce_grad
            group_hook_state: dict[str, Any] = {
                "target_group": None,
                "originals": {},
            }

            def install_group_lifecycle_hooks(
                stage: Any,
                _group_hook_state: dict[str, Any] = group_hook_state,
            ) -> None:
                if _group_hook_state["target_group"] is not None:
                    return
                target_group, target_fsdp_param = find_target_fsdp_group(stage)
                _group_hook_state["target_group"] = target_group

                for method_name, event_code in FSDP_LIFECYCLE_EVENT_CODES.items():
                    original_group_method = getattr(target_group, method_name)

                    @wraps(original_group_method)
                    def group_method_with_capture(
                        *method_args: Any,
                        _original_group_method: Any = original_group_method,
                        _event_code: int = event_code,
                        _fsdp_param: Any = target_fsdp_param,
                        **method_kwargs: Any,
                    ) -> Any:
                        before_state = sharded_state_code(_fsdp_param)
                        before_numel = registered_parameter_local_numel(_fsdp_param)
                        result = _original_group_method(*method_args, **method_kwargs)
                        capture["fsdp_lifecycle_events"].append(
                            [
                                float(_event_code),
                                before_state,
                                sharded_state_code(_fsdp_param),
                                before_numel,
                                registered_parameter_local_numel(_fsdp_param),
                            ]
                        )
                        return result

                    _group_hook_state["originals"][method_name] = (
                        original_group_method
                    )
                    setattr(target_group, method_name, group_method_with_capture)

            @wraps(original_forward)
            def forward_with_capture(
                *forward_args: Any,
                _original: Any = original_forward,
                _stage: Any = stage,
                _install: Any = install_group_lifecycle_hooks,
                **forward_kwargs: Any,
            ) -> Any:
                result = _original(*forward_args, **forward_kwargs)
                _install(_stage)
                return result

            @wraps(original_backward)
            def backward_with_capture(
                *backward_args: Any,
                _original: Any = original_backward,
                _stage: Any = stage,
                **backward_kwargs: Any,
            ) -> Any:
                last_backward = backward_kwargs.get("last_backward")
                if last_backward is None and len(backward_args) >= 3:
                    last_backward = backward_args[2]
                if last_backward is None:
                    last_backward = False
                if capture["forced_native_last_backward_sync"] and last_backward:
                    result = run_backward_with_native_last_sync(
                        _stage,
                        _original,
                        backward_args,
                        backward_kwargs,
                    )
                else:
                    result = _original(*backward_args, **backward_kwargs)
                capture["after_backward"].append(snapshot_local_gradient())
                capture["backward_last_flags"].append(float(bool(last_backward)))
                capture["reduce_enabled_after_backward"].append(
                    float(fsdp_reduce_enabled(_stage))
                )
                capture["fsdp_internal_after_backward"].append(
                    snapshot_fsdp_internal(_stage)
                )
                return result

            @wraps(original_reduce)
            def reduce_with_capture(
                *reduce_args: Any,
                _original: Any = original_reduce,
                _stage: Any = stage,
                **reduce_kwargs: Any,
            ) -> Any:
                if capture["reduce_called"]:
                    raise RuntimeError("final norm transition observed multiple reductions")
                capture["reduce_called"] = True
                capture["before_reduce"] = snapshot_local_gradient()
                capture["reduce_enabled_before_reduce"] = float(
                    fsdp_reduce_enabled(_stage)
                )
                capture["fsdp_internal_before_reduce"] = snapshot_fsdp_internal(
                    _stage
                )
                if capture["forced_pre_reduce_sync"]:
                    import torch

                    torch.accelerator.synchronize()
                if capture["forced_sharded_grad_all_reduce"]:
                    all_reduce_sharded_gradient(_stage)
                result = _original(*reduce_args, **reduce_kwargs)
                capture["after_reduce"] = snapshot_local_gradient()
                capture["reduce_enabled_after_reduce"] = float(
                    fsdp_reduce_enabled(_stage)
                )
                capture["fsdp_internal_after_reduce"] = snapshot_fsdp_internal(
                    _stage
                )
                return result

            stage.forward_maybe_with_nosync = forward_with_capture
            stage.backward_maybe_with_nosync = backward_with_capture
            stage.perform_reduce_grad = reduce_with_capture
            originals.append(
                (
                    stage,
                    original_forward,
                    original_backward,
                    original_reduce,
                    group_hook_state,
                )
            )

        def restore() -> None:
            for (
                stage,
                original_forward,
                original_backward,
                original_reduce,
                group_hook_state,
            ) in originals:
                stage.forward_maybe_with_nosync = original_forward
                stage.backward_maybe_with_nosync = original_backward
                stage.perform_reduce_grad = original_reduce
                target_group = group_hook_state["target_group"]
                if target_group is not None:
                    for method_name, original_group_method in group_hook_state[
                        "originals"
                    ].items():
                        setattr(target_group, method_name, original_group_method)

        return restore

    def install_router_state_hooks(trainer: Any) -> None:
        instance_marker = "_glm5_msprobe_router_state_handles"
        if hasattr(trainer, instance_marker):
            return

        import torch

        handles = []
        discovered: set[int] = set()
        buffers: dict[tuple[int, str], list[Any]] = {}
        gradient_buffers: dict[tuple[int, str], dict[int, Any]] = {}
        gradient_sources: dict[tuple[int, str], str] = {}
        setattr(trainer, "_glm5_msprobe_router_state_buffers", buffers)
        setattr(
            trainer,
            "_glm5_msprobe_router_state_gradient_buffers",
            gradient_buffers,
        )
        setattr(
            trainer,
            "_glm5_msprobe_router_state_gradient_sources",
            gradient_sources,
        )

        def capture_forward(layer: int, field: str, value: Any) -> int:
            logical = logical_tensor(value)
            field_buffers = buffers.setdefault((layer, field), [])
            invocation = len(field_buffers)
            field_buffers.append(logical.clone())
            return invocation

        def register_gradient(
            layer: int,
            field: str,
            source: str,
            value: Any,
            invocation: int,
        ) -> None:
            if not isinstance(value, torch.Tensor) or not value.requires_grad:
                raise RuntimeError(
                    f"GLM5 router layer {layer} {field} does not require grad"
                )
            gradient_sources[(layer, field)] = source

            def capture(gradient: Any) -> Any:
                import torch.distributed as dist

                summed = gradient.detach()
                try:
                    from torch.distributed.tensor import DTensor
                except ImportError:  # pragma: no cover
                    DTensor = ()
                if isinstance(summed, DTensor):
                    summed = summed.to_local()
                summed = summed.clone()
                if trainer.parallel_dims.tp > 1:
                    dist.all_reduce(
                        summed,
                        group=trainer.parallel_dims.get_mesh("tp").get_group(),
                    )
                logical = logical_tensor(gradient)
                gradient_buffers.setdefault((layer, field), {})[
                    invocation
                ] = logical.clone()
                summed_field = f"{field}_tp_sum"
                gradient_sources[(layer, summed_field)] = source
                gradient_buffers.setdefault((layer, summed_field), {})[
                    invocation
                ] = summed
                return gradient

            value.register_hook(capture)

        for model_part in trainer.model_parts:
            for module_name, module in model_part.named_modules():
                match = re.search(r"(?:^|\.)layers\.(\d+)\.moe\.router$", module_name)
                if match is None:
                    continue
                layer = int(match.group(1))
                if layer in discovered:
                    continue
                discovered.add(layer)

                def capture_gate(
                    _module: Any,
                    args: tuple[Any, ...],
                    output: Any,
                    *,
                    index: int = layer,
                ) -> None:
                    if not args:
                        raise RuntimeError(f"GLM5 router gate layer {index} has no input")
                    invocation = capture_forward(index, "gate_input", args[0])
                    output_invocation = capture_forward(index, "gate_logits", output)
                    if invocation != output_invocation:
                        raise RuntimeError(
                            f"GLM5 router gate layer {index} invocation mismatch"
                        )
                    register_gradient(
                        index,
                        "gate_grad_input",
                        "gate_input",
                        args[0],
                        invocation,
                    )
                    register_gradient(
                        index,
                        "gate_grad_logits",
                        "gate_logits",
                        output,
                        invocation,
                    )

                def capture_router(
                    router: Any,
                    args: tuple[Any, ...],
                    output: Any,
                    *,
                    index: int = layer,
                ) -> None:
                    if not isinstance(output, tuple) or len(output) != 3:
                        raise RuntimeError(
                            f"GLM5 router layer {index} returned an unexpected value"
                        )
                    topk_scores, topk_ids, scores = output
                    scores_logical = logical_tensor(scores)
                    topk_scores_logical = logical_tensor(topk_scores)
                    topk_ids_logical = logical_tensor(topk_ids).long()
                    invocation = capture_forward(index, "scores", scores)
                    capture_forward(index, "topk_scores", topk_scores)

                    selection_map = torch.zeros_like(scores_logical).scatter(
                        -1,
                        topk_ids_logical,
                        1.0,
                    )
                    weighted_map = torch.zeros_like(scores_logical).scatter(
                        -1,
                        topk_ids_logical,
                        topk_scores_logical,
                    )
                    capture_forward(index, "selection_map", selection_map)
                    capture_forward(index, "weighted_map", weighted_map)

                    scores_for_choice = scores_logical
                    if len(args) > 1 and args[1] is not None:
                        scores_for_choice = scores_for_choice + logical_tensor(args[1])
                    choice_values = torch.topk(
                        scores_for_choice,
                        k=router.top_k + 1,
                        dim=-1,
                        sorted=True,
                    ).values
                    margin = (
                        choice_values[..., router.top_k - 1]
                        - choice_values[..., router.top_k]
                    )
                    capture_forward(index, "topk_margin", margin)

                    register_gradient(
                        index,
                        "scores_grad",
                        "scores",
                        scores,
                        invocation,
                    )
                    register_gradient(
                        index,
                        "topk_scores_grad",
                        "topk_scores",
                        topk_scores,
                        invocation,
                    )

                handles.append(module.gate.register_forward_hook(capture_gate))
                handles.append(module.register_forward_hook(capture_router))

        # A pipeline stage may legitimately contain only dense layers (the
        # eight-stage debug model's first stage owns dense layer 0). Other
        # stages still save their local routers and the combined dump count is
        # validated against the single-card reference.
        if not discovered and getattr(trainer.parallel_dims, "pp", 1) == 1:
            raise RuntimeError("no GLM5 MoE routers found for msProbe capture")
        setattr(trainer, instance_marker, tuple(handles))

    def clear_router_state_buffers(trainer: Any) -> None:
        getattr(trainer, "_glm5_msprobe_router_state_buffers").clear()
        getattr(trainer, "_glm5_msprobe_router_state_gradient_buffers").clear()
        getattr(trainer, "_glm5_msprobe_router_state_gradient_sources").clear()

    def save_global_step_router_state(trainer: Any) -> None:
        buffers = getattr(trainer, "_glm5_msprobe_router_state_buffers")
        gradient_buffers = getattr(
            trainer, "_glm5_msprobe_router_state_gradient_buffers"
        )
        gradient_sources = getattr(
            trainer, "_glm5_msprobe_router_state_gradient_sources"
        )
        owns_stage_dump = stage_dump_owner(trainer)
        data_parallel_size = (
            trainer.parallel_dims.dp_replicate * trainer.parallel_dims.dp_shard
        )
        expected_local_rows = expected_local_step_rows(trainer)

        for (layer, field), chunks in sorted(buffers.items()):
            if not chunks:
                raise RuntimeError(f"GLM5 router layer {layer} {field} is empty")
            global_step = global_step_tensor(
                trainer,
                chunks,
                allow_leading_extra=True,
            )
            if owns_stage_dump:
                debugger.save(
                    global_step,
                    f"router_{layer:02d}_{field}",
                    save_backward=False,
                )

        for (layer, field), captured in sorted(gradient_buffers.items()):
            source = gradient_sources[(layer, field)]
            forward_chunks = buffers[(layer, source)]
            chunk_rows = forward_chunks[0].shape[0]
            expected_calls = expected_local_rows // chunk_rows
            extra_calls = len(forward_chunks) - expected_calls
            expected_indexes = range(extra_calls, len(forward_chunks))
            missing = [index for index in expected_indexes if index not in captured]
            if missing:
                raise RuntimeError(
                    f"GLM5 router layer {layer} {field} is missing calls {missing}"
                )
            global_step = global_step_tensor(
                trainer,
                [captured[index] for index in expected_indexes],
                allow_leading_extra=False,
            )
            if owns_stage_dump:
                debugger.save(
                    global_step,
                    f"router_{layer:02d}_{field}",
                    save_backward=False,
                )

    def named_trainable_parameters(trainer: Any) -> dict[str, Any]:
        parameters: dict[str, Any] = {}
        for model_part in trainer.model_parts:
            for name, parameter in model_part.named_parameters():
                if not parameter.requires_grad:
                    continue
                if name in parameters and parameters[name] is not parameter:
                    raise RuntimeError(f"duplicate trainable parameter name: {name}")
                parameters[name] = parameter
        if not parameters:
            raise RuntimeError("no trainable parameters found for msProbe capture")
        return parameters

    def snapshot_trainable_parameters(trainer: Any) -> dict[str, Any]:
        return {
            name: parameter.detach().clone()
            for name, parameter in named_trainable_parameters(trainer).items()
        }

    def parameter_debug_name(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]+", "__", name).strip("_")

    def save_parameter_state(
        trainer: Any,
        initial_parameters: dict[str, Any],
    ) -> None:
        import torch

        parameters = named_trainable_parameters(trainer)
        if parameters.keys() != initial_parameters.keys():
            raise RuntimeError("trainable parameter names changed during train_step")
        owns_stage_dump = stage_dump_owner(trainer)
        debug_names: set[str] = set()

        for name, parameter in sorted(parameters.items()):
            debug_name = parameter_debug_name(name)
            if debug_name in debug_names:
                raise RuntimeError(f"parameter debug-name collision: {debug_name}")
            debug_names.add(debug_name)
            initial = logical_tensor(initial_parameters[name])
            updated = logical_tensor(parameter)
            update = updated - initial
            if owns_stage_dump:
                grad_present = torch.tensor(
                    [float(parameter.grad is not None), float(parameter.grad is None)],
                    dtype=torch.float32,
                    device=updated.device,
                )
                debugger.save(
                    grad_present,
                    f"parameter_grad_present__{debug_name}",
                    save_backward=False,
                )
            if parameter.grad is not None:
                gradient = logical_tensor(parameter.grad)
                if owns_stage_dump:
                    debugger.save(
                        gradient,
                        f"parameter_grad__{debug_name}",
                        save_backward=False,
                    )
            if owns_stage_dump:
                debugger.save(
                    update,
                    f"parameter_update__{debug_name}",
                    save_backward=False,
                )
                debugger.save(
                    updated,
                    f"parameter_after__{debug_name}",
                    save_backward=False,
                )

    def optimizer_probe_targets(trainer: Any) -> dict[str, Any]:
        parameters = named_trainable_parameters(trainer)
        targets: dict[str, Any] = {}
        for requested_name in OPTIMIZER_DIAGNOSTIC_PARAMETERS:
            matches = [
                (name, parameter)
                for name, parameter in parameters.items()
                if name == requested_name or name.endswith(f".{requested_name}")
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected one optimizer diagnostic parameter matching "
                    f"{requested_name!r}, found {[name for name, _ in matches]}"
                )
            name, parameter = matches[0]
            targets[name] = parameter
        return targets

    def optimizer_group_hyperparameters(
        trainer: Any,
        targets: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        by_parameter: dict[int, tuple[Any, dict[str, Any]]] = {}
        for optimizer in trainer.optimizers:
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if id(parameter) in by_parameter:
                        raise RuntimeError(
                            "parameter belongs to multiple optimizer groups"
                        )
                    by_parameter[id(parameter)] = (optimizer, group)

        captured: dict[str, dict[str, Any]] = {}
        for name, parameter in targets.items():
            if id(parameter) not in by_parameter:
                raise RuntimeError(f"no optimizer group found for {name}")
            optimizer, group = by_parameter[id(parameter)]
            if optimizer.__class__.__name__ != "AdamW":
                raise RuntimeError(
                    f"optimizer diagnostic requires AdamW, got "
                    f"{optimizer.__class__.__name__} for {name}"
                )
            if group.get("amsgrad", False) or group.get("maximize", False):
                raise RuntimeError(
                    "optimizer diagnostic supports only standard AdamW without "
                    f"amsgrad/maximize: {name}"
                )
            captured[name] = {
                "optimizer": optimizer,
                "lr": float(group["lr"]),
                "betas": tuple(float(value) for value in group["betas"]),
                "eps": float(group["eps"]),
                "weight_decay": float(group["weight_decay"]),
            }
        return captured

    def save_optimizer_state(
        trainer: Any,
        initial_parameters: dict[str, Any],
        capture: dict[str, Any],
    ) -> None:
        import torch

        if not capture.get("clip_called"):
            raise RuntimeError("optimizer diagnostic did not observe gradient clipping")
        owns_stage_dump = stage_dump_owner(trainer)
        targets = capture["targets"]
        hyperparameters = capture["hyperparameters"]

        if owns_stage_dump:
            debugger.save(
                capture["clip_stats"],
                "optimizer_clip_norm_and_coefficient",
                save_backward=False,
            )

        for name, parameter in sorted(targets.items()):
            debug_name = parameter_debug_name(name)
            settings = hyperparameters[name]
            state = settings["optimizer"].state[parameter]
            if not {"step", "exp_avg", "exp_avg_sq"}.issubset(state):
                raise RuntimeError(f"AdamW state is incomplete after step for {name}")
            step_value = logical_tensor(state["step"])
            step = float(
                step_value.item() if hasattr(step_value, "item") else step_value
            )
            initial = logical_tensor(initial_parameters[name])
            updated = logical_tensor(parameter)
            exp_avg = logical_tensor(state["exp_avg"])
            exp_avg_sq = logical_tensor(state["exp_avg_sq"])
            adaptive, decay, intended = adamw_update_components(
                initial,
                exp_avg,
                exp_avg_sq,
                step=step,
                lr=settings["lr"],
                betas=settings["betas"],
                eps=settings["eps"],
                weight_decay=settings["weight_decay"],
            )
            actual = updated.float() - initial.float()
            single_cast = (
                (initial.float() + intended).to(dtype=initial.dtype).float()
                - initial.float()
            )
            rounding_residual = actual - intended
            hparams = torch.tensor(
                [
                    step,
                    settings["lr"],
                    settings["betas"][0],
                    settings["betas"][1],
                    settings["eps"],
                    settings["weight_decay"],
                ],
                dtype=torch.float32,
                device=updated.device,
            )
            values = {
                "preclip_grad": capture["preclip_grad"][name],
                "postclip_grad": capture["postclip_grad"][name],
                "exp_avg": exp_avg,
                "exp_avg_sq": exp_avg_sq,
                "adaptive_delta_fp32": adaptive,
                "decay_delta_fp32": decay,
                "intended_delta_fp32": intended,
                "actual_delta_fp32": actual,
                "single_cast_delta_fp32": single_cast,
                "rounding_residual_fp32": rounding_residual,
                "step_hparams": hparams,
            }
            if owns_stage_dump:
                for field, value in values.items():
                    debugger.save(
                        value,
                        f"optimizer_{field}__{debug_name}",
                        save_backward=False,
                    )

    @wraps(original_train_step)
    def train_step_with_msprobe(self: Any, *args: Any, **kwargs: Any) -> Any:
        if os.environ.get(MSPROBE_BLOCK_BOUNDARIES_ENV) == "1":
            install_block_boundary_hooks(self)
            if os.environ.get(MSPROBE_BLOCK_GLOBAL_STEP_ENV) == "1":
                clear_block_boundary_buffers(self)
        if os.environ.get(MSPROBE_ROUTER_STATE_ENV) == "1":
            install_router_state_hooks(self)
            clear_router_state_buffers(self)
        if os.environ.get(MSPROBE_FINAL_NORM_STATE_ENV) == "1":
            install_final_norm_state_hooks(self)
            clear_final_norm_state_buffers(self)
        initial_parameters = None
        if os.environ.get(MSPROBE_PARAMETER_STATE_ENV) == "1":
            initial_parameters = snapshot_trainable_parameters(self)
        optimizer_capture = None
        final_norm_capture = None
        restore_reduce_transition = None
        clip_captures: list[dict[str, Any]] = []
        distributed_utils = None
        original_clip_grad_norm = None
        if os.environ.get(MSPROBE_OPTIMIZER_STATE_ENV) == "1":
            if initial_parameters is None:
                raise RuntimeError(
                    "optimizer-state capture requires parameter-state capture"
                )
            from torchtitan.distributed import utils as distributed_utils

            targets = optimizer_probe_targets(self)
            optimizer_capture = {
                "targets": targets,
                "hyperparameters": optimizer_group_hyperparameters(self, targets),
                "clip_called": False,
            }
            clip_captures.append(optimizer_capture)
        if os.environ.get(MSPROBE_FINAL_NORM_STATE_ENV) == "1":
            targets = final_norm_target(self)
            if targets:
                final_norm_capture = {
                    "targets": targets,
                    "initial_weight": logical_tensor(
                        next(iter(targets.values()))
                    ).clone(),
                    "clip_called": False,
                }
                clip_captures.append(final_norm_capture)
                if (
                    os.environ.get(MSPROBE_FINAL_NORM_REDUCE_TRANSITION_ENV)
                    == "1"
                ):
                    restore_reduce_transition = (
                        install_final_norm_reduce_transition_hooks(
                            self,
                            final_norm_capture,
                        )
                    )
        if clip_captures:
            from torchtitan.distributed import utils as distributed_utils

            original_clip_grad_norm = distributed_utils.clip_grad_norm_

            @wraps(original_clip_grad_norm)
            def capture_clip_grad_norm(
                parameters: Any,
                max_norm: float,
                *clip_args: Any,
                **clip_kwargs: Any,
            ) -> Any:
                import torch

                if any(capture["clip_called"] for capture in clip_captures):
                    raise RuntimeError(
                        "gradient diagnostic observed multiple clipping calls"
                    )
                for capture in clip_captures:
                    preclip = {}
                    for name, parameter in capture["targets"].items():
                        if parameter.grad is None:
                            raise RuntimeError(f"missing pre-clip gradient for {name}")
                        preclip[name] = logical_tensor(parameter.grad).clone()
                    capture["preclip_grad"] = preclip
                total_norm = original_clip_grad_norm(
                    parameters,
                    max_norm,
                    *clip_args,
                    **clip_kwargs,
                )
                logical_norm = logical_tensor(total_norm).float().reshape(1)
                coefficient = (float(max_norm) / (logical_norm + 1e-6)).clamp(max=1.0)
                for capture in clip_captures:
                    postclip = {}
                    for name, parameter in capture["targets"].items():
                        if parameter.grad is None:
                            raise RuntimeError(f"missing post-clip gradient for {name}")
                        postclip[name] = logical_tensor(parameter.grad).clone()
                    capture.update(
                        {
                            "postclip_grad": postclip,
                            "clip_stats": torch.cat((logical_norm, coefficient)),
                            "clip_called": True,
                        }
                    )
                return total_norm

            distributed_utils.clip_grad_norm_ = capture_clip_grad_norm
        # PrecisionDebugger explicitly accepts a list/tuple of model parts and
        # prefixes their module names with the local part index.
        debugger.start(model=self.model_parts)
        try:
            result = original_train_step(self, *args, **kwargs)
            if os.environ.get(MSPROBE_BLOCK_GLOBAL_STEP_ENV) == "1":
                save_global_step_boundaries(self)
            if os.environ.get(MSPROBE_BLOCK_BACKWARD_ENV) == "1":
                save_global_step_gradients(self)
            if os.environ.get(MSPROBE_ROUTER_STATE_ENV) == "1":
                save_global_step_router_state(self)
            if initial_parameters is not None:
                save_parameter_state(self, initial_parameters)
            if optimizer_capture is not None:
                save_optimizer_state(self, initial_parameters, optimizer_capture)
            save_final_norm_state(self, final_norm_capture)
            return result
        finally:
            try:
                if distributed_utils is not None:
                    distributed_utils.clip_grad_norm_ = original_clip_grad_norm
                if restore_reduce_transition is not None:
                    restore_reduce_transition()
                debugger.stop()
            finally:
                debugger.step()

    setattr(Trainer, marker, original_train_step)
    setattr(Trainer, "_glm5_msprobe_debugger", debugger)
    Trainer.train_step = train_step_with_msprobe
    return debugger


def validate_dump_directory(path: str | Path) -> Path:
    """Require a completed L0/mix dump suitable for both visualizers."""

    directory = Path(path)
    dump_files = list(directory.rglob("dump.json")) if directory.is_dir() else []
    construct_files = (
        list(directory.rglob("construct.json")) if directory.is_dir() else []
    )
    if not dump_files:
        raise RuntimeError(f"no msProbe dump.json found under {directory}")
    if not construct_files:
        raise RuntimeError(f"no msProbe construct.json found under {directory}")
    stack_files = list(directory.rglob("stack.json")) if directory.is_dir() else []
    if not stack_files:
        raise RuntimeError(f"no msProbe stack.json found under {directory}")
    return directory


def validate_debug_dump_directory(path: str | Path) -> Path:
    """Require at least one non-empty public ``PrecisionDebugger.save`` dump."""

    directory = Path(path)
    debug_files = list(directory.rglob("debug.json")) if directory.is_dir() else []
    if not debug_files:
        raise RuntimeError(f"no msProbe debug.json found under {directory}")
    if not any(
        json.loads(item.read_text(encoding="utf-8")).get("data")
        for item in debug_files
    ):
        raise RuntimeError(f"no saved msProbe debug tensors found under {directory}")
    return directory


def _dump_leaf_directories(path: Path) -> tuple[Path, ...]:
    dump_parents = {item.parent for item in path.rglob("dump.json")}
    construct_parents = {item.parent for item in path.rglob("construct.json")}
    return tuple(sorted(dump_parents & construct_parents))


def _resolve_executable(name: str, explicit: str | Path | None) -> str:
    if explicit is not None:
        candidate = Path(explicit)
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return str(candidate.resolve())
    candidate = shutil.which(name)
    if candidate is None:
        raise RuntimeError(f"required executable is not on PATH: {name}")
    return candidate


def _package_version() -> str:
    try:
        return metadata.version("mindstudio-probe")
    except metadata.PackageNotFoundError:
        return "unknown"


def tensorboard_plugins() -> tuple[str, ...]:
    """Return installed TensorBoard plugin entry-point names."""

    entry_points = metadata.entry_points()
    if hasattr(entry_points, "select"):
        selected = entry_points.select(group="tensorboard_plugins")
    else:  # pragma: no cover - Python/importlib compatibility path
        selected = entry_points.get("tensorboard_plugins", ())
    return tuple(sorted(entry_point.name for entry_point in selected))


def require_visualization_plugins() -> tuple[str, ...]:
    plugins = tensorboard_plugins()
    by_lower_name = {name.lower() for name in plugins}
    required = {"graph_ascend", "trendvis"}
    missing = sorted(required - by_lower_name)
    if missing:
        raise RuntimeError(
            "mindstudio-probe is missing TensorBoard visualization plugins: "
            f"{missing}; install a build containing tb_graph_ascend and "
            "trend_analyzer"
        )
    return plugins


def _run(command: Sequence[str]) -> None:
    subprocess.run(list(command), check=True)


def _move_single_trend_database(source: Path, destination: Path) -> None:
    matches = list(source.glob("*.trend.db"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one .trend.db from msprobe data2db in {source}, got {matches}"
        )
    shutil.move(str(matches[0]), destination)
    shutil.rmtree(source)


def build_tensorboard_assets(
    *,
    reference_dump: str | Path,
    candidate_dump: str | Path,
    output: str | Path,
    reference_parallel: MsprobeParallelSpec,
    candidate_parallel: MsprobeParallelSpec,
    force: bool = False,
    resume: bool = False,
    msprobe_executable: str | Path | None = None,
) -> Path:
    """Build an official cross-partition msProbe graph comparison and trends."""

    if force and resume:
        raise ValueError("force and resume are mutually exclusive")
    reference = validate_dump_directory(reference_dump).resolve()
    candidate = validate_dump_directory(candidate_dump).resolve()
    validate_parallel_merge_pair(reference_parallel, candidate_parallel)
    reference_leaves = _dump_leaf_directories(reference)
    candidate_leaves = _dump_leaf_directories(candidate)
    reference_steps = {path.parent for path in reference_leaves}
    candidate_steps = {path.parent for path in candidate_leaves}
    if len(reference_steps) != 1 or len(candidate_steps) != 1:
        raise ValueError(
            "the MindStudio baseline comparison requires exactly one captured step"
        )
    graph_reference = next(iter(reference_steps))
    graph_candidate = next(iter(candidate_steps))
    destination = Path(output).resolve()
    manifest_path = destination / "msprobe_tensorboard.json"
    expected = (
        destination / "reference.trend.db",
        destination / "candidate.trend.db",
    )
    if destination.exists():
        if resume:
            completed = False
            if manifest_path.is_file() and all(path.is_file() for path in expected):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                completed = (
                    manifest.get("reference_dump") == str(reference)
                    and manifest.get("candidate_dump") == str(candidate)
                    and bool(list(destination.glob("*.vis.db")))
                )
            if completed:
                return destination
            shutil.rmtree(destination)
        elif not force:
            raise FileExistsError(
                f"TensorBoard output already exists; use force or resume: {destination}"
            )
        else:
            shutil.rmtree(destination)
    destination.mkdir(parents=True)

    executable = _resolve_executable("msprobe", msprobe_executable)
    commands: list[list[str]] = []

    graph_command = [
        executable,
        "graph_visualize",
        "-tp",
        str(graph_candidate),
        "-gp",
        str(graph_reference),
        "-o",
        str(destination),
        "--rank_size",
        str(candidate_parallel.rank_size),
        str(reference_parallel.rank_size),
        "--tp",
        str(candidate_parallel.tensor_parallel),
        str(reference_parallel.tensor_parallel),
        "--pp",
        str(candidate_parallel.pipeline_parallel),
        str(reference_parallel.pipeline_parallel),
        "--vpp",
        str(candidate_parallel.virtual_pipeline_parallel),
        str(reference_parallel.virtual_pipeline_parallel),
    ]
    _run(graph_command)
    commands.append(graph_command)

    for label, dump_path in (("reference", reference), ("candidate", candidate)):
        scratch = destination / f".{label}-data2db"
        command = [
            executable,
            "data2db",
            "--data",
            str(dump_path),
            "--db",
            str(scratch),
            "--format",
            "dump",
        ]
        _run(command)
        commands.append(command)
        _move_single_trend_database(
            scratch, destination / f"{label}.trend.db"
        )

    graph_databases = sorted(destination.glob("*.vis.db"))
    if not graph_databases:
        raise RuntimeError(
            f"msprobe graph_visualize produced no .vis.db in {destination}"
        )

    payload = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "mindstudio_probe_version": _package_version(),
        "reference_dump": str(reference),
        "candidate_dump": str(candidate),
        "graph_reference_dump": str(graph_reference),
        "graph_candidate_dump": str(graph_candidate),
        "output": str(destination),
        "assets": sorted(path.name for path in destination.glob("*.db")),
        "commands": commands,
        "comparison": {
            "kind": "msprobe_parallel_merge",
            "task": "statistics",
            "level": "mix",
            "steps": [0],
            "reference_parallel": reference_parallel.__dict__,
            "candidate_parallel": candidate_parallel.__dict__,
            "framework_validation": (
                "msProbe documents graph merging for Megatron/MindSpeed-LLM; "
                "TorchTitan compatibility must be established by this run"
            ),
        },
        "tensorboard": {
            "logdir": str(destination),
            "tabs": ["GRAPH_ASCEND", "TREND ANALYZER"],
        },
        "warning": (
            "Diagnostic msProbe capture may add synchronization; do not use its "
            "metrics as formal precision or throughput evidence."
        ),
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def tensorboard_command(
    logdir: str | Path,
    *,
    port: int = 6006,
    bind_all: bool = False,
    tensorboard_executable: str | Path | None = None,
) -> list[str]:
    if not 1 <= port <= 65535:
        raise ValueError("TensorBoard port must be in [1, 65535]")
    executable = _resolve_executable("tensorboard", tensorboard_executable)
    command = [
        executable,
        "--logdir",
        str(Path(logdir).resolve()),
        "--port",
        str(port),
    ]
    if bind_all:
        command.append("--bind_all")
    return command


def serve_tensorboard(
    logdir: str | Path,
    *,
    port: int = 6006,
    bind_all: bool = False,
    tensorboard_executable: str | Path | None = None,
) -> None:
    require_visualization_plugins()
    _run(
        tensorboard_command(
            logdir,
            port=port,
            bind_all=bind_all,
            tensorboard_executable=tensorboard_executable,
        )
    )


__all__ = [
    "MSPROBE_BLOCK_BOUNDARIES_ENV",
    "MSPROBE_CONFIG_PATH_ENV",
    "MsprobeCaptureConfig",
    "MsprobeParallelSpec",
    "build_tensorboard_assets",
    "install_trainer_capture",
    "require_visualization_plugins",
    "serve_tensorboard",
    "tensorboard_command",
    "validate_parallel_merge_pair",
    "validate_dump_directory",
    "validate_debug_dump_directory",
    "write_capture_config",
]
