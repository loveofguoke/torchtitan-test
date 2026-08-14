"""Runtime hooks injected into TorchTitan for a single DDP training step."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


_INSTALLED = False
_LAYER_RE = re.compile(r"(?:^|\.)layers\.\d+$")


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    local = value.detach()
    to_local = getattr(local, "to_local", None)
    if callable(to_local):
        local = to_local()
    return local


def _canonical_cpu(value: torch.Tensor) -> torch.Tensor:
    local = _local_tensor(value)
    if local.is_floating_point() or local.is_complex():
        local = local.float()
    return local.to("cpu").contiguous().clone()


def _summary(value: torch.Tensor) -> dict[str, Any]:
    cpu = _canonical_cpu(value)
    raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    result: dict[str, Any] = {
        "shape": list(cpu.shape),
        "source_dtype": str(value.dtype),
        "stored_dtype": str(cpu.dtype),
        "numel": cpu.numel(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if cpu.numel() == 0:
        return result
    if cpu.is_floating_point():
        numeric = cpu.double()
        finite = torch.isfinite(numeric)
        result.update(
            {
                "finite": bool(finite.all()),
                "min": float(numeric.min()),
                "max": float(numeric.max()),
                "mean": float(numeric.mean()),
                "norm": float(torch.linalg.vector_norm(numeric)),
                "sample": [float(item) for item in numeric.flatten()[:8]],
            }
        )
    else:
        result["sample"] = [int(item) for item in cpu.flatten()[:8]]
    return result


def _tensor_leaves(value: Any, prefix: str = "") -> Iterator[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield prefix or "tensor", value
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _tensor_leaves(value[key], child)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            child = f"{prefix}.{index}" if prefix else str(index)
            yield from _tensor_leaves(item, child)


def _clean_name(name: str) -> str:
    for fragment in ("._orig_mod", ".module"):
        name = name.replace(fragment, "")
    return name.strip(".") or "model"


def _trace_module(name: str) -> bool:
    clean = _clean_name(name)
    return (
        clean == "model"
        or clean.endswith(("tok_embeddings", "norm", "lm_head"))
        or bool(_LAYER_RE.search(clean))
        or clean.endswith((".attention", ".attention.indexer", ".moe.router"))
    )


def _trace_parameter(name: str) -> bool:
    clean = _clean_name(name)
    return (
        "tok_embeddings" in clean
        or "lm_head" in clean
        or ".layers.0." in f".{clean}."
        or "router" in clean
        or clean.endswith("norm.weight")
    )


class _RankCapture:
    def __init__(self, trainer: Any):
        self.trainer = trainer
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.output = Path(os.environ["TORCHTITAN_DIAGNOSTIC_DIR"]).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.tensors: dict[str, torch.Tensor] = {}
        self.records: list[dict[str, Any]] = []
        self.seen: set[str] = set()
        self.handles: list[Any] = []
        self.parameter_before: dict[str, torch.Tensor] = {}

    def store(self, key: str, value: torch.Tensor, *, kind: str) -> None:
        key = key.replace("/", ".")
        if key in self.seen:
            return
        self.seen.add(key)
        cpu = _canonical_cpu(value)
        self.tensors[key] = cpu
        self.records.append({"key": key, "kind": kind, **_summary(value)})

    def store_tree(self, prefix: str, value: Any, *, kind: str) -> None:
        for suffix, tensor in _tensor_leaves(value):
            self.store(f"{prefix}.{suffix}", tensor, kind=kind)

    def register_forward_hooks(self) -> None:
        for part_index, model in enumerate(self.trainer.model_parts):
            for raw_name, module in model.named_modules():
                clean = _clean_name(raw_name)
                semantic = f"part{part_index}.{clean}"
                if not _trace_module(clean):
                    continue

                def capture_output(
                    _module: torch.nn.Module,
                    _inputs: tuple[Any, ...],
                    output: Any,
                    *,
                    module_name: str = semantic,
                ) -> None:
                    if self.trainer.step == 1:
                        self.store_tree(
                            f"forward.{module_name}", output, kind="activation"
                        )

                self.handles.append(module.register_forward_hook(capture_output))

    def capture_parameters_before(self) -> None:
        if self.rank != 0:
            return
        for part_index, model in enumerate(self.trainer.model_parts):
            for raw_name, parameter in model.named_parameters():
                name = f"part{part_index}.{_clean_name(raw_name)}"
                if not _trace_parameter(name):
                    continue
                before = _canonical_cpu(parameter)
                self.parameter_before[name] = before
                self.store(f"parameter.before.{name}", parameter, kind="parameter")

    def capture_parameters_after(self) -> None:
        if self.rank != 0:
            return
        selected: list[tuple[str, torch.nn.Parameter]] = []
        for part_index, model in enumerate(self.trainer.model_parts):
            for raw_name, parameter in model.named_parameters():
                name = f"part{part_index}.{_clean_name(raw_name)}"
                if name not in self.parameter_before:
                    continue
                selected.append((name, parameter))
        # The optimizer has already run, but gradients are retained. Record them
        # first so the report follows the logical train-step dataflow.
        for name, parameter in selected:
            if parameter.grad is not None:
                self.store(
                    f"gradient.post_reduce_clip.{name}",
                    parameter.grad,
                    kind="gradient",
                )
        for name, parameter in selected:
            after = _canonical_cpu(parameter)
            self.store(f"parameter.after.{name}", after, kind="parameter")
            self.store(
                f"parameter.delta.{name}",
                after - self.parameter_before[name],
                kind="parameter_delta",
            )

    def finish(self, *, status: str, failure: str = "") -> None:
        for handle in self.handles:
            handle.remove()
        tensor_path = self.output / f"rank-{self.rank:03d}.safetensors"
        json_path = self.output / f"rank-{self.rank:03d}.json"
        if self.tensors:
            save_file(self.tensors, tensor_path)
        payload = {
            "schema": "torchtitan.ddp-step-diagnostic",
            "schema_version": 1,
            "status": status,
            "failure": failure,
            "rank": self.rank,
            "world_size": self.world_size,
            "backend": os.environ.get("TORCHTITAN_DEVICE", "unknown"),
            "step": self.trainer.step,
            "records": self.records,
            "tensor_file": tensor_path.name,
        }
        temporary = json_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, json_path)


class _CapturingIterator:
    def __init__(self, delegate: Iterator[Any], capture: _RankCapture):
        self.delegate = delegate
        self.capture = capture
        self.count = 0

    def __iter__(self) -> "_CapturingIterator":
        return self

    def __next__(self) -> Any:
        item = next(self.delegate)
        if self.count == 0:
            input_dict, labels = item
            self.capture.store_tree("input", input_dict, kind="input")
            self.capture.store("labels", labels, kind="input")
        self.count += 1
        return item


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from torchtitan.trainer import Trainer

    original_init = Trainer.__init__
    original_train_step = Trainer.train_step
    original_forward_backward = Trainer.forward_backward_step

    def init_with_diagnostics(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        capture = _RankCapture(self)
        capture.register_forward_hooks()
        self._ddp_step_diagnostic = capture

    def forward_backward_with_diagnostics(
        self: Any, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        loss = original_forward_backward(self, *args, **kwargs)
        if self.step == 1:
            self._ddp_step_diagnostic.store("loss.local", loss, kind="local_loss")
        return loss

    def train_step_with_diagnostics(self: Any, data_iterator: Iterator[Any]) -> Any:
        capture: _RankCapture = self._ddp_step_diagnostic
        if self.step != 1:
            return original_train_step(self, data_iterator)
        capture.capture_parameters_before()
        try:
            result = original_train_step(
                self, _CapturingIterator(data_iterator, capture)
            )
            capture.capture_parameters_after()
            capture.finish(status="success")
            return result
        except BaseException as error:
            capture.finish(
                status="failed", failure=f"{type(error).__name__}: {error}"
            )
            raise

    Trainer.__init__ = init_with_diagnostics
    Trainer.forward_backward_step = forward_backward_with_diagnostics
    Trainer.train_step = train_step_with_diagnostics
