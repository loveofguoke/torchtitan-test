# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Checkpoint fault injection installed only inside benchmark workers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import time
from typing import Any

import torch.distributed as dist


FAULT_MODE_ENV = "GLM5_CHECKPOINT_FAULT_MODE"
FAULT_STEP_ENV = "GLM5_CHECKPOINT_FAULT_STEP"
FAULT_RANK_ENV = "GLM5_CHECKPOINT_FAULT_RANK"
READY_PATH_ENV = "GLM5_CHECKPOINT_READY_PATH"
EVENTS_DIR_ENV = "GLM5_CHECKPOINT_EVENTS_DIR"
BOUNDARY_STEP_ENV = "GLM5_CHECKPOINT_BOUNDARY_STEP"

EXTERNAL_FAILURE_MODES = {"sigint", "sigterm", "sigkill", "incomplete"}
RANK_FAILURE_MODES = {"rank-error", "rank-sigkill"}
FAILURE_MODES = {"graceful", *EXTERNAL_FAILURE_MODES, *RANK_FAILURE_MODES}


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _record_event(event: str, *, step: int | None, **values: Any) -> None:
    directory_value = os.environ.get(EVENTS_DIR_ENV)
    if not directory_value:
        return
    directory = Path(directory_value)
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "event": event,
        "rank": _rank(),
        "step": step,
        **values,
    }
    with (directory / f"rank-{_rank()}.jsonl").open(
        "a", encoding="utf-8", buffering=1
    ) as stream:
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
        stream.write("\n")


def _publish_ready(mode: str, step: int) -> None:
    ready_value = os.environ.get(READY_PATH_ENV)
    if not ready_value:
        raise RuntimeError(f"{READY_PATH_ENV} is required for {mode}")
    ready_path = Path(ready_value)
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ready_path.with_suffix(f".rank-{_rank()}.tmp")
    temporary.write_text(
        json.dumps({"mode": mode, "rank": _rank(), "step": step}) + "\n",
        encoding="utf-8",
    )
    temporary.replace(ready_path)


def _wait_for_launcher_failure() -> None:
    while True:
        time.sleep(60)


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _missing_optimizer_state(checkpointer: Any) -> list[str]:
    container = checkpointer.states.get("optimizer")
    optimizers = getattr(container, "optimizers", ())
    missing: list[str] = []
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            names = group.get("param_names", ())
            for name, parameter in zip(names, group["params"], strict=True):
                if parameter.requires_grad and not optimizer.state.get(parameter):
                    missing.append(str(name))
    return sorted(missing)


def _fingerprint_leaf(value: Any) -> dict[str, Any]:
    """Return a stable, local fingerprint for one checkpoint-state leaf."""

    import torch

    if isinstance(value, torch.Tensor):
        local = value.to_local() if hasattr(value, "to_local") else value
        local = local.detach().cpu().contiguous()
        payload = local.reshape(-1).view(torch.uint8).numpy()
        return {
            "kind": "tensor",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "local_shape": list(local.shape),
            "numel": int(local.numel()),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "size": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
    ).encode("utf-8")
    return {
        "kind": "value",
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "value": (
            value
            if value is None or isinstance(value, (bool, int, float, str))
            else repr(value)
        ),
    }


def _walk_fingerprint_leaves(
    value: Any,
    path: tuple[str, ...] = (),
):
    import torch

    if isinstance(value, torch.Tensor) or isinstance(value, bytes):
        yield path, _fingerprint_leaf(value)
    elif isinstance(value, dict):
        if not value:
            yield path, _fingerprint_leaf({})
        for key in sorted(value, key=str):
            yield from _walk_fingerprint_leaves(value[key], (*path, str(key)))
    elif isinstance(value, (list, tuple)):
        if not value:
            yield path, _fingerprint_leaf([])
        for index, item in enumerate(value):
            yield from _walk_fingerprint_leaves(item, (*path, str(index)))
    else:
        yield path, _fingerprint_leaf(value)


def _state_fingerprint(value: Any) -> dict[str, Any]:
    state = value.state_dict() if hasattr(value, "state_dict") else value
    leaves = {
        "/".join(path) or "$": leaf
        for path, leaf in _walk_fingerprint_leaves(state)
    }
    digest = hashlib.sha256()
    for path, leaf in sorted(leaves.items()):
        digest.update(path.encode("utf-8"))
        digest.update(
            json.dumps(leaf, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
    return {
        "sha256": digest.hexdigest(),
        "leaf_count": len(leaves),
        "tensor_leaf_count": sum(
            leaf["kind"] == "tensor" for leaf in leaves.values()
        ),
        "tensor_elements": sum(
            int(leaf.get("numel", 0)) for leaf in leaves.values()
        ),
        "leaves": leaves,
    }


def _checkpoint_state_fingerprints(checkpointer: Any) -> dict[str, Any]:
    """Fingerprint the live local shard of every registered checkpoint scope."""

    return {
        str(scope): _state_fingerprint(value)
        for scope, value in sorted(checkpointer.states.items())
        if value is not None
    }


def _record_state_fingerprint(
    checkpointer: Any,
    *,
    phase: str,
    step: int | None,
) -> None:
    _record_event(
        "state-fingerprint",
        step=step,
        phase=phase,
        scopes=_checkpoint_state_fingerprints(checkpointer),
    )


def install_checkpoint_fault_injection() -> None:
    """Instrument checkpoint load/save and inject the requested process failure."""

    mode = os.environ.get(FAULT_MODE_ENV)
    if mode is not None and mode not in FAILURE_MODES:
        raise ValueError(f"unsupported checkpoint fault mode: {mode!r}")
    fault_step = int(os.environ[FAULT_STEP_ENV]) if mode is not None else -1
    fault_rank = int(os.environ.get(FAULT_RANK_ENV, "0"))
    boundary_value = os.environ.get(BOUNDARY_STEP_ENV)
    boundary_step = int(boundary_value) if boundary_value else None

    from torchtitan.components.checkpointer import CheckpointManager

    original_load = CheckpointManager.load
    original_save = CheckpointManager.save

    def load_with_evidence(self: CheckpointManager, step: int = -1) -> bool:
        loaded = original_load(self, step)
        train_state = self.states.get("train_state")
        restored_step = getattr(train_state, "step", None)
        restored_step = int(restored_step) if restored_step is not None else None
        _record_event(
            "load",
            step=restored_step,
            loaded=bool(loaded),
        )
        if loaded and (boundary_step is None or restored_step == boundary_step):
            _record_state_fingerprint(self, phase="load", step=restored_step)
        return loaded

    def save_with_fault(
        self: CheckpointManager,
        curr_step: int,
        last_step: bool = False,
    ) -> bool:
        if mode is None:
            saved = original_save(self, curr_step, last_step)
            _record_event("save-return", step=curr_step, saved=bool(saved))
            if saved:
                _record_event(
                    "optimizer-state-audit",
                    step=curr_step,
                    missing=_missing_optimizer_state(self),
                )
                if boundary_step is None or curr_step == boundary_step:
                    _record_state_fingerprint(self, phase="save", step=curr_step)
            return saved
        if mode == "incomplete" and curr_step == fault_step:
            # Model a job that dies after a new distributed-checkpoint directory
            # becomes visible but before DCP publishes its commit metadata. The
            # loader must ignore this higher, incomplete step on restart.
            self.maybe_wait_for_staging()
            self.maybe_wait_for_saving()
            partial = Path(self._create_checkpoint_id(curr_step))
            partial.mkdir(parents=True, exist_ok=True)
            (partial / f".incomplete-rank-{_rank()}").write_text(
                "fault injection before DCP metadata commit\n", encoding="utf-8"
            )
            _barrier()
            _record_event("fault-ready", step=curr_step, mode=mode, saved=False)
            if _rank() == fault_rank:
                _publish_ready(mode, curr_step)
            _wait_for_launcher_failure()

        saved = original_save(self, curr_step, last_step)
        _record_event("save-return", step=curr_step, saved=bool(saved))
        if saved:
            _record_event(
                "optimizer-state-audit",
                step=curr_step,
                missing=_missing_optimizer_state(self),
            )
            if boundary_step is None or curr_step == boundary_step:
                _record_state_fingerprint(self, phase="save", step=curr_step)
        if curr_step != fault_step or mode == "graceful":
            return saved

        self.maybe_wait_for_staging()
        self.maybe_wait_for_saving()
        _barrier()
        _record_event("fault-ready", step=curr_step, mode=mode, saved=bool(saved))
        if _rank() == fault_rank:
            _publish_ready(mode, curr_step)
            if mode == "rank-error":
                raise RuntimeError(
                    f"injected checkpoint benchmark failure on rank {fault_rank}"
                )
            if mode == "rank-sigkill":
                os.kill(os.getpid(), signal.SIGKILL)
        _wait_for_launcher_failure()

    CheckpointManager.load = load_with_evidence
    CheckpointManager.save = save_with_fault


def wait_for_completed_checkpoint(trainer: Any, stop_after: int) -> bool:
    """Finish asynchronous checkpoint work before a controlled normal exit."""

    if trainer.step < stop_after:
        return True
    trainer.checkpointer.maybe_wait_for_staging()
    trainer.checkpointer.maybe_wait_for_saving()
    _record_event("graceful-stop", step=trainer.step)
    return False
