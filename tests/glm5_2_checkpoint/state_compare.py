# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Inspect and compare TorchTitan distributed checkpoints on CPU."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import tempfile
from typing import Any


_AUXILIARY_SCOPES = {
    "optimizer",
    "dataloader",
    "lr_scheduler",
    "train_state",
}
_REQUIRED_SCOPES = {"model", *_AUXILIARY_SCOPES}


def _scope(path: tuple[str, ...]) -> str:
    first = path[0] if path else ""
    root = first.replace("/", ".").split(".", 1)[0]
    return root if root in _AUXILIARY_SCOPES else "model"


def _walk(value: Any, path: tuple[str, ...] = ()):
    import torch

    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            yield from _walk(value[key], (*path, str(key)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk(item, (*path, str(index)))
    else:
        yield path, value


def _load_dcp(path: Path, destination: Path) -> dict[str, Any]:
    import torch
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    dcp_to_torch_save(path, destination)
    value = torch.load(destination, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError(f"converted checkpoint is not a dictionary: {path}")
    return value


def summarize_checkpoint(path: Path) -> dict[str, Any]:
    """Convert one DCP checkpoint and summarize the persisted state scopes."""

    with tempfile.TemporaryDirectory(prefix="glm5-checkpoint-inspect-") as temporary:
        state = _load_dcp(path, Path(temporary) / "checkpoint.pt")
    scope_counts: dict[str, int] = defaultdict(int)
    tensor_counts: dict[str, int] = defaultdict(int)
    for item_path, value in _walk(state):
        scope = _scope(item_path)
        scope_counts[scope] += 1
        if hasattr(value, "numel"):
            tensor_counts[scope] += 1
    present = set(scope_counts)
    return {
        "path": str(path),
        "present_scopes": sorted(present),
        "missing_required_scopes": sorted(_REQUIRED_SCOPES - present),
        "leaf_counts": dict(sorted(scope_counts.items())),
        "tensor_counts": dict(sorted(tensor_counts.items())),
    }


def compare_checkpoints(
    reference: Path,
    candidate: Path,
    *,
    exact: bool,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    """Compare complete logical DCP states after consolidating them on CPU."""

    import torch

    with tempfile.TemporaryDirectory(prefix="glm5-checkpoint-compare-") as temporary:
        root = Path(temporary)
        left = _load_dcp(reference, root / "reference.pt")
        right = _load_dcp(candidate, root / "candidate.pt")

    left_items = {path: value for path, value in _walk(left)}
    right_items = {path: value for path, value in _walk(right)}
    missing_candidate = sorted(
        "/".join(path) for path in left_items.keys() - right_items.keys()
    )
    missing_reference = sorted(
        "/".join(path) for path in right_items.keys() - left_items.keys()
    )
    scope_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "tensor_leaves": 0,
            "tensor_elements": 0,
            "exact_tensor_mismatches": 0,
            "tolerance_tensor_mismatches": 0,
            "non_tensor_mismatches": 0,
            "max_absolute": 0.0,
            "absolute_error_sum": 0.0,
            "relative_error_sum": 0.0,
        }
    )
    mismatch_examples: list[str] = []
    for path in sorted(left_items.keys() & right_items.keys()):
        left_value = left_items[path]
        right_value = right_items[path]
        scope = _scope(path)
        stats = scope_stats[scope]
        label = "/".join(path)
        if isinstance(left_value, torch.Tensor) and isinstance(
            right_value, torch.Tensor
        ):
            stats["tensor_leaves"] += 1
            if (
                left_value.shape != right_value.shape
                or left_value.dtype != right_value.dtype
            ):
                stats["exact_tensor_mismatches"] += 1
                stats["tolerance_tensor_mismatches"] += 1
                if len(mismatch_examples) < 20:
                    mismatch_examples.append(
                        f"{label}: tensor metadata differs "
                        f"({left_value.shape}, {left_value.dtype}) != "
                        f"({right_value.shape}, {right_value.dtype})"
                    )
                continue
            numel = left_value.numel()
            stats["tensor_elements"] += numel
            exact_match = torch.equal(left_value, right_value)
            tolerance_match = torch.allclose(
                left_value,
                right_value,
                atol=absolute_tolerance,
                rtol=relative_tolerance,
                equal_nan=True,
            )
            stats["exact_tensor_mismatches"] += int(not exact_match)
            stats["tolerance_tensor_mismatches"] += int(not tolerance_match)
            if numel:
                difference = (
                    left_value.to(torch.float64) - right_value.to(torch.float64)
                ).abs()
                stats["max_absolute"] = max(
                    stats["max_absolute"], float(difference.max().item())
                )
                stats["absolute_error_sum"] += float(difference.sum().item())
                denominator = left_value.to(torch.float64).abs().clamp_min(1e-30)
                relative = difference / denominator
                stats["relative_error_sum"] += float(relative.sum().item())
            selected_mismatch = not exact_match if exact else not tolerance_match
            if selected_mismatch and len(mismatch_examples) < 20:
                mismatch_examples.append(f"{label}: tensor values differ")
        elif isinstance(left_value, torch.Tensor) != isinstance(
            right_value, torch.Tensor
        ):
            stats["non_tensor_mismatches"] += 1
            if len(mismatch_examples) < 20:
                mismatch_examples.append(f"{label}: value types differ")
        else:
            try:
                equal = left_value == right_value
                if hasattr(equal, "all"):
                    equal = bool(equal.all())
                else:
                    equal = bool(equal)
            except (TypeError, ValueError):
                equal = False
            stats["non_tensor_mismatches"] += int(not equal)
            if not equal and len(mismatch_examples) < 20:
                mismatch_examples.append(f"{label}: non-tensor values differ")

    normalized: dict[str, dict[str, Any]] = {}
    for scope, stats in sorted(scope_stats.items()):
        elements = stats.pop("tensor_elements")
        absolute_sum = stats.pop("absolute_error_sum")
        relative_sum = stats.pop("relative_error_sum")
        normalized[scope] = {
            **stats,
            "tensor_elements": elements,
            "mean_absolute": absolute_sum / elements if elements else 0.0,
            "mean_absolute_relative": relative_sum / elements if elements else 0.0,
        }

    present = set(normalized)
    missing_scopes = sorted(_REQUIRED_SCOPES - present)
    structural_pass = (
        not missing_candidate and not missing_reference and not missing_scopes
    )
    state_pass = structural_pass
    for scope, stats in normalized.items():
        numeric_mismatches = (
            stats["exact_tensor_mismatches"]
            if exact
            else stats["tolerance_tensor_mismatches"]
        )
        # Step, dataloader, and scheduler metadata must be exact even when
        # numerical tensors are evaluated with BF16 tolerances.
        if scope in {"dataloader", "lr_scheduler", "train_state"}:
            numeric_mismatches = stats["exact_tensor_mismatches"]
        state_pass &= numeric_mismatches == 0 and stats["non_tensor_mismatches"] == 0
    return {
        "passed": bool(state_pass),
        "mode": "exact" if exact else "tolerance",
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "missing_from_candidate": missing_candidate[:20],
        "missing_from_reference": missing_reference[:20],
        "missing_required_scopes": missing_scopes,
        "scopes": normalized,
        "mismatch_examples": mismatch_examples,
    }
