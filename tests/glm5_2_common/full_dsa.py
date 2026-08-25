# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Shared Full DSA model and optional-operator experiment selection."""

from __future__ import annotations

from collections.abc import Sequence


FULL_DSA_CONFIG = "glm5_full_dsa_debugmodel"
REFERENCE_KERNEL = "reference"
TRITON_KERNEL = "triton"
NPU_SPARSE_KERNEL = "npu-sparse"
FULL_DSA_KERNELS = (
    REFERENCE_KERNEL,
    "auto",
    TRITON_KERNEL,
    NPU_SPARSE_KERNEL,
)

GPU_TRITON_OVERRIDES = (
    "torchtitan.models.glm5.ops.triton.triton_dsa_indexer",
    "torchtitan.models.glm5.ops.triton.triton_sparse_mla",
)
NPU_TRITON_OVERRIDES = (
    "torchtitanturbo.models.glm5.ops.triton.npu_triton_dsa_indexer",
    "torchtitanturbo.models.glm5.ops.triton.npu_triton_sparse_mla",
)
NPU_SPARSE_OVERRIDES = (
    "torchtitanturbo.models.glm5.ops.sparse_mla.npu_sparse_mla",
)


def normalize_device_type(device_type: str) -> str:
    if device_type == "gpu":
        return "cuda"
    if device_type not in {"cuda", "npu"}:
        raise ValueError(f"unsupported Full DSA device: {device_type!r}")
    return device_type


def resolve_full_dsa_config(
    *,
    enabled: bool,
    requested_config: str | None,
    default_config: str = "glm5_debugmodel",
) -> str:
    """Resolve the model flavor without silently replacing custom configs."""

    if not enabled:
        return requested_config or default_config
    if requested_config not in {None, FULL_DSA_CONFIG}:
        raise ValueError(
            "--full-dsa requires --config glm5_full_dsa_debugmodel; "
            f"got {requested_config!r}"
        )
    return FULL_DSA_CONFIG


def full_dsa_override_targets(
    *,
    device_type: str,
    kernel: str,
) -> tuple[str, ...]:
    """Resolve a device-specific implementation for one Full DSA endpoint."""

    device_type = normalize_device_type(device_type)
    if kernel not in FULL_DSA_KERNELS:
        raise ValueError(f"unknown Full DSA kernel: {kernel!r}")
    if kernel == "auto":
        kernel = TRITON_KERNEL
    if kernel == REFERENCE_KERNEL:
        return ()
    if kernel == TRITON_KERNEL:
        return (
            GPU_TRITON_OVERRIDES
            if device_type == "cuda"
            else NPU_TRITON_OVERRIDES
        )
    if kernel == NPU_SPARSE_KERNEL:
        if device_type != "npu":
            raise ValueError("the NPU SparseMLA kernel requires an NPU endpoint")
        return NPU_SPARSE_OVERRIDES
    raise AssertionError(f"unreachable Full DSA kernel: {kernel}")


def merge_override_imports(
    current: str | None,
    targets: Sequence[str],
) -> str | None:
    """Merge comma-separated override targets while preserving their order."""

    values = [] if current is None else current.split(",")
    values.extend(targets)
    deduplicated = tuple(
        dict.fromkeys(value.strip() for value in values if value.strip())
    )
    return ",".join(deduplicated) if deduplicated else None


def merge_override_argument(
    arguments: Sequence[str],
    targets: Sequence[str],
) -> tuple[str, ...]:
    """Add targets to one endpoint's ``--override.imports`` argument."""

    prefix = "--override.imports="
    result = []
    current = None
    for argument in arguments:
        if argument.startswith(prefix):
            if current is not None:
                raise ValueError("multiple --override.imports arguments are ambiguous")
            current = argument.removeprefix(prefix)
        else:
            result.append(argument)
    merged = merge_override_imports(current, targets)
    if merged is not None:
        result.append(prefix + merged)
    return tuple(result)


__all__ = [
    "FULL_DSA_CONFIG",
    "FULL_DSA_KERNELS",
    "GPU_TRITON_OVERRIDES",
    "NPU_SPARSE_KERNEL",
    "NPU_SPARSE_OVERRIDES",
    "NPU_TRITON_OVERRIDES",
    "REFERENCE_KERNEL",
    "TRITON_KERNEL",
    "full_dsa_override_targets",
    "merge_override_argument",
    "merge_override_imports",
    "normalize_device_type",
    "resolve_full_dsa_config",
]
