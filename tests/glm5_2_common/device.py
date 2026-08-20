# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Accelerator selection shared by all experiment entry points."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Mapping


DeviceType = Literal["cuda", "npu"]


@dataclass(frozen=True)
class AcceleratorSelection:
    device_type: DeviceType
    visibility_variable: str
    visible_devices: str

    @property
    def torchtitan_device(self) -> str:
        return "npu" if self.device_type == "npu" else "gpu"

    @property
    def count(self) -> int:
        return len(
            [value for value in self.visible_devices.split(",") if value.strip()]
        )


def resolve_device_type(
    requested: str | None,
    *,
    environment: Mapping[str, str] | None = None,
) -> DeviceType:
    """Resolve a device type without requiring a visibility value."""

    if requested not in {None, "auto"}:
        if requested not in {"cuda", "npu"}:
            raise ValueError(f"unsupported accelerator: {requested!r}")
        return requested
    values = os.environ if environment is None else environment
    cuda = values.get("CUDA_VISIBLE_DEVICES", "").strip()
    npu = values.get("ASCEND_RT_VISIBLE_DEVICES", "").strip()
    if bool(cuda) != bool(npu):
        return "cuda" if cuda else "npu"
    try:
        import torch

        if getattr(torch, "npu", None) is not None and torch.npu.is_available():
            return "npu"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    raise ValueError(
        "export exactly one visibility variable or select cuda|npu"
    )


def resolve_accelerator(
    requested: str | None,
    *,
    environment: Mapping[str, str] | None = None,
) -> AcceleratorSelection:
    """Resolve CUDA/NPU from one explicit request or exported visibility."""

    values = os.environ if environment is None else environment
    cuda = values.get("CUDA_VISIBLE_DEVICES", "").strip()
    npu = values.get("ASCEND_RT_VISIBLE_DEVICES", "").strip()
    requested = resolve_device_type(requested, environment=values)
    if requested == "cuda":
        if not cuda:
            raise ValueError("CUDA_VISIBLE_DEVICES must be exported")
        return AcceleratorSelection("cuda", "CUDA_VISIBLE_DEVICES", cuda)
    if requested == "npu":
        if not npu:
            raise ValueError("ASCEND_RT_VISIBLE_DEVICES must be exported")
        return AcceleratorSelection("npu", "ASCEND_RT_VISIBLE_DEVICES", npu)
    raise AssertionError(f"unreachable accelerator: {requested!r}")
