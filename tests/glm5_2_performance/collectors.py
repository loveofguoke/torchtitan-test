# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Official Ascend performance collector contracts.

The registry intentionally separates tool discovery from execution. Only
collectors whose whole-training launch contract has been verified are allowed
to wrap a TorchTitan job. Other MindStudio collectors remain visible so the
CLI can explain their correct scope instead of inventing an unsupported
command line.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import os
from pathlib import Path
import shutil
from typing import Any


class PerformanceCollector(StrEnum):
    """MindStudio collection entry points relevant to AI Infra work."""

    MSPROF = "msprof"
    TORCH_NPU_PROFILER = "torch_npu_profiler"
    MSOPPROF = "msopprof"
    MSMEMSCOPE = "msmemscope"
    SERVICE_PROFILER = "service_profiler"


@dataclass(frozen=True)
class CollectorCapability:
    """Audited execution boundary for one collector."""

    collector: PerformanceCollector
    executable_for_training: bool
    layer: str
    intended_scope: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_CAPABILITIES = {
    PerformanceCollector.MSPROF: CollectorCapability(
        collector=PerformanceCollector.MSPROF,
        executable_for_training=True,
        layer="CANN and NPU",
        intended_scope="whole TorchTitan training launch",
        reason=(
            "The official msProf launch contract accepts an application after "
            "its collection options. It is the default generic collector."
        ),
    ),
    PerformanceCollector.TORCH_NPU_PROFILER: CollectorCapability(
        collector=PerformanceCollector.TORCH_NPU_PROFILER,
        executable_for_training=True,
        layer="PyTorch, CANN, and NPU",
        intended_scope="scheduled in-process TorchTitan training window",
        reason=(
            "The source-installed Turbo adapter maps TorchTitan's profiler "
            "lifecycle to the public torch_npu.profiler API."
        ),
    ),
    PerformanceCollector.MSOPPROF: CollectorCapability(
        collector=PerformanceCollector.MSOPPROF,
        executable_for_training=False,
        layer="single operator and kernel",
        intended_scope="operator on-board or simulator investigation",
        reason=(
            "msOpProf is a drill-down tool for an isolated operator, not a "
            "verified whole-training launcher for TorchTitan."
        ),
    ),
    PerformanceCollector.MSMEMSCOPE: CollectorCapability(
        collector=PerformanceCollector.MSMEMSCOPE,
        executable_for_training=False,
        layer="memory",
        intended_scope="memory-specific investigation",
        reason=(
            "No CANN-version-pinned TorchTitan launch contract has been "
            "validated for msMemScope in this repository."
        ),
    ),
    PerformanceCollector.SERVICE_PROFILER: CollectorCapability(
        collector=PerformanceCollector.SERVICE_PROFILER,
        executable_for_training=False,
        layer="online service",
        intended_scope="service and inference request profiling",
        reason=(
            "Service profiling targets online request pipelines rather than "
            "the current offline TorchTitan training process."
        ),
    ),
}


def collector_capability(
    value: PerformanceCollector | str,
) -> CollectorCapability:
    """Return the audited capability record for ``value``."""

    collector = PerformanceCollector(value)
    return _CAPABILITIES[collector]


def require_training_collector(
    value: PerformanceCollector | str,
) -> PerformanceCollector:
    """Reject a registered collector without a verified training contract."""

    capability = collector_capability(value)
    if not capability.executable_for_training:
        raise NotImplementedError(
            f"collector {capability.collector.value!r} is registered for "
            f"{capability.intended_scope}, but its whole-training TorchTitan "
            f"entry is not implemented: {capability.reason} Server validation "
            "and an explicit launch contract are required before enabling it."
        )
    return capability.collector


def collector_inventory() -> list[dict[str, Any]]:
    """Return a stable JSON-serializable collector inventory."""

    return [
        _CAPABILITIES[collector].as_dict()
        for collector in PerformanceCollector
    ]


def resolve_msprof_executable() -> str:
    """Resolve msProf exactly like the MindStudio doctor does."""

    override = os.environ.get("TORCHTITAN_MSPROF")
    if override:
        executable = Path(override).expanduser().resolve()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise FileNotFoundError(
                "TORCHTITAN_MSPROF must name an executable file: "
                f"{executable}"
            )
        return str(executable)
    executable = shutil.which("msprof")
    if executable is None:
        raise FileNotFoundError(
            "msprof is not installed or is not on PATH; source the CANN "
            "Toolkit environment or set TORCHTITAN_MSPROF"
        )
    return executable
