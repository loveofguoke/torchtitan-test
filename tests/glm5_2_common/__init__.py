# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Shared GLM 5.2 experiment primitives with no experiment dependencies."""

from .cli import run_all_topologies
from .device import AcceleratorSelection, resolve_accelerator, resolve_device_type
from .execution import ExecutionPlan, TrainingFeature, compose_execution
from .topology import (
    DISTRIBUTED_TOPOLOGY_NAMES,
    ParallelTopology,
    select_topologies,
    standard_topologies,
)

__all__ = [
    "AcceleratorSelection",
    "DISTRIBUTED_TOPOLOGY_NAMES",
    "ExecutionPlan",
    "ParallelTopology",
    "TrainingFeature",
    "compose_execution",
    "run_all_topologies",
    "resolve_accelerator",
    "resolve_device_type",
    "select_topologies",
    "standard_topologies",
]
