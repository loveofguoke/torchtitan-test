# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Shared GLM 5.2 experiment primitives with no experiment dependencies."""

from .cli import archive_previous_output, replace_topology, run_all_topologies
from .device import AcceleratorSelection, resolve_accelerator, resolve_device_type
from .execution import ExecutionPlan, TrainingFeature, compose_execution
from .naming import config_digest, config_name, slug
from .topology import (
    DISTRIBUTED_TOPOLOGY_NAMES,
    ParallelTopology,
    training_command_args,
    select_topologies,
    standard_topologies,
)

__all__ = [
    "AcceleratorSelection",
    "DISTRIBUTED_TOPOLOGY_NAMES",
    "ExecutionPlan",
    "ParallelTopology",
    "training_command_args",
    "TrainingFeature",
    "archive_previous_output",
    "compose_execution",
    "config_digest",
    "config_name",
    "replace_topology",
    "run_all_topologies",
    "resolve_accelerator",
    "resolve_device_type",
    "select_topologies",
    "slug",
    "standard_topologies",
]
