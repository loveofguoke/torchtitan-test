# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Canonical TorchTitan topology definitions shared by every experiment."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not result:
        raise ValueError(f"value does not produce a usable name: {value!r}")
    return result


@dataclass(frozen=True)
class ParallelTopology:
    """One authoritative set of TorchTitan parallel degrees."""

    name: str
    world_size: int
    data_parallel_replicate_degree: int = 1
    data_parallel_shard_degree: int = 1
    context_parallel_degree: int = 1
    tensor_parallel_degree: int = 1
    pipeline_parallel_degree: int = 1
    expert_parallel_degree: int = 1
    pipeline_parallel_schedule: str = "1F1B"
    pipeline_parallel_microbatch_size: int = 1
    extra_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        degrees = (
            self.world_size,
            self.data_parallel_replicate_degree,
            self.data_parallel_shard_degree,
            self.context_parallel_degree,
            self.tensor_parallel_degree,
            self.pipeline_parallel_degree,
            self.expert_parallel_degree,
            self.pipeline_parallel_microbatch_size,
        )
        if any(degree < 1 for degree in degrees):
            raise ValueError("parallel degrees and world size must be positive")
        dense_world_size = (
            self.data_parallel_replicate_degree
            * self.data_parallel_shard_degree
            * self.context_parallel_degree
            * self.tensor_parallel_degree
            * self.pipeline_parallel_degree
        )
        if dense_world_size != self.world_size:
            raise ValueError(
                f"topology {self.name!r} covers {dense_world_size} dense ranks, "
                f"not world_size={self.world_size}"
            )
        sparse_width = (
            self.data_parallel_shard_degree
            * self.context_parallel_degree
            * self.tensor_parallel_degree
        )
        if sparse_width % self.expert_parallel_degree:
            raise ValueError(
                "dp_shard * cp * tp must be divisible by expert_parallel_degree: "
                f"{sparse_width} % {self.expert_parallel_degree} != 0"
            )

    @property
    def slug(self) -> str:
        return _slug(self.name)

    @property
    def data_parallel_degree(self) -> int:
        return (
            self.data_parallel_replicate_degree
            * self.data_parallel_shard_degree
        )

    # Compact aliases preserve the old performance API while every experiment
    # now receives the same topology object.
    @property
    def dp_replicate(self) -> int:
        return self.data_parallel_replicate_degree

    @property
    def dp_shard(self) -> int:
        return self.data_parallel_shard_degree

    @property
    def tp(self) -> int:
        return self.tensor_parallel_degree

    @property
    def cp(self) -> int:
        return self.context_parallel_degree

    @property
    def pp(self) -> int:
        return self.pipeline_parallel_degree

    @property
    def ep(self) -> int:
        return self.expert_parallel_degree

    @property
    def pp_schedule(self) -> str:
        return self.pipeline_parallel_schedule

    @property
    def pp_microbatch_size(self) -> int:
        return self.pipeline_parallel_microbatch_size

    def command_args(self) -> list[str]:
        args = [
            "--parallelism.spmd_backend=partial_dtensor",
            "--parallelism.data_parallel_replicate_degree="
            f"{self.data_parallel_replicate_degree}",
            "--parallelism.data_parallel_shard_degree="
            f"{self.data_parallel_shard_degree}",
            f"--parallelism.context_parallel_degree={self.context_parallel_degree}",
            f"--parallelism.tensor_parallel_degree={self.tensor_parallel_degree}",
            f"--parallelism.pipeline_parallel_degree={self.pipeline_parallel_degree}",
            f"--parallelism.expert_parallel_degree={self.expert_parallel_degree}",
        ]
        if self.tensor_parallel_degree > 1:
            args.append("--parallelism.no-enable-sequence-parallel")
        if self.pipeline_parallel_degree > 1:
            args.append(
                "--parallelism.pipeline_parallel_schedule="
                f"{self.pipeline_parallel_schedule}"
            )
        args.extend(self.extra_args)
        return args


def training_command_args(
    *,
    local_batch_size: int,
    global_batch_size: int,
    sequence_length: int,
    topology: ParallelTopology,
) -> list[str]:
    """Translate sample-batch settings to TorchTitan's token-budget CLI."""

    microbatch_size = (
        topology.pipeline_parallel_microbatch_size
        if topology.pipeline_parallel_degree > 1
        else local_batch_size
    )
    if local_batch_size % microbatch_size:
        raise ValueError(
            "local_batch_size must be divisible by the pipeline microbatch size"
        )
    num_pp_microbatches = local_batch_size // microbatch_size
    args = [
        "--training.num_tokens_per_microbatch_per_dp_rank="
        f"{microbatch_size * sequence_length}",
        "--training.num_tokens_per_train_step="
        f"{global_batch_size * sequence_length}",
        f"--training.max_context_length={sequence_length}",
    ]
    if topology.pipeline_parallel_degree > 1:
        args.append(
            f"--parallelism.num_pp_microbatches={num_pp_microbatches}"
        )
    return args


def standard_topologies() -> dict[str, ParallelTopology]:
    """Return the single source of truth for supported GLM topologies."""

    return {
        "single": ParallelTopology("single", 1),
        "ddp2": ParallelTopology(
            "ddp2", 2, data_parallel_replicate_degree=2
        ),
        "ddp8": ParallelTopology(
            "ddp8", 8, data_parallel_replicate_degree=8
        ),
        "fsdp8": ParallelTopology("fsdp8", 8, data_parallel_shard_degree=8),
        "tp8": ParallelTopology("tp8", 8, tensor_parallel_degree=8),
        "cp8": ParallelTopology("cp8", 8, context_parallel_degree=8),
        "pp8": ParallelTopology("pp8", 8, pipeline_parallel_degree=8),
        "ep8": ParallelTopology(
            "ep8",
            8,
            data_parallel_shard_degree=8,
            expert_parallel_degree=8,
        ),
        "fsdp2-tp4": ParallelTopology(
            "fsdp2-tp4",
            8,
            data_parallel_shard_degree=2,
            tensor_parallel_degree=4,
        ),
        "fsdp2-cp4": ParallelTopology(
            "fsdp2-cp4",
            8,
            data_parallel_shard_degree=2,
            context_parallel_degree=4,
        ),
        "tp2-cp4": ParallelTopology(
            "tp2-cp4",
            8,
            context_parallel_degree=4,
            tensor_parallel_degree=2,
        ),
        "fsdp4-tp2": ParallelTopology(
            "fsdp4-tp2",
            8,
            data_parallel_shard_degree=4,
            tensor_parallel_degree=2,
        ),
        "fsdp2-pp4": ParallelTopology(
            "fsdp2-pp4",
            8,
            data_parallel_shard_degree=2,
            pipeline_parallel_degree=4,
        ),
        "fsdp2-tp2-pp2": ParallelTopology(
            "fsdp2-tp2-pp2",
            8,
            data_parallel_shard_degree=2,
            tensor_parallel_degree=2,
            pipeline_parallel_degree=2,
        ),
        "fsdp2-tp4-ep8": ParallelTopology(
            "fsdp2-tp4-ep8",
            8,
            data_parallel_shard_degree=2,
            tensor_parallel_degree=4,
            expert_parallel_degree=8,
        ),
        "ddp16": ParallelTopology(
            "ddp16", 16, data_parallel_replicate_degree=16
        ),
        "fsdp16": ParallelTopology(
            "fsdp16", 16, data_parallel_shard_degree=16
        ),
    }


DISTRIBUTED_TOPOLOGY_NAMES = tuple(
    name
    for name, topology in standard_topologies().items()
    if topology.world_size <= 8 and name != "single"
)


def select_topologies(
    *,
    available: Sequence[str],
    topology: str | None = None,
    topologies: str | None = None,
    default: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Resolve one, a comma-separated subset, or the all topology selector."""

    available_names = tuple(available)
    if topology and topologies:
        raise ValueError("use either topology or topologies, not both")
    value = topology or topologies
    if value == "all":
        return available_names
    if topology:
        selected = (topology,)
    elif topologies:
        selected = tuple(part.strip() for part in topologies.split(",") if part.strip())
    else:
        selected = tuple(default or available_names)
    unknown = sorted(set(selected) - set(available_names))
    if unknown:
        raise ValueError(f"unknown topologies: {unknown}")
    if not selected:
        raise ValueError("at least one topology must be selected")
    return selected
