# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Orthogonal training features and conflict-checked command composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .topology import ParallelTopology


@dataclass(frozen=True)
class TrainingFeature:
    """One independent contribution to a TorchTitan execution."""

    name: str
    arguments: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _argument_key(argument: str) -> str | None:
    if not argument.startswith("--"):
        return None
    return argument.split("=", 1)[0]


@dataclass(frozen=True)
class ExecutionPlan:
    """A topology plus independent features assembled at one boundary."""

    topology: ParallelTopology
    features: tuple[TrainingFeature, ...] = ()

    def __post_init__(self) -> None:
        seen_arguments: dict[str, str] = {}
        seen_environment: dict[str, tuple[str, str]] = {}
        for feature in self.features:
            for argument in feature.arguments:
                key = _argument_key(argument)
                if key is None:
                    continue
                owner = seen_arguments.get(key)
                if owner is not None:
                    raise ValueError(
                        f"training option {key!r} is supplied by both "
                        f"{owner!r} and {feature.name!r}"
                    )
                seen_arguments[key] = feature.name
            for key, value in feature.environment.items():
                previous = seen_environment.get(key)
                if previous is not None and previous[1] != value:
                    raise ValueError(
                        f"environment variable {key!r} conflicts between "
                        f"{previous[0]!r} and {feature.name!r}"
                    )
                seen_environment[key] = (feature.name, value)

    def command_args(self) -> tuple[str, ...]:
        return tuple(
            argument
            for feature in self.features
            for argument in feature.arguments
        )

    def environment(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for feature in self.features:
            result.update(feature.environment)
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "topology": self.topology.name,
            "features": [
                {
                    "name": feature.name,
                    "arguments": list(feature.arguments),
                    "environment": dict(feature.environment),
                    "metadata": dict(feature.metadata),
                }
                for feature in self.features
            ],
        }


def compose_execution(
    topology: ParallelTopology,
    features: Sequence[TrainingFeature],
) -> ExecutionPlan:
    """Compose features without giving any feature knowledge of another."""

    return ExecutionPlan(topology=topology, features=tuple(features))
