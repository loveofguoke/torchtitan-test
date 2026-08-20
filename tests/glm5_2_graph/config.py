# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Compatibility exports for graph features owned by the combination layer."""

from tests.glm5_2_combination.config import GraphFeatureConfig, GraphMode


def graph_modes() -> dict[str, GraphFeatureConfig]:
    modes: tuple[GraphMode, ...] = ("eager", "inductor", "npugraphs")
    return {
        mode: GraphFeatureConfig(mode)
        for mode in modes
    }


__all__ = [
    "GraphFeatureConfig",
    "GraphMode",
    "graph_modes",
]
