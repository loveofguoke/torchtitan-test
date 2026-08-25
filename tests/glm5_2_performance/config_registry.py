# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Test-owned GLM-5 configs for isolated performance trade-off experiments."""

from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.glm5.config_registry import glm5_full_dsa_debugmodel


def glm5_full_dsa_full_ac_debugmodel():
    """Enable full-block recomputation without adding a core model flavor."""

    config = glm5_full_dsa_debugmodel()
    config.activation_checkpoint = FullAC.Config()
    return config


__all__ = ["glm5_full_dsa_full_ac_debugmodel"]
