# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Pytest discovery entry for the reusable GLM-5.2 parity harness."""

from tests.glm5_2_parity import suite as _suite


# Preserve the historical module API used by the focused CPU regression tests.
# The implementation now lives with the parity framework instead of inside the
# unit-test discovery tree.
for _name in dir(_suite):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_suite, _name)


class TestGlm5Parity(_suite.Glm5ParitySuite):
    """Expose the unchanged pytest node used by paired and offline workflows."""
    pass
