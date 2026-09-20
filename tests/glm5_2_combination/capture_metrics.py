#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Run one training job with synchronized data and optional profiling.

Installation order mirrors feature ownership: configure compiler diagnostics,
load device-specific Turbo/rank filtering, install formal metric capture and
fixed-token data, then enter the unmodified TorchTitan train entry point. This
makes the same worker simultaneously observable by precision, graph, and
performance tooling without one feature secretly configuring another.
"""

from __future__ import annotations

import os
import sys


def _install_comm_timeout_override() -> None:
    """Forward the graph launcher's process-group timeout to TorchTitan."""

    raw_timeout = os.environ.get("TORCHTITAN_COMM_INIT_TIMEOUT_SECONDS")
    if raw_timeout is None:
        return
    try:
        timeout_seconds = int(raw_timeout)
    except ValueError as error:
        raise ValueError(
            "TORCHTITAN_COMM_INIT_TIMEOUT_SECONDS must be a positive integer; "
            f"got {raw_timeout!r}"
        ) from error
    if timeout_seconds <= 0:
        raise ValueError(
            "TORCHTITAN_COMM_INIT_TIMEOUT_SECONDS must be a positive integer; "
            f"got {raw_timeout!r}"
        )
    if not any(
        argument == "--comm.init_timeout_seconds"
        or argument.startswith("--comm.init_timeout_seconds=")
        for argument in sys.argv
    ):
        sys.argv.append(f"--comm.init_timeout_seconds={timeout_seconds}")


def main() -> None:
    _install_comm_timeout_override()

    from tests.glm5_2_graph.visualization import configure_graph_diagnostics

    configure_graph_diagnostics()
    device_type = os.environ.get("TORCHTITAN_DEVICE", "gpu")
    if device_type == "npu":
        import torchtitanturbo  # noqa: F401
    elif device_type == "gpu":
        if os.environ.get("GLM5_PERFORMANCE_PROFILE_RANKS"):
            from tests.glm5_2_performance.capture_metrics import (
                _install_gpu_rank_filter,
            )

            _install_gpu_rank_filter()
    else:
        raise ValueError(f"TORCHTITAN_DEVICE must be gpu or npu, got {device_type!r}")

    from tests.glm5_2_precision.capture_metrics import (
        _install_jsonl_metrics_capture,
    )
    from tests.glm5_2_precision.fixed_token_dataloader import (
        install_fixed_token_dataloader,
    )

    install_fixed_token_dataloader()
    _install_jsonl_metrics_capture()

    from torchtitan.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
