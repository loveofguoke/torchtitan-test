#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Run one training job with synchronized data and optional profiling."""

from __future__ import annotations

import os


def main() -> None:
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
