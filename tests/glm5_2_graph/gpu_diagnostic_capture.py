#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""CUDA precision capture with optional Block 0 tensor tracing."""

from __future__ import annotations

import os


def main() -> None:
    if os.environ.get("TORCHTITAN_DEVICE", "gpu") != "gpu":
        raise ValueError("gpu_diagnostic_capture supports only CUDA endpoints")

    from tests.glm5_2_precision.capture_metrics import (
        _install_jsonl_metrics_capture,
    )
    from tests.glm5_2_precision.fixed_token_dataloader import (
        install_fixed_token_dataloader,
    )

    install_fixed_token_dataloader()
    _install_jsonl_metrics_capture()

    if os.environ.get("GLM5_GPU_BLOCK_TRACE_PATH"):
        from tests.glm5_2_graph.gpu_block_trace import install_gpu_block_trace

        install_gpu_block_trace()

    if os.environ.get("GLM5_GPU_INTERNAL_TRACE_PATH"):
        from tests.glm5_2_graph.gpu_internal_block_trace import (
            install_gpu_internal_block_trace,
        )

        install_gpu_internal_block_trace()

    if os.environ.get("GLM5_GPU_MATERIALIZE_DENSE_SILU") == "1":
        from tests.glm5_2_graph.gpu_silu_materialization import (
            install_gpu_silu_materialization,
        )

        install_gpu_silu_materialization()

    if os.environ.get("GLM5_GPU_ATTENTION_TRACE_PATH"):
        from tests.glm5_2_graph.gpu_attention_internal_trace import (
            install_gpu_attention_internal_trace,
        )

        install_gpu_attention_internal_trace()

    from torchtitan.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
