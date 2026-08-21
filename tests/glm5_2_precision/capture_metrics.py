#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Run TorchTitan while preserving full-precision training metrics as JSONL."""

from __future__ import annotations

import inspect
import json
import math
import os
from pathlib import Path
from typing import Any


METRICS_PATH_ENV = "GLM5_PRECISION_METRICS_PATH"


def _declare_npu_router_compute_dtype_support() -> None:
    """Bridge older TorchTitanTurbo implementations with the gate contract.

    Some Turbo revisions implement the explicit ``compute_dtype`` keyword but
    predate TorchTitan's capability declaration. Only declare support after
    checking the actual forward signature; never bypass the contract for an
    implementation that cannot honor the keyword.
    """

    try:
        from torchtitanturbo.models.glm5.patch import NpuFp32RouterLinear
    except ImportError:
        return
    if "compute_dtype" not in inspect.signature(
        NpuFp32RouterLinear.forward
    ).parameters:
        raise RuntimeError(
            "NpuFp32RouterLinear does not implement the compute_dtype contract"
        )
    NpuFp32RouterLinear.supports_compute_dtype = True


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (float, int)):
        result = float(value)
    elif hasattr(value, "item"):
        result = float(value.item())
    else:
        return None
    return result if math.isfinite(result) else None


def _install_jsonl_metrics_capture() -> None:
    metrics_path_value = os.environ.get(METRICS_PATH_ENV)
    if not metrics_path_value:
        raise RuntimeError(f"{METRICS_PATH_ENV} must name the JSONL metrics file")
    metrics_path = Path(metrics_path_value)

    from torchtitan.components.metrics import TensorBoardLogger

    original_log = TensorBoardLogger.log

    def log_with_jsonl(
        self: TensorBoardLogger,
        metrics: dict[str, Any],
        step: int,
    ) -> None:
        original_log(self, metrics, step)
        required = {
            "loss_metrics/global_avg_loss",
            "loss_metrics/global_max_loss",
            "grad_norm",
        }
        if not required.issubset(metrics):
            return
        numeric_metrics = {
            key: converted
            for key, value in metrics.items()
            if (converted := _to_float(value)) is not None
        }
        record = {
            "step": int(step),
            "rank": int(os.environ.get("RANK", "0")),
            "metrics": numeric_metrics,
        }
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            stream.write("\n")

    TensorBoardLogger.log = log_with_jsonl


def main() -> None:
    device_type = os.environ.get("TORCHTITAN_DEVICE", "gpu")
    if device_type == "npu":
        import torchtitanturbo  # noqa: F401

        _declare_npu_router_compute_dtype_support()
    elif device_type != "gpu":
        raise ValueError(f"TORCHTITAN_DEVICE must be gpu or npu, got {device_type!r}")

    from .fixed_token_dataloader import install_fixed_token_dataloader

    install_fixed_token_dataloader()
    _install_jsonl_metrics_capture()
    if os.environ.get("GLM5_PRECISION_TRACE_STEPS"):
        from .sampled_trace import install_sampled_trace

        install_sampled_trace()

    from torchtitan.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
