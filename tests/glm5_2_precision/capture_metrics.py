#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Run TorchTitan while preserving full-precision training metrics as JSONL."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any


METRICS_PATH_ENV = "GLM5_PRECISION_METRICS_PATH"


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
    rank = int(os.environ.get("RANK", "0"))
    owner_rank = int(os.environ.get("LOG_RANK", "0"))
    capture_enabled = rank == owner_rank
    if capture_enabled:
        # Each launcher process owns one phase file. Truncate stale output from
        # an earlier failed attempt instead of appending a second copy of steps.
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text("", encoding="utf-8")

    from torchtitan.components.metrics import TensorBoardLogger

    original_log = TensorBoardLogger.log
    last_captured_step: int | None = None

    def log_with_jsonl(
        self: TensorBoardLogger,
        metrics: dict[str, Any],
        step: int,
    ) -> None:
        nonlocal last_captured_step
        original_log(self, metrics, step)
        if not capture_enabled:
            return
        required = {
            "loss_metrics/global_avg_loss",
            "loss_metrics/global_max_loss",
            "grad_norm",
        }
        if not required.issubset(metrics):
            return
        captured_step = int(step)
        if last_captured_step is not None and captured_step <= last_captured_step:
            raise RuntimeError(
                "training metrics must be emitted once in strictly increasing "
                f"step order: previous={last_captured_step}, current={captured_step}"
            )
        last_captured_step = captured_step
        numeric_metrics = {
            key: converted
            for key, value in metrics.items()
            if (converted := _to_float(value)) is not None
        }
        record = {
            "step": captured_step,
            "rank": rank,
            "metrics": numeric_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            stream.write("\n")

    TensorBoardLogger.log = log_with_jsonl


def main() -> None:
    device_type = os.environ.get("TORCHTITAN_DEVICE", "gpu")
    if device_type == "npu":
        import torchtitanturbo  # noqa: F401
    elif device_type != "gpu":
        raise ValueError(f"TORCHTITAN_DEVICE must be gpu or npu, got {device_type!r}")

    from .fixed_token_dataloader import install_fixed_token_dataloader

    install_fixed_token_dataloader()
    _install_jsonl_metrics_capture()

    from torchtitan.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
