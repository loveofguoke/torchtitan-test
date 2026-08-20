#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Run TorchTitan and capture finite training metrics for stability tests."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
from typing import Any


METRICS_PATH_ENV = "GLM5_STABILITY_METRICS_PATH"


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (float, int)):
        result = float(value)
    elif hasattr(value, "item"):
        result = float(value.item())
    else:
        return None
    return result if math.isfinite(result) else None


def _install_metrics_capture() -> None:
    metrics_path = Path(os.environ[METRICS_PATH_ENV])
    rank = int(os.environ.get("RANK", "0"))
    owner_rank = int(os.environ.get("LOG_RANK", "0"))
    capture_enabled = rank == owner_rank
    if capture_enabled:
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
        required = (
            "loss_metrics/global_avg_loss",
            "loss_metrics/global_max_loss",
            "grad_norm",
        )
        if not all(key in metrics for key in required):
            return
        invalid = [key for key in required if _finite_float(metrics[key]) is None]
        if invalid:
            raise FloatingPointError(
                f"non-finite required training metrics at step {step}: {invalid}"
            )
        captured_step = int(step)
        if last_captured_step is not None and captured_step <= last_captured_step:
            raise RuntimeError(
                "stability metrics must be emitted once in strictly increasing "
                f"step order: previous={last_captured_step}, current={captured_step}"
            )
        last_captured_step = captured_step
        numeric_metrics = {
            key: converted
            for key, value in metrics.items()
            if (converted := _finite_float(value)) is not None
        }
        record = {
            "step": captured_step,
            "rank": rank,
            "timestamp_unix": time.time(),
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
        raise ValueError(f"unsupported TorchTitan device: {device_type!r}")

    from tests.glm5_2_precision.fixed_token_dataloader import (
        install_fixed_token_dataloader,
    )

    install_fixed_token_dataloader()
    _install_metrics_capture()

    from torchtitan.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
