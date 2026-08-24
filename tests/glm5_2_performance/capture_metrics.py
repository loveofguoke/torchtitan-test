#!/usr/bin/env python3
"""Run TorchTitan while recording numeric training performance metrics."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Literal


METRICS_PATH_ENV = "GLM5_PERFORMANCE_METRICS_PATH"
PROFILE_RANKS_ENV = "GLM5_PERFORMANCE_PROFILE_RANKS"
REDUCE_DTYPE_OVERRIDE_ENV = "GLM5_PERFORMANCE_REDUCE_DTYPE_OVERRIDE"


def _install_reduction_dtype_override() -> None:
    """Expose reduced-precision FSDP reductions only to this experiment."""

    value = os.environ.get(REDUCE_DTYPE_OVERRIDE_ENV)
    if not value:
        return
    supported = ("float32", "float16", "bfloat16")
    if value not in supported:
        raise ValueError(
            f"unsupported {REDUCE_DTYPE_OVERRIDE_ENV}={value!r}; "
            f"expected one of {supported}"
        )

    # TorchTitan deliberately exposes only float32 in its public config. The
    # underlying MixedPrecisionPolicy already accepts these torch dtypes, so
    # widen Tyro's runtime dataclass metadata for this isolated performance
    # process without changing TorchTitan's generic Trainer or config source.
    from torchtitan.config.configs import TrainingConfig

    choices = Literal["float32", "float16", "bfloat16"]
    TrainingConfig.__annotations__["mixed_precision_reduce"] = choices
    TrainingConfig.__dataclass_fields__["mixed_precision_reduce"].type = choices


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
    elif hasattr(value, "item"):
        result = float(value.item())
    else:
        return None
    return result if math.isfinite(result) else None


def _install_metrics_capture() -> None:
    path_value = os.environ.get(METRICS_PATH_ENV)
    if not path_value:
        raise RuntimeError(f"{METRICS_PATH_ENV} must name the metrics JSONL file")
    path = Path(path_value)

    from torchtitan.components.metrics import TensorBoardLogger

    original_log = TensorBoardLogger.log

    def log_with_jsonl(
        self: TensorBoardLogger,
        metrics: dict[str, Any],
        step: int,
    ) -> None:
        original_log(self, metrics, step)
        numeric = {
            key: converted
            for key, value in metrics.items()
            if (converted := _to_float(value)) is not None
        }
        if not numeric:
            return
        record = {
            "step": int(step),
            "rank": int(os.environ.get("RANK", "0")),
            "metrics": numeric,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            stream.write("\n")

    TensorBoardLogger.log = log_with_jsonl


def _install_gpu_rank_filter() -> None:
    """Apply the same rank-selection contract to TorchTitan's GPU profiler."""

    value = os.environ.get(PROFILE_RANKS_ENV, "all").strip().lower()
    if value in {"all", "*", "-1"}:
        return
    selected_ranks = {int(part.strip()) for part in value.split(",")}

    import torch
    from torchtitan.tools.profiler import Profiler

    original_build = Profiler.build_torch_profiler

    def rank_filtered_build(self: Profiler, **kwargs: Any) -> Any:
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available()
            and torch.distributed.is_initialized()
            else 0
        )
        if rank not in selected_ranks:
            return None
        return original_build(self, **kwargs)

    Profiler.build_torch_profiler = rank_filtered_build


def main() -> None:
    if os.environ.get("TORCHTITAN_DEVICE") == "npu":
        import torchtitanturbo  # noqa: F401
    else:
        _install_gpu_rank_filter()

    _install_reduction_dtype_override()
    _install_metrics_capture()

    from torchtitan.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
