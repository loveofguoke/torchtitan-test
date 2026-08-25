#!/usr/bin/env python3
"""Start TorchTitan with CUDA profiling and JSONL metric capture enabled."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
from typing import Any


METRICS_PATH_ENV = "GLM5_GPU_PERFORMANCE_METRICS_PATH"
PROFILE_RANKS_ENV = "GLM5_GPU_PROFILE_RANKS"
PROFILE_RECORD_SHAPES_ENV = "GLM5_GPU_PROFILE_RECORD_SHAPES"
PROFILE_MEMORY_ENV = "GLM5_GPU_PROFILE_MEMORY"
PROFILE_STACK_ENV = "GLM5_GPU_PROFILE_WITH_STACK"
PROFILE_FLOPS_ENV = "GLM5_GPU_PROFILE_WITH_FLOPS"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _selected_ranks() -> set[int] | None:
    value = os.environ.get(PROFILE_RANKS_ENV, "0").strip().lower()
    if value in {"all", "*", "-1"}:
        return None
    ranks = {int(part.strip()) for part in value.split(",") if part.strip()}
    if not ranks or min(ranks) < 0:
        raise ValueError(f"invalid {PROFILE_RANKS_ENV}: {value!r}")
    return ranks


def _install_cuda_profiler() -> None:
    """Replace only Profiler construction while retaining its lifecycle."""

    import torch
    from torchtitan.tools.logging import logger
    from torchtitan.tools.profiler import Profiler

    selected_ranks = _selected_ranks()

    def build_cuda_profiler(
        self: Profiler,
        *,
        global_step: int,
        base_folder: str,
        leaf_folder: str,
    ) -> Any:
        cfg = self._config
        if not cfg.enable_profiling:
            return None
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        if selected_ranks is not None and rank not in selected_ranks:
            logger.info(f"CUDA profiling disabled for unselected rank {rank}")
            return None
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA profiler requested but torch.cuda is unavailable")

        trace_dir = Path(base_folder) / cfg.save_traces_folder
        trace_dir.mkdir(parents=True, exist_ok=True)

        def trace_handler(prof: Any) -> None:
            destination = trace_dir / f"iteration_{prof.step_num}" / leaf_folder
            destination.mkdir(parents=True, exist_ok=True)
            output = destination / f"rank{rank}_trace.json"
            started = time.monotonic()
            prof.export_chrome_trace(str(output))
            logger.info(
                f"Exported CUDA profiler trace {output} in "
                f"{time.monotonic() - started:.2f} seconds"
            )

        schedule_options = {
            key: value
            for key, value in (
                ("repeat", cfg.profiler_repeat),
                ("skip_first", cfg.profiler_skip_first),
                ("skip_first_wait", cfg.profiler_skip_first_wait),
            )
            if value is not None
        }
        wait = cfg.profile_freq - (cfg.profiler_warmup + cfg.profiler_active)
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=wait,
                warmup=cfg.profiler_warmup,
                active=cfg.profiler_active,
                **schedule_options,
            ),
            on_trace_ready=trace_handler,
            record_shapes=_env_bool(PROFILE_RECORD_SHAPES_ENV, True),
            profile_memory=_env_bool(PROFILE_MEMORY_ENV, False),
            with_stack=_env_bool(PROFILE_STACK_ENV, False),
            with_flops=_env_bool(PROFILE_FLOPS_ENV, True),
        )
        profiler.__enter__()
        profiler.step_num = global_step
        return profiler

    Profiler.build_torch_profiler = build_cuda_profiler


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
    destination = os.environ.get(METRICS_PATH_ENV)
    if not destination:
        raise RuntimeError(f"{METRICS_PATH_ENV} must be set")
    path = Path(destination)

    from torchtitan.components.metrics import TensorBoardLogger

    original_log = TensorBoardLogger.log

    def log_with_jsonl(
        self: TensorBoardLogger, metrics: dict[str, Any], step: int
    ) -> None:
        original_log(self, metrics, step)
        numeric = {
            key: converted
            for key, value in metrics.items()
            if (converted := _to_float(value)) is not None
        }
        if not numeric:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "step": int(step),
            "rank": int(os.environ.get("RANK", "0")),
            "metrics": numeric,
        }
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            stream.write("\n")

    TensorBoardLogger.log = log_with_jsonl


def main() -> None:
    _install_cuda_profiler()
    _install_metrics_capture()
    from torchtitan.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
