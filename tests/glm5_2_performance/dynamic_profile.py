"""Configuration control for torch_npu non-intrusive dynamic profiling."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .config import ProfilerPreset


_LEVEL_NAMES = {
    "level_none": "Level_none",
    "level0": "Level0",
    "level1": "Level1",
    "level2": "Level2",
}
_AIC_METRIC_NAMES = {
    "none": "AiCoreNone",
    "pipe_utilization": "PipeUtilization",
    "arithmetic_utilization": "ArithmeticUtilization",
    "memory": "Memory",
    "memory_l0": "MemoryL0",
    "memory_ub": "MemoryUB",
    "resource_conflict_ratio": "ResourceConflictRatio",
    "l2_cache": "L2Cache",
    "memory_access": "MemoryAccess",
}


def build_dynamic_profile_config(
    *,
    preset: ProfilerPreset,
    profile_directory: Path,
    start_step: int,
    active_steps: int,
    warmup_steps: int,
    enabled: bool,
) -> dict[str, Any]:
    """Translate a performance preset to the official dynamic JSON schema."""

    if start_step < 0:
        raise ValueError("dynamic profile start_step must be non-negative")
    if active_steps < 1 or warmup_steps < 0:
        raise ValueError("dynamic profile active/warmup steps are invalid")
    if preset.profile_ranks in {"all", "*", "-1"}:
        is_rank = False
        rank_list: list[int] = []
    else:
        is_rank = True
        rank_list = sorted(
            {int(value.strip()) for value in preset.profile_ranks.split(",")}
        )
    if is_rank and preset.parse_mode == "sync":
        raise ValueError(
            "rank-filtered dynamic profiling requires async or offline parsing"
        )

    return {
        "is_valid": enabled,
        "activities": ["CPU", "NPU"],
        "prof_dir": str(profile_directory.resolve()),
        "analyse": preset.parse_mode != "offline",
        "async_mode": preset.parse_mode == "async",
        "record_shapes": preset.record_shapes,
        "profile_memory": preset.profile_memory,
        "with_stack": preset.with_stack,
        "with_flops": False,
        "with_modules": preset.with_modules,
        "active": active_steps,
        "warmup": warmup_steps,
        "start_step": start_step,
        "is_rank": is_rank,
        "rank_list": rank_list,
        "experimental_config": {
            "profiler_level": _LEVEL_NAMES[preset.level],
            "aic_metrics": _AIC_METRIC_NAMES[preset.aic_metrics],
            "l2_cache": preset.l2_cache,
            "op_attr": preset.op_attr,
            "gc_detect_threshold": preset.gc_detect_threshold,
            "data_simplification": preset.data_simplification,
            "record_op_args": preset.record_op_args,
            "export_type": ["text", "db"],
            "msprof_tx": False,
            "mstx": False,
            "host_sys": list(preset.host_system),
            "mstx_domain_include": [],
            "mstx_domain_exclude": [],
            "sys_io": preset.system_io,
            "sys_interconnection": preset.system_interconnection,
        },
    }


def write_dynamic_profile_config(
    config_directory: Path,
    config: dict[str, Any],
) -> Path:
    """Atomically update the configuration polled by dynamic_profile."""

    config_directory.mkdir(parents=True, exist_ok=True)
    if config_directory.is_symlink():
        raise ValueError("dynamic profile config directory must not be a symlink")
    destination = config_directory / "profiler_config.json"
    temporary = config_directory / ".profiler_config.json.tmp"
    temporary.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination
