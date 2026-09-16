# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Compiler-cache routing shared by experiment launch entrypoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping


def configure_rank_local_compiler_cache(
    environment: MutableMapping[str, str] | None = None,
) -> tuple[Path, Path] | None:
    """Route Inductor and Triton caches below the current distributed rank."""
    environment = os.environ if environment is None else environment
    configured_root = environment.get("TORCHTITAN_COMPILER_CACHE_ROOT")
    if not configured_root:
        return None

    rank = environment.get("RANK", environment.get("LOCAL_RANK", "0"))
    cache_root = Path(configured_root) / f"rank{rank}"
    inductor_cache = cache_root / "inductor"
    triton_cache = cache_root / "triton"
    inductor_cache.mkdir(parents=True, exist_ok=True)
    triton_cache.mkdir(parents=True, exist_ok=True)
    environment["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)
    environment["TRITON_CACHE_DIR"] = str(triton_cache)
    return inductor_cache, triton_cache
