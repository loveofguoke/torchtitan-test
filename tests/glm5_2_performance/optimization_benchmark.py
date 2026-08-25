#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Profile one opt-in GLM-5 Full DSA optimization against a stable baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_performance.config import (  # noqa: E402
    PerformanceConfig,
    performance_topologies,
)
from tests.glm5_2_performance.optimizations import (  # noqa: E402
    optimization_profiles,
)
from tests.glm5_2_performance.workflow import run_profiler_cli  # noqa: E402


BASE_CONFIG = PerformanceConfig(
    name="glm5-full-dsa-optimization",
    model_config="glm5_full_dsa_debugmodel",
    topology="single",
    preset="comparison",
    steps=30,
    skip_steps=10,
    warmup_steps=2,
    active_steps=5,
    local_batch_size=8,
    global_batch_size=64,
    sequence_length=128,
)


def _select_profile(argv: list[str]) -> tuple[PerformanceConfig, list[str]]:
    profiles = optimization_profiles()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--optimization",
        choices=tuple(profiles),
        default="reference",
    )
    parser.add_argument("--topology")
    parser.add_argument("--topologies")
    known, remaining = parser.parse_known_args(argv)
    requested = (
        tuple(part.strip() for part in known.topologies.split(",") if part.strip())
        if known.topologies
        else (known.topology or BASE_CONFIG.topology,)
    )
    profile = profiles[known.optimization]
    topologies = performance_topologies()
    full_dsa_topologies = tuple(
        name
        for name, topology in topologies.items()
        if topology.world_size <= 8 and topology.pipeline_parallel_degree == 1
    )
    if "all" in requested:
        selected = profile.allowed_topologies or full_dsa_topologies
    else:
        selected = requested
    unknown = sorted(set(selected) - set(full_dsa_topologies))
    if unknown:
        raise ValueError(
            "Full DSA optimization experiments support only known non-PP "
            f"topologies; unsupported selections: {unknown}"
        )
    profile.validate_topologies(selected)
    forwarded = list(remaining)
    if len(selected) == 1:
        forwarded.extend(("--topology", selected[0]))
    else:
        forwarded.extend(("--topologies", ",".join(selected)))
    return profile.apply(BASE_CONFIG), forwarded


if __name__ == "__main__":
    CONFIG, forwarded_arguments = _select_profile(sys.argv[1:])
    sys.argv = [sys.argv[0], *forwarded_arguments]
    run_profiler_cli(CONFIG, __file__)
