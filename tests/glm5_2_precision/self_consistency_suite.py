# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Backward-compatible entry points for the generalized topology suite."""

from .topology_suite import compare_topology_suite, run_topology_suite_cli


compare_suite = compare_topology_suite


def run_suite_cli(
    base,
    script_path,
    *,
    topologies,
    candidate_topologies,
):
    return run_topology_suite_cli(
        base,
        script_path,
        topologies=topologies,
        topology_names=candidate_topologies,
    )


__all__ = ["compare_suite", "run_suite_cli"]
