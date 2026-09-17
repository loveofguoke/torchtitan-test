#!/usr/bin/env python3
"""Measure rank-local compute and communication phases without a model."""

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


def percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "p95_ms": percentile_95(values),
        "max_ms": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-size", type=int, default=8192)
    parser.add_argument("--collective-size-mib", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    dist.init_process_group("hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.npu.set_device(rank)
    device = torch.device(f"npu:{rank}")
    matrix = torch.randn(
        args.matrix_size,
        args.matrix_size,
        dtype=torch.bfloat16,
        device=device,
    )
    weight = torch.randn_like(matrix)
    num_collective_elements = args.collective_size_mib * 1024 * 1024 // 2
    collective = torch.ones(
        num_collective_elements,
        dtype=torch.bfloat16,
        device=device,
    )

    for _ in range(args.warmup):
        torch.mm(matrix, weight)
        dist.all_reduce(collective)
    torch.npu.synchronize()
    dist.barrier()

    barrier_release_ms = []
    compute_enqueue_ms = []
    compute_synchronized_ms = []
    collective_enqueue_ms = []
    collective_synchronized_ms = []
    total_ms = []
    for _ in range(args.iterations):
        before_barrier = time.perf_counter_ns()
        dist.barrier()
        released = time.perf_counter_ns()

        compute_start = time.perf_counter_ns()
        torch.mm(matrix, weight)
        compute_enqueued = time.perf_counter_ns()
        torch.npu.synchronize()
        compute_completed = time.perf_counter_ns()

        collective_start = time.perf_counter_ns()
        dist.all_reduce(collective)
        collective_enqueued = time.perf_counter_ns()
        torch.npu.synchronize()
        collective_completed = time.perf_counter_ns()

        barrier_release_ms.append((released - before_barrier) / 1_000_000.0)
        compute_enqueue_ms.append(
            (compute_enqueued - compute_start) / 1_000_000.0
        )
        compute_synchronized_ms.append(
            (compute_completed - compute_start) / 1_000_000.0
        )
        collective_enqueue_ms.append(
            (collective_enqueued - collective_start) / 1_000_000.0
        )
        collective_synchronized_ms.append(
            (collective_completed - collective_start) / 1_000_000.0
        )
        total_ms.append((collective_completed - released) / 1_000_000.0)

    visible_devices = [
        value.strip()
        for value in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    ]
    row = {
        "benchmark": "synthetic_ddp_step",
        "visible_devices": ",".join(visible_devices),
        "rank": rank,
        "physical_device": visible_devices[rank],
        "world_size": world_size,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "matrix_size": args.matrix_size,
        "collective_size_mib": args.collective_size_mib,
        "iterations": args.iterations,
        "barrier": summarize(barrier_release_ms),
        "compute_enqueue": summarize(compute_enqueue_ms),
        "compute_synchronized": summarize(compute_synchronized_ms),
        "collective_enqueue": summarize(collective_enqueue_ms),
        "collective_synchronized": summarize(collective_synchronized_ms),
        "total": summarize(total_ms),
    }
    gathered: list[dict | None] = [None] * world_size
    dist.all_gather_object(gathered, row)
    if rank == 0:
        for rank_row in gathered:
            print(json.dumps(rank_row), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
