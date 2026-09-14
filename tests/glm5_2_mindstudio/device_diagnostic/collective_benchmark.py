#!/usr/bin/env python3
"""Measure HCCL collective latency for the currently visible devices."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mib", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    dist.init_process_group("hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.npu.set_device(rank)
    num_elements = args.size_mib * 1024 * 1024 // 2
    tensor = torch.ones(num_elements, dtype=torch.bfloat16, device=f"npu:{rank}")

    for _ in range(args.warmup):
        dist.all_reduce(tensor)
    torch.npu.synchronize()
    dist.barrier()

    samples = []
    for _ in range(args.iterations):
        dist.barrier()
        start = time.perf_counter()
        dist.all_reduce(tensor)
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)

    gathered: list[list[float] | None] = [None] * world_size
    dist.all_gather_object(gathered, samples)
    if rank == 0:
        for measured_rank, rank_samples in enumerate(gathered):
            assert rank_samples is not None
            print(json.dumps({
                "benchmark": "hccl_all_reduce",
                "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "unset"),
                "world_size": world_size,
                "rank": measured_rank,
                "size_mib": args.size_mib,
                "median_ms": statistics.median(rank_samples),
                "min_ms": min(rank_samples),
                "p95_ms": percentile_95(rank_samples),
                "max_ms": max(rank_samples),
                "iterations": len(rank_samples),
            }), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
