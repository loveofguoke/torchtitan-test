#!/usr/bin/env python3
"""Measure compute and memory behavior on one visible Ascend device."""

import argparse
import json
import os
import statistics
import time

import torch
import torch_npu  # noqa: F401


def measure(operation, *, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        operation()
    torch.npu.synchronize()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def summary(name: str, samples: list[float], **extra) -> dict:
    ordered = sorted(samples)
    result = {
        "benchmark": name,
        "physical_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "unset"),
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": ordered[0],
        "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "max_ms": ordered[-1],
        "iterations": len(samples),
    }
    result.update(extra)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-size", type=int, default=8192)
    parser.add_argument("--memory-mib", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--round-index", type=int, default=0)
    parser.add_argument("--launch-batch", type=int, default=100)
    args = parser.parse_args()

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    print(json.dumps({
        "benchmark": "host_context",
        "physical_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "unset"),
        "round_index": args.round_index,
        "pid": os.getpid(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "load_average": os.getloadavg(),
        "torch_num_threads": torch.get_num_threads(),
    }), flush=True)
    matrix_a = torch.randn(args.matrix_size, args.matrix_size, device=device, dtype=torch.bfloat16)
    matrix_b = torch.randn(args.matrix_size, args.matrix_size, device=device, dtype=torch.bfloat16)
    matrix_out = torch.empty_like(matrix_a)
    matmul_samples = measure(
        lambda: torch.mm(matrix_a, matrix_b, out=matrix_out),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    operations = 2 * args.matrix_size**3
    print(json.dumps(summary(
        "bf16_matmul",
        matmul_samples,
        matrix_size=args.matrix_size,
        round_index=args.round_index,
        tflops=operations / (statistics.median(matmul_samples) / 1000.0) / 1e12,
    )), flush=True)

    num_elements = args.memory_mib * 1024 * 1024 // 2
    source = torch.randn(num_elements, device=device, dtype=torch.bfloat16)
    destination = torch.empty_like(source)
    copy_samples = measure(
        lambda: destination.copy_(source),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    bytes_moved = num_elements * 2 * 2
    print(json.dumps(summary(
        "bf16_copy",
        copy_samples,
        memory_mib=args.memory_mib,
        round_index=args.round_index,
        gib_per_second=bytes_moved / (statistics.median(copy_samples) / 1000.0) / 2**30,
    )), flush=True)

    launch_source = torch.ones(1024, device=device, dtype=torch.float32)
    launch_destination = torch.empty_like(launch_source)
    for _ in range(args.warmup):
        torch.add(launch_source, 1.0, out=launch_destination)
    torch.npu.synchronize()
    enqueue_samples = []
    synchronized_samples = []
    for _ in range(args.iterations):
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(args.launch_batch):
            torch.add(launch_source, 1.0, out=launch_destination)
        enqueued = time.perf_counter()
        torch.npu.synchronize()
        completed = time.perf_counter()
        enqueue_samples.append(
            (enqueued - start) * 1_000_000.0 / args.launch_batch
        )
        synchronized_samples.append(
            (completed - start) * 1_000_000.0 / args.launch_batch
        )
    print(json.dumps({
        "benchmark": "host_launch",
        "physical_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "unset"),
        "round_index": args.round_index,
        "launch_batch": args.launch_batch,
        "iterations": args.iterations,
        "enqueue_us_per_op": statistics.median(enqueue_samples),
        "enqueue_p95_us_per_op": sorted(enqueue_samples)[
            min(len(enqueue_samples) - 1, int(len(enqueue_samples) * 0.95))
        ],
        "synchronized_us_per_op": statistics.median(synchronized_samples),
        "synchronized_p95_us_per_op": sorted(synchronized_samples)[
            min(len(synchronized_samples) - 1, int(len(synchronized_samples) * 0.95))
        ],
    }), flush=True)


if __name__ == "__main__":
    main()
