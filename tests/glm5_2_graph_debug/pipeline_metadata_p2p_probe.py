#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Reproduce PipelineStage metadata object P2P without model compilation."""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
import time

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from torch.distributed.pipelining.stage import (
    _StageForwardMeta,
    extract_tensor_metas,
)


def _parse_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-batch", type=_parse_bool, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl", timeout=timedelta(seconds=60))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("npu", local_rank)

    elapsed: list[float] = []
    for iteration in range(args.iterations):
        started = time.monotonic()
        if rank == 0:
            sample = torch.empty((8, 128, 512), dtype=torch.bfloat16, device=device)
            metadata = _StageForwardMeta(extract_tensor_metas((sample,)))
        else:
            received: list[object | None] = [None]
            dist.recv_object_list(
                received,
                src=rank - 1,
                device=device,
                use_batch=args.use_batch,
            )
            metadata = received[0]
            if not isinstance(metadata, _StageForwardMeta):
                raise TypeError(
                    f"rank {rank} received {type(metadata).__name__}, "
                    "expected _StageForwardMeta"
                )
            tensor_meta = metadata.forward_metas[0]
            if tuple(tensor_meta.shape) != (8, 128, 512):
                raise ValueError(
                    f"rank {rank} received corrupt shape {tuple(tensor_meta.shape)}"
                )
            if tensor_meta.dtype is not torch.bfloat16:
                raise ValueError(
                    f"rank {rank} received corrupt dtype {tensor_meta.dtype}"
                )
        if rank + 1 < world_size:
            dist.send_object_list(
                [metadata],
                dst=rank + 1,
                device=device,
                use_batch=args.use_batch,
            )
        dist.barrier()
        elapsed.append(time.monotonic() - started)

    print(
        json.dumps(
            {
                "rank": rank,
                "world_size": world_size,
                "use_batch": args.use_batch,
                "task_queue": os.environ.get("TASK_QUEUE_ENABLE"),
                "elapsed_seconds": elapsed,
                "status": "passed",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
