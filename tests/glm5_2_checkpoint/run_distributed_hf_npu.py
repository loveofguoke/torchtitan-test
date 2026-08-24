# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""torchrun entry point for the GLM-5 distributed HF checkpoint NPU test."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from distributed_hf_roundtrip import (
    run_pp_tp_hf_roundtrip,
    run_tp_hf_roundtrip,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("tp2", "ep2-tp2", "pp2-tp2"), required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    expected_world_size = 2 if args.case == "tp2" else 4
    if world_size != expected_world_size:
        raise ValueError(
            f"{args.case} requires world size {expected_world_size}, got {world_size}"
        )

    args.checkpoint_path.mkdir(parents=True, exist_ok=True)
    common_args = (
        rank,
        world_size,
        "env://",
        str(args.checkpoint_path),
    )
    if args.case == "pp2-tp2":
        run_pp_tp_hf_roundtrip(*common_args, backend="hccl", device_type="npu")
    else:
        run_tp_hf_roundtrip(
            *common_args,
            expert_parallel=args.case == "ep2-tp2",
            backend="hccl",
            device_type="npu",
        )


if __name__ == "__main__":
    main()
