#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

NGPU=${NGPU:-8}
export LOG_RANK=${LOG_RANK:-0}
MODULE=${MODULE:-llama3}
CONFIG=${CONFIG:-llama3_debugmodel}
COMM_MODE=${COMM_MODE:-}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-http://localhost:29510}

DEVICE=${TORCHTITAN_DEVICE:-auto}
if [[ "$DEVICE" == "auto" ]]; then
    if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
        DEVICE=npu
    else
        DEVICE=gpu
    fi
fi

case "$DEVICE" in
    gpu)
        TRAIN_ENTRY=(-m torchtitan.train)
        export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
        ;;
    npu)
        TRAIN_ENTRY=("$SCRIPT_DIR/train_npu.py")
        ;;
    *)
        echo "TORCHTITAN_DEVICE must be auto, gpu, or npu; got: $DEVICE" >&2
        exit 2
        ;;
esac

echo "TorchTitan backend: $DEVICE"
if [[ -n "$COMM_MODE" ]]; then
    NGPU="$NGPU" LOCAL_RANK=0 python3 "${TRAIN_ENTRY[@]}" \
        --module "$MODULE" --config "$CONFIG" "$@" \
        --comm.mode="$COMM_MODE" --training.steps 1
else
    TORCHFT_LIGHTHOUSE="$TORCHFT_LIGHTHOUSE" \
        torchrun --nproc_per_node="$NGPU" \
        --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
        --local-ranks-filter "$LOG_RANK" --role rank --tee 3 \
        "${TRAIN_ENTRY[@]}" --module "$MODULE" --config "$CONFIG" "$@"
fi
