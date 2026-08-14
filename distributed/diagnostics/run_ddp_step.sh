#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

NGPU=${NGPU:-4}
export LOG_RANK=${LOG_RANK:-0}
export TORCHTITAN_DEVICE=${TORCHTITAN_DEVICE:-gpu}

torchrun --nproc_per_node="$NGPU" \
  --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
  --local-ranks-filter "$LOG_RANK" --role rank --tee 3 \
  "$SCRIPT_DIR/entry.py" --module glm5 --config glm5_debugmodel "$@"
