#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export TORCHTITAN_DEVICE=gpu
exec "$SCRIPT_DIR/run_train.sh" "$@"
