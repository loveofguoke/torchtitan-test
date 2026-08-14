#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

PYTHON_BIN=${PYTHON_BIN:-python3}
ACTION=${1:-all}
GPU_DEVICES=${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}
REFERENCE_GPU=${REFERENCE_GPU:-${GPU_DEVICES%%,*}}
PRECISION=${PRECISION:-bf16}
STEPS=${STEPS:-1000}
TOPOLOGY=${TOPOLOGY:-fsdp4}
FORCE=${FORCE:-0}
RESUME=${RESUME:-0}

BENCHMARK=tests/glm5_2_precision/single_vs_distributed_gpu_benchmark.py
COMMON_ARGS=(
    --device cuda
    --reference-topology single
    --candidate-topology "$TOPOLOGY"
    --reference-visible-devices "$REFERENCE_GPU"
    --candidate-visible-devices "$GPU_DEVICES"
    --precision "$PRECISION"
)

if [[ ! "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "STEPS must be a positive integer; got: $STEPS" >&2
    exit 2
fi

IFS=',' read -r -a GPU_DEVICE_LIST <<< "$GPU_DEVICES"
if [[ "${#GPU_DEVICE_LIST[@]}" -ne 4 ]]; then
    echo "GPU_DEVICES must contain exactly four comma-separated GPU IDs" >&2
    exit 2
fi

export GLM5_ALIGNMENT_STEPS="$STEPS"

if [[ "$FORCE" == "1" ]]; then
    COMMON_ARGS+=(--force)
elif [[ "$FORCE" != "0" ]]; then
    echo "FORCE must be 0 or 1; got: $FORCE" >&2
    exit 2
fi

if [[ "$RESUME" == "1" ]]; then
    COMMON_ARGS+=(--resume)
elif [[ "$RESUME" != "0" ]]; then
    echo "RESUME must be 0 or 1; got: $RESUME" >&2
    exit 2
fi

if [[ "$FORCE" == "1" && "$RESUME" == "1" ]]; then
    echo "FORCE and RESUME cannot both be 1" >&2
    exit 2
fi

run_benchmark() {
    echo "+ $PYTHON_BIN $BENCHMARK $* ${COMMON_ARGS[*]}"
    "$PYTHON_BIN" "$BENCHMARK" "$@" "${COMMON_ARGS[@]}"
}

prepare() {
    run_benchmark --data
}

capture_reference() {
    capture_reference_repeat 1
    capture_reference_repeat 2
}

capture_candidate() {
    capture_candidate_repeat 1
    capture_candidate_repeat 2
}

capture_reference_repeat() {
    run_benchmark --capture reference --repeat "$1"
}

capture_candidate_repeat() {
    run_benchmark --capture candidate --repeat "$1"
}

compare() {
    run_benchmark --compare
}

case "$ACTION" in
    prepare)
        prepare
        ;;
    capture-reference)
        capture_reference
        ;;
    capture-reference-1)
        capture_reference_repeat 1
        ;;
    capture-reference-2)
        capture_reference_repeat 2
        ;;
    capture-candidate)
        capture_candidate
        ;;
    capture-candidate-1)
        capture_candidate_repeat 1
        ;;
    capture-candidate-2)
        capture_candidate_repeat 2
        ;;
    compare)
        compare
        ;;
    all)
        prepare
        capture_reference
        capture_candidate
        compare
        ;;
    *)
        echo "Usage: $0 [all|prepare|capture-reference[-1|-2]|capture-candidate[-1|-2]|compare]" >&2
        exit 2
        ;;
esac
