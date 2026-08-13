#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

PYTHON_BIN=${PYTHON_BIN:-python3}
ACTION=${1:-all}
REFERENCE_TOPOLOGY=${REFERENCE_TOPOLOGY:-single}
CANDIDATE_TOPOLOGY=${CANDIDATE_TOPOLOGY:-fsdp4}
GPU_DEVICES=${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}
REFERENCE_GPU=${REFERENCE_GPU:-${GPU_DEVICES%%,*}}
PRECISION=${PRECISION:-bf16}
FORCE=${FORCE:-0}

BENCHMARK=tests/glm5_2_precision/self_consistency_benchmark.py
COMMON_ARGS=(
    --device cuda
    --reference-topology "$REFERENCE_TOPOLOGY"
    --candidate-topology "$CANDIDATE_TOPOLOGY"
    --reference-visible-devices "$REFERENCE_GPU"
    --candidate-visible-devices "$GPU_DEVICES"
    --precision "$PRECISION"
)

if [[ "$FORCE" == "1" ]]; then
    COMMON_ARGS+=(--force)
elif [[ "$FORCE" != "0" ]]; then
    echo "FORCE must be 0 or 1; got: $FORCE" >&2
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
    run_benchmark --capture reference --repeat 1
    run_benchmark --capture reference --repeat 2
}

capture_candidate() {
    run_benchmark --capture candidate --repeat 1
    run_benchmark --capture candidate --repeat 2
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
    capture-candidate)
        capture_candidate
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
        echo "Usage: $0 [all|prepare|capture-reference|capture-candidate|compare]" >&2
        exit 2
        ;;
esac
