#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

ACTION=${1:-all}
GPU_DEVICES=${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}
REFERENCE_GPU=${REFERENCE_GPU:-${GPU_DEVICES%%,*}}
PRECISION=${PRECISION:-bf16}
STEPS=${STEPS:-800}
FORCE=${FORCE:-0}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
TOPOLOGIES=${TOPOLOGIES:-"ddp4 hsdp2x2 tp4 pp4 fsdp2-tp2 pp2-fsdp2 pp2-tp2 fsdp4-ep2 fsdp4-ep4 fsdp2-tp2-ep2 pp2-fsdp2-ep2 pp2-tp2-ep2"}

if [[ "$CONTINUE_ON_ERROR" != "0" && "$CONTINUE_ON_ERROR" != "1" ]]; then
    echo "CONTINUE_ON_ERROR must be 0 or 1; got: $CONTINUE_ON_ERROR" >&2
    exit 2
fi

read -r -a TOPOLOGY_LIST <<< "$TOPOLOGIES"
if [[ "${#TOPOLOGY_LIST[@]}" -eq 0 ]]; then
    echo "TOPOLOGIES must contain at least one topology" >&2
    exit 2
fi

failed=()
for index in "${!TOPOLOGY_LIST[@]}"; do
    topology=${TOPOLOGY_LIST[$index]}
    echo
    echo "===== [$((index + 1))/${#TOPOLOGY_LIST[@]}] single vs $topology ($PRECISION, $STEPS steps) ====="
    if ! TOPOLOGY="$topology" \
        GPU_DEVICES="$GPU_DEVICES" \
        REFERENCE_GPU="$REFERENCE_GPU" \
        PRECISION="$PRECISION" \
        STEPS="$STEPS" \
        FORCE="$FORCE" \
        "$SCRIPT_DIR/align_glm5_debugmodel_single_vs_distributed_gpu.sh" "$ACTION"; then
        failed+=("$topology")
        echo "FAILED: $topology" >&2
        if [[ "$CONTINUE_ON_ERROR" == "0" ]]; then
            exit 1
        fi
    fi
done

echo
if [[ "${#failed[@]}" -gt 0 ]]; then
    echo "Failed topologies: ${failed[*]}" >&2
    exit 1
fi
echo "All topologies completed successfully."
