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
RESUME=${RESUME:-1}
RESET=${RESET:-0}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
TOPOLOGIES=${TOPOLOGIES:-"ddp4 hsdp2x2 tp4 pp4 fsdp2-tp2 pp2-fsdp2 pp2-tp2 fsdp4-ep2 fsdp4-ep4 fsdp2-tp2-ep2 pp2-fsdp2-ep2 pp2-tp2-ep2"}
SHARED_REFERENCE_GROUP=${SHARED_REFERENCE_GROUP:-four-gpu-matrix}

if [[ "$FORCE" != "0" && "$FORCE" != "1" ]]; then
    echo "FORCE must be 0 or 1; got: $FORCE" >&2
    exit 2
fi
if [[ "$RESUME" != "0" && "$RESUME" != "1" ]]; then
    echo "RESUME must be 0 or 1; got: $RESUME" >&2
    exit 2
fi
if [[ "$RESET" != "0" && "$RESET" != "1" ]]; then
    echo "RESET must be 0 or 1; got: $RESET" >&2
    exit 2
fi
if [[ "$FORCE" == "1" ]]; then
    echo "NOTE: FORCE=1 is ignored by the resumable matrix runner; use RESET=1 to rerun completed work." >&2
fi
if [[ "$RESET" == "1" ]]; then
    WORKER_FORCE=1
    WORKER_RESUME=0
else
    WORKER_FORCE=0
    WORKER_RESUME=$RESUME
fi

if [[ "$CONTINUE_ON_ERROR" != "0" && "$CONTINUE_ON_ERROR" != "1" ]]; then
    echo "CONTINUE_ON_ERROR must be 0 or 1; got: $CONTINUE_ON_ERROR" >&2
    exit 2
fi

read -r -a TOPOLOGY_LIST <<< "$TOPOLOGIES"
if [[ "${#TOPOLOGY_LIST[@]}" -eq 0 ]]; then
    echo "TOPOLOGIES must contain at least one topology" >&2
    exit 2
fi

run_worker() {
    local topology=$1
    local action=$2
    TOPOLOGY="$topology" \
        GPU_DEVICES="$GPU_DEVICES" \
        REFERENCE_GPU="$REFERENCE_GPU" \
        PRECISION="$PRECISION" \
        STEPS="$STEPS" \
        FORCE="$WORKER_FORCE" \
        RESUME="$WORKER_RESUME" \
        GLM5_ALIGNMENT_SHARED_REFERENCE_GROUP="$SHARED_REFERENCE_GROUP" \
        "$SCRIPT_DIR/align_glm5_debugmodel_single_vs_distributed_gpu.sh" "$action"
}

run_shared_stage() {
    local action=$1
    local topology=${TOPOLOGY_LIST[0]}
    echo
    echo "===== shared single stage: $action ($PRECISION, $STEPS steps) ====="
    run_worker "$topology" "$action"
}

run_candidate_stage() {
    local action=$1
    local failed=()
    local index topology
    for index in "${!TOPOLOGY_LIST[@]}"; do
        topology=${TOPOLOGY_LIST[$index]}
        echo
        echo "===== [$((index + 1))/${#TOPOLOGY_LIST[@]}] single vs $topology: $action ($PRECISION, $STEPS steps) ====="
        if [[ "$action" == "all-candidates" ]]; then
            if run_worker "$topology" capture-candidate; then
                run_worker "$topology" compare || status=$?
            else
                status=$?
            fi
        else
            run_worker "$topology" "$action" || status=$?
        fi
        if [[ "${status:-0}" -ne 0 ]]; then
            failed+=("$topology")
            echo "FAILED: $topology" >&2
            if [[ "$CONTINUE_ON_ERROR" == "0" ]]; then
                exit 1
            fi
        fi
        unset status
    done

    echo
    if [[ "${#failed[@]}" -gt 0 ]]; then
        echo "Failed topologies: ${failed[*]}" >&2
        return 1
    fi
    echo "All candidate topologies completed stage: $action"
}

case "$ACTION" in
    prepare|capture-reference|capture-reference-1|capture-reference-2)
        run_shared_stage "$ACTION"
        ;;
    capture-candidate|capture-candidate-1|capture-candidate-2|compare)
        run_candidate_stage "$ACTION"
        ;;
    all)
        run_shared_stage prepare
        run_shared_stage capture-reference
        run_candidate_stage all-candidates
        ;;
    *)
        echo "Usage: $0 [all|prepare|capture-reference[-1|-2]|capture-candidate[-1|-2]|compare]" >&2
        exit 2
        ;;
esac
