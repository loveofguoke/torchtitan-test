#!/usr/bin/env bash
# Preserve a short gate and long-run metrics without changing running checkouts.
set -eu
source /root/miniconda3/etc/profile.d/conda.sh
conda activate titan040_baseline
cd /workspace/yyb/torchtitan
TEST_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=$(mktemp -d /workspace/yyb/outputs/glm5-a5-capture-XXXXXXXX)
export TORCHINDUCTOR_CACHE_DIR="$ROOT/inductor-cache"
export TRITON_CACHE_DIR="$ROOT/triton-cache"
exec > "$ROOT/capture.log" 2>&1
trap 'printf "capture_exit_code=%s\n" "$?"' EXIT
date -u
printf 'capture_root=%s test_root=%s\n' "$ROOT" "$TEST_ROOT"
for repository in torchtitan TorchTitanTurbo torchtitan-test; do
    git -C "/workspace/yyb/$repository" rev-parse HEAD
done
npu-smi info > "$ROOT/npu-before.log"
export PYTHONPATH="${GLM_NPU_OVERLAY:+$GLM_NPU_OVERLAY:}$TEST_ROOT:/workspace/yyb/TorchTitanTurbo:/workspace/yyb/torchtitan"
export OMP_NUM_THREADS=4 TORCHTITAN_CONFIG_FIRST_CLI=1 GLM_RUNTIME=turbo
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-auto}
export ASCEND_RT_VISIBLE_DEVICES=${GLM_CAPTURE_DEVICE:-7}
export TORCHTITAN_TURBO_PATCHES=${GLM_CAPTURE_PATCHES:-inductor_scheduler,inductor_reduction,remat}
export GLM_TOPOLOGY=single GLM_REMAT=selective GLM_PRECISION=bfloat16
export GLM_SEQ_LEN=128 GLM_MICROBATCH_TOKENS=1024 GLM_SEED=61
export GLM_DETERMINISTIC=${GLM_CAPTURE_DETERMINISTIC:-1}
export GLM_AB_FIXTURE="$ROOT/initial-model.pt"
printf 'device=%s patches=%s deterministic=%s\n' "$ASCEND_RT_VISIBLE_DEVICES" "$TORCHTITAN_TURBO_PATCHES" "$GLM_DETERMINISTIC"
for steps in 2 500; do
    export GLM_STEPS=$steps GLM_OUTPUT="$ROOT/steps-$steps"
    export GLM_INPUT_CAPTURE="$GLM_OUTPUT/inputs"
    mkdir -p "$GLM_OUTPUT"
    timeout --kill-after=20s 3600s python -m torch.distributed.run \
        --standalone --nproc-per-node=1 --log-dir="$GLM_OUTPUT/torchrun" \
        --redirects=3 --tee=3 "$TEST_ROOT/tests/glm5_ascend040/worker.py" \
        > "$GLM_OUTPUT/runtime.log" 2>&1
    printf 'PASSED steps=%s\n' "$steps"
done
