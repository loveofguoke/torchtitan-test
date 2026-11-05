#!/usr/bin/env bash
# Each A/B endpoint is a fresh process on the same explicitly selected stack.
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate titan040_baseline
cd /workspace/yyb/torchtitan
mkdir -p /workspace/yyb/outputs
ROOT=$(mktemp -d /workspace/yyb/outputs/glm5-turbo-a5-ab-XXXXXXXX)
printf '%s\n' "$ROOT" > /workspace/yyb/outputs/glm5-turbo-a5-ab-latest.path
exec > "$ROOT/suite.log" 2>&1
trap 'printf "suite_exit_code=%s\n" "$?"' EXIT
export PYTHONPATH=/workspace/yyb/torchtitan-test:/workspace/yyb/TorchTitanTurbo:/workspace/yyb/torchtitan
export OMP_NUM_THREADS=4
export TORCHTITAN_CONFIG_FIRST_CLI=1
export GLM_RUNTIME=turbo
date -u
for repository in torchtitan TorchTitanTurbo torchtitan-test torchtitan-ascend; do
    git -C "/workspace/yyb/$repository" rev-parse HEAD
    git -C "/workspace/yyb/$repository" status --short
done

check_idle() {
    npu-smi info > "$ROOT/npu-before-$1.log"
    if [ "$(grep -c 'No running processes found in NPU' "$ROOT/npu-before-$1.log")" -ne 8 ]; then
        printf 'STOP: devices are not all idle\n'
        exit 2
    fi
}

check_idle unit
export ASCEND_RT_VISIBLE_DEVICES=0
export TORCHTITAN_TURBO_PATCHES=none
if python -m pytest --noconftest /workspace/yyb/TorchTitanTurbo/tests/unit_tests/test_glm5_patch.py /workspace/yyb/TorchTitanTurbo/tests/unit_tests/test_npu_rope.py /workspace/yyb/TorchTitanTurbo/tests/unit_tests/test_remat.py -q --tb=short > "$ROOT/unit.log" 2>&1; then
    printf 'PASSED unit\n'
else
    printf 'FAILED unit; see unit.log\n'
fi

for mode in none all; do
    check_idle "native-$mode"
    export ASCEND_RT_VISIBLE_DEVICES=0,1
    export TORCHTITAN_TURBO_PATCHES="$mode"
    mkdir -p "$ROOT/native-$mode"
    if timeout --kill-after=20s 300s python -m torch.distributed.run --standalone --nproc-per-node=2 --log-dir="$ROOT/native-$mode/torchrun" --redirects=3 --tee=3 -m pytest -p tests.glm5_ascend040.turbo_plugin --noconftest /workspace/yyb/torchtitan-ascend/tests/integration_tests/test_glm5_native_ops.py -q --tb=short > "$ROOT/native-$mode/runtime.log" 2>&1; then
        printf 'PASSED native-%s\n' "$mode"
    else
        printf 'FAILED native-%s rc=%s\n' "$mode" "$?"
    fi
done

smoke() {
    local label=$1 patches=$2 count=$3 topology=$4 remat=$5
    check_idle "$label"
    export ASCEND_RT_VISIBLE_DEVICES=$(seq -s, 0 "$((count - 1))")
    export TORCHTITAN_TURBO_PATCHES="$patches"
    export GLM_TOPOLOGY="$topology" GLM_REMAT="$remat" GLM_MTP=0
    export GLM_OUTPUT="$ROOT/$label"
    mkdir -p "$GLM_OUTPUT"
    printf 'START %s patches=%s\n' "$label" "$patches"
    if timeout --kill-after=20s 300s python -m torch.distributed.run --standalone --nproc-per-node="$count" --log-dir="$GLM_OUTPUT/torchrun" --redirects=3 --tee=3 /workspace/yyb/torchtitan-test/tests/glm5_ascend040/worker.py > "$GLM_OUTPUT/runtime.log" 2>&1; then
        printf 'PASSED %s\n' "$label"
        return 0
    else
        local rc=$?
        printf 'FAILED %s rc=%s\n' "$label" "$rc"
        return "$rc"
    fi
}

# Do not compare different random initializations as numerical A/B evidence.
# These first runs establish functionality only; component tests compare gradients.
all_ok=0
none_ok=0
if smoke all-single all 1 single selective; then all_ok=1; fi
if smoke none-single none 1 single selective; then none_ok=1; fi
if [ "$all_ok" -eq 1 ]; then
    smoke without-init remat,rope,glm_loss 1 single selective || true
    smoke without-loss remat,rope,glm_init 1 single selective || true
    smoke without-rope remat,glm_init,glm_loss 1 single selective || true
    smoke without-remat rope,glm_init,glm_loss 1 single selective || true
fi
if [ "$none_ok" -eq 1 ]; then
    smoke none-full none 1 single full || true
    smoke none-tp2-ep2 none 2 tp2-ep2 selective || true
    smoke none-cp2 none 2 cp2 selective || true
    smoke none-fsdp2 none 2 fsdp2 selective || true
fi
printf 'A/B collection ended; inspect individual cases, not just this exit code.\n'
