#!/usr/bin/env bash
# Each A/B endpoint is a fresh process on the same explicitly selected stack.
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate titan040_baseline
cd /workspace/yyb/torchtitan
mkdir -p /workspace/yyb/outputs
ROOT=$(mktemp -d /workspace/yyb/outputs/glm5-turbo-a5-ab-XXXXXXXX)
export TORCHINDUCTOR_CACHE_DIR="$ROOT/inductor-cache"
export TRITON_CACHE_DIR="$ROOT/triton-cache"
printf '%s\n' "$ROOT" > /workspace/yyb/outputs/glm5-turbo-a5-ab-latest.path
exec > "$ROOT/suite.log" 2>&1
active_pid=
cleanup() {
    local rc=$?
    if [ -n "$active_pid" ]; then
        kill -TERM "$active_pid" 2>/dev/null || true
        wait "$active_pid" 2>/dev/null || true
    fi
    printf 'suite_exit_code=%s\n' "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
bounded() {
    timeout --kill-after=20s 300s "$@" &
    active_pid=$!
    local rc=0
    wait "$active_pid" || rc=$?
    active_pid=
    return "$rc"
}
export PYTHONPATH=/workspace/yyb/torchtitan-test:/workspace/yyb/TorchTitanTurbo:/workspace/yyb/torchtitan
export PYTHONPATH="/workspace/yyb/torchtitan-test/tests/glm5_ascend040:$PYTHONPATH"
export PYTHONPATH="${GLM_NPU_OVERLAY:+$GLM_NPU_OVERLAY:}$PYTHONPATH"
printf 'npu_overlay=%s\n' "${GLM_NPU_OVERLAY:-<installed>}"
export OMP_NUM_THREADS=4
export TORCHTITAN_CONFIG_FIRST_CLI=1
export GLM_RUNTIME=turbo
IFS=, read -r -a physical_devices <<< "${GLM_AB_DEVICES:-0,1,2,3,4,5,6,7}"
COMMON_PATCHES=${GLM_AB_COMMON_PATCHES:-}
printf 'physical_devices=%s common_patches=%s\n' "${physical_devices[*]}" "$COMMON_PATCHES"
PHASE=${GLM_AB_PHASE:-all}
case "$PHASE" in all|components|smoke|matrix) ;; *) printf 'Unknown phase: %s\n' "$PHASE"; exit 2 ;; esac
SHARED=${GLM_AB_SHARED_DEVICES:-0}
case "$SHARED" in 0|1) ;; *) printf 'Invalid shared-device policy\n'; exit 2 ;; esac
printf 'shared_devices=%s (shared runs are not performance evidence)\n' "$SHARED"
if [ "$SHARED" = 1 ]; then
    export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-auto}
    printf 'hccl_npu_socket_port_range=%s\n' "$HCCL_NPU_SOCKET_PORT_RANGE"
fi
printf 'phase=%s\n' "$PHASE"
date -u
for repository in torchtitan TorchTitanTurbo torchtitan-test torchtitan-ascend; do
    git -C "/workspace/yyb/$repository" rev-parse HEAD
    git -C "/workspace/yyb/$repository" status --short
done

check_idle() {
    local count=${2:-1}
    if [ "$count" -gt "${#physical_devices[@]}" ]; then
        printf 'STOP: insufficient explicitly selected devices\n'
        exit 2
    fi
    ASCEND_RT_VISIBLE_DEVICES=$(IFS=,; echo "${physical_devices[*]:0:$count}")
    export ASCEND_RT_VISIBLE_DEVICES
    npu-smi info > "$ROOT/npu-before-$1.log"
    if ! python - "$ROOT/npu-before-$1.log" "$ASCEND_RT_VISIBLE_DEVICES" "$SHARED" <<'PY'
import re, sys
from pathlib import Path
table = Path(sys.argv[1]).read_text()
devices = sys.argv[2].split(',')
assert len(devices) == len(set(devices)) and all(x.isdigit() for x in devices)
for device in devices:
    if sys.argv[3] == '0':
        assert re.search(r'No running processes found in NPU ' + re.escape(device) + r'\b', table), f'NPU {device} is not confirmed idle'
PY
    then
        printf 'STOP: selected devices are not confirmed idle\n'
        exit 2
    fi
}

select_patches() {
    local requested=$1
    if [ -z "$COMMON_PATCHES" ]; then
        TORCHTITAN_TURBO_PATCHES=$requested
    else
        case "$requested" in
            all) requested=remat,rope,glm_init,glm_loss ;;
            none) requested= ;;
        esac
        TORCHTITAN_TURBO_PATCHES="$COMMON_PATCHES${requested:+,$requested}"
    fi
    export TORCHTITAN_TURBO_PATCHES
}

if [ "$PHASE" = all ] || [ "$PHASE" = components ]; then
check_idle unit
export TORCHTITAN_TURBO_PATCHES=none
if python -m pytest --noconftest /workspace/yyb/TorchTitanTurbo/tests/unit_tests/test_glm5_patch.py /workspace/yyb/TorchTitanTurbo/tests/unit_tests/test_npu_rope.py /workspace/yyb/TorchTitanTurbo/tests/unit_tests/test_remat.py /workspace/yyb/TorchTitanTurbo/tests/unit_tests/test_inductor_scheduler.py -q --tb=short > "$ROOT/unit.log" 2>&1; then
    printf 'PASSED unit\n'
else
    printf 'FAILED unit; see unit.log\n'
fi

for mode in none all; do
    check_idle "native-$mode" 2
    select_patches "$mode"
    mkdir -p "$ROOT/native-$mode"
    if bounded python -m torch.distributed.run --standalone --nproc-per-node=2 --log-dir="$ROOT/native-$mode/torchrun" --redirects=3 --tee=3 -m pytest -p turbo_plugin --noconftest /workspace/yyb/torchtitan-ascend/tests/integration_tests/test_glm5_native_ops.py -q --tb=short > "$ROOT/native-$mode/runtime.log" 2>&1; then
        printf 'PASSED native-%s\n' "$mode"
    else
        printf 'FAILED native-%s rc=%s\n' "$mode" "$?"
    fi
done
fi

if [ "$PHASE" = components ]; then exit 0; fi
export GLM_AB_FIXTURE="$ROOT/shared-model-fixture.pt"

smoke() {
    local label=$1 patches=$2 count=$3 topology=$4 remat=$5 mtp=${6:-0}
    check_idle "$label" "$count"
    select_patches "$patches"
    export GLM_TOPOLOGY="$topology" GLM_REMAT="$remat" GLM_MTP="$mtp"
    export GLM_OUTPUT="$ROOT/$label"
    mkdir -p "$GLM_OUTPUT"
    printf 'START %s patches=%s devices=%s\n' "$label" "$TORCHTITAN_TURBO_PATCHES" "$ASCEND_RT_VISIBLE_DEVICES"
    if bounded python -m torch.distributed.run --standalone --nproc-per-node="$count" --log-dir="$GLM_OUTPUT/torchrun" --redirects=3 --tee=3 /workspace/yyb/torchtitan-test/tests/glm5_ascend040/worker.py > "$GLM_OUTPUT/runtime.log" 2>&1; then
        printf 'PASSED %s\n' "$label"
        return 0
    else
        local rc=$?
        printf 'FAILED %s rc=%s\n' "$label" "$rc"
        return "$rc"
    fi
}

if [ "$PHASE" = matrix ]; then
    # Native initialization, not A/B fixture equivalence. PP owns split model parts.
    unset GLM_AB_FIXTURE
    matrix_cases=${GLM_AB_MATRIX_CASES:-ddp2:2 ddp8:8 fsdp8:8 ep8:8 tp8:8 cp4:4 cp8:8 hsdp2x4:8 hsdp4x2:8 fsdp2-tp4:8 fsdp2-cp4:8 tp2-cp4:8 fsdp4-tp2:8 fsdp2-tp4-ep8:8 fsdp2-tp2-cp2-ep2:8 pp8:8 fsdp2-pp4:8 fsdp2-tp2-pp2:8}
    printf 'matrix_cases=%s extras=%s\n' "$matrix_cases" "${GLM_AB_MATRIX_EXTRAS:-1}"
    for case in $matrix_cases; do
        if [[ ! "$case" =~ ^[a-z0-9-]+:[1-8]$ ]]; then
            printf 'Invalid matrix case: %s\n' "$case"; exit 2
        fi
        topology=${case%:*}
        count=${case#*:}
        smoke "none-$topology" none "$count" "$topology" selective || true
    done
    if [ "${GLM_AB_MATRIX_EXTRAS:-1}" = 0 ]; then exit 0; fi
    smoke none-combined-full none 8 fsdp2-tp2-cp2-ep2 full || true
    smoke none-combined-mtp none 8 fsdp2-tp2-cp2-ep2 selective 1 || true
    export GLM_PRECISION=float32
    smoke none-fsdp8-fp32 none 8 fsdp8 selective || true
    printf 'Matrix collection ended; inspect every rank verdict and exit status.\n'
    exit 0
fi

# Every endpoint builds normally, then restores the same full model fixture.
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
python - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
baseline = root / 'all-single/rank-0.json'
report = {}
if baseline.exists():
    ref = json.loads(baseline.read_text())
    for file in root.glob('*/rank-0.json'):
        candidate = json.loads(file.read_text())
        if ref['status'] != 'passed' or candidate['status'] != 'passed':
            continue
        assert candidate['fixture_sha256'] == ref['fixture_sha256']
        # Only single-rank cases have the same topology and token budget here.
        if candidate['world_size'] != 1:
            continue
        deltas = {name: max(abs(a[name] - b[name]) for a, b in zip(ref['metrics'], candidate['metrics'], strict=True)) for name in ('loss', 'grad_norm')}
        report[file.parent.name] = {'max_abs_delta': deltas, 'bitwise_metrics_equal': all(value == 0 for value in deltas.values()), 'fixture_sha256': ref['fixture_sha256']}
(root / 'same-fixture-comparison.json').write_text(json.dumps(report, indent=2))
PY
printf 'A/B collection ended; inspect individual cases, not just this exit code.\n'
