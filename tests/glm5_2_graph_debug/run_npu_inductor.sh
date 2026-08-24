#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
WORKSPACE_ROOT=$(dirname -- "$PROJECT_ROOT")

ACTION=${1:-smoke}
if [[ $# -gt 0 ]]; then
    shift
fi

usage() {
    cat <<'EOF'
Usage: run_npu_inductor.sh ACTION [extra arguments]

  smoke  Run the existing GLM-5.2 graph probe-sized training workload:
         10 steps, batch 1, sequence length 32. This is the default.
  smoke-ddp   Run the same workload with replicated data parallelism.
  smoke-fsdp  Run the same workload with sharded data parallelism.
  train  Run the repository's normal GLM-5.2 training entry point with the
         three Inductor graph-mode flags. Extra arguments go to run_train.sh.
  train-ddp   Run normal training with replicated data parallelism.
  train-fsdp  Run normal training with sharded data parallelism.
  probe  Run compile_probe.py's data, eager reference, Inductor candidate,
         and comparison stages. Extra arguments go to every probe stage.
  env    Print the isolated software stack and cache paths without training.

Environment overrides:
  ASCEND_RT_VISIBLE_DEVICES   Physical NPU id; default: 2
                              Multi-card default: 3,4
  GRAPH_NGPU                 World size; multi-card default: 2. It must equal
                              the number of visible devices.
  GRAPH_HCCL_NPU_SOCKET_PORT_RANGE
                              Per-process HCCL NPU port selection; default: auto
  GRAPH_CANN_ROOT             CANN root; default: /usr/local/Ascend/cann-9.1.0
  GRAPH_CONDA_ENV             Conda env; default: torchtitan-0803-graph-adapt
  GRAPH_CACHE_ROOT            Persistent compiler cache location
  GRAPH_INDUCTOR_CACHE_DIR    Optional exact Inductor cache directory
  GRAPH_TRITON_CACHE_DIR      Optional exact Triton cache directory
  GRAPH_COMPILE_DEBUG_DIR     Optional exact torch compile-debug directory
  GRAPH_RUN_ROOT              Per-run logs and reports location
  TORCH_COMPILE_DEBUG         Set to 1 to emit debug data under .cache
EOF
}

if [[ "$ACTION" == "-h" || "$ACTION" == "--help" || "$ACTION" == "help" ]]; then
    usage
    exit 0
fi

case "$ACTION" in
    smoke|smoke-ddp|smoke-fsdp|train|train-ddp|train-fsdp|probe|env) ;;
    *)
        echo "Unknown action: $ACTION" >&2
        usage >&2
        exit 2
        ;;
esac

CANN_ROOT=${GRAPH_CANN_ROOT:-/usr/local/Ascend/cann-9.1.0}
CONDA_ENV=${GRAPH_CONDA_ENV:-/root/miniconda3/envs/torchtitan-0803-graph-adapt}
ATB_ROOT=${GRAPH_ATB_ROOT:-/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1}
case "$ACTION" in
    smoke-ddp|smoke-fsdp|train-ddp|train-fsdp)
        DEFAULT_VISIBLE_DEVICES=3,4
        DEFAULT_WORLD_SIZE=2
        ;;
    *)
        DEFAULT_VISIBLE_DEVICES=2
        DEFAULT_WORLD_SIZE=1
        ;;
esac
VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-$DEFAULT_VISIBLE_DEVICES}
IFS=',' read -r -a VISIBLE_DEVICE_LIST <<<"$VISIBLE_DEVICES"
VISIBLE_DEVICE_COUNT=${#VISIBLE_DEVICE_LIST[@]}
WORLD_SIZE=${GRAPH_NGPU:-$DEFAULT_WORLD_SIZE}
HCCL_NPU_PORT_RANGE=${GRAPH_HCCL_NPU_SOCKET_PORT_RANGE:-auto}
if ! [[ "$WORLD_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "GRAPH_NGPU must be a positive integer, got: $WORLD_SIZE" >&2
    exit 2
fi
if [[ $WORLD_SIZE -ne $VISIBLE_DEVICE_COUNT ]]; then
    echo "GRAPH_NGPU=$WORLD_SIZE but ASCEND_RT_VISIBLE_DEVICES exposes $VISIBLE_DEVICE_COUNT device(s): $VISIBLE_DEVICES" >&2
    exit 2
fi
CACHE_ROOT=${GRAPH_CACHE_ROOT:-$WORKSPACE_ROOT/.cache/torchtitan-test/graph_mode/cann91-torch214-triton321}
RUN_ROOT=${GRAPH_RUN_ROOT:-$PROJECT_ROOT/graph_debug_runs}
INDUCTOR_CACHE_DIR=${GRAPH_INDUCTOR_CACHE_DIR:-$CACHE_ROOT/inductor}
TRITON_CACHE_DIR=${GRAPH_TRITON_CACHE_DIR:-$CACHE_ROOT/triton}
COMPILE_DEBUG_DIR=${GRAPH_COMPILE_DEBUG_DIR:-$CACHE_ROOT/torch_compile_debug}
COMPILE_DEBUG=${TORCH_COMPILE_DEBUG:-0}

for required_path in \
    "$CANN_ROOT/set_env.sh" \
    "$CONDA_ENV/bin/python" \
    "$PROJECT_ROOT/run_train.sh"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Missing required path: $required_path" >&2
        exit 1
    fi
done

# Re-exec once with a minimal environment. This prevents the container's global
# CANN 9.0 paths from leaking into an otherwise CANN 9.1 graph-mode process.
if [[ ${TORCHTITAN_GRAPH_ENV_READY:-0} != 1 ]]; then
    exec env -i \
        HOME=/root \
        USER=root \
        LANG=C.UTF-8 \
        ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES" \
        GRAPH_NGPU="$WORLD_SIZE" \
        GRAPH_HCCL_NPU_SOCKET_PORT_RANGE="$HCCL_NPU_PORT_RANGE" \
        HCCL_NPU_SOCKET_PORT_RANGE="$HCCL_NPU_PORT_RANGE" \
        GRAPH_CANN_ROOT="$CANN_ROOT" \
        GRAPH_CONDA_ENV="$CONDA_ENV" \
        GRAPH_ATB_ROOT="$ATB_ROOT" \
        GRAPH_CACHE_ROOT="$CACHE_ROOT" \
        GRAPH_INDUCTOR_CACHE_DIR="$INDUCTOR_CACHE_DIR" \
        GRAPH_TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
        GRAPH_COMPILE_DEBUG_DIR="$COMPILE_DEBUG_DIR" \
        GRAPH_RUN_ROOT="$RUN_ROOT" \
        TORCH_COMPILE_DEBUG="$COMPILE_DEBUG" \
        TORCHTITAN_GRAPH_ENV_READY=1 \
        PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
        /bin/bash --noprofile --norc "$0" "$ACTION" "$@"
fi

# CANN's generated set_env.sh reads optional variables before defining them.
# Temporarily disable nounset so a clean env does not emit misleading warnings.
set +u
source "$CANN_ROOT/set_env.sh" >/dev/null
set -u
export PATH="$CONDA_ENV/bin:$ATB_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$ATB_ROOT/lib:${LD_LIBRARY_PATH:-}"
export TORCHTITAN_DEVICE=npu
export NGPU="$WORLD_SIZE"
export LOG_RANK=0
export MODULE=glm5
export CONFIG=glm5_debugmodel
unset CUDA_VISIBLE_DEVICES

mkdir -p "$INDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$COMPILE_DEBUG_DIR" "$RUN_ROOT"
export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_CACHE_DIR"
export TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
export TORCH_COMPILE_DEBUG_DIR="$COMPILE_DEBUG_DIR"

cd "$PROJECT_ROOT"

print_environment() {
    "$CONDA_ENV/bin/python" - <<'PY'
import importlib.metadata as metadata
import os
import torch
import torch_npu

print(f"torch={torch.__version__}")
print(f"torch_npu={torch_npu.__version__}")
print(f"triton_ascend={metadata.version('triton-ascend')}")
print(f"triton_dist={metadata.version('triton')}")
print(f"ASCEND_RT_VISIBLE_DEVICES={os.environ['ASCEND_RT_VISIBLE_DEVICES']}")
print(f"NGPU={os.environ['NGPU']}")
print(f"HCCL_NPU_SOCKET_PORT_RANGE={os.environ['HCCL_NPU_SOCKET_PORT_RANGE']}")
print(f"TORCHINDUCTOR_CACHE_DIR={os.environ['TORCHINDUCTOR_CACHE_DIR']}")
print(f"TRITON_CACHE_DIR={os.environ['TRITON_CACHE_DIR']}")
print(f"TORCH_COMPILE_DEBUG_DIR={os.environ['TORCH_COMPILE_DEBUG_DIR']}")
PY
    echo "CANN_ROOT=$CANN_ROOT"
    echo "CONDA_ENV=$CONDA_ENV"
    echo "GRAPH_RUN_ROOT=$RUN_ROOT"
}

if [[ "$ACTION" == "env" ]]; then
    print_environment
    exit 0
fi

RUN_STAMP=$(date +%Y%m%d-%H%M%S)
RUN_DIR="$RUN_ROOT/${ACTION}-${RUN_STAMP}-$$"
RUNTIME_LOG="$RUN_DIR/logs/runtime.log"
REPORT_FILE="$RUN_DIR/reports/report.md"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/reports"

START_EPOCH=$(date +%s)
STATUS=0

TOPOLOGY=single
PARALLEL_ARGS=()
case "$ACTION" in
    smoke-ddp|train-ddp)
        TOPOLOGY="ddp${WORLD_SIZE}"
        PARALLEL_ARGS=(
            --parallelism.spmd_backend=partial_dtensor
            --parallelism.data_parallel_replicate_degree="$WORLD_SIZE"
            --parallelism.data_parallel_shard_degree=1
            --parallelism.context_parallel_degree=1
            --parallelism.tensor_parallel_degree=1
            --parallelism.pipeline_parallel_degree=1
            --parallelism.expert_parallel_degree=1
        )
        ;;
    smoke-fsdp|train-fsdp)
        TOPOLOGY="fsdp${WORLD_SIZE}"
        PARALLEL_ARGS=(
            --parallelism.spmd_backend=partial_dtensor
            --parallelism.data_parallel_replicate_degree=1
            --parallelism.data_parallel_shard_degree="$WORLD_SIZE"
            --parallelism.context_parallel_degree=1
            --parallelism.tensor_parallel_degree=1
            --parallelism.pipeline_parallel_degree=1
            --parallelism.expert_parallel_degree=1
        )
        ;;
esac

case "$ACTION" in
    smoke|smoke-ddp|smoke-fsdp)
        export TORCHTITAN_RUN_LOG="$RUNTIME_LOG"
        set +e
        ./run_train.sh \
            --training.steps=10 \
            --training.num_tokens_per_microbatch_per_dp_rank=32 \
            --training.num_tokens_per_train_step="$((32 * WORLD_SIZE))" \
            --training.max_context_length=32 \
            --training.dtype=float32 \
            --training.mixed_precision_param=bfloat16 \
            --compile.enable \
            --compile.components=model \
            --compile.backend=inductor \
            "${PARALLEL_ARGS[@]}" \
            "$@"
        STATUS=$?
        set -e
        ;;
    train|train-ddp|train-fsdp)
        export TORCHTITAN_RUN_LOG="$RUNTIME_LOG"
        set +e
        ./run_train.sh \
            --compile.enable \
            --compile.components=model \
            --compile.backend=inductor \
            "${PARALLEL_ARGS[@]}" \
            "$@"
        STATUS=$?
        set -e
        ;;
    probe)
        COMMON_ARGS=(
            --topology single
            --repeat 1
            --objectives precision
            --reference-graph eager
            --candidate-graph inductor
        )
        set +e
        (
            set -e
            python tests/glm5_2_graph/compile_probe.py \
                --data --data-device npu "${COMMON_ARGS[@]}" "$@"
            python tests/glm5_2_graph/compile_probe.py \
                --capture reference --compiler-diagnostics \
                "${COMMON_ARGS[@]}" "$@"
            python tests/glm5_2_graph/compile_probe.py \
                --capture candidate --compiler-diagnostics \
                "${COMMON_ARGS[@]}" "$@"
            python tests/glm5_2_graph/compile_probe.py \
                --compare --compiler-diagnostics --require-all \
                "${COMMON_ARGS[@]}" "$@"
        ) 2>&1 | tee "$RUNTIME_LOG"
        STATUS=${PIPESTATUS[0]}
        set -e
        ;;
esac

END_EPOCH=$(date +%s)
DURATION=$((END_EPOCH - START_EPOCH))
RESULT=FAILED
if [[ $STATUS -eq 0 ]]; then
    RESULT=PASSED
fi

TORCH_VERSION=$(python -c 'import importlib.metadata as m; print(m.version("torch"))')
TORCH_NPU_VERSION=$(python -c 'import importlib.metadata as m; print(m.version("torch-npu"))')
TRITON_ASCEND_VERSION=$(python -c 'import importlib.metadata as m; print(m.version("triton-ascend"))')
STEP_COUNT=$(grep -c 'step:' "$RUNTIME_LOG" || true)
COMPLETED_COUNT=$(grep -c 'Training completed' "$RUNTIME_LOG" || true)
COMPILER_ERROR_COUNT=$(grep -Eic 'BackendCompilerFailed|InductorError|No valid triton configs' "$RUNTIME_LOG" || true)

cat >"$REPORT_FILE" <<EOF
# NPU Inductor graph-mode run

- Result: **$RESULT**
- Action: \`$ACTION\`
- Exit code: $STATUS
- Duration: $DURATION seconds
- Started: $RUN_STAMP
- Topology: \`$TOPOLOGY\`
- World size: $WORLD_SIZE
- Physical NPUs: \`$VISIBLE_DEVICES\`
- HCCL NPU socket ports: \`$HCCL_NPU_PORT_RANGE\`
- Conda: \`$CONDA_ENV\`
- CANN: \`$CANN_ROOT\`
- torch: \`$TORCH_VERSION\`
- torch_npu: \`$TORCH_NPU_VERSION\`
- triton_ascend: \`$TRITON_ASCEND_VERSION\`
- Inductor cache: \`$TORCHINDUCTOR_CACHE_DIR\`
- Triton cache: \`$TRITON_CACHE_DIR\`
- Compile debug: \`$TORCH_COMPILE_DEBUG_DIR\`
- Runtime log: \`$RUNTIME_LOG\`
- Logged training steps: $STEP_COUNT
- \`Training completed\` records: $COMPLETED_COUNT
- Compiler/backend error records: $COMPILER_ERROR_COUNT

The process was launched from a minimal environment and used only the isolated
CANN 9.1 and cloned Conda environment. See \`runtime.log\` for the exact command,
training steps, compiler diagnostics, and any exception traceback.
EOF

echo "Graph-mode result: $RESULT"
echo "Graph-mode report: $REPORT_FILE"
echo "Graph-mode runtime log: $RUNTIME_LOG"
exit "$STATUS"
