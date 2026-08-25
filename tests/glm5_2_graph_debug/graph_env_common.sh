#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Shared CANN 9.1 graph-mode environment. Source this file, then call:
#   graph_env_init BACKEND REEXEC_SCRIPT [REEXEC_SCRIPT_ARGS...]
# BACKEND is eager, inductor, or npugraphs. The function re-executes the caller
# once with a clean environment, then sources CANN before Python can import
# torch_npu. It intentionally does not launch an experiment by itself.

if [[ -n ${TORCHTITAN_GRAPH_ENV_COMMON_LOADED:-} ]]; then
    return 0 2>/dev/null || exit 0
fi
TORCHTITAN_GRAPH_ENV_COMMON_LOADED=1

GRAPH_ENV_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$GRAPH_ENV_SCRIPT_DIR/../.." && pwd)
WORKSPACE_ROOT=$(dirname -- "$PROJECT_ROOT")

graph_env_usage() {
    cat <<'EOF'
Shared graph environment variables:
  ASCEND_RT_VISIBLE_DEVICES   Physical NPUs, strictly ascending; default: 0..7
  GRAPH_CANN_ROOT             CANN root; default: /usr/local/Ascend/cann-9.1.0
  GRAPH_CONDA_ENV             Conda prefix; default: graph adaptation env
  GRAPH_ATB_ROOT              ATB C++ ABI runtime root
  GRAPH_CACHE_ROOT            Persistent compiler cache outside the Git repo
  GRAPH_INDUCTOR_CACHE_DIR    Inductor cache override
  GRAPH_TRITON_CACHE_DIR      Triton cache override
  GRAPH_COMPILE_DEBUG_DIR     torch_compile_debug override
  GRAPH_RUN_ROOT              Invocation reports; default: graph_debug_runs
  GRAPH_HCCL_NPU_SOCKET_PORT_RANGE
                              NPU NIC ports; default: auto
  GRAPH_HCCL_IF_BASE_PORT     Host socket base; default: per-process 62000-63584
  GRAPH_HCCL_CONNECT_TIMEOUT  HCCL connection timeout; default: 600 seconds
  GRAPH_COMM_INIT_TIMEOUT_SECONDS
                              PyTorch process-group timeout; default: 2400
  GRAPH_TORCHINDUCTOR_NPU_BACKEND
                              Inductor NPU codegen; default: default
  GRAPH_NPU_INDUCTOR_FALLBACK_LIST
                              Inductor default: aten.sum and c10d all_reduce
  GRAPH_TASK_QUEUE_ENABLE     Graph profiles default: 0; eager default: 1
  GRAPH_NPUGRAPH_CAPTURE_ERROR_MODE
                              Optional capture mode override
  GRAPH_NPUGRAPH_UNSAFE_OPS   NPUGraphs default: aten._grouped_mm.default
  GRAPH_NPUGRAPH_SKIP_ALL     Disable replay but keep AOT graph execution; default: 1
  GRAPH_REGISTER_COMPLEX_DTENSOR_STRATEGY
                              Register TP aten.complex strategy; default: 1
  GRAPH_PIPELINE_META_USE_BATCH
                              Pipeline metadata batched P2P; default: 0
  GRAPH_SAFE_EMPTY_GROUPED_MM Avoid CANN zero-core grouped-mm launch; default: 1
  GRAPH_SAFE_ZERO_NUMEL_TRITON
                              Skip zero-grid NPU Triton launches; default: 1
  GRAPH_COMPILE_THREADS       Inductor workers per rank; default: 1
  GRAPH_ASCEND_LAUNCH_BLOCKING
                              Synchronous CANN launch diagnostic; default: 0
  GRAPH_LOG_RANK              Rank used by test logging; default: 0
EOF
}

graph_env_require_boolean() {
    local name=$1
    local value=$2
    if [[ "$value" != 0 && "$value" != 1 ]]; then
        echo "$name must be 0 or 1: $value" >&2
        return 2
    fi
}

graph_env_init() {
    if [[ $# -lt 2 ]]; then
        echo "graph_env_init requires BACKEND and REEXEC_SCRIPT" >&2
        return 2
    fi
    GRAPH_BACKEND=$1
    local reexec_script=$2
    shift 2
    if [[ "$GRAPH_BACKEND" != eager \
        && "$GRAPH_BACKEND" != inductor \
        && "$GRAPH_BACKEND" != npugraphs ]]; then
        echo "graph backend must be eager, inductor, or npugraphs: $GRAPH_BACKEND" >&2
        return 2
    fi

    CANN_ROOT=${GRAPH_CANN_ROOT:-/usr/local/Ascend/cann-9.1.0}
    CONDA_ENV=${GRAPH_CONDA_ENV:-/root/miniconda3/envs/torchtitan-0803-graph-adapt}
    ATB_ROOT=${GRAPH_ATB_ROOT:-/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1}
    VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
    local previous_device_id=-1
    local device_id
    local -a visible_device_ids
    IFS=',' read -r -a visible_device_ids <<<"$VISIBLE_DEVICES"
    for device_id in "${visible_device_ids[@]}"; do
        if [[ ! "$device_id" =~ ^(0|[1-9][0-9]*)$ ]] \
            || (( device_id <= previous_device_id )); then
            echo "ASCEND_RT_VISIBLE_DEVICES must be strictly ascending: $VISIBLE_DEVICES" >&2
            return 2
        fi
        previous_device_id=$device_id
    done

    HCCL_NPU_PORT_RANGE=${GRAPH_HCCL_NPU_SOCKET_PORT_RANGE:-auto}
    HCCL_HOST_BASE_PORT=${GRAPH_HCCL_IF_BASE_PORT:-$((62000 + ($$ % 100) * 16))}
    HCCL_CONNECT_TIMEOUT_SECONDS=${GRAPH_HCCL_CONNECT_TIMEOUT:-600}
    COMM_INIT_TIMEOUT_SECONDS=${GRAPH_COMM_INIT_TIMEOUT_SECONDS:-2400}
    NPU_INDUCTOR_BACKEND=${GRAPH_TORCHINDUCTOR_NPU_BACKEND:-default}
    local default_fallback_list=
    local default_task_queue=1
    local default_unsafe_ops=
    local default_npugraph_skip_all=0
    if [[ "$GRAPH_BACKEND" == inductor ]]; then
        default_fallback_list=aten.sum,_c10d_functional.all_reduce
        default_task_queue=0
    elif [[ "$GRAPH_BACKEND" == npugraphs ]]; then
        default_unsafe_ops=aten._grouped_mm.default,torchtitan_graph_debug.safe_grouped_mm.default
        default_npugraph_skip_all=1
        # The default compatibility profile skips replay and therefore does
        # not require NPUGraph's asynchronous queue mode.  Queue 0 also avoids
        # a TP+PP coalesced/all-gather deadlock on this toolchain.  Native
        # capture experiments can opt back in with GRAPH_TASK_QUEUE_ENABLE=1.
        default_task_queue=0
    fi
    NPU_FALLBACK_LIST=${GRAPH_NPU_INDUCTOR_FALLBACK_LIST-$default_fallback_list}
    TASK_QUEUE_MODE=${GRAPH_TASK_QUEUE_ENABLE:-$default_task_queue}
    NPUGRAPH_CAPTURE_MODE=${GRAPH_NPUGRAPH_CAPTURE_ERROR_MODE-}
    NPUGRAPH_UNSAFE_OPS=${GRAPH_NPUGRAPH_UNSAFE_OPS-$default_unsafe_ops}
    NPUGRAPH_SKIP_ALL=${GRAPH_NPUGRAPH_SKIP_ALL:-$default_npugraph_skip_all}
    REGISTER_COMPLEX_DTENSOR_STRATEGY=${GRAPH_REGISTER_COMPLEX_DTENSOR_STRATEGY:-1}
    PIPELINE_META_USE_BATCH=${GRAPH_PIPELINE_META_USE_BATCH:-0}
    SAFE_EMPTY_GROUPED_MM=${GRAPH_SAFE_EMPTY_GROUPED_MM:-1}
    SAFE_ZERO_NUMEL_TRITON=${GRAPH_SAFE_ZERO_NUMEL_TRITON:-1}
    COMPILE_THREADS=${GRAPH_COMPILE_THREADS:-1}
    ASCEND_LAUNCH_BLOCKING_VALUE=${GRAPH_ASCEND_LAUNCH_BLOCKING:-0}
    LOG_RANK_VALUE=${GRAPH_LOG_RANK:-0}
    graph_env_require_boolean GRAPH_TASK_QUEUE_ENABLE "$TASK_QUEUE_MODE"
    graph_env_require_boolean GRAPH_REGISTER_COMPLEX_DTENSOR_STRATEGY \
        "$REGISTER_COMPLEX_DTENSOR_STRATEGY"
    graph_env_require_boolean GRAPH_PIPELINE_META_USE_BATCH \
        "$PIPELINE_META_USE_BATCH"
    graph_env_require_boolean GRAPH_SAFE_EMPTY_GROUPED_MM \
        "$SAFE_EMPTY_GROUPED_MM"
    graph_env_require_boolean GRAPH_SAFE_ZERO_NUMEL_TRITON \
        "$SAFE_ZERO_NUMEL_TRITON"
    graph_env_require_boolean GRAPH_NPUGRAPH_SKIP_ALL "$NPUGRAPH_SKIP_ALL"
    graph_env_require_boolean GRAPH_ASCEND_LAUNCH_BLOCKING \
        "$ASCEND_LAUNCH_BLOCKING_VALUE"
    if [[ ! "$COMPILE_THREADS" =~ ^[1-9][0-9]*$ ]]; then
        echo "GRAPH_COMPILE_THREADS must be a positive integer: $COMPILE_THREADS" >&2
        return 2
    fi
    CACHE_ROOT=${GRAPH_CACHE_ROOT:-$WORKSPACE_ROOT/.cache/torchtitan-test/graph_mode/cann91-torch214-triton321}
    RUN_ROOT=${GRAPH_RUN_ROOT:-$PROJECT_ROOT/graph_debug_runs}
    INDUCTOR_CACHE_DIR=${GRAPH_INDUCTOR_CACHE_DIR:-$CACHE_ROOT/inductor}
    TRITON_CACHE_DIR=${GRAPH_TRITON_CACHE_DIR:-$CACHE_ROOT/triton}
    COMPILE_DEBUG_DIR=${GRAPH_COMPILE_DEBUG_DIR:-$CACHE_ROOT/torch_compile_debug}
    COMPILE_DEBUG=${TORCH_COMPILE_DEBUG:-0}

    local required_path
    for required_path in \
        "$CANN_ROOT/set_env.sh" \
        "$CONDA_ENV/bin/python"; do
        if [[ ! -e "$required_path" ]]; then
            echo "Missing required graph environment path: $required_path" >&2
            return 1
        fi
    done

    if [[ ${TORCHTITAN_GRAPH_ENV_READY:-} != "$GRAPH_BACKEND" ]]; then
        local -a passthrough_env=()
        local variable_name
        while IFS= read -r variable_name; do
            case "$variable_name" in
                GLM5_*|MASTER_ADDR|MASTER_PORT|NNODES|NODE_RANK|MODULE|CONFIG|NGPU|LOG_RANK|\
                HTTP_PROXY|HTTPS_PROXY|NO_PROXY|http_proxy|https_proxy|no_proxy|\
                TORCH_LOGS|TORCHDYNAMO_VERBOSE)
                    passthrough_env+=("$variable_name=${!variable_name}")
                    ;;
            esac
        done < <(compgen -e)
        exec env -i \
            "${passthrough_env[@]}" \
            HOME=/root \
            USER=root \
            LANG=C.UTF-8 \
            ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES" \
            HCCL_NPU_SOCKET_PORT_RANGE="$HCCL_NPU_PORT_RANGE" \
            HCCL_IF_BASE_PORT="$HCCL_HOST_BASE_PORT" \
            HCCL_CONNECT_TIMEOUT="$HCCL_CONNECT_TIMEOUT_SECONDS" \
            TORCHTITAN_COMM_INIT_TIMEOUT_SECONDS="$COMM_INIT_TIMEOUT_SECONDS" \
            TORCHINDUCTOR_NPU_BACKEND="$NPU_INDUCTOR_BACKEND" \
            NPU_INDUCTOR_FALLBACK_LIST="$NPU_FALLBACK_LIST" \
            TASK_QUEUE_ENABLE="$TASK_QUEUE_MODE" \
            TORCHTITAN_TASK_QUEUE_ENABLE="$TASK_QUEUE_MODE" \
            TORCHTITAN_NPUGRAPH_CAPTURE_ERROR_MODE="$NPUGRAPH_CAPTURE_MODE" \
            TORCHTITAN_NPUGRAPH_UNSAFE_OPS="$NPUGRAPH_UNSAFE_OPS" \
            TORCHTITAN_NPUGRAPH_SKIP_ALL="$NPUGRAPH_SKIP_ALL" \
            TORCHTITAN_REGISTER_COMPLEX_DTENSOR_STRATEGY="$REGISTER_COMPLEX_DTENSOR_STRATEGY" \
            TORCHTITAN_PIPELINE_META_USE_BATCH="$PIPELINE_META_USE_BATCH" \
            TORCHTITAN_SAFE_EMPTY_GROUPED_MM="$SAFE_EMPTY_GROUPED_MM" \
            TORCHTITAN_SAFE_ZERO_NUMEL_TRITON="$SAFE_ZERO_NUMEL_TRITON" \
            TORCHINDUCTOR_COMPILE_THREADS="$COMPILE_THREADS" \
            ASCEND_LAUNCH_BLOCKING="$ASCEND_LAUNCH_BLOCKING_VALUE" \
            LOG_RANK="$LOG_RANK_VALUE" \
            GRAPH_BACKEND="$GRAPH_BACKEND" \
            GRAPH_CANN_ROOT="$CANN_ROOT" \
            GRAPH_CONDA_ENV="$CONDA_ENV" \
            GRAPH_ATB_ROOT="$ATB_ROOT" \
            GRAPH_CACHE_ROOT="$CACHE_ROOT" \
            GRAPH_INDUCTOR_CACHE_DIR="$INDUCTOR_CACHE_DIR" \
            GRAPH_TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
            GRAPH_COMPILE_DEBUG_DIR="$COMPILE_DEBUG_DIR" \
            GRAPH_RUN_ROOT="$RUN_ROOT" \
            GRAPH_HCCL_NPU_SOCKET_PORT_RANGE="$HCCL_NPU_PORT_RANGE" \
            GRAPH_HCCL_IF_BASE_PORT="$HCCL_HOST_BASE_PORT" \
            GRAPH_HCCL_CONNECT_TIMEOUT="$HCCL_CONNECT_TIMEOUT_SECONDS" \
            GRAPH_COMM_INIT_TIMEOUT_SECONDS="$COMM_INIT_TIMEOUT_SECONDS" \
            GRAPH_TORCHINDUCTOR_NPU_BACKEND="$NPU_INDUCTOR_BACKEND" \
            GRAPH_NPU_INDUCTOR_FALLBACK_LIST="$NPU_FALLBACK_LIST" \
            GRAPH_TASK_QUEUE_ENABLE="$TASK_QUEUE_MODE" \
            GRAPH_NPUGRAPH_CAPTURE_ERROR_MODE="$NPUGRAPH_CAPTURE_MODE" \
            GRAPH_NPUGRAPH_UNSAFE_OPS="$NPUGRAPH_UNSAFE_OPS" \
            GRAPH_NPUGRAPH_SKIP_ALL="$NPUGRAPH_SKIP_ALL" \
            GRAPH_REGISTER_COMPLEX_DTENSOR_STRATEGY="$REGISTER_COMPLEX_DTENSOR_STRATEGY" \
            GRAPH_PIPELINE_META_USE_BATCH="$PIPELINE_META_USE_BATCH" \
            GRAPH_SAFE_EMPTY_GROUPED_MM="$SAFE_EMPTY_GROUPED_MM" \
            GRAPH_SAFE_ZERO_NUMEL_TRITON="$SAFE_ZERO_NUMEL_TRITON" \
            GRAPH_COMPILE_THREADS="$COMPILE_THREADS" \
            GRAPH_ASCEND_LAUNCH_BLOCKING="$ASCEND_LAUNCH_BLOCKING_VALUE" \
            GRAPH_LOG_RANK="$LOG_RANK_VALUE" \
            TORCH_COMPILE_DEBUG="$COMPILE_DEBUG" \
            PYTHONFAULTHANDLER=1 \
            PYTHONUNBUFFERED=1 \
            TORCHTITAN_GRAPH_ENV_READY="$GRAPH_BACKEND" \
            PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
            /bin/bash --noprofile --norc "$reexec_script" "$@"
    fi

    set +u
    # shellcheck disable=SC1090
    source "$CANN_ROOT/set_env.sh" >/dev/null
    set -u
    export PATH="$CONDA_ENV/bin:$ATB_ROOT/bin:$PATH"
    export LD_LIBRARY_PATH="$ATB_ROOT/lib:${LD_LIBRARY_PATH:-}"
    export TORCHTITAN_DEVICE=npu
    export HCCL_NPU_SOCKET_PORT_RANGE="$HCCL_NPU_PORT_RANGE"
    export HCCL_IF_BASE_PORT="$HCCL_HOST_BASE_PORT"
    export HCCL_CONNECT_TIMEOUT="$HCCL_CONNECT_TIMEOUT_SECONDS"
    export TORCHTITAN_COMM_INIT_TIMEOUT_SECONDS="$COMM_INIT_TIMEOUT_SECONDS"
    export TORCHINDUCTOR_NPU_BACKEND="$NPU_INDUCTOR_BACKEND"
    export NPU_INDUCTOR_FALLBACK_LIST="$NPU_FALLBACK_LIST"
    export TASK_QUEUE_ENABLE="$TASK_QUEUE_MODE"
    export TORCHTITAN_TASK_QUEUE_ENABLE="$TASK_QUEUE_MODE"
    export TORCHTITAN_NPUGRAPH_CAPTURE_ERROR_MODE="$NPUGRAPH_CAPTURE_MODE"
    export TORCHTITAN_NPUGRAPH_UNSAFE_OPS="$NPUGRAPH_UNSAFE_OPS"
    export TORCHTITAN_NPUGRAPH_SKIP_ALL="$NPUGRAPH_SKIP_ALL"
    export TORCHTITAN_REGISTER_COMPLEX_DTENSOR_STRATEGY="$REGISTER_COMPLEX_DTENSOR_STRATEGY"
    export TORCHTITAN_PIPELINE_META_USE_BATCH="$PIPELINE_META_USE_BATCH"
    export TORCHTITAN_SAFE_EMPTY_GROUPED_MM="$SAFE_EMPTY_GROUPED_MM"
    export TORCHTITAN_SAFE_ZERO_NUMEL_TRITON="$SAFE_ZERO_NUMEL_TRITON"
    export TORCHINDUCTOR_COMPILE_THREADS="$COMPILE_THREADS"
    export ASCEND_LAUNCH_BLOCKING="$ASCEND_LAUNCH_BLOCKING_VALUE"
    export LOG_RANK="$LOG_RANK_VALUE"
    export TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_CACHE_DIR"
    export TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
    export TORCH_COMPILE_DEBUG_DIR="$COMPILE_DEBUG_DIR"
    export PYTHONFAULTHANDLER=1
    export PYTHONUNBUFFERED=1
    unset CUDA_VISIBLE_DEVICES
    mkdir -p "$INDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$COMPILE_DEBUG_DIR" "$RUN_ROOT"
}

graph_env_markdown() {
    cat <<EOF
- Backend profile: \`$GRAPH_BACKEND\`
- Physical NPUs: \`$VISIBLE_DEVICES\`
- CANN: \`$CANN_ROOT\`
- ASCEND_HOME_PATH: \`${ASCEND_HOME_PATH:-unset}\`
- ASCEND_TOOLKIT_HOME: \`${ASCEND_TOOLKIT_HOME:-unset}\`
- ASCEND_AICPU_PATH: \`${ASCEND_AICPU_PATH:-unset}\`
- ASCEND_OPP_PATH: \`${ASCEND_OPP_PATH:-unset}\`
- TOOLCHAIN_HOME: \`${TOOLCHAIN_HOME:-unset}\`
- Conda: \`$CONDA_ENV\`
- ATB: \`$ATB_ROOT\`
- HCCL NPU socket ports: \`$HCCL_NPU_PORT_RANGE\`
- HCCL host base port: \`$HCCL_HOST_BASE_PORT\`
- HCCL connect timeout: \`$HCCL_CONNECT_TIMEOUT_SECONDS seconds\`
- Process-group timeout: \`$COMM_INIT_TIMEOUT_SECONDS seconds\`
- NPU Inductor codegen: \`$NPU_INDUCTOR_BACKEND\`
- User Inductor fallbacks: \`${NPU_FALLBACK_LIST:-none}\`
- Task queue mode: \`$TASK_QUEUE_MODE\`
- NPUGraph capture error mode: \`${NPUGRAPH_CAPTURE_MODE:-unchanged}\`
- NPUGraph unsafe ops: \`${NPUGRAPH_UNSAFE_OPS:-none}\`
- NPUGraph replay disabled: \`$NPUGRAPH_SKIP_ALL\`
- aten.complex DTensor strategy: \`$REGISTER_COMPLEX_DTENSOR_STRATEGY\`
- Pipeline metadata batched P2P: \`$PIPELINE_META_USE_BATCH\`
- Empty grouped-mm guard: \`$SAFE_EMPTY_GROUPED_MM\`
- Zero-numel Triton guard: \`$SAFE_ZERO_NUMEL_TRITON\`
- Inductor compile threads per rank: \`$COMPILE_THREADS\`
- CANN synchronous launch: \`$ASCEND_LAUNCH_BLOCKING_VALUE\`
- Inductor cache: \`$INDUCTOR_CACHE_DIR\`
- Triton cache: \`$TRITON_CACHE_DIR\`
- Compile debug: \`$COMPILE_DEBUG_DIR\`
EOF
}
