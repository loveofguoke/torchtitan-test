#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=graph_env_common.sh
source "$SCRIPT_DIR/graph_env_common.sh"

usage() {
    cat <<'EOF'
Usage: run_smoke_graph.sh BACKEND [train_smoke.py arguments]

Run the existing GLM-5.2 smoke suite through the common isolated NPU graph
environment. BACKEND is inductor or npugraphs; default: inductor.

Unless --topology/--topologies is supplied, all standard topologies up to
eight NPUs are selected. Per-topology results remain under smoke_runs/ and the
invocation environment report is written under graph_debug_runs/.
EOF
    echo
    graph_env_usage
}

BACKEND=${1:-inductor}
if [[ $# -gt 0 ]]; then
    shift
fi
if [[ "$BACKEND" == -h || "$BACKEND" == --help || "$BACKEND" == help ]]; then
    usage
    exit 0
fi
if [[ "$BACKEND" != inductor && "$BACKEND" != npugraphs ]]; then
    echo "BACKEND must be inductor or npugraphs: $BACKEND" >&2
    exit 2
fi

HAS_TOPOLOGY=0
for argument in "$@"; do
    case "$argument" in
        --topology|--topology=*|--topologies|--topologies=*) HAS_TOPOLOGY=1 ;;
        --device|--device=*|--graph|--graph=*)
            echo "$argument is controlled by this wrapper" >&2
            exit 2
            ;;
    esac
done

graph_env_init "$BACKEND" "$0" "$BACKEND" "$@"
if [[ ! -f "$PROJECT_ROOT/tests/glm5_2_smoke/train_smoke.py" ]]; then
    echo "Missing smoke runner under $PROJECT_ROOT" >&2
    exit 1
fi
cd "$PROJECT_ROOT"

RUN_STAMP=$(date +%Y%m%d-%H%M%S)
RUN_DIR="$RUN_ROOT/smoke-suite-${BACKEND}-${RUN_STAMP}-$$"
RUNTIME_LOG="$RUN_DIR/logs/runtime.log"
REPORT_FILE="$RUN_DIR/reports/report.md"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/reports"

SMOKE_ARGS=(--device npu --graph "$BACKEND")
if [[ $HAS_TOPOLOGY -eq 0 ]]; then
    SMOKE_ARGS+=(--topology all)
fi
SMOKE_ARGS+=("$@")

START_EPOCH=$(date +%s)
set +e
"$CONDA_ENV/bin/python" tests/glm5_2_smoke/train_smoke.py \
    "${SMOKE_ARGS[@]}" 2>&1 | tee "$RUNTIME_LOG"
STATUS=${PIPESTATUS[0]}
set -e
DURATION=$(($(date +%s) - START_EPOCH))

TORCH_VERSION=$(python -c 'import importlib.metadata as m; print(m.version("torch"))')
TORCH_NPU_VERSION=$(python -c 'import importlib.metadata as m; print(m.version("torch-npu"))')
TRITON_ASCEND_VERSION=$(python -c 'import importlib.metadata as m; print(m.version("triton-ascend"))')
STARTED_COUNT=$(grep -c '^Starting smoke topology:' "$RUNTIME_LOG" || true)
PASSED_COUNT=$(grep -c '^Passed smoke topology ' "$RUNTIME_LOG" || true)
SKIPPED_COUNT=$(grep -c '^Skip completed topology ' "$RUNTIME_LOG" || true)
COMPILER_ERROR_COUNT=$(grep -Eic 'BackendCompilerFailed|InductorError|No valid triton configs|ACL stream synchronize failed|NPU graph capture|does not have a sharding strategy|Tried to allocate more than 1EB|EOFError' "$RUNTIME_LOG" || true)
RESOURCE_ERROR_COUNT=$(grep -Eic 'EE1023|Too many streams|resource alloc fail|Failed to apply for memory|Inner_Error_Device_Subprocess_Startup_Timeout|out of memory' "$RUNTIME_LOG" || true)
NPUGRAPH_SKIP_COUNT=$(grep -Ec 'NPUGraph: skipped' "$RUNTIME_LOG" || true)
NUMERICAL_ERROR_COUNT=$(grep -Eic 'Loss or gradient norm is not finite|(loss|grad_norm): *-?([0-9]{7,}|nan|inf)' "$RUNTIME_LOG" || true)

RESULT=FAILED
if [[ $STATUS -eq 0 && $NUMERICAL_ERROR_COUNT -eq 0 ]]; then
    RESULT=PASSED
elif [[ $STATUS -eq 0 ]]; then
    RESULT=FAILED_NUMERICS
    STATUS=86
elif [[ $RESOURCE_ERROR_COUNT -gt 0 ]]; then
    RESULT=FAILED_INFRA
fi

{
    echo "# NPU smoke graph-suite run"
    echo
    echo "- Result: **$RESULT**"
    echo "- Exit code: $STATUS"
    echo "- Duration: $DURATION seconds"
    echo "- Started: $RUN_STAMP"
    graph_env_markdown
    echo "- Log rank: \`$LOG_RANK_VALUE\`"
    echo "- torch: \`$TORCH_VERSION\`"
    echo "- torch_npu: \`$TORCH_NPU_VERSION\`"
    echo "- triton_ascend: \`$TRITON_ASCEND_VERSION\`"
    echo "- Wrapper runtime log: \`$RUNTIME_LOG\`"
    echo "- NPUGraph capture skips: $NPUGRAPH_SKIP_COUNT"
    echo "- Topologies started in this invocation: $STARTED_COUNT"
    echo "- Topologies passed in this invocation: $PASSED_COUNT"
    echo "- Previously completed topologies skipped: $SKIPPED_COUNT"
    echo "- Compiler/backend error records: $COMPILER_ERROR_COUNT"
    echo "- Infrastructure resource-error records: $RESOURCE_ERROR_COUNT"
    echo "- Non-finite/exploding-gradient records: $NUMERICAL_ERROR_COUNT"
    echo
    echo "The existing smoke runner owns per-topology logs, trainer output, and"
    echo "manifests under \`smoke_runs/\`. This wrapper adds the common CANN 9.1"
    echo "environment and an invocation-level report without copying intermediates."
    echo
    echo "## Command"
    echo
    printf '`python tests/glm5_2_smoke/train_smoke.py'
    printf ' %q' "${SMOKE_ARGS[@]}"
    echo '`'
} >"$REPORT_FILE"

echo "Smoke graph-suite result: $RESULT"
echo "Smoke graph-suite report: $REPORT_FILE"
echo "Smoke graph-suite runtime log: $RUNTIME_LOG"
exit "$STATUS"
