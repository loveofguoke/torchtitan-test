#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=graph_env_common.sh
source "$SCRIPT_DIR/graph_env_common.sh"

usage() {
    cat <<'EOF'
Usage: run_graph_mode.sh BACKEND TARGET [existing target arguments]

BACKEND:
  inductor | npugraphs

TARGET:
  train         Existing ./run_train.sh; graph compile flags are injected
  smoke         Existing tests/glm5_2_smoke/train_smoke.py
  compile-probe Existing tests/glm5_2_graph/compile_probe.py
  precision     Existing tests/glm5_2_graph/precision_benchmark.py
  performance   Existing tests/glm5_2_graph/performance_benchmark.py
  combination   Existing tests/glm5_2_combination/combination_benchmark.py
  command       Arbitrary command after --; no graph CLI flags are injected
  env           Print the resolved isolated environment and package versions

Examples:
  run_graph_mode.sh inductor train --training.steps=10
  run_graph_mode.sh inductor smoke --topology single
  run_graph_mode.sh npugraphs smoke --topologies ddp2,fsdp8
  run_graph_mode.sh inductor performance --capture candidate --topology single
  run_graph_mode.sh inductor combination --capture candidate --topology fsdp8
  run_graph_mode.sh inductor command -- ./run_train.sh --compile.enable \
    --compile.components=model --compile.backend=inductor

For train, --compile.enable, --compile.components=model, and the selected
--compile.backend are injected only when missing. For benchmark and combination targets,
--reference-graph=eager and
--candidate-graph=BACKEND are injected only when the caller did not provide
those original options. All other arguments are forwarded unchanged.
EOF
    echo
    graph_env_usage
}

if [[ $# -eq 0 || ${1:-} == -h || ${1:-} == --help ]]; then
    usage
    exit 0
fi
if [[ $# -lt 2 ]]; then
    usage >&2
    exit 2
fi

BACKEND=$1
TARGET=$2
shift 2
if [[ "$BACKEND" != inductor && "$BACKEND" != npugraphs ]]; then
    echo "BACKEND must be inductor or npugraphs: $BACKEND" >&2
    exit 2
fi

if [[ "$TARGET" == smoke ]]; then
    exec "$SCRIPT_DIR/run_smoke_graph.sh" "$BACKEND" "$@"
fi

graph_env_init "$BACKEND" "$0" "$BACKEND" "$TARGET" "$@"
cd "$PROJECT_ROOT"

if [[ "$TARGET" == env ]]; then
    RUN_STAMP=$(date +%Y%m%d-%H%M%S)
    RUN_DIR="$RUN_ROOT/launcher-env-${BACKEND}-${RUN_STAMP}-$$"
    RUNTIME_LOG="$RUN_DIR/logs/runtime.log"
    REPORT_FILE="$RUN_DIR/reports/report.md"
    mkdir -p "$RUN_DIR/logs" "$RUN_DIR/reports"
    {
        echo "Resolved graph environment"
        graph_env_markdown
        "$CONDA_ENV/bin/python" - <<'PY'
import importlib.metadata as metadata

for package in ("torch", "torch-npu", "triton-ascend"):
    print(f"- {package}: `{metadata.version(package)}`")
PY
    } | tee "$RUNTIME_LOG"
    {
        echo "# NPU graph-mode environment check"
        echo
        echo "- Result: **PASSED**"
        echo "- Target: \`env\`"
        echo "- Exit code: 0"
        echo "- Started: $RUN_STAMP"
        graph_env_markdown
        echo "- Runtime log: \`$RUNTIME_LOG\`"
    } >"$REPORT_FILE"
    echo "Graph-mode environment result: PASSED"
    echo "Graph-mode environment report: $REPORT_FILE"
    echo "Graph-mode environment runtime log: $RUNTIME_LOG"
    exit 0
fi

has_option() {
    local wanted=$1
    shift
    local argument
    for argument in "$@"; do
        if [[ "$argument" == "$wanted" || "$argument" == "$wanted="* ]]; then
            return 0
        fi
    done
    return 1
}

option_value() {
    local wanted=$1
    shift
    local expect_value=0
    local argument
    for argument in "$@"; do
        if [[ $expect_value -eq 1 ]]; then
            printf '%s\n' "$argument"
            return 0
        fi
        if [[ "$argument" == "$wanted" ]]; then
            expect_value=1
        elif [[ "$argument" == "$wanted="* ]]; then
            printf '%s\n' "${argument#*=}"
            return 0
        fi
    done
    return 1
}

COMMAND=()
case "$TARGET" in
    train)
        COMMAND=(./run_train.sh)
        if ! has_option --compile.enable "$@"; then
            COMMAND+=(--compile.enable)
        fi
        if ! has_option --compile.components "$@"; then
            COMMAND+=(--compile.components=model)
        fi
        if has_option --compile.backend "$@"; then
            if ! requested_backend=$(option_value --compile.backend "$@"); then
                echo "--compile.backend requires a value" >&2
                exit 2
            fi
            if [[ "$requested_backend" != "$BACKEND" ]]; then
                echo "--compile.backend=$requested_backend conflicts with common backend $BACKEND" >&2
                exit 2
            fi
        else
            COMMAND+=("--compile.backend=$BACKEND")
        fi
        COMMAND+=("$@")
        ;;
    compile-probe|precision|performance|combination)
        case "$TARGET" in
            compile-probe) entry=tests/glm5_2_graph/compile_probe.py ;;
            precision) entry=tests/glm5_2_graph/precision_benchmark.py ;;
            performance) entry=tests/glm5_2_graph/performance_benchmark.py ;;
            combination) entry=tests/glm5_2_combination/combination_benchmark.py ;;
        esac
        COMMAND=("$CONDA_ENV/bin/python" "$entry")
        if ! has_option --reference-graph "$@"; then
            COMMAND+=(--reference-graph eager)
        fi
        if ! has_option --candidate-graph "$@"; then
            COMMAND+=(--candidate-graph "$BACKEND")
        fi
        COMMAND+=("$@")
        ;;
    command)
        if [[ ${1:-} == -- ]]; then
            shift
        fi
        if [[ $# -eq 0 ]]; then
            echo "command target requires a command after --" >&2
            exit 2
        fi
        COMMAND=("$@")
        ;;
    *)
        echo "Unknown TARGET: $TARGET" >&2
        usage >&2
        exit 2
        ;;
esac

RUN_STAMP=$(date +%Y%m%d-%H%M%S)
RUN_DIR="$RUN_ROOT/launcher-${TARGET}-${BACKEND}-${RUN_STAMP}-$$"
RUNTIME_LOG="$RUN_DIR/logs/runtime.log"
REPORT_FILE="$RUN_DIR/reports/report.md"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/reports"

START_EPOCH=$(date +%s)
set +e
"${COMMAND[@]}" 2>&1 | tee "$RUNTIME_LOG"
STATUS=${PIPESTATUS[0]}
set -e
DURATION=$(($(date +%s) - START_EPOCH))
COMPILER_ERROR_COUNT=$(grep -Eic 'BackendCompilerFailed|InductorError|No valid triton configs|NPU graph capture|does not have a sharding strategy' "$RUNTIME_LOG" || true)
RESOURCE_ERROR_COUNT=$(grep -Eic 'EE1023|Too many streams|resource alloc fail|Failed to apply for memory|Inner_Error_Device_Subprocess_Startup_Timeout|out of memory' "$RUNTIME_LOG" || true)
NUMERICAL_ERROR_COUNT=$(grep -Eic 'Loss or gradient norm is not finite|(loss|grad_norm): *-?([0-9]{7,}|nan|inf)' "$RUNTIME_LOG" || true)
RESULT=FAILED
if [[ $STATUS -eq 0 ]]; then
    if [[ "$BACKEND" == npugraphs && "${NPUGRAPH_SKIP_ALL:-0}" == 1 ]]; then
        RESULT=PASSED_AOT_COMPAT
    else
        RESULT=PASSED
    fi
fi
if [[ $STATUS -eq 0 && $NUMERICAL_ERROR_COUNT -gt 0 ]]; then
    RESULT=FAILED_NUMERICS
    STATUS=86
elif [[ $STATUS -ne 0 && $RESOURCE_ERROR_COUNT -gt 0 ]]; then
    RESULT=FAILED_INFRA
fi

{
    echo "# NPU graph-mode launcher run"
    echo
    echo "- Result: **$RESULT**"
    echo "- Target: \`$TARGET\`"
    echo "- Exit code: $STATUS"
    echo "- Duration: $DURATION seconds"
    echo "- Started: $RUN_STAMP"
    graph_env_markdown
    echo "- Runtime log: \`$RUNTIME_LOG\`"
    echo "- Compiler/backend error records: $COMPILER_ERROR_COUNT"
    echo "- Infrastructure resource-error records: $RESOURCE_ERROR_COUNT"
    echo "- Non-finite/exploding-gradient records: $NUMERICAL_ERROR_COUNT"
    echo
    echo "## Command"
    echo
    printf '`'
    printf '%q ' "${COMMAND[@]}"
    echo '`'
} >"$REPORT_FILE"

echo "Graph-mode launcher result: $RESULT"
echo "Graph-mode launcher report: $REPORT_FILE"
echo "Graph-mode launcher runtime log: $RUNTIME_LOG"
exit "$STATUS"
