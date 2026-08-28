#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -euo pipefail

backend="${1:-inductor}"
action="${2:-all}"
topology="${TOPOLOGY:-all}"
repeat="${REPEAT:-}"
compiler_diagnostics="${COMPILER_DIAGNOSTICS:-0}"

case "${backend}" in
  inductor|npugraphs) ;;
  *)
    echo "backend must be inductor or npugraphs" >&2
    exit 2
    ;;
esac

case "${action}" in
  data|reference|candidate|compare|all) ;;
  *)
    echo "action must be data, reference, candidate, compare, or all" >&2
    exit 2
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
graph_launcher="${repo_root}/tests/glm5_2_graph_debug/run_graph_mode.sh"

common_args=(
  --topology "${topology}"
  --objectives precision
  --reference-graph eager
  --candidate-graph "${backend}"
  --profiler-preset off
)

if [[ -n "${repeat}" ]]; then
  common_args+=(--repeat "${repeat}")
fi
if [[ "${compiler_diagnostics}" == "1" ]]; then
  common_args+=(--compiler-diagnostics)
elif [[ "${compiler_diagnostics}" != "0" ]]; then
  echo "COMPILER_DIAGNOSTICS must be 0 or 1" >&2
  exit 2
fi

run_stage() {
  local stage="$1"
  case "${stage}" in
    data)
      "${graph_launcher}" "${backend}" combination \
        --data --data-device npu "${common_args[@]}"
      ;;
    reference|candidate)
      "${graph_launcher}" "${backend}" combination \
        --capture "${stage}" "${common_args[@]}"
      ;;
    compare)
      "${graph_launcher}" "${backend}" combination \
        --compare --require-all "${common_args[@]}"
      ;;
  esac
}

cd "${repo_root}"
if [[ "${action}" == "all" ]]; then
  for stage in data reference candidate compare; do
    run_stage "${stage}"
  done
else
  run_stage "${action}"
fi
