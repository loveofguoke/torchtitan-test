#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_gpu_silu_materialization.sh [probe|benchmark] [--force]

  probe      10-step BF16 eager versus SiLU-materialized Inductor A/B.
  benchmark  1000-step formal BF16 A/B.

CUDA_VISIBLE_DEVICES defaults to 0. Existing complete captures are reused;
pass --force only when intentionally replacing them.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

mode=$1
force_arg=()
if [[ ${2:-} == "--force" ]]; then
  force_arg=(--force)
elif [[ $# -eq 2 ]]; then
  usage >&2
  exit 2
fi

case "$mode" in
  probe) entry=tests/glm5_2_graph/gpu_silu_materialization_probe.py ;;
  benchmark) entry=tests/glm5_2_graph/gpu_silu_materialization_benchmark.py ;;
  *) usage >&2; exit 2 ;;
esac

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
unset ASCEND_RT_VISIBLE_DEVICES

python_bin=${PYTHON_BIN:-python}
common_args=(
  --topology single
  --objectives precision
  --reference-graph eager
  --candidate-graph inductor
  --compiler-diagnostics
  --precision bf16
)

"$python_bin" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(f"torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")'

echo "==> $mode: prepare shared BF16 fixture"
"$python_bin" "$entry" --data --data-device cuda \
  "${common_args[@]}" "${force_arg[@]}"
echo "==> $mode: capture unmodified eager reference"
"$python_bin" "$entry" --capture reference \
  "${common_args[@]}" "${force_arg[@]}"
echo "==> $mode: capture SiLU-materialized Inductor candidate"
"$python_bin" "$entry" --capture candidate \
  "${common_args[@]}" "${force_arg[@]}"
echo "==> $mode: compare"
"$python_bin" "$entry" --compare --require-all "${common_args[@]}"
