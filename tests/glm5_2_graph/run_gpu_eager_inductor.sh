#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_gpu_eager_inductor.sh [probe|benchmark] [fp32|bf16|all] [--force]

  probe      10 steps: verifies that eager and Inductor both run.
  benchmark  1000 steps: runs the formal precision comparison.

CUDA_VISIBLE_DEVICES defaults to 0. Captures are reused when complete. Fixture
generation requires a new output directory; use --force to replace an existing
fixture and all requested captures intentionally.
EOF
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
  usage >&2
  exit 2
fi

mode=$1
precision=${2:-all}
force_arg=()
if [[ ${3:-} == "--force" || ${2:-} == "--force" ]]; then
  force_arg=(--force)
  if [[ ${2:-} == "--force" ]]; then
    precision=all
  fi
elif [[ $# -eq 3 ]]; then
  usage >&2
  exit 2
fi

case "$mode" in
  probe)
    entry=tests/glm5_2_graph/gpu_compile_probe.py
    ;;
  benchmark)
    entry=tests/glm5_2_graph/gpu_precision_benchmark.py
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

case "$precision" in
  fp32|bf16)
    precisions=("$precision")
    ;;
  all)
    precisions=(fp32 bf16)
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
unset ASCEND_RT_VISIBLE_DEVICES

python_bin=${PYTHON_BIN:-python}
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python executable not found: $python_bin (override with PYTHON_BIN)" >&2
  exit 1
fi

"$python_bin" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(f"torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")'

common_args=(
  --topology single
  --objectives precision
  --reference-graph eager
  --candidate-graph inductor
  --compiler-diagnostics
)

for current_precision in "${precisions[@]}"; do
  echo "==> ${mode}: ${current_precision}: prepare fixture"
  "$python_bin" "$entry" --data --data-device cuda \
    --precision "$current_precision" "${common_args[@]}" "${force_arg[@]}"

  echo "==> ${mode}: ${current_precision}: capture eager reference (2 repeats)"
  "$python_bin" "$entry" --capture reference \
    --precision "$current_precision" "${common_args[@]}" "${force_arg[@]}"

  echo "==> ${mode}: ${current_precision}: capture Inductor candidate (2 repeats)"
  "$python_bin" "$entry" --capture candidate \
    --precision "$current_precision" "${common_args[@]}" "${force_arg[@]}"

  echo "==> ${mode}: ${current_precision}: compare"
  "$python_bin" "$entry" --compare --require-all \
    --precision "$current_precision" "${common_args[@]}"
done
