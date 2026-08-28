#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
unset ASCEND_RT_VISIBLE_DEVICES

python_bin=${PYTHON_BIN:-python}
run_id=${GPU_DIAG_RUN_ID:-h20-silu-symmetric-control-v1}
output_root="$repo_root/graph_debug_runs/gpu-inductor/$run_id"
mkdir -p "$output_root"
entry=tests/glm5_2_graph/gpu_silu_symmetric_probe.py
common_args=(
  --topology single
  --objectives precision
  --reference-graph eager
  --candidate-graph inductor
  --compiler-diagnostics
  --precision bf16
)

"$python_bin" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(f"torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")'
echo "Diagnostic output: $output_root"

echo "==> symmetric SiLU control: prepare fixture"
"$python_bin" "$entry" --data --data-device cuda "${common_args[@]}" --force

capture() {
  role=$1
  repeat=$2
  trace_name=$3
  cache_name=${4:-}
  trace_path="$output_root/$trace_name.jsonl"
  command=(
    "$python_bin" "$entry" --capture "$role" --repeat "$repeat"
    "${common_args[@]}" --force
  )
  if [[ -n "$cache_name" ]]; then
    cache_root="$output_root/$cache_name"
    GLM5_GPU_BLOCK_TRACE_PATH="$trace_path" \
    GLM5_GPU_BLOCK_TRACE_STEPS=1 \
    TORCHINDUCTOR_CACHE_DIR="$cache_root/inductor" \
    TRITON_CACHE_DIR="$cache_root/triton" \
    "${command[@]}"
  else
    GLM5_GPU_BLOCK_TRACE_PATH="$trace_path" \
    GLM5_GPU_BLOCK_TRACE_STEPS=1 \
    "${command[@]}"
  fi
}

echo "==> symmetric SiLU control: eager repeats"
capture reference 1 bf16-silu-control-eager-r1
capture reference 2 bf16-silu-control-eager-r2

echo "==> symmetric SiLU control: independent cold Inductor repeats"
capture candidate 1 bf16-silu-control-inductor-cold-r1 \
  bf16-silu-control-inductor-cold-r1-cache
capture candidate 2 bf16-silu-control-inductor-cold-r2 \
  bf16-silu-control-inductor-cold-r2-cache

echo "==> symmetric SiLU control: compare formal metrics"
"$python_bin" "$entry" --compare --require-all "${common_args[@]}"

echo "==> symmetric SiLU control: compare Step 1 gradients"
"$python_bin" tests/glm5_2_graph/compare_gpu_silu_control_traces.py \
  --eager-r1 "$output_root/bf16-silu-control-eager-r1.jsonl" \
  --eager-r2 "$output_root/bf16-silu-control-eager-r2.jsonl" \
  --inductor-r1 "$output_root/bf16-silu-control-inductor-cold-r1.jsonl" \
  --inductor-r2 "$output_root/bf16-silu-control-inductor-cold-r2.jsonl" \
  --output "$output_root/bf16-silu-control-trace-comparison.json"

echo "Diagnostics complete: $output_root"
