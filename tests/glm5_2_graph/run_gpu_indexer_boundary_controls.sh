#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: run_gpu_indexer_boundary_controls.sh [minimal|downstream|upstream|cumulative|all]"
}

selection=${1:-upstream}
case "$selection" in
  minimal) controls=(q-rope-k-norm) ;;
  downstream) controls=(original decomposed-none qk relu weighted masked) ;;
  upstream) controls=(q-proj k-proj k-norm q-rope k-rope final-q final-k weights) ;;
  cumulative) controls=(cum-qk-proj cum-k-norm cum-rope cum-final-qk cum-weights) ;;
  all)
    controls=(
      original decomposed-none qk relu weighted masked
      q-proj k-proj k-norm q-rope k-rope final-q final-k weights
      cum-qk-proj cum-k-norm cum-rope cum-final-qk cum-weights
    )
    ;;
  *) usage >&2; exit 2 ;;
esac

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
unset ASCEND_RT_VISIBLE_DEVICES
python_bin=${PYTHON_BIN:-python}
run_id=${GPU_DIAG_RUN_ID:-h20-indexer-controls-v1}
base_output="$repo_root/graph_debug_runs/gpu-inductor/$run_id"
entry=tests/glm5_2_graph/gpu_silu_staged_control.py
common_args=(
  --topology single
  --objectives precision
  --reference-graph eager
  --candidate-graph inductor
  --compiler-diagnostics
  --precision bf16
)

"$python_bin" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(f"torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")'
echo "Diagnostic output: $base_output"

GLM5_GPU_SILU_STAGE_VARIANT=attention-residual \
GLM5_GPU_SILU_STAGE_LENGTH=step1 \
"$python_bin" "$entry" --data --data-device cuda \
  "${common_args[@]}" --force

for control in "${controls[@]}"; do
  output_root="$base_output/$control"
  mkdir -p "$output_root"
  echo "==> indexer control $control"

  capture() {
    role=$1
    repeat=$2
    trace_name=$3
    cache_name=${4:-}
    index_trace="$output_root/$trace_name-index.jsonl"
    layer_trace="$output_root/$trace_name-layers.jsonl"
    environment=(
      GLM5_GPU_SILU_STAGE_VARIANT=attention-residual
      GLM5_GPU_SILU_STAGE_LENGTH=step1
      GLM5_GPU_INDEXER_CONTROL_VARIANT="$control"
      GLM5_GPU_INDEXER_CONTROL_TRACE_PATH="$index_trace"
      GLM5_GPU_INDEXER_CONTROL_LAYER=1
      GLM5_GPU_INDEXER_CONTROL_STEP=1
      GLM5_GPU_LAYER_BOUNDARY_TRACE_PATH="$layer_trace"
      GLM5_GPU_LAYER_BOUNDARY_TRACE_STEPS=1
    )
    if [[ -n "$cache_name" ]]; then
      environment+=(
        TORCHINDUCTOR_CACHE_DIR="$output_root/$cache_name/inductor"
        TRITON_CACHE_DIR="$output_root/$cache_name/triton"
      )
    fi
    env "${environment[@]}" \
      "$python_bin" "$entry" --capture "$role" --repeat "$repeat" \
      "${common_args[@]}" --force
  }

  capture reference 1 eager-r1
  capture reference 2 eager-r2
  capture candidate 1 inductor-r1 inductor-r1-cache
  capture candidate 2 inductor-r2 inductor-r2-cache

  "$python_bin" tests/glm5_2_graph/compare_gpu_indexer_control_traces.py \
    --eager-r1 "$output_root/eager-r1-index.jsonl" \
    --eager-r2 "$output_root/eager-r2-index.jsonl" \
    --inductor-r1 "$output_root/inductor-r1-index.jsonl" \
    --inductor-r2 "$output_root/inductor-r2-index.jsonl" \
    --output "$output_root/index-comparison.json"
  "$python_bin" tests/glm5_2_graph/compare_gpu_layer_boundary_traces.py \
    --eager-r1 "$output_root/eager-r1-layers.jsonl" \
    --eager-r2 "$output_root/eager-r2-layers.jsonl" \
    --inductor-r1 "$output_root/inductor-r1-layers.jsonl" \
    --inductor-r2 "$output_root/inductor-r2-layers.jsonl" \
    --output "$output_root/layer-comparison.json"
done

echo "Diagnostics complete: $base_output"
