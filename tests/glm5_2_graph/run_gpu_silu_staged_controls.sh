#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_gpu_silu_staged_controls.sh [step1|backward|moe|probe] VARIANT

  step1  One training step with Block 0 forward/backward/gradient tracing.
  backward  One step with all-layer forward/backward boundary tracing.
  moe      One step with ordered Layer 1 attention/router/expert tracing.
  probe  Ten training steps plus the Step 1 trace.
  VARIANT is one named variant, matrix (FFN), or frontier-matrix.
EOF
}

if [[ $# -ne 2 ]]; then
  usage >&2
  exit 2
fi
length=$1
selection=$2
case "$length" in step1|backward|moe|probe) ;; *) usage >&2; exit 2 ;; esac
case "$selection" in
  silu-product|silu-product-down|all|ffn-input|attention-output|attention-residual|block-frontier)
    variants=("$selection")
    ;;
  matrix) variants=(silu-product silu-product-down all) ;;
  frontier-matrix)
    variants=(ffn-input attention-output attention-residual block-frontier)
    ;;
  *) usage >&2; exit 2 ;;
esac

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
unset ASCEND_RT_VISIBLE_DEVICES
python_bin=${PYTHON_BIN:-python}
run_id=${GPU_DIAG_RUN_ID:-h20-silu-staged-v1}
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

run_variant() {
  variant=$1
  output_root="$base_output/$length-$variant"
  mkdir -p "$output_root"
  echo "==> $length $variant: prepare fixture"
  GLM5_GPU_SILU_STAGE_VARIANT="$variant" \
  GLM5_GPU_SILU_STAGE_LENGTH="$length" \
  "$python_bin" "$entry" --data --data-device cuda \
    "${common_args[@]}" --force

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
    if [[ "$length" == "backward" ]]; then
      trace_environment=(
        GLM5_GPU_LAYER_BOUNDARY_TRACE_PATH="$trace_path"
        GLM5_GPU_LAYER_BOUNDARY_TRACE_STEPS=1
      )
    elif [[ "$length" == "moe" ]]; then
      trace_environment=(
        GLM5_GPU_MOE_LAYER_TRACE_PATH="$trace_path"
        GLM5_GPU_MOE_LAYER_TRACE_STEP=1
        GLM5_GPU_MOE_LAYER=1
      )
    else
      trace_environment=(
        GLM5_GPU_BLOCK_TRACE_PATH="$trace_path"
        GLM5_GPU_BLOCK_TRACE_STEPS=1
      )
    fi
    capture_environment=(
      GLM5_GPU_SILU_STAGE_VARIANT="$variant"
      GLM5_GPU_SILU_STAGE_LENGTH="$length"
    )
    if [[ -n "$cache_name" ]]; then
      cache_root="$output_root/$cache_name"
      capture_environment+=(
        TORCHINDUCTOR_CACHE_DIR="$cache_root/inductor"
        TRITON_CACHE_DIR="$cache_root/triton"
      )
      env "${capture_environment[@]}" "${trace_environment[@]}" \
      "${command[@]}"
    else
      env "${capture_environment[@]}" "${trace_environment[@]}" \
      "${command[@]}"
    fi
  }

  echo "==> $length $variant: eager repeats"
  capture reference 1 eager-r1
  capture reference 2 eager-r2
  echo "==> $length $variant: independent cold Inductor repeats"
  capture candidate 1 inductor-r1 inductor-r1-cache
  capture candidate 2 inductor-r2 inductor-r2-cache

  echo "==> $length $variant: compare formal metrics"
  GLM5_GPU_SILU_STAGE_VARIANT="$variant" \
  GLM5_GPU_SILU_STAGE_LENGTH="$length" \
  "$python_bin" "$entry" --compare --require-all "${common_args[@]}"

  if [[ "$length" == "backward" ]]; then
    comparator=tests/glm5_2_graph/compare_gpu_layer_boundary_traces.py
    comparison_label="ordered all-layer boundaries and gradients"
  elif [[ "$length" == "moe" ]]; then
    comparator=tests/glm5_2_graph/compare_gpu_moe_layer_traces.py
    comparison_label="ordered Layer 1 attention/router/expert stages"
  else
    comparator=tests/glm5_2_graph/compare_gpu_silu_control_traces.py
    comparison_label="Step 1 Block/gradients"
  fi
  echo "==> $length $variant: compare $comparison_label"
  "$python_bin" "$comparator" \
    --eager-r1 "$output_root/eager-r1.jsonl" \
    --eager-r2 "$output_root/eager-r2.jsonl" \
    --inductor-r1 "$output_root/inductor-r1.jsonl" \
    --inductor-r2 "$output_root/inductor-r2.jsonl" \
    --output "$output_root/trace-comparison.json"
}

for variant in "${variants[@]}"; do
  run_variant "$variant"
done
echo "Diagnostics complete: $base_output"
