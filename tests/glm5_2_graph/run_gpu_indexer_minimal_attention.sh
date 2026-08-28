#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
unset ASCEND_RT_VISIBLE_DEVICES
python_bin=${PYTHON_BIN:-python}
run_id=${GPU_DIAG_RUN_ID:-h20-indexer-minimal-attention-v1}
boundary_run_id="${run_id}-boundary"
output_root="$repo_root/graph_debug_runs/gpu-inductor/$run_id/attention"
entry=tests/glm5_2_graph/gpu_silu_staged_control.py
control=q-rope-k-norm
common_args=(
  --topology single
  --objectives precision
  --reference-graph eager
  --candidate-graph inductor
  --compiler-diagnostics
  --precision bf16
)

"$python_bin" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(f"torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")'
echo "==> verify minimal q-rope + k-norm boundary set"
GPU_DIAG_RUN_ID="$boundary_run_id" \
  tests/glm5_2_graph/run_gpu_indexer_boundary_controls.sh minimal

mkdir -p "$output_root"
echo "==> trace Layer 1 attention with $control"
GLM5_GPU_SILU_STAGE_VARIANT=attention-residual \
GLM5_GPU_SILU_STAGE_LENGTH=attention1 \
"$python_bin" "$entry" --data --data-device cuda \
  "${common_args[@]}" --force

capture() {
  role=$1
  repeat=$2
  trace_name=$3
  cache_name=${4:-}
  environment=(
    GLM5_GPU_SILU_STAGE_VARIANT=attention-residual
    GLM5_GPU_SILU_STAGE_LENGTH=attention1
    GLM5_GPU_INDEXER_CONTROL_VARIANT="$control"
    GLM5_GPU_SHARED_ATTENTION_TRACE_PATH="$output_root/$trace_name.jsonl"
    GLM5_GPU_SHARED_ATTENTION_TRACE_STEP=1
    GLM5_GPU_SHARED_ATTENTION_LAYER=1
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

"$python_bin" tests/glm5_2_graph/compare_gpu_shared_attention_traces.py \
  --eager-r1 "$output_root/eager-r1.jsonl" \
  --eager-r2 "$output_root/eager-r2.jsonl" \
  --inductor-r1 "$output_root/inductor-r1.jsonl" \
  --inductor-r2 "$output_root/inductor-r2.jsonl" \
  --output "$output_root/trace-comparison.json"

echo "Diagnostics complete: $repo_root/graph_debug_runs/gpu-inductor/$run_id"
