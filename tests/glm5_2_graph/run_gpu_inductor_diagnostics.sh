#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_gpu_inductor_diagnostics.sh [minimal|trace|internal|attention|all] [fp32|bf16|all]

  minimal  Run minimal SiLU/multiply and linear/residual eager/Inductor checks
           in two independent cold caches.
  trace    Capture eager r1/r2 and independently cold-compiled Inductor r1/r2
           Block 0 forward/backward/parameter-gradient fingerprints.
  internal Capture the ordered Block 0 internal forward stages for five steps
           and report the first eager/Inductor divergence at each step.
  attention Run a one-step FP32 trace inside absorbed SparseMLA attention.
  all      Run minimal, Block trace, and internal trace (not attention).

CUDA_VISIBLE_DEVICES defaults to 0. Set GPU_DIAG_RUN_ID to choose a stable
output identity; otherwise a UTC timestamp is used.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

action=$1
precision=${2:-all}
case "$action" in
  minimal|trace|internal|attention|all) ;;
  *) usage >&2; exit 2 ;;
esac
case "$precision" in
  fp32|bf16) precisions=("$precision") ;;
  all) precisions=(fp32 bf16) ;;
  *) usage >&2; exit 2 ;;
esac
if [[ "$action" == "attention" && "$precision" != "fp32" ]]; then
  echo "attention localization requires precision=fp32" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
unset ASCEND_RT_VISIBLE_DEVICES

python_bin=${PYTHON_BIN:-python}
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python executable not found: $python_bin (override with PYTHON_BIN)" >&2
  exit 1
fi
"$python_bin" -c 'import torch,triton; assert torch.cuda.is_available(), "CUDA is unavailable"; print(f"torch={torch.__version__} triton={triton.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")'

run_id=${GPU_DIAG_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
output_root="$repo_root/graph_debug_runs/gpu-inductor/$run_id"
mkdir -p "$output_root"
echo "Diagnostic output: $output_root"

run_minimal() {
  for current_precision in "${precisions[@]}"; do
    for repeat in 1 2; do
      cache_root="$output_root/minimal-$current_precision-r$repeat-cache"
      result="$output_root/minimal-$current_precision-r$repeat.jsonl"
      echo "==> minimal $current_precision cold repeat $repeat"
      TORCHINDUCTOR_CACHE_DIR="$cache_root/inductor" \
      TRITON_CACHE_DIR="$cache_root/triton" \
      "$python_bin" tests/glm5_2_graph/gpu_minimal_inductor_regressions.py \
        --dtype "$current_precision" --output "$result"
    done
    "$python_bin" tests/glm5_2_graph/compare_gpu_minimal_results.py \
      --first "$output_root/minimal-$current_precision-r1.jsonl" \
      --second "$output_root/minimal-$current_precision-r2.jsonl" \
      --output "$output_root/minimal-$current_precision-comparison.json"
  done
}

common_args=(
  --topology single
  --objectives precision
  --reference-graph eager
  --candidate-graph inductor
  --compiler-diagnostics
)

capture_trace() {
  role=$1
  repeat=$2
  current_precision=$3
  trace_name=$4
  cache_name=${5:-}
  trace_path="$output_root/$trace_name.jsonl"
  command=(
    "$python_bin" tests/glm5_2_graph/gpu_block_trace_probe.py
    --capture "$role"
    --repeat "$repeat"
    --precision "$current_precision"
    "${common_args[@]}"
    --force
  )
  if [[ -n "$cache_name" ]]; then
    cache_root="$output_root/$cache_name"
    GLM5_GPU_BLOCK_TRACE_PATH="$trace_path" \
    GLM5_GPU_BLOCK_TRACE_STEPS=1,2,8,9,10 \
    TORCHINDUCTOR_CACHE_DIR="$cache_root/inductor" \
    TRITON_CACHE_DIR="$cache_root/triton" \
    "${command[@]}"
  else
    GLM5_GPU_BLOCK_TRACE_PATH="$trace_path" \
    GLM5_GPU_BLOCK_TRACE_STEPS=1,2,8,9,10 \
    "${command[@]}"
  fi
}

run_trace() {
  for current_precision in "${precisions[@]}"; do
    echo "==> trace $current_precision prepare fixture"
    "$python_bin" tests/glm5_2_graph/gpu_block_trace_probe.py \
      --data --data-device cuda --precision "$current_precision" \
      "${common_args[@]}" --force

    echo "==> trace $current_precision eager repeats"
    capture_trace reference 1 "$current_precision" "$current_precision-eager-r1"
    capture_trace reference 2 "$current_precision" "$current_precision-eager-r2"

    echo "==> trace $current_precision independent cold Inductor repeats"
    capture_trace candidate 1 "$current_precision" \
      "$current_precision-inductor-cold-r1" \
      "$current_precision-inductor-cold-r1-cache"
    capture_trace candidate 2 "$current_precision" \
      "$current_precision-inductor-cold-r2" \
      "$current_precision-inductor-cold-r2-cache"

    echo "==> trace $current_precision compare formal metrics"
    "$python_bin" tests/glm5_2_graph/gpu_block_trace_probe.py \
      --compare --require-all --precision "$current_precision" \
      "${common_args[@]}"

    echo "==> trace $current_precision compare Block 0 fingerprints"
    "$python_bin" tests/glm5_2_graph/compare_gpu_block_traces.py \
      --eager-r1 "$output_root/$current_precision-eager-r1.jsonl" \
      --eager-r2 "$output_root/$current_precision-eager-r2.jsonl" \
      --inductor-r1 "$output_root/$current_precision-inductor-cold-r1.jsonl" \
      --inductor-r2 "$output_root/$current_precision-inductor-cold-r2.jsonl" \
      --output "$output_root/$current_precision-block-trace-comparison.json"
  done
}

capture_internal_trace() {
  role=$1
  repeat=$2
  current_precision=$3
  trace_name=$4
  cache_name=${5:-}
  trace_path="$output_root/$trace_name.jsonl"
  command=(
    "$python_bin" tests/glm5_2_graph/gpu_internal_trace_probe.py
    --capture "$role"
    --repeat "$repeat"
    --precision "$current_precision"
    "${common_args[@]}"
    --force
  )
  if [[ -n "$cache_name" ]]; then
    cache_root="$output_root/$cache_name"
    GLM5_GPU_INTERNAL_TRACE_PATH="$trace_path" \
    GLM5_GPU_INTERNAL_TRACE_STEPS=1,2,8,9,10 \
    TORCHINDUCTOR_CACHE_DIR="$cache_root/inductor" \
    TRITON_CACHE_DIR="$cache_root/triton" \
    "${command[@]}"
  else
    GLM5_GPU_INTERNAL_TRACE_PATH="$trace_path" \
    GLM5_GPU_INTERNAL_TRACE_STEPS=1,2,8,9,10 \
    "${command[@]}"
  fi
}

run_internal_trace() {
  for current_precision in "${precisions[@]}"; do
    echo "==> internal trace $current_precision prepare fixture"
    "$python_bin" tests/glm5_2_graph/gpu_internal_trace_probe.py \
      --data --data-device cuda --precision "$current_precision" \
      "${common_args[@]}" --force

    echo "==> internal trace $current_precision eager repeats"
    capture_internal_trace reference 1 "$current_precision" \
      "$current_precision-internal-eager-r1"
    capture_internal_trace reference 2 "$current_precision" \
      "$current_precision-internal-eager-r2"

    echo "==> internal trace $current_precision independent cold Inductor repeats"
    capture_internal_trace candidate 1 "$current_precision" \
      "$current_precision-internal-inductor-cold-r1" \
      "$current_precision-internal-inductor-cold-r1-cache"
    capture_internal_trace candidate 2 "$current_precision" \
      "$current_precision-internal-inductor-cold-r2" \
      "$current_precision-internal-inductor-cold-r2-cache"

    echo "==> internal trace $current_precision compare ordered forward stages"
    "$python_bin" tests/glm5_2_graph/compare_gpu_internal_traces.py \
      --eager-r1 "$output_root/$current_precision-internal-eager-r1.jsonl" \
      --eager-r2 "$output_root/$current_precision-internal-eager-r2.jsonl" \
      --inductor-r1 "$output_root/$current_precision-internal-inductor-cold-r1.jsonl" \
      --inductor-r2 "$output_root/$current_precision-internal-inductor-cold-r2.jsonl" \
      --output "$output_root/$current_precision-internal-trace-comparison.json"
  done
}

capture_attention_trace() {
  role=$1
  repeat=$2
  trace_name=$3
  cache_name=${4:-}
  trace_path="$output_root/$trace_name.jsonl"
  command=(
    "$python_bin" tests/glm5_2_graph/gpu_attention_trace_probe.py
    --capture "$role"
    --repeat "$repeat"
    --precision fp32
    "${common_args[@]}"
    --force
  )
  if [[ -n "$cache_name" ]]; then
    cache_root="$output_root/$cache_name"
    GLM5_GPU_ATTENTION_TRACE_PATH="$trace_path" \
    GLM5_GPU_ATTENTION_TRACE_STEP=1 \
    TORCHINDUCTOR_CACHE_DIR="$cache_root/inductor" \
    TRITON_CACHE_DIR="$cache_root/triton" \
    "${command[@]}"
  else
    GLM5_GPU_ATTENTION_TRACE_PATH="$trace_path" \
    GLM5_GPU_ATTENTION_TRACE_STEP=1 \
    "${command[@]}"
  fi
}

run_attention_trace() {
  echo "==> attention trace fp32 prepare fixture"
  "$python_bin" tests/glm5_2_graph/gpu_attention_trace_probe.py \
    --data --data-device cuda --precision fp32 "${common_args[@]}" --force

  echo "==> attention trace fp32 eager repeats"
  capture_attention_trace reference 1 fp32-attention-eager-r1
  capture_attention_trace reference 2 fp32-attention-eager-r2

  echo "==> attention trace fp32 independent cold Inductor repeats"
  capture_attention_trace candidate 1 fp32-attention-inductor-cold-r1 \
    fp32-attention-inductor-cold-r1-cache
  capture_attention_trace candidate 2 fp32-attention-inductor-cold-r2 \
    fp32-attention-inductor-cold-r2-cache

  echo "==> attention trace fp32 compare ordered stages"
  "$python_bin" tests/glm5_2_graph/compare_gpu_attention_traces.py \
    --eager-r1 "$output_root/fp32-attention-eager-r1.jsonl" \
    --eager-r2 "$output_root/fp32-attention-eager-r2.jsonl" \
    --inductor-r1 "$output_root/fp32-attention-inductor-cold-r1.jsonl" \
    --inductor-r2 "$output_root/fp32-attention-inductor-cold-r2.jsonl" \
    --output "$output_root/fp32-attention-trace-comparison.json"
}

case "$action" in
  minimal) run_minimal ;;
  trace) run_trace ;;
  internal) run_internal_trace ;;
  attention) run_attention_trace ;;
  all) run_minimal; run_trace; run_internal_trace ;;
esac

echo "Diagnostics complete: $output_root"
