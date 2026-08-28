# GPU eager/Inductor precision runbook

## Scope

This runbook compares GLM-5.2 eager training with model-block
`torch.compile(backend="inductor")` on one CUDA GPU. Both sides use the same
step-0 checkpoint, fixed token plan, seed, optimizer configuration, and two
independent repeats. Whole-step CUDA Graph is disabled.

The 10-step probe is only a run-through gate. The 1000-step benchmark is the
minimum formal precision run.

## Environment preflight

Use the same GPU environment that previously ran the distributed baseline, and
run from the `torchtitan-test` repository root:

```bash
cd /path/to/torchtitan-test
conda activate <existing-gpu-environment>

nvidia-smi -L
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
python -c 'import torchtitan; print(torchtitan.__file__)'
```

The installed `torchtitan` must point to the GLM-5.2 checkout used by the prior
distributed experiment. If it does not, reinstall that checkout into the active
environment:

```bash
python -m pip install -e ../torchtitan --no-deps
```

## Recommended execution

Select one physical GPU. The wrapper prints the selected PyTorch/CUDA/device
before creating any artifacts.

```bash
export CUDA_VISIBLE_DEVICES=0
unset ASCEND_RT_VISIBLE_DEVICES

tests/glm5_2_graph/run_gpu_eager_inductor.sh probe all
tests/glm5_2_graph/run_gpu_eager_inductor.sh benchmark all
```

`all` runs FP32 followed by BF16. To shorten the first attempt, select one:

```bash
tests/glm5_2_graph/run_gpu_eager_inductor.sh probe bf16
tests/glm5_2_graph/run_gpu_eager_inductor.sh benchmark bf16
```

If the environment's Python executable has a nonstandard name:

```bash
PYTHON_BIN=/path/to/python \
  tests/glm5_2_graph/run_gpu_eager_inductor.sh probe bf16
```

The fixture phase refuses to overwrite an existing fixture. Use `--force` only
when intentionally replacing the fixture and both endpoint captures:

```bash
tests/glm5_2_graph/run_gpu_eager_inductor.sh probe bf16 --force
```

## Manual phases

The wrapper is equivalent to the following four phases. Use these commands when
rerunning only one failed phase. Replace `gpu_compile_probe.py` with
`gpu_precision_benchmark.py` for the 1000-step run.

```bash
ENTRY=tests/glm5_2_graph/gpu_compile_probe.py
COMMON="--topology single --objectives precision --reference-graph eager --candidate-graph inductor --compiler-diagnostics --precision bf16"

python "$ENTRY" --data --data-device cuda $COMMON
python "$ENTRY" --capture reference $COMMON
python "$ENTRY" --capture candidate $COMMON
python "$ENTRY" --compare --require-all $COMMON
```

Omit `--force` for capture retries: complete captures are reused, while an
incomplete capture is replaced automatically.

## Results

- `combination_runs/`: full runtime logs and raw per-step metrics.
- `combination_artifacts/`: portable eager and Inductor capture artifacts.
- `combination_reports/precision/`: HTML reports and `suite_summary.json`.

Find the newest machine-readable CUDA result with:

```bash
find combination_reports/precision -path '*self-cuda*' \
  -name suite_summary.json -printf '%T@ %p\n' | sort -nr | head
```

For the probe, a formal FAIL caused only by fewer than 1000 observations is
expected; confirm that all four captures completed and inspect the numerical
metrics. For the benchmark, require `"passed": true` in `suite_summary.json`.
Also inspect repeatability and the first/mean loss plus grad-norm criteria; do
not treat successful process exit alone as precision alignment.

If Inductor compilation fails, start with the candidate `runtime.log` and the
`TORCH_LOGS=graph_breaks,recompiles,dynamic` diagnostics recorded by the entry
point. Do not enable whole-step CUDA Graph while diagnosing eager/Inductor
precision, because that introduces a second execution difference.

## Post-failure CUDA localization

After a reproducible 1000-step eager/Inductor loss failure, run the CUDA
diagnostic sequence below. It reuses the NPU investigation's proven order while
keeping all CUDA hooks and outputs independent from NPU patches:

1. Minimal FP32/BF16 SiLU-multiply and linear-residual checks.
2. Block 0 input/output forward and backward-gradient fingerprints at steps
   1, 2, 8, 9, and 10.
3. Block 0 parameter gradients immediately before clipping.
4. Two Inductor repeats compiled in separate Inductor and Triton cache roots.
5. Ordered internal forward stages for attention, residuals, norms, and dense
   SwiGLU projections after a Block-internal divergence is established.

Run the complete FP32 and BF16 diagnostic in one command:

```bash
GPU_DIAG_RUN_ID=h20-eager-inductor-v1 tests/glm5_2_graph/run_gpu_inductor_diagnostics.sh all all
```

The stable run ID makes the output location predictable:

```text
graph_debug_runs/gpu-inductor/h20-eager-inductor-v1/
```

Run only the minimal fusion checks or only the Block trace when needed:

```bash
GPU_DIAG_RUN_ID=h20-minimal-v1 tests/glm5_2_graph/run_gpu_inductor_diagnostics.sh minimal all
GPU_DIAG_RUN_ID=h20-trace-v1 tests/glm5_2_graph/run_gpu_inductor_diagnostics.sh trace all
GPU_DIAG_RUN_ID=h20-internal-v1 tests/glm5_2_graph/run_gpu_inductor_diagnostics.sh internal all
```

Each minimal precision/repeat writes JSONL containing exact hashes, mismatch
counts, mean/max errors, CUDA versions, TF32 flags, and reduced-precision
reduction flags. Each trace precision writes a compact comparison JSON and also
prints these four comparisons:

```text
eager-r1 vs eager-r2
inductor-r1 vs inductor-r2
eager-r1 vs inductor-r1
eager-r1 vs inductor-r2
```

The two Inductor cache directories are distinct, so exact Inductor r1/r2 traces
prove cold-build reproducibility rather than only same-cache replay. An exact
Block input followed by a mismatching Block output localizes the first observed
difference inside Block 0. A mismatching parameter gradient before clipping
proves the difference has reached the first optimizer update.

The internal trace prints the first divergent stage at steps 1, 2, 8, 9, and
10. Its intermediate tensors are additional graph outputs and therefore
materialization points; use it to localize the first stage, then use the
minimal A/B checks to establish fusion causality.

After FP32 first diverges at `attention_output`, localize the one-step absorbed
SparseMLA path through q/KV projections, normalization, RoPE, top-k selection,
inner attention, value unabsorption, and output projection:

```bash
GPU_DIAG_RUN_ID=h20-attention-v1 tests/glm5_2_graph/run_gpu_inductor_diagnostics.sh attention fp32
```

For the BF16 SiLU hypothesis, run the real training path with only the dense
SwiGLU SiLU output behind an autograd-enabled materialization custom op. The
reference remains unmodified eager and the candidate is materialized Inductor:

```bash
tests/glm5_2_graph/run_gpu_silu_materialization.sh probe
tests/glm5_2_graph/run_gpu_silu_materialization.sh benchmark
```

Run the 10-step probe first. Start the 1000-step benchmark only after the probe
reduces the original eager/Inductor drift without introducing instability.

If the candidate-only intervention does not improve both loss and gradient
alignment, run the strict symmetric control. It materializes the same SiLU
output in eager and Inductor, uses independent cold compiler caches, and prints
all Step 1 Block 0 forward/backward and trainable parameter-gradient matches:

```bash
GPU_DIAG_RUN_ID=h20-silu-control-v1 tests/glm5_2_graph/run_gpu_silu_symmetric_control.sh
```

If SiLU-only materialization reduces but does not eliminate the Block output
difference, run the progressive one-step matrix before any longer experiment:

```bash
GPU_DIAG_RUN_ID=h20-silu-staged-v1 tests/glm5_2_graph/run_gpu_silu_staged_controls.sh step1 matrix
```

The variants add boundaries in order: `silu-product`,
`silu-product-down`, then `all` (`w1`, `w3`, SiLU, product, and down).
After selecting the smallest variant that aligns Block output and gradients,
run only that variant for 10 steps by replacing `step1` with `probe`.

If all FFN-only variants fail, first run the one-step BF16 attention trace,
then test the Block-frontier matrix (`ffn_norm` output, attention output, and
attention residual):

```bash
GPU_DIAG_RUN_ID=h20-attention-bf16-v1 tests/glm5_2_graph/run_gpu_inductor_diagnostics.sh attention bf16
GPU_DIAG_RUN_ID=h20-block-frontier-v1 tests/glm5_2_graph/run_gpu_silu_staged_controls.sh step1 frontier-matrix
```

When `attention-residual` is the smallest variant that makes Block forward
exact, run its symmetric real-training probe and its ordered backward trace:

```bash
GPU_DIAG_RUN_ID=h20-attention-residual-probe-v1 tests/glm5_2_graph/run_gpu_silu_staged_controls.sh probe attention-residual
GPU_DIAG_RUN_ID=h20-attention-residual-backward-v1 tests/glm5_2_graph/run_gpu_silu_staged_controls.sh backward attention-residual
```

The backward comparison walks from `block_output` toward `block_input` and
reports the first eager/Inductor intermediate-gradient mismatch. Both endpoints
use the same materialized program; the experiment does not compare an altered
candidate against an unaltered eager reference.
