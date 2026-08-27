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
