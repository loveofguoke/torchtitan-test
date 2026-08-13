# Distributed GPU/NPU training comparison

This workflow runs the same TorchTitan distributed scenario on separate GPU and
NPU machines. It compares small, exact TensorBoard scalar captures first; it
does not dump per-layer tensors unless a later investigation needs them.

The portable fixture contains the local input dataset, tokenizer assets, and a
TorchTitan seed checkpoint. Every file is protected by SHA-256. Both captures
refuse a modified, partially copied, or scenario-incompatible fixture.

## Four-card GLM-5 matrix

The supplied suite contains 15 four-card scenarios: DDP, FSDP, HSDP, TP, PP,
PP+FSDP, PP+TP, FSDP+TP, several EP two-dimensional and three-dimensional
layouts, and representative FSDP/FSDP+TP BF16 runs. List the resolved matrix
without running training:

```bash
python -m distributed.suite \
  --suite distributed/scenarios/glm5_debugmodel_4card_suite.json \
  --output-root distributed_runs/glm5_debugmodel_4card \
  list
```

All scenarios use four processes, 20 steps, five warmup steps, local batch size
4, global batch size 16, and sequence length 128. Sequence length 128 is the
maximum supported by the current GLM-5 debugmodel RoPE configuration.

Common values live in `_base_fp32.json`; individual files append only their
topology. Keep the complete scenario directory unchanged on both machines.
Important fields are:

- `world_size`: identical process count on GPU and NPU.
- `steps`: at least 10 for a useful performance sample.
- `performance_warmup_steps`: steps excluded from performance medians.
- `deterministic`: keep `true` for numerical diagnosis. A separate performance
  scenario may set it to `false` when production kernels are non-deterministic.
- `seed_args`: model-shape overrides required while building the unsharded
  checkpoint; never put DP/TP/PP options here.
- `training_args`: identical TorchTitan parallelism, batch, precision, and
  compile options.
- `tolerances`: precision pass/fail thresholds. GPU is the reference.

Do not put backend-specific optimization flags in `training_args`; pass them
through the machine environment so the shared scenario identity stays honest.

## 1. Prepare the matrix fixtures

Run this once on the GPU machine. Seed checkpoint creation intentionally uses
one device and stores an unsharded step-0 checkpoint, so every later layout can
load weights initialized from the same seed.

```bash
python -m distributed.suite \
  --suite distributed/scenarios/glm5_debugmodel_4card_suite.json \
  --output-root distributed_runs/glm5_debugmodel_4card \
  --skip-existing \
  prepare --backend gpu
```

Copy the complete scenario directory and `distributed_runs/glm5_debugmodel_4card`
to the NPU machine. Do not regenerate fixtures independently.

## 2. Capture each machine

GPU machine:

```bash
python -m distributed.suite \
  --suite distributed/scenarios/glm5_debugmodel_4card_suite.json \
  --output-root distributed_runs/glm5_debugmodel_4card \
  --skip-existing \
  --continue-on-error \
  capture --backend gpu
```

NPU machine:

```bash
python -m distributed.suite \
  --suite distributed/scenarios/glm5_debugmodel_4card_suite.json \
  --output-root distributed_runs/glm5_debugmodel_4card \
  --skip-existing \
  --continue-on-error \
  capture --backend npu
```

Each command also leaves a sibling run directory containing the full runtime
log and TensorBoard event. Capture uses the scenario's deterministic setting
and seed, one metric record per step, and model-only loading from the shared
checkpoint. Performance numbers always describe those exact settings; use a
separate scenario when the production performance configuration differs.
`--continue-on-error` records failed captures, proceeds with later scenarios,
and returns exit code 1 after the matrix finishes. Completed immutable captures
remain usable.

## 3. Compare offline

Copy the NPU capture JSON files back beside the GPU captures, then run:

```bash
python -m distributed.suite \
  --suite distributed/scenarios/glm5_debugmodel_4card_suite.json \
  --output-root distributed_runs/glm5_debugmodel_4card \
  compare
```

Precision comparison covers global average loss, global maximum rank loss, and
gradient norm at every step. A tolerance failure returns exit code 1.
Performance is descriptive and reports steady-state medians and NPU/GPU ratios
for throughput, end-to-end step time, TFLOPS, MFU, and peak memory when present.

## Deterministic mode

`deterministic: true` asks PyTorch to use deterministic implementations and to
reject known non-deterministic operations. The seed, input data, and initial
weights are fixed independently of this switch. Deterministic mode additionally
makes repeated runs on the same backend reproducible by avoiding kernels such
as atomic or split-reduction implementations whose accumulation order may vary.

This is the correct first setting for numerical diagnosis, but deterministic
kernels can be slower than production kernels. The supplied matrix therefore
measures performance under accuracy-diagnostic settings. After accuracy passes,
copy the desired scenario, set `deterministic` to `false`, and repeat it for the
production-oriented performance result. With it disabled, small run-to-run
floating-point differences are expected even though data and weights stay fixed.

## Run one scenario

The lower-level `distributed.workflow` CLI remains available when only one
topology is needed. Pass one scenario JSON, one fixture path, and one capture or
report path; `python -m distributed.workflow --help` lists the arguments.

The first escalation after a failure should reuse the existing
`tests/glm5_2_parity` layer/component capture framework with the failing model,
precision, and input narrowed from this run.
