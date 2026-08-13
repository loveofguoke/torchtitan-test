# GLM 5.2 formal training precision benchmark

This package is the delivery-oriented complement to
`tests/glm5_2_parity`. The parity suite captures many intermediate tensors to
locate a numerical divergence. This suite decides whether a long-running
training process is accurate and stable enough for migration or distributed
self-consistency.

The two suites are intentionally independent. They share TorchTitan model and
data assets, and a formal report can link sampled exploratory reports through
`exploratory_reports`, but exploratory trace results never decide the formal
PASS/FAIL result.

`FormalTrainingConfig.exploratory_steps` is the explicit sampling plan recorded
in every formal training contract. Keep it empty for delivery runs, or list the
steps at which separate parity/parameter diagnostics will be captured. This
framework deliberately does not dump every parameter at every step: doing so
would make long multi-card runs impractical. Use the existing parity suite at
those sampled steps and add its report paths to `exploratory_reports`.

## Error definitions

For reference sequence
$X=(x_1,x_2,\ldots,x_n)$ and candidate sequence
$Y=(y_1,y_2,\ldots,y_n)$, where $x_i,y_i\in\mathbb{R}$:

$$
\operatorname{Normal}(X,Y)
=\frac{1}{n}\sum_{i=1}^{n}(x_i-y_i)
$$

$$
\operatorname{Absolute}(X,Y)
=\frac{1}{n}\sum_{i=1}^{n}|x_i-y_i|
$$

$$
\operatorname{RelativeNormal}(X,Y)
=\frac{1}{n}\sum_{i=1}^{n}\frac{x_i-y_i}{x_i}
$$

$$
\operatorname{RelativeAbsolute}(X,Y)
=\frac{1}{n}\sum_{i=1}^{n}\frac{|x_i-y_i|}{x_i}
$$

`X` is always the benchmark/reference. The formal series here are loss and
global grad norm, so their reference denominators are nonnegative. Reports
also include max absolute error and absolute-error quantiles.

The reusable default migration standard follows the internal guidance:

- first loss: mean absolute `<= 0.005` OR mean absolute relative `<= 0.5%`;
- all retained losses: mean absolute `<= 0.01` OR mean absolute relative
  `<= 1%`;
- global grad norm signed mean relative error in `[-5%, 5%]`;
- both reference and candidate are captured at least twice, with exactly equal
  loss and grad norm series within each endpoint.

The example migration script selects the documented strict customer profile:
cumulative mean absolute loss error `<= 0.005` after warmup, P99/P99.9/P99.99
tail limits, and optional removal of the largest 0.5% step errors. All
thresholds live in `standards.py` or in the experiment script's
`PrecisionStandard`; none are hard-coded in report or decision code.

## Reproducible training contract

`--data` creates one unsharded TorchTitan seed checkpoint with
`checkpoint.create_seed_checkpoint`. Both endpoints load that synchronized
checkpoint. The fixture manifest records checkpoint and local dataset/tokenizer
SHA-256 digests. Captures also fix:

- model module and config;
- seed and deterministic algorithms;
- number of steps and sequence length;
- local and global batch size;
- training, parameter, and reduction dtypes;
- parallel topology;
- any extra TorchTitan arguments.

The global batch size is fixed across topologies. Gradient accumulation makes a
single-card run consume the same global batch size as a data-parallel run.
Comparison rejects artifacts with different common contracts; migration also
requires identical parallel topology. Self-consistency intentionally permits
different topologies.

Full-precision scalar values are captured before console formatting by wrapping
TorchTitan's TensorBoard logger. The runtime package is imported, not copied or
modified. NPU capture imports TorchTitanTurbo before TorchTitan training begins.

## Migration workflow

Configure `CONFIG` once in `migration_benchmark.py`. Each command derives its
fixture, artifact, run, and report directory from the experiment kind, devices,
topology, and precision.

Single-card fixture and the two GPU reference repetitions:

```bash
python tests/glm5_2_precision/migration_benchmark.py \
  --data --topology single --reference-visible-devices 7

python tests/glm5_2_precision/migration_benchmark.py \
  --capture reference --repeat 1 --topology single \
  --reference-visible-devices 7

python tests/glm5_2_precision/migration_benchmark.py \
  --capture reference --repeat 2 --topology single \
  --reference-visible-devices 7
```

Synchronize the scenario directory under `precision_fixtures/` to the NPU
server. Then run the two required candidate repetitions:

```bash
python tests/glm5_2_precision/migration_benchmark.py \
  --capture candidate --repeat 1 --topology single \
  --candidate-visible-devices 4

python tests/glm5_2_precision/migration_benchmark.py \
  --capture candidate --repeat 2 --topology single \
  --candidate-visible-devices 4
```

Synchronize all small scenario directories under `precision_artifacts/` to one
machine and compare on CPU:

```bash
python tests/glm5_2_precision/migration_benchmark.py \
  --compare --topology single
```

Choose any registered multi-card topology without copying the script:

```bash
python tests/glm5_2_precision/migration_benchmark.py --list-topologies
python tests/glm5_2_precision/migration_benchmark.py \
  --capture reference --repeat 1 --topology fsdp8
python tests/glm5_2_precision/migration_benchmark.py \
  --capture candidate --repeat 1 --topology fsdp2-tp4-ep8
```

The final two commands illustrate topology selection only; a migration pair
must use the same `--topology` at data, both captures, and compare stages.

Use `--precision fp32`, `--precision bf16`, or `--precision full-bf16` to
override both endpoints consistently. `bf16` means FP32 master training with
BF16 mixed-precision parameters; `full-bf16` sets the full training dtype.

## Distributed self-consistency

The self-consistency script defaults to CUDA single-card versus CUDA FSDP8.
Select both topologies from one file:

```bash
python tests/glm5_2_precision/self_consistency_benchmark.py \
  --data --device cuda --reference-topology single \
  --candidate-topology fsdp8

python tests/glm5_2_precision/self_consistency_benchmark.py \
  --capture reference --repeat 1 --device cuda \
  --reference-topology single --candidate-topology fsdp8
python tests/glm5_2_precision/self_consistency_benchmark.py \
  --capture reference --repeat 2 --device cuda \
  --reference-topology single --candidate-topology fsdp8

python tests/glm5_2_precision/self_consistency_benchmark.py \
  --capture candidate --repeat 1 --device cuda \
  --reference-topology single --candidate-topology fsdp8
python tests/glm5_2_precision/self_consistency_benchmark.py \
  --capture candidate --repeat 2 --device cuda \
  --reference-topology single --candidate-topology fsdp8

python tests/glm5_2_precision/self_consistency_benchmark.py \
  --compare --device cuda --reference-topology single \
  --candidate-topology fsdp8
```

Replace `--device cuda` with `--device npu` for NPU self-consistency. When a
change does not affect randomness, set `randomness_impacted=False`; the formal
standard then requires bitwise-identical reference/candidate loss and grad norm.
Parallel decomposition changes reduction order, so the supplied example uses
the migration fallback standard while still requiring exact candidate repeats.

## Multi-node capture

Define or select a topology whose `world_size` covers every node. The built-in
`ddp16` and `fsdp16` entries target two eight-device nodes. Run the same command
on both nodes with a shared rendezvous endpoint and the appropriate node rank:

```bash
python tests/glm5_2_precision/migration_benchmark.py \
  --capture reference --repeat 1 --topology fsdp16 \
  --num-nodes 2 --node-rank 0 --rdzv-endpoint host0:29500

python tests/glm5_2_precision/migration_benchmark.py \
  --capture reference --repeat 1 --topology fsdp16 \
  --num-nodes 2 --node-rank 1 --rdzv-endpoint host0:29500
```

Only the node owning TorchTitan's metrics rank writes the portable artifact.
Each node keeps a separate runtime directory. The synchronized fixture must be
present at the same repository-relative location on every node.

## Outputs

- `precision_fixtures/`: shared seed or converged checkpoint plus checksums;
- `precision_runs/`: local TensorBoard files, structured logs, and raw capture;
- `precision_artifacts/`: small checksummed offline artifacts, including the
  exact JSONL series and runtime log;
- `precision_reports/`: a self-contained HTML report and machine-readable JSON
  summary.

The model checkpoint is large; synchronize fixtures with a shared filesystem,
`rsync`, or another binary transfer tool rather than ordinary Git. Formal
artifacts are intentionally small and suitable for normal source-control based
synchronization when desired.
