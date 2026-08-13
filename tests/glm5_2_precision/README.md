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
- both reference and candidate are captured at least twice. Repeat-run bitwise
  equality is reported as a diagnostic; BF16 formal decisions use the migration
  tolerances above.

The example migration script uses the internal migration standard above after
the configured warmup. P99/P99.9/P99.99 and the four historical customer
profiles remain visible as supplemental diagnostics, but do not decide formal
PASS/FAIL. All thresholds live in `standards.py` or in the experiment script's
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

The standard `CUDA_VISIBLE_DEVICES` and `ASCEND_RT_VISIBLE_DEVICES` exports are
the primary device selection interface. CLI visibility overrides are retained
only for automation. `--capture` runs every configured repetition by default;
use `--repeat N` only to rerun or debug one repetition.

The fixture is not owned by reference or candidate. `--data` uses whichever
accelerator visibility variable is exported. If neither or both are exported,
the command stops instead of assuming an endpoint; `--data-device cuda|npu`
is the explicit override for that uncommon case.

The shared fixture can be generated on either endpoint. For example, generate
it on NPU and capture both NPU candidate repetitions:

```bash
export ASCEND_RT_VISIBLE_DEVICES=4

python tests/glm5_2_precision/migration_benchmark.py \
  --data --topology single

python tests/glm5_2_precision/migration_benchmark.py \
  --capture candidate --topology single
```

Synchronize the scenario directory under `precision_fixtures/` to the GPU
server, then capture both GPU reference repetitions:

```bash
export CUDA_VISIBLE_DEVICES=7

python tests/glm5_2_precision/migration_benchmark.py \
  --capture reference --topology single
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
  --capture reference --topology fsdp8
python tests/glm5_2_precision/migration_benchmark.py \
  --capture candidate --topology fsdp2-tp4-ep8
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
  --data --reference-topology single \
  --candidate-topology fsdp8

python tests/glm5_2_precision/self_consistency_benchmark.py \
  --capture reference \
  --reference-topology single --candidate-topology fsdp8

python tests/glm5_2_precision/self_consistency_benchmark.py \
  --capture candidate \
  --reference-topology single --candidate-topology fsdp8

python tests/glm5_2_precision/self_consistency_benchmark.py \
  --compare --reference-topology single \
  --candidate-topology fsdp8
```

Export `ASCEND_RT_VISIBLE_DEVICES` instead for NPU self-consistency. When a
change does not affect randomness, set `randomness_impacted=False`; the formal
standard then requires bitwise-identical reference/candidate loss and grad norm.
Parallel decomposition changes reduction order, so the supplied example uses
the migration fallback standard. Exact repeatability remains visible as a
diagnostic and does not fail this BF16 self-consistency experiment.

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

- `precision_fixtures/`: synchronized seed or converged checkpoint plus
  checksums; keep this directory in Git when another environment needs it;
- `precision_runs/`: ignored local working files, including TensorBoard output,
  runtime logs, and the raw metric stream;
- `precision_artifacts/`: synchronized checksummed comparison inputs. Each
  capture contains only `manifest.json`, normalized `metrics.jsonl`, and
  `training_contract.json`;
- `precision_reports/`: a self-contained HTML report and machine-readable JSON
  summary; it is generated locally and ignored by default.

Directory names omit the model name and redundant endpoint labels. For example:
`migration-cuda-npu-single-bf16-random-s1000-b16-seq128-seed61/candidate-r1`.
The full descriptive scenario remains in manifests and reports.

Existing outputs made by an older version can be renamed without rerunning:

```bash
# Preview every fixture, capture, run, and report migration.
python tests/glm5_2_precision/migrate_legacy_outputs.py

# Apply after reviewing the paths.
python tests/glm5_2_precision/migrate_legacy_outputs.py --apply
```

The converter also removes redundant artifact attachments. Local raw metrics
and runtime logs remain under the ignored `precision_runs/` directory.
If artifacts were copied through a system that converted line endings, add
`--repair-artifacts` to recompute checksums without changing metric values.

The model checkpoint is large; synchronize fixtures with a shared filesystem,
`rsync`, or another binary transfer tool rather than ordinary Git. Formal
artifacts are intentionally small and suitable for normal source-control based
synchronization when desired.
