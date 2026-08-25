# GLM 5.2 formal training precision benchmark

This package is the delivery-oriented complement to
`tests/glm5_2_parity`. The parity suite captures many intermediate tensors to
locate a numerical divergence. This suite decides whether a long-running
training process is accurate and stable enough for migration or distributed
self-consistency.

This package owns ordinary eager CUDA/NPU migration and distributed
self-consistency. Graph-mode precision keeps the same formal fixture, metric,
standard, and report implementation, but is composed through
`tests/glm5_2_graph` and `tests/glm5_2_combination` so compiler policy does not
leak into this workflow. Compiled graph execution is currently NPU-only.

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
`checkpoint.create_seed_checkpoint`. It also runs the configured text loader and
tokenizer once with `dp_world_size=1`, then stores a fixed token plan with shape
`[steps, global_batch_size, sequence_length]` for inputs, labels, and positions,
plus per-step SHA-256 digests. Inputs and labels are stored independently because
packed text batches can cross document boundaries and are not always one globally
shifted token row. Both endpoints load that synchronized checkpoint and token plan.
The fixture manifest records checkpoint, token plan, and local dataset/tokenizer
SHA-256 digests. Captures also fix:

- model module and config;
- seed and deterministic algorithms;
- number of steps and sequence length;
- local and global batch size;
- training, parameter, and reduction dtypes;
- parallel topology;
- any extra TorchTitan arguments.

The runtime test dataloader maps each optimizer step's global sample slots onto
the current DP rank, gradient-accumulation group, and PP microbatch. Therefore a
single-card run, DDP/FSDP/EP run, and mixed TP/PP topology consume exactly the
same ordered global token batch at every optimizer step; only ownership changes.
Every rank writes an input-contract stream. Capture verifies every slot and token
hash against the fixture before it creates a portable artifact. A comparison is
rejected if either side lacks a valid fixed-token contract.
Comparison rejects artifacts with different common contracts; migration also
requires identical parallel topology. Self-consistency intentionally permits
different topologies.

Full-precision scalar values are captured before console formatting by wrapping
TorchTitan's TensorBoard logger. The runtime package is imported, not copied or
modified. NPU capture imports TorchTitanTurbo before TorchTitan training begins.

Because the token plan is now part of the formal data contract, fixtures made by
an earlier framework version cannot be reused for new captures. Recreate the
fixture once with `--data --force`, then rerun reference and candidate captures.

## Command-line parameters

The maintained `migration_benchmark.py`, `full_dsa_migration_benchmark.py`, and
`self_consistency_benchmark.py` entry points share this suite CLI.

| Parameter | Purpose and choices | Default |
|---|---|---|
| `--data` | Generate one shared checkpoint/token fixture and exit. | no action; one action is required |
| `--capture reference\|candidate` | Capture one endpoint. Omit `--repeat` to run both configured repeats. | no action; one action is required |
| `--capture-all` | Compatibility alias for `--capture candidate --topology all`. | disabled |
| `--compare` | Compare available captures and generate detailed plus suite reports. | no action; one action is required |
| `--list-topologies` | Print topology degrees and the derived batch schedule. | no action; one action is required |
| `--topology` | One topology or `all`. Migration applies it to both devices; self-consistency applies it to the distributed candidate. | `all` |
| `--topologies` | Comma-separated subset or `all`; mutually exclusive with `--topology`. | unset |
| `--repeat` | Capture only repeat `N`; omit it for every configured repeat. | all configured repeats (`2`) |
| `--precision` | `fp32`, `bf16`, or `full-bf16`. | `bf16` from the experiment config |
| `--data-device` | Fixture backend override: `cuda` or `npu`. Normally inferred from exactly one exported visibility variable. | inferred |
| `--force` | Start a new generation for the selected fixture/capture range. All selected members are removed before execution; rerun without it after interruption to resume only that generation. | disabled |
| `--require-all` | Require every selected topology and repeat for a final suite decision. Without it, missing members are reported as `NOT RUN`. | disabled |

Both maintained experiments define 5000 steps, local batch 8, global batch 64,
sequence length 128, seed 61, deterministic execution, BF16 parameters with
FP32 master/reduction, and two captures per endpoint. Those values are part of
the maintained experiment contract and are edited in the corresponding Python
`CONFIG`, not overridden ad hoc on the command line.

For one experiment, data, captures, and compare must use the same precision and
training config. Topology selection may be narrowed during capture and expanded
later during compare because every topology shares the same fixture. `--force`
is never required merely to resume an interrupted suite.

## Report contents and acceptance

Every detailed report shows the complete loss and global-grad-norm curves,
warmup boundary, repeat-to-repeat determinism diagnostics, first and worst
divergent steps, the four error formulas, max error, and configured error
quantiles. It also records device/topology/dtype/source metadata and validates
the fixed-token input contract before comparing numerical series.

The migration PASS/FAIL decision uses the configured first-loss, all-loss, and
grad-norm standards. P99/P99.9/P99.99 distributions and historical customer
profiles are supplemental diagnostics unless the experiment's
`PrecisionStandard` explicitly promotes them. The suite index shows one row per
topology, links each detailed report, and distinguishes `PASS`, `FAIL`,
`PARTIAL PASS`, and `NOT RUN`; `--require-all` turns missing selected members
into an incomplete final deliverable.

## Migration workflow

Configure `CONFIG` once in `migration_benchmark.py`. Each command derives its
fixture, artifact, run, and report directory from the experiment kind, devices,
topology, and precision.

The built-in migration profile runs 5000 optimizer steps with local batch 8
and global batch 64, so the same training contract and one topology-independent
fixture are valid for every registered topology up to eight devices, including
DP8 and `1F1B` PP8.

Use `--topology all` to run the complete suite. Data is generated once, each
server captures all topologies sequentially with one command, and compare writes
one suite index plus a detailed report per topology:

```bash
unset CUDA_VISIBLE_DEVICES
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_precision/migration_benchmark.py \
  --data --data-device npu --topology all
python tests/glm5_2_precision/migration_benchmark.py \
  --capture candidate --topology all

unset ASCEND_RT_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_precision/migration_benchmark.py \
  --capture reference --topology all

python tests/glm5_2_precision/migration_benchmark.py \
  --compare --topology all --require-all
```

The commands above are the complete all-topology migration workflow. For a
focused single-card GPU/NPU migration, run the same data, two capture, and
compare stages with `single`:

```bash
# NPU: prepare shared data and capture the candidate.
unset CUDA_VISIBLE_DEVICES
export ASCEND_RT_VISIBLE_DEVICES=4
python tests/glm5_2_precision/migration_benchmark.py \
  --data --data-device npu --topology single
python tests/glm5_2_precision/migration_benchmark.py \
  --capture candidate --topology single

# GPU: after synchronizing precision_fixtures, capture the reference.
unset ASCEND_RT_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=7
python tests/glm5_2_precision/migration_benchmark.py \
  --capture reference --topology single

# CPU: after synchronizing precision_artifacts.
python tests/glm5_2_precision/migration_benchmark.py \
  --compare --topology single --require-all
```

Distributed GPU/NPU migration example (`fsdp8`; replace it with any registered
topology):

```bash
# NPU.
unset CUDA_VISIBLE_DEVICES
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_precision/migration_benchmark.py \
  --data --data-device npu --topology fsdp8
python tests/glm5_2_precision/migration_benchmark.py \
  --capture candidate --topology fsdp8

# GPU, after fixture synchronization.
unset ASCEND_RT_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_precision/migration_benchmark.py \
  --capture reference --topology fsdp8

# CPU, after artifact synchronization.
python tests/glm5_2_precision/migration_benchmark.py \
  --compare --topology fsdp8 --require-all
```

`--topologies single,fsdp8,tp8` selects a subset. Completed captures are skipped,
so rerunning an interrupted all-topology command resumes at the first incomplete
topology/repeat.

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
  --capture candidate --topology fsdp8
```

Each migration comparison uses the same topology on both endpoints. `--data`
is topology-independent, so the one fixture can be reused by all topology
pairs in the suite.

Use `--precision fp32`, `--precision bf16`, or `--precision full-bf16` to
override both endpoints consistently. `bf16` means FP32 master training with
BF16 mixed-precision parameters; `full-bf16` sets the full training dtype.

## Full DSA migration workflow

`full_dsa_migration_benchmark.py` is independent from the established eager
migration baseline. It selects `glm5_full_dsa_debugmodel`, preserves the same
5000-step/token-plan/checkpoint/error-standard contract, and deliberately
excludes PP topologies because cross-stage top-k transport is not implemented.
For this entry point, `--topology all` means every registered non-PP topology.

Generate the shared fixture on either server, then capture all NPU candidates:

```bash
unset CUDA_VISIBLE_DEVICES
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_precision/full_dsa_migration_benchmark.py \
  --data --data-device npu --topology all
python tests/glm5_2_precision/full_dsa_migration_benchmark.py \
  --capture candidate --topology all
```

After synchronizing the fixture, capture the matching GPU references:

```bash
unset ASCEND_RT_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_precision/full_dsa_migration_benchmark.py \
  --capture reference --topology all
```

After synchronizing the portable artifacts, compare on CPU:

```bash
python tests/glm5_2_precision/full_dsa_migration_benchmark.py \
  --compare --topology all --require-all
```

Replace `all` with `single`, one distributed topology, or use
`--topologies single,fsdp8,tp8` for a subset. Completed captures are reused and
incomplete captures are retried under the normal precision workflow rules.

Operator implementation is an independent factor. The defaults keep both
endpoints on the readable reference. To compare CUDA Triton with the same
kernel source compiled by Triton-Ascend, append the same flags to data, both
captures, and compare:

```bash
--reference-kernel triton --candidate-kernel triton
```

The optimized captures have distinct names but reuse the reference Full DSA
fixture because the checkpoint and token plan are unchanged. See
`tests/glm5_2_performance/OPTIMIZATION.md` for the complete A/B matrix.

## Distributed self-consistency

The CUDA self-consistency suite shares one fixture and one single-card
reference across every distributed topology.

The built-in profile runs 5000 optimizer steps with local batch 8, global batch
64, and PP microbatch size 1. These are the smallest common batch settings that
support both eight-way data parallelism and eight-stage `1F1B` while preserving
one shared single-card reference. The workflow validates the DP accumulation
and PP microbatch constraints before starting `torchrun`.

Prepare and capture the reference only once:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python tests/glm5_2_precision/self_consistency_benchmark.py \
  --data

python tests/glm5_2_precision/self_consistency_benchmark.py \
  --capture reference
```

The reference is always the single-card baseline. The self-consistency
candidate registry contains distributed topologies, so this script does not
accept `--topology single`; the command above is the single-card operation.

Complete GPU single-reference versus one distributed candidate workflow
(`fsdp8` in this example):

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset ASCEND_RT_VISIBLE_DEVICES

python tests/glm5_2_precision/self_consistency_benchmark.py --data
python tests/glm5_2_precision/self_consistency_benchmark.py \
  --capture reference --topology fsdp8
python tests/glm5_2_precision/self_consistency_benchmark.py \
  --capture candidate --topology fsdp8
python tests/glm5_2_precision/self_consistency_benchmark.py \
  --compare --topology fsdp8 --require-all
```

Complete GPU single-reference versus every distributed topology workflow:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset ASCEND_RT_VISIBLE_DEVICES

python tests/glm5_2_precision/self_consistency_benchmark.py --data
python tests/glm5_2_precision/self_consistency_benchmark.py \
  --capture reference
python tests/glm5_2_precision/self_consistency_benchmark.py --capture-all
python tests/glm5_2_precision/self_consistency_benchmark.py \
  --compare --require-all
```

Run candidates independently so completed work is preserved and can be
compared immediately:

```bash
export CUDA_VISIBLE_DEVICES=0,1
python tests/glm5_2_precision/self_consistency_benchmark.py \
  --capture candidate --topology ddp2

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_precision/self_consistency_benchmark.py \
  --capture candidate --topology ddp8

python tests/glm5_2_precision/self_consistency_benchmark.py \
  --capture candidate --topology fsdp8

python tests/glm5_2_precision/self_consistency_benchmark.py \
  --capture candidate --topology fsdp2-tp4
```

Or run every configured candidate sequentially:

```bash
python tests/glm5_2_precision/self_consistency_benchmark.py --capture-all
```

Candidates run sequentially rather than concurrently. `--capture-all` requires
enough visible devices for the largest selected topology. Individual captures
only validate the endpoint being executed, so a single-card reference needs one
visible device and `ddp2` needs two. Completed topology/repeat artifacts are
skipped, making the command safe to resume after an interruption.

Compare every completed candidate against the shared single-card reference:

```bash
python tests/glm5_2_precision/self_consistency_benchmark.py --compare
```

The suite report shows uncompleted topologies as `NOT RUN` and labels a clean
incomplete run `PARTIAL PASS`; only a complete clean suite receives `PASS`.
Add `--require-all` for a final deliverable that must contain all configured
topologies. Use `--topologies ddp8,fsdp8,tp8` to capture or compare a subset.
The complete built-in candidate list is:

```text
ddp2
ddp8
fsdp8
tp8
cp8
pp8
ep8
fsdp2-tp4
fsdp2-cp4
tp2-cp4
fsdp4-tp2
fsdp2-pp4
fsdp2-tp2-pp2
fsdp2-tp4-ep8
```

The storage layout is shared by the entire suite:

```text
precision_artifacts/self-cuda-bf16-random-s5000-b64-seq128-seed61/
  reference-r1/
  reference-r2/
  ddp8-r1/
  ddp8-r2/
  fsdp8-r1/
  ...
```

Each detailed topology report and the suite summary are generated together:

```text
precision_reports/self-cuda-bf16-random-s5000-b64-seq128-seed61/
  self-cuda-single-vs-distributed-bf16-suite.html
  suite_summary.json
  ddp8/self-cuda-single-vs-cuda-ddp8-bf16.html
  fsdp8/self-cuda-single-vs-cuda-fsdp8-bf16.html
  ...
```

The previous one-pair command remains available through the workflow module,
but the suite CLI is the recommended interface for GPU distributed baselines.

To list the exact topology settings:

```bash
python tests/glm5_2_precision/self_consistency_benchmark.py --list-topologies
```

To run only repeat 2 for one topology:

```bash
python tests/glm5_2_precision/self_consistency_benchmark.py \
  --capture candidate --topology fsdp8 --repeat 2
```

To regenerate an existing capture explicitly:

```bash
python tests/glm5_2_precision/self_consistency_benchmark.py \
  --capture candidate --topology fsdp8 --force
```

For the final suite report:

```bash
python tests/glm5_2_precision/self_consistency_benchmark.py \
  --compare --require-all
```

Parallel decomposition changes reduction order, so the supplied BF16 suite
uses migration tolerances instead of requiring bitwise equality. Repeat-run
bitwise differences remain visible as diagnostics.

For NPU self-consistency, use a separate suite configuration and export
`ASCEND_RT_VISIBLE_DEVICES`; do not mix CUDA and NPU candidates into this CUDA
self-consistency suite.

## Multi-node status

The maintained topology-suite entry points currently expose single-node
topologies through eight local ranks. The common registry retains `ddp16` and
`fsdp16` for future multi-node work, but they are intentionally not selectable
from these scripts yet. Do not pass legacy `--num-nodes`, `--node-rank`, or
`--rdzv-endpoint` examples to the suite CLI; multi-node artifact coordination
needs a separately defined shared-storage and launcher contract first.

## Outputs

- `precision_fixtures/`: synchronized seed or converged checkpoint, fixed token
  plan, and checksums; transfer it with `release_artifacts.py`, shared storage,
  or `rsync` when another environment needs it;
- `precision_runs/`: ignored local working files, including TensorBoard output,
  runtime logs, and the raw metric stream;
- `precision_artifacts/`: synchronized checksummed comparison inputs. Each
  capture contains `manifest.json`, normalized `metrics.jsonl`,
  `training_contract.json`, and a compact validated input-contract summary;
- `precision_reports/`: a self-contained HTML report and machine-readable JSON
  summary; it is generated locally and ignored by Git.

Directory names omit the model name and redundant endpoint labels. For example:
`migration-cuda-npu-single-bf16-random-s5000-b64-seq128-seed61/candidate-r1`.
The full descriptive scenario remains in manifests and reports.

Existing outputs made by an older version can be renamed without rerunning:

```bash
# Preview every fixture, capture, run, and report migration.
python tests/glm5_2_precision/migrate_legacy_outputs.py

# Apply after reviewing the paths.
python tests/glm5_2_precision/migrate_legacy_outputs.py --apply
```

The converter removes redundant runtime-log and raw-metric attachments while
preserving a current artifact's compact input-contract summary. Local raw
metrics and runtime logs remain under the ignored `precision_runs/` directory.
If artifacts were copied through a system that converted line endings, add
`--repair-artifacts` to recompute checksums without changing metric values.

The model checkpoint is large; synchronize fixtures with a shared filesystem,
`rsync`, or another binary transfer tool rather than ordinary Git. Formal
artifacts are intentionally small and suitable for normal source-control based
synchronization when desired.
