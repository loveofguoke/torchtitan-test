# torchtitan-test

This repository keeps execution and parity testing outside the two runtime
packages:

- `torchtitan` contains device-independent model and trainer code.
- `torchtitanturbo` depends on `torchtitan` and owns NPU patches, acceleration,
  and profiling.
- `torchtitan-test` owns launch scripts, fixtures, captures, comparisons, and
  reports.

Install the repositories from source into the same environment. A GPU-only
environment needs TorchTitan and the test requirements:

```bash
pip install -e ../torchtitan --no-deps
pip install -r requirements-test.txt
```

An NPU environment also needs the matching TorchTitanTurbo checkout and its
NPU runtime dependencies:

```bash
pip install -e ../torchtitan --no-deps
pip install -e ../TorchTitanTurbo --no-deps
pip install -r requirements-test.txt
```

## Training

`run_train.sh` selects NPU when `ASCEND_RT_VISIBLE_DEVICES` is set and otherwise
selects GPU. The existing commands therefore remain unchanged.

GPU:

```bash
export CUDA_VISIBLE_DEVICES=4
NGPU=1 LOG_RANK=0 MODULE=glm5 CONFIG=glm5_debugmodel ./run_train.sh
```

NPU:

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
# export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
NGPU=1 LOG_RANK=0 MODULE=glm5 CONFIG=glm5_debugmodel ./run_train.sh
```

The NPU entry imports `torchtitanturbo` before `torchtitan.train`, so Turbo's
NPU integration is active before the trainer is constructed. Use
`run_train_gpu.sh` or `run_train_npu.sh` when an explicit backend is preferable,
or set `TORCHTITAN_DEVICE=gpu|npu` when calling `run_train.sh`.

Short eager or NPU graph-mode launch checks across `single`, a selected
topology/subset, or every topology through eight ranks are provided by
[tests/glm5_2_smoke/README.md](tests/glm5_2_smoke/README.md).

## GLM-5.2 parity

Run all parity commands from this repository root. Existing pytest commands and
environment variables are unchanged because this repository preserves the
`tests/unit_tests/test_glm5_parity.py` path. The framework imports the
source-installed `torchtitan` package; an NPU capture additionally imports
`torchtitanturbo` before selecting the device backend.

The offline workflow records this test repository, TorchTitan, optional
TorchTitanTurbo, and package versions as diagnostic metadata. Commits and dirty
worktrees do not gate execution. Experiment identity is enforced by the
scenario configuration, suite version, test plan, and exact fixture checksums.

See [tests/glm5_2_parity/README.md](tests/glm5_2_parity/README.md) for paired,
fixture, capture, and CPU comparison commands.

## GLM-5.2 formal training precision

The formal benchmark is separate from exploratory tensor parity. It records
full-precision loss and global grad norm for long, deterministic training runs,
supports single-card, DDP, FSDP, TP, PP, EP, combined, and multi-node
topologies, and produces a CPU-only offline report with curves, four error
metrics, max absolute error, distributions, thresholds, and repeatability
checks.

See
[tests/glm5_2_precision/README.md](tests/glm5_2_precision/README.md) for the
`--data`, `--capture`, and `--compare` workflows for migration and distributed
self-consistency.

## Performance, graph, and combined experiments

The standalone Ascend Profiler workflow is documented in
[tests/glm5_2_performance/README.md](tests/glm5_2_performance/README.md).
Compiled graph and standalone performance execution currently support NPU
only; CUDA values remain reserved interfaces that fail explicitly. CUDA/NPU
eager precision remains device-neutral.

Graph mode owns only graph execution policy and convenience entry points. Its
primary correctness experiment compares NPU eager single-card reference data
with NPU graph-mode single or distributed candidates. See
[tests/glm5_2_graph/README.md](tests/glm5_2_graph/README.md).

Use [tests/glm5_2_combination/README.md](tests/glm5_2_combination/README.md)
when one training execution must independently select eager/graph modes and
produce precision, performance, or both objectives, with optional Profiler
collection.

## Stability and checkpoint validation

The stability soak test records step progress, duration, finite loss, and
finite global grad norm for single-card or distributed GPU/NPU training. See
[tests/glm5_2_stability/README.md](tests/glm5_2_stability/README.md).

The checkpoint test compares uninterrupted training with a real process
restart at an intermediate full DCP checkpoint. It verifies exact fixed-token
data continuation, loss and grad norm, trainer/dataloader/scheduler state, and
the final model and optimizer state. See
[tests/glm5_2_checkpoint/README.md](tests/glm5_2_checkpoint/README.md).

Result ownership at a glance:

| Suite | Result contents | What decides success |
|---|---|---|
| smoke | runtime log, exact launch contract, return code | requested training process exits successfully |
| parity | intermediate tensors/gradients, parameters, logits/loss, top-k scores and boundary diagnostics | requested decisive components satisfy exploratory tolerances |
| precision | multi-step loss/grad-norm curves, four error formulas, distributions, repeats and input-contract validation | configured migration or self-consistency standard |
| performance | step time, throughput, memory/MFU when available, profiler window, operator/kernel and official Ascend analysis links | diagnostic only unless an external regression target is declared |
| graph | the formal precision/performance reports with eager/graph policy metadata | same-device eager-versus-graph precision standard; performance is separate |
| combination | linked precision suite plus per-topology endpoint timing and speedup | precision verdict remains authoritative; objectives do not weaken one another |
| stability | completed steps, duration, stalls, process status, finite loss/grad norm | normal completion, all steps, no stall/nonfinite values, minimum duration |
| checkpoint | failure matrix, selected checkpoint per rank, replay, boundary/final state and cleanup | every requested restart/fault scenario satisfies exact or tolerance contract |

Each suite README lists every public parameter, its choices, default, output
layout, and the distinction between PASS/FAIL evidence and diagnostic-only
information.

## Experiment output synchronization

Generated `*_reports/` directories are tracked by Git so HTML, JSON, Markdown,
and other compact analysis results can be reviewed, compared, and synchronized
with the code that produced them. Fixtures, metric artifacts, raw runs, profiler
outputs, and other large generated data remain excluded from Git and are
transferred with GitHub Releases.

Install and authenticate GitHub CLI first (`gh auth login`) when transferring
the large outputs. Reports remain accepted in release archives as a convenient
self-contained experiment snapshot, even though Git is their primary sync path.
The release tag, release title, and archive name are derived from the experiment
name. Uploading collects matching directories and directly named report files
under every standard parity, precision, performance, stability, checkpoint,
combination/graph, smoke, and direct-training output root:

```bash
python release_artifacts.py upload \
  migration-cuda-npu-single-bf16-random-s5000-b64-seq128-seed61
```

Most experiment families use one name across fixtures, runs, artifacts, and
reports. When an experiment intentionally reuses a fixture with a different
identity, or a legacy parity report has a custom name, add each dependency to
the same archive explicitly:

```bash
python release_artifacts.py upload <experiment-id> \
  --include <shared-fixture-id> \
  --include <custom-report-name-without-extension>
```

`--include` is repeatable. It changes archive contents only; the Release tag
and asset name remain `<experiment-id>`.

Release CLI parameters:

| Parameter | Purpose and choices | Default |
|---|---|---|
| `--repo OWNER/REPO` | GitHub repository used for upload/download. This global option appears before the subcommand. | `loveofguoke/torchtitan-test` |
| `upload <experiment>` | Discover the named output across every standard experiment root, archive it, write SHA-256, and create/update the same-named Release. | required upload action |
| `upload --include NAME` | Include another fixture/report/output identity in the same archive; repeat as needed. | none |
| `upload --repository-root PATH` | Repository root used for output discovery and relative archive paths. | current directory |
| `download <experiment>` | Download, verify, and restore one release archive. | required download action |
| `download --backend` | `gh` or trusted-network `wget`. | `gh` |
| `download --destination PATH` | Root below which repository-relative paths are restored. | current directory |
| `download --overwrite` | Permit archive files to replace existing local files. | disabled |

If a release with that name already exists, its archive and checksum assets are
replaced. Download and restore the original repository-relative directories:

```bash
python release_artifacts.py download \
  migration-cuda-npu-single-bf16-random-s5000-b64-seq128-seed61
```

On machines where `gh` cannot download because of certificate validation,
retain `gh` for uploads and select the `wget` download backend:

```bash
python release_artifacts.py download \
  migration-cuda-npu-single-bf16-random-s5000-b64-seq128-seed61 \
  --backend wget
```

The `wget` backend uses `--no-check-certificate`, so it should only be used on
a trusted network for this known GitHub repository. Both backends verify the
downloaded archive against its release SHA-256 asset, extract it into the
original repository-relative directories, and remove the temporary archive and
checksum file automatically.

The download refuses to replace existing files by default. Add `--overwrite`
only when the local copies are intentionally being refreshed. Use
`--repo OWNER/REPO` before the subcommand to select a repository other than
`loveofguoke/torchtitan-test`.
