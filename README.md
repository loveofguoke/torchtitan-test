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

## Stability and checkpoint validation

The stability soak test records step progress, duration, finite loss, and
finite global grad norm for single-card or distributed GPU/NPU training. See
[tests/glm5_2_stability/README.md](tests/glm5_2_stability/README.md).

The checkpoint test compares uninterrupted training with a real process
restart at an intermediate full DCP checkpoint. It verifies exact fixed-token
data continuation, loss and grad norm, trainer/dataloader/scheduler state, and
the final model and optimizer state. See
[tests/glm5_2_checkpoint/README.md](tests/glm5_2_checkpoint/README.md).

## Experiment artifact releases

Large generated experiment directories are transferred with GitHub Releases
instead of Git. Install and authenticate GitHub CLI first (`gh auth login`).
The release tag, release title, and archive name are derived from the experiment
directory name. Uploading collects matching directories under the standard
artifact, fixture, and run roots:

```bash
python release_artifacts.py upload \
  migration-cuda-npu-single-bf16-random-s5000-b64-seq128-seed61
```

If a release with that name already exists, its archive and checksum assets are
replaced. Download and restore the original repository-relative directories:

```bash
python release_artifacts.py download \
  migration-cuda-npu-single-bf16-random-s5000-b64-seq128-seed61
```

The download refuses to replace existing files by default. Add `--overwrite`
only when the local copies are intentionally being refreshed. Use
`--repo OWNER/REPO` before the subcommand to select a repository other than
`loveofguoke/torchtitan-test`.
