# GLM 5.2 checkpoint validation

This benchmark tests TorchTitan's existing distributed-checkpoint manager. It
does not implement an alternative checkpoint format.

The default test runs two equivalent jobs:

1. A baseline trains continuously from step 1 to `N`.
2. A resumed job is configured for the same `N` steps, trains to step `K`,
   saves a full interval checkpoint, exits normally, starts a new process,
   loads that checkpoint, and continues from `K + 1` to `N`.

Keeping `training.steps=N` in both processes is important: changing it to `K`
would change the LR schedule and invalidate the comparison before the restart.

The report checks:

- the step-`K` checkpoint contains model, optimizer, LR scheduler, dataloader,
  and trainer state;
- the fixed-token input sequence after restart is exactly the baseline input;
- all loss and global grad-norm steps match;
- the final logical DCP state matches, including model and optimizer tensors.
- a new process started after normal completion loads the final checkpoint on
  every rank and exits without repeating a training step.

Use `--failure-mode all` for the fault-recovery suite. It reuses one continuous
baseline and validates these interruption paths independently:

- controlled normal exit immediately after a completed interval checkpoint;
- terminal-style `SIGINT` (Ctrl+C), `SIGTERM`, and `SIGKILL` delivered to the
  complete `torchrun` process group;
- an uncaught exception or `SIGKILL` on one worker rank for distributed jobs;
- a higher checkpoint directory that is visible without DCP commit metadata.

Abrupt failures are injected one step after the last complete checkpoint. The
restart must select the earlier complete step on every rank, discard the
uncommitted work, replay the same fixed tokens, and converge to the exact same
logical distributed checkpoint as the uninterrupted baseline. The benchmark
also audits that every trainable parameter registered with an optimizer has
persisted state; a directory containing `.metadata` is not treated as
sufficient evidence by itself. Frozen model parameters, including the pretrained
GLM DSA indexer, remain in the model checkpoint but are excluded from this
optimizer-state audit.

Each launched training job owns a separate POSIX process group. The controller
always cleans that group in a `finally` path, escalates from SIGTERM to SIGKILL
when needed, and acts as a Linux child subreaper so abruptly killed torchrun
workers do not remain as orphaned or zombie processes. Every runtime log ends
with a `Process cleanup` record; a surviving process group fails the test.

## Command-line parameters

| Parameter | Purpose and choices | Default |
|---|---|---|
| `--data` | Generate the shared step-0 checkpoint and fixed token plan, then exit. | disabled; ordinary execution may create a missing fixture for compatibility |
| `--device` | `cuda` or `npu`; normally inferred from the exported visibility variable when omitted. | inferred |
| `--topology` | One common topology or `all`. | `single` |
| `--precision` | `fp32`, `bf16`, or `full-bf16`. | `bf16` |
| `--total-steps` | Final optimizer step for uninterrupted and resumed jobs. | `20` |
| `--split-step` | Complete checkpoint step from which the restarted job must resume. | `10` |
| `--local-batch-size` | Per-DP-rank local batch used by the shared token contract. | `8` |
| `--global-batch-size` | Global samples per optimizer step. | `64` |
| `--sequence-length` | Tokens per sample. | `128` |
| `--seed` | Model, data, and trainer seed. | `61` |
| `--comparison` | `exact` for bitwise equality, or `tolerance` for numerical state/metric thresholds while structural and input checks remain exact. | `exact` |
| `--state-atol`, `--state-rtol` | Absolute and relative checkpoint-tensor tolerances in tolerance mode. | `0.01`, `0.01` |
| `--loss-atol`, `--loss-rtol` | Absolute and relative loss tolerances in tolerance mode. | `0.01`, `0.01` |
| `--grad-relative-limit` | Maximum relative grad-norm difference in tolerance mode. | `0.05` |
| `--async-mode` | TorchTitan save path: `disabled`, `async`, or `async_with_pinned_mem`. | `disabled` |
| `--failure-mode` | Repeatable: `graceful`, `sigint`, `sigterm`, `sigkill`, `incomplete`, `rank-error`, `rank-sigkill`, or `all`. | `graceful` |
| `--failure-rank` | Target rank for `rank-error` and `rank-sigkill`. | rank 1 distributed; rank 0 single-card |
| `--interrupt-step` | Step at which abrupt failure is injected. Must be after the split checkpoint. | `split-step + 1` |
| `--failure-timeout` | Seconds allowed for a failure scenario before the controller aborts it. | `1800` |
| `--restart-delay` | Seconds allowed for accelerator/process cleanup before restart. | `2` |
| `--extra-train-arg` | Append one raw TorchTitan argument; repeat for graph or other orthogonal features. | none |
| `--run-tag` | Short output-identity label for feature variants such as `inductor`. | unset |
| `--force` | Start a new generation for the selected topology range. All selected members are removed before execution; rerun without it after interruption to skip completed members and continue. | disabled |

The fixture-defining options (`precision`, steps, batches, sequence length,
seed, run tag, extra training arguments, async mode, comparison mode, and
requested failure modes) must match between `--data` and execution. For
`--topology all`, one topology-independent fixture is created and all members
are stored below one suite directory.

## Report contents and acceptance

The report contains one row per failure mode with the injected interruption,
observed exit status, actual checkpoint loaded on every rank, incomplete-latest
fallback result, process-cleanup result, and scenario verdict. Expanded
diagnostics show loss/grad-norm comparisons before and after restart, the first
divergent step, token replay, state fingerprints at save/load boundaries, and
final DCP differences grouped by model, optimizer, scheduler, dataloader, and
trainer state.

`exact` requires identical logical state and trajectory. `tolerance` applies
the configured numerical limits but never relaxes checkpoint selection,
structure, loaded step, dataloader/input order, or cleanup correctness. The
topology-suite index records completed, failed, and pending members; a member
passes only when every requested failure mode passes.

## Prepare synchronized inputs

Checkpoint equivalence requires more than using the same seed. First prepare a
shared fixture containing the step-0 model checkpoint, a fixed token plan for
every training step and global-batch position, and their integrity metadata:

```bash
python tests/glm5_2_checkpoint/checkpoint_benchmark.py \
  --data --device cuda \
  --precision bf16 --total-steps 20 \
  --local-batch-size 8 --global-batch-size 64 \
  --sequence-length 128 --seed 61
```

The fixture can instead be generated with `--device npu`. Generate it once on
either host and synchronize its directory under `checkpoint_fixtures/` when
different servers participate. The fixture is topology-independent: single,
DDP, FSDP, TP, PP, EP, and combined topologies reuse it when the precision,
step count, batch configuration, sequence length, and seed are unchanged.

`--data --force` recreates only the fixture. A benchmark command still prepares
a missing fixture automatically for backward compatibility, but the explicit
`--data` phase is recommended for reproducible experiments.

## Single-card GPU

```bash
export CUDA_VISIBLE_DEVICES=7
unset ASCEND_RT_VISIBLE_DEVICES

python tests/glm5_2_checkpoint/checkpoint_benchmark.py \
  --device cuda --topology single \
  --precision bf16 --total-steps 20 --split-step 10 \
  --local-batch-size 8 --global-batch-size 64 \
  --sequence-length 128 --seed 61
```

## Single-card NPU

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
unset CUDA_VISIBLE_DEVICES

python tests/glm5_2_checkpoint/checkpoint_benchmark.py \
  --device npu --topology single \
  --precision bf16 --total-steps 20 --split-step 10 \
  --local-batch-size 8 --global-batch-size 64 \
  --sequence-length 128 --seed 61
```

## Distributed topology

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python tests/glm5_2_checkpoint/checkpoint_benchmark.py \
  --device cuda --topology fsdp8 \
  --precision bf16 --total-steps 20 --split-step 10 \
  --local-batch-size 8 --global-batch-size 64 \
  --sequence-length 128 --seed 61 \
  --failure-mode all
```

The same command shape works for NPU. Set
`ASCEND_RT_VISIBLE_DEVICES`, select `--device npu`, and keep the chosen topology
and training configuration unchanged. To run one failure path, use for example
`--failure-mode sigint` or `--failure-mode rank-sigkill`. Omitting the option
keeps the original controlled-resume behavior and its command unchanged.

After preparing the shared fixture above, the corresponding NPU distributed
validation changes only the device selection. This example uses `fsdp8`, which
can be replaced by any registered topology:

```bash
unset CUDA_VISIBLE_DEVICES
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python tests/glm5_2_checkpoint/checkpoint_benchmark.py \
  --device npu --topology fsdp8 \
  --precision bf16 --total-steps 20 --split-step 10 \
  --local-batch-size 8 --global-batch-size 64 \
  --sequence-length 128 --seed 61 --failure-mode all
```

To run every registered GPU topology through eight ranks with one fixture, use:

```bash
unset ASCEND_RT_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python tests/glm5_2_checkpoint/checkpoint_benchmark.py \
  --device cuda --topology all \
  --precision bf16 --total-steps 20 --split-step 10 \
  --local-batch-size 8 --global-batch-size 64 \
  --sequence-length 128 --seed 61 --failure-mode all
```

The corresponding NPU all-topology workflow is:

```bash
unset CUDA_VISIBLE_DEVICES
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python tests/glm5_2_checkpoint/checkpoint_benchmark.py \
  --device npu --topology all \
  --precision bf16 --total-steps 20 --split-step 10 \
  --local-batch-size 8 --global-batch-size 64 \
  --sequence-length 128 --seed 61 --failure-mode all
```

To run the complete topology suite, select `--topology all`. All members use one
suite directory, with `single/`, `ddp8/`, `fsdp8/`, and the other topology
subdirectories beneath it. A later invocation skips members whose matching JSON
summary is complete and PASS, archives incomplete or failed member output, and
retries those members. `--force` is the only option that reruns completed
members. If a member fails, the suite writes a partial topology index and stops;
running the same command again retries that member and then continues with the
remaining topology directories.

`--interrupt-step` defaults to `split-step + 1`. This deliberately creates
work that was computed but not committed, so the test verifies replay rather
than only restarting exactly on a save boundary. `--failure-rank` defaults to
rank 1 for distributed jobs and rank 0 for a single-card job.

The default `--comparison exact` is the strict checkpoint criterion. If a
backend is already known to be non-bitwise deterministic, use
`--comparison tolerance`; input order, restored step, dataloader state, and
non-numerical checkpoint state remain exact requirements.

Use `--async-mode async` or `--async-mode async_with_pinned_mem` to validate
TorchTitan's asynchronous save paths separately. Local trainer output and
temporary consolidated checkpoints stay under ignored `checkpoint_runs/`;
portable HTML and JSON reports are written to `checkpoint_reports/`.

The current TorchTitan checkpoint schema does not expose a standalone RNG
state entry. This benchmark therefore verifies every state scope that the
manager currently persists and checks the resulting training trajectory
end-to-end. If stochastic GLM training features are introduced, direct RNG
state persistence should be added upstream and made a required scope here.

## Resume-boundary diagnostics

The benchmark fingerprints the live local shard of every checkpoint scope at
the configured split-boundary save and again immediately after
`CheckpointManager.load()` in the new process. This instrumentation records
evidence only; TorchTitan still owns
checkpoint selection, loading, and training resumption. Fingerprints cover
model, optimizer, LR scheduler, dataloader, and train state. Changed optimizer
leaves retain their FQN and state-field path, such as `exp_avg`, `exp_avg_sq`,
or `step`.

The HTML report distinguishes three failure locations:

1. The saved step-`K` checkpoint differs from the uninterrupted baseline.
2. The in-memory state immediately after load differs from the state before
   process restart.
3. The boundary state is exact, but the first resumed training step diverges.

It also reports the first differing loss and grad-norm step, separate error
statistics through step `K` and after restart, and final DCP differences grouped
by state scope.

For TP localization, run only the controlled restart first. This avoids paying
the state-fingerprint cost for every fault mode while preserving the same
checkpoint and resume path:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset CUDA_VISIBLE_DEVICES

python tests/glm5_2_checkpoint/checkpoint_benchmark.py \
  --device npu --topology tp8 \
  --precision bf16 --total-steps 100 --split-step 50 \
  --local-batch-size 8 --global-batch-size 64 \
  --sequence-length 128 --seed 61 \
  --comparison exact \
  --async-mode disabled \
  --failure-mode graceful
```

Once the shared resume path is exact, rerun with `--failure-mode all` to cover
the external-signal, single-rank, and incomplete-checkpoint cases.

## NPU graph-mode checkpoint validation

Checkpoint validation remains owned by this package. Graph mode is an
independent TorchTitan training feature passed through `--extra-train-arg`, so
the checkpoint test still controls only interruption and observation while
TorchTitan owns save, selection, load, and resume. Current compiled execution
is NPU-only.

Use the same graph arguments and `--run-tag` in the data and execution phases:

```bash
unset CUDA_VISIBLE_DEVICES
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python tests/glm5_2_checkpoint/checkpoint_benchmark.py \
  --data --device npu --topology all \
  --precision bf16 --total-steps 100 --split-step 50 \
  --local-batch-size 8 --global-batch-size 64 \
  --sequence-length 128 --seed 61 \
  --failure-mode graceful --run-tag inductor \
  --extra-train-arg=--compile.enable \
  --extra-train-arg=--compile.components=model \
  --extra-train-arg=--compile.backend=inductor

python tests/glm5_2_checkpoint/checkpoint_benchmark.py \
  --device npu --topology all \
  --precision bf16 --total-steps 100 --split-step 50 \
  --local-batch-size 8 --global-batch-size 64 \
  --sequence-length 128 --seed 61 \
  --failure-mode graceful --run-tag inductor \
  --extra-train-arg=--compile.enable \
  --extra-train-arg=--compile.components=model \
  --extra-train-arg=--compile.backend=inductor
```

Replace `all` with `single` or one registered topology for focused debugging.
After graceful resume passes, change both commands to `--failure-mode all` for
the complete failure-recovery matrix; that is a distinct experiment identity.
