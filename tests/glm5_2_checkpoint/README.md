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
