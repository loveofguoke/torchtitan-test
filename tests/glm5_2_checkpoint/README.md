# GLM 5.2 checkpoint validation

This benchmark tests TorchTitan's existing distributed-checkpoint manager. It
does not implement an alternative checkpoint format.

The test runs two equivalent jobs:

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

## Single-card GPU

```bash
export CUDA_VISIBLE_DEVICES=7
unset ASCEND_RT_VISIBLE_DEVICES

python tests/glm5_2_checkpoint/checkpoint_benchmark.py \
  --device cuda --topology single \
  --precision bf16 --total-steps 20 --split-step 10
```

## Single-card NPU

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
unset CUDA_VISIBLE_DEVICES

python tests/glm5_2_checkpoint/checkpoint_benchmark.py \
  --device npu --topology single \
  --precision bf16 --total-steps 20 --split-step 10
```

## Distributed topology

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python tests/glm5_2_checkpoint/checkpoint_benchmark.py \
  --device cuda --topology fsdp8 \
  --precision bf16 --total-steps 20 --split-step 10
```

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
