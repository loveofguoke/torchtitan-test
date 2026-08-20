# GLM 5.2 training stability benchmark

This soak test is separate from precision comparison. It verifies that one
training run completes every requested optimizer step, emits finite loss and
global grad norm values, does not stop making progress, exits normally, and
runs for at least the requested wall-clock duration.

The default run requests 20,000 steps and at least one hour. A faster run that
finishes every step in less than one hour is reported as
`INSUFFICIENT_DURATION`, not `PASS`; increase `--steps` and rerun.

## Prepare the reproducible training fixture

Prepare one shared step-0 checkpoint and fixed token plan before the soak run.
The token plan covers all requested steps and every global-batch position:

```bash
python tests/glm5_2_stability/stability_benchmark.py \
  --data --device npu \
  --precision bf16 --steps 20000 \
  --local-batch-size 8 --global-batch-size 64 \
  --sequence-length 128 --seed 61
```

Either GPU or NPU may generate the fixture. Synchronize its directory under
`stability_fixtures/` when the execution host differs. All topologies reuse the
same fixture when these training settings match. `--data --force` rebuilds the
fixture; an ordinary run automatically creates it only when it is missing.

Single-card NPU BF16:

```bash
unset CUDA_VISIBLE_DEVICES
export ASCEND_RT_VISIBLE_DEVICES=4

python tests/glm5_2_stability/stability_benchmark.py \
  --device npu \
  --topology single \
  --precision bf16 \
  --steps 20000 \
  --local-batch-size 8 --global-batch-size 64 \
  --sequence-length 128 --seed 61 \
  --minimum-hours 1
```

Single-card GPU BF16:

```bash
unset ASCEND_RT_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=7

python tests/glm5_2_stability/stability_benchmark.py \
  --device cuda \
  --topology single \
  --precision bf16 \
  --steps 20000 \
  --local-batch-size 8 --global-batch-size 64 \
  --sequence-length 128 --seed 61 \
  --minimum-hours 1
```

The same entry point supports registered multi-card topologies. For example,
an eight-card NPU FSDP soak test is:

```bash
unset CUDA_VISIBLE_DEVICES
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python tests/glm5_2_stability/stability_benchmark.py \
  --device npu \
  --topology fsdp8 \
  --precision bf16 \
  --steps 20000 \
  --local-batch-size 8 --global-batch-size 64 \
  --sequence-length 128 --seed 61 \
  --minimum-hours 1
```

Progress is printed once per minute. By default, a run with no new metric for
20 minutes is terminated and marked `FAIL`; override this using
`--stall-timeout-minutes`. Local trainer output and the complete runtime log are
stored under `stability_runs/` and ignored by Git. The portable HTML and JSON
summaries are stored under `stability_reports/` with device, topology,
precision, steps, batch size, sequence length, and seed in their names.
With `--topology all`, every topology is a child of one shared suite directory.
Completed PASS members are skipped, while incomplete, failed, or
`INSUFFICIENT_DURATION` members are archived and retried. Use `--force` only
when completed members must also run again.
