# GLM 5.2 training stability benchmark

This soak test is separate from precision comparison. It verifies that one
training run completes every requested optimizer step, emits finite loss and
global grad norm values, does not stop making progress, exits normally, and
runs for at least the requested wall-clock duration.

The default run requests 20,000 steps and at least one hour. A faster run that
finishes every step in less than one hour is reported as
`INSUFFICIENT_DURATION`, not `PASS`; increase `--steps` and rerun.

## Command-line parameters

| Parameter | Purpose and choices | Default |
|---|---|---|
| `--data` | Generate the shared step-0 checkpoint and fixed token plan, then exit. | disabled; an ordinary run creates it only when missing |
| `--device` | `cuda` or `npu`; inferred from the exported visibility variable when omitted. | inferred |
| `--topology` | One common topology or `all`. | `single` |
| `--precision` | `fp32`, `bf16`, or `full-bf16`. | `bf16` |
| `--steps` | Requested optimizer steps. | `20000` |
| `--minimum-hours` | Minimum wall-clock duration required for `PASS`. | `1.0` |
| `--stall-timeout-minutes` | Fail and clean up when no new metric arrives for this long. | `20.0` |
| `--local-batch-size` | Per-DP-rank local batch. | `8` |
| `--global-batch-size` | Global samples per optimizer step. | `64` |
| `--sequence-length` | Tokens per sample. | `128` |
| `--seed` | Model, data, and trainer seed. | `61` |
| `--extra-train-arg` | Append one raw TorchTitan argument; repeat for multiple graph or training feature arguments. | none |
| `--run-tag` | Short label included in output identity, for example `inductor`. | unset |
| `--force` | Replace completed PASS members. Without it, incomplete/failed/insufficient members are archived and retried. | disabled |

Keep precision, steps, batch sizes, sequence length, seed, run tag, and every
extra training argument identical between `--data` and execution. `--topology`
does not change the fixture; `--data --topology all` therefore still creates
one reusable fixture.

## Report contents and acceptance

Each topology report records requested and completed steps, wall-clock start
and end times, elapsed hours, progress/stall observations, process exit status,
and finite-value checks for every captured loss and global grad norm. Runtime
logs and raw metrics remain in `stability_runs`; the portable JSON/HTML summary
contains the evidence needed for the stability verdict.

`PASS` requires normal process completion, every requested step, finite loss
and grad norm, no stall timeout, and elapsed duration at least
`--minimum-hours`. A clean but too-short job is
`INSUFFICIENT_DURATION`, not PASS. With `--topology all`, every topology writes
its own report below the shared suite directory; rerunning the same command
skips PASS members and resumes only missing or unsuccessful members.

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

The same entry point supports every registered multi-card topology. For
example, the following NPU command uses `fsdp8`:

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

The corresponding GPU distributed example keeps the same `fsdp8` topology and
training contract:

```bash
unset ASCEND_RT_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python tests/glm5_2_stability/stability_benchmark.py \
  --device cuda --topology fsdp8 \
  --precision bf16 --steps 20000 --minimum-hours 1 \
  --local-batch-size 8 --global-batch-size 64 \
  --sequence-length 128 --seed 61
```

To run every registered GPU topology through eight ranks with one fixture, use:

```bash
unset ASCEND_RT_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python tests/glm5_2_stability/stability_benchmark.py \
  --device cuda --topology all \
  --precision bf16 --steps 20000 --minimum-hours 1 \
  --local-batch-size 8 --global-batch-size 64 \
  --sequence-length 128 --seed 61
```

The corresponding NPU all-topology workflow is:

```bash
unset CUDA_VISIBLE_DEVICES
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python tests/glm5_2_stability/stability_benchmark.py \
  --device npu --topology all \
  --precision bf16 --steps 20000 --minimum-hours 1 \
  --local-batch-size 8 --global-batch-size 64 \
  --sequence-length 128 --seed 61
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

## NPU graph-mode stability

Stability accepts graph arguments without owning compiler policy. Repeat the
same `--extra-train-arg` values and `--run-tag` in the data and run phases so
the generated suite identity is unambiguous. Current compiled execution is
NPU-only.

```bash
unset CUDA_VISIBLE_DEVICES
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python tests/glm5_2_stability/stability_benchmark.py \
  --data --device npu --topology all \
  --precision bf16 --steps 20000 \
  --local-batch-size 8 --global-batch-size 64 \
  --sequence-length 128 --seed 61 \
  --run-tag inductor \
  --extra-train-arg=--compile.enable \
  --extra-train-arg=--compile.components=model \
  --extra-train-arg=--compile.backend=inductor

python tests/glm5_2_stability/stability_benchmark.py \
  --device npu --topology all \
  --precision bf16 --steps 20000 \
  --local-batch-size 8 --global-batch-size 64 \
  --sequence-length 128 --seed 61 --minimum-hours 1 \
  --run-tag inductor \
  --extra-train-arg=--compile.enable \
  --extra-train-arg=--compile.components=model \
  --extra-train-arg=--compile.backend=inductor
```

Use `--topology single`, one distributed topology, or `--topology all` with
the same command shape.
