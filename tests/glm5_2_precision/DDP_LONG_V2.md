# DDP long-run convergence alignment V2

`ddp_long_v2.py` re-scores one existing GPU/NPU distributed scenario, and
`distributed_long_convergence_v2.py` applies the same decision profile to the
complete eight-topology 5000-step matrix. Neither entry launches training or
modifies a fixture.

This profile answers whether the overall training loss curves have equivalent
shape and convergence. It deliberately does not require raw loss values at each
optimizer step to remain numerically close. Pointwise loss error, global grad
norm, and bitwise repeat equality remain visible as diagnostics for numerical
localization, but they do not independently decide PASS or FAIL.

The limits below are project convergence-equivalence guardrails. They are not a
claim of a universal MindStudio or hardware-vendor delivery threshold.

## Default decision profile

- at least two GPU artifacts and two NPU artifacts;
- identical checksummed training contracts and a multi-rank topology;
- 5000 matching optimizer-step observations;
- finite loss and global grad norm at every step;
- whole-run loss area-under-curve relative difference at most 2%;
- final-20% mean loss relative difference at most 2%;
- Pearson correlation of the post-warmup 100-step moving-average loss curves at
  least 0.99;
- every rolling 500-step mean-loss relative difference at most 3%.

Repeat runs on each platform are checked with the same curve-level conditions.
If either platform's own repeats do not meet them, the cross-platform result is
`INCONCLUSIVE` rather than attributing the instability to GPU/NPU migration.

## What no longer gates the result

The following values are reported but do not independently determine the V2
result:

- raw pointwise loss MARE, P95, P99, and maximum relative error;
- first-step loss difference;
- global grad-norm error;
- bitwise equality between repeats.

These diagnostics remain useful for locating a numerical divergence. The V2
delivery decision instead reflects overall curve shape and convergence.

## Offline comparison

The four paths must be artifact directories containing `manifest.json`,
`training_contract.json`, and `metrics.jsonl`:

```bash
python3 -m tests.glm5_2_precision.ddp_long_v2 \
  --gpu-artifact precision_artifacts/SCENARIO/reference-r1 \
  --gpu-artifact precision_artifacts/SCENARIO/reference-r2 \
  --npu-artifact precision_artifacts/SCENARIO/candidate-r1 \
  --npu-artifact precision_artifacts/SCENARIO/candidate-r2 \
  --output-dir precision_reports/SCENARIO/ddp-long-v2
```

The command writes:

- `ddp_long_v2_summary.json`, for automation;
- `ddp_long_v2_report.md`, for review.

Exit codes are `0` for PASS, `1` for FAIL, and `2` for INCONCLUSIVE or invalid
input. A nonzero status is therefore expected for a valid convergence failure;
do not hide it with `|| true` in CI.

## Eight-topology matrix reassessment

The batch entry evaluates DDP8, EP8, FSDP2-TP4, FSDP2-TP4-EP8, FSDP4-TP2,
FSDP8, PP8, and TP8 with one profile:

```bash
python3 -m tests.glm5_2_precision.distributed_long_convergence_v2 \
  --artifact-root precision_artifacts \
  --output-root precision_reports/distributed-long-convergence-v2
```

It writes an individual report below `<output-root>/<topology>/` and the matrix
files `distributed_long_convergence_v2_summary.json` and
`distributed_long_convergence_v2_report.md` at the output root. Missing or
invalid topology artifacts are recorded as `INVALID`, and make the matrix result
`INCONCLUSIVE`; the remaining topologies are still assessed.

## Loss-step plots

The same four artifacts per topology can be rendered without matplotlib or any
other plotting dependency:

```bash
python3 -m tests.glm5_2_precision.plot_distributed_loss_curves \
  --artifact-root precision_artifacts \
  --output-root precision_reports/distributed-long-convergence-v2/loss-curves
```

The command creates `loss_step_overview.svg` with the 100-step moving-average
GPU/NPU curves for all eight topologies, plus one `loss_step_<topology>.svg` per
topology. Each detailed chart contains both raw repeat curves as faint lines and
the repeat-mean 100-step moving-average curves as emphasized lines. The y-axis
uses the observed range for that topology so a persistent curve offset remains
visible. `loss_step_plots.json` records the source artifacts and generated plot
paths.

## Compact result summary

After the per-topology reports have been generated, their key decision values
can be condensed into one short table:

```bash
python3 -m tests.glm5_2_precision.summarize_distributed_long_results \
  --report-root precision_reports/distributed-long-convergence-v2
```

The table is printed to the terminal and also written as
`distributed_long_key_results.md` and `distributed_long_key_results.json` in the
report root. It retains only the topology result, step count, loss AUC
difference, final-20% loss difference, smoothed correlation, and worst
500-step-window loss difference.

## Reuse policy

Existing data, tokenizer, token plan, seed checkpoint, and formal training
artifacts remain reusable because this command changes only offline decision
logic. A new training run is required only when the old artifacts lack a needed
metric, or when code, fixture, topology, or training contract changes.
