# DDP long-run precision V2

`ddp_long_v2.py` re-scores existing GPU/NPU DDP artifacts. It never launches
training and does not modify or regenerate a fixture.

The core gate follows the public MindStudio signal that first-step or
long-stable loss mean error above 1% requires precision investigation. The
window and P99 checks below are project guardrails, not Huawei product-line
delivery thresholds.

## Default decision profile

- at least two GPU artifacts and two NPU artifacts;
- identical checksummed training contracts and a pure multi-rank DDP topology;
- 5000 matching optimizer-step observations;
- finite loss and global grad norm at every step;
- first-step, first-100-step, post-warmup, and final-20% loss MARE at most 1%;
- post-warmup pointwise relative-error P99 at most 5%;
- fewer than three consecutive non-overlapping 100-step windows whose loss
  MARE exceeds 1%;
- no removal of the largest errors.

Global grad norm and bitwise repeat equality are reported as diagnostics. They
do not independently decide PASS or FAIL. If either platform's repeat loss
MARE exceeds 1%, the result is INCONCLUSIVE instead of attributing the
instability to GPU/NPU migration.

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
input. A nonzero status is therefore expected for a valid precision failure;
do not hide it with `|| true` in CI.

## Reuse policy

Existing data, tokenizer, token plan, seed checkpoint, and formal training
artifacts remain reusable because this command changes only offline decision
logic. A new training run is required only for future rank-level parameter
synchronization or validation-set metrics that were not captured in the old
artifacts.
