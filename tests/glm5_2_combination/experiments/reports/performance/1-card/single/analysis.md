# single eager versus Inductor

## Aggregate

| mode | median step | p90 step | tok/s/job | repeat drift | peak HBM |
|---|---:|---:|---:|---:|---:|
| eager | 3.058331 s | 4.243902 s | 2678.59 | 4.04% | 1.323 GiB |
| Inductor | 3.004804 s | 3.715388 s | 2726.30 | 4.99% | 1.161 GiB |

Steady-state speedup: **1.0178x** (+1.78%).
Cold-start break-even estimate: **7405 steps**.

The steady-state change is 1.78%. Treat changes below 5% as diagnostic on this node because several NPUs report Alarm and the eight-card graph batch overlapped an external NPU0 job.

## Per-run evidence

| mode | repeat | median step (s) | p90 (s) | tok/s/job | peak HBM (GiB) | record |
|---|---:|---:|---:|---:|---:|---|
| eager | 1 | 3.118828 | 5.434489 | 2626.63 | 1.323 | [process](../../../../runs/performance/1-card/single/eager-r1/readme.md) |
| eager | 2 | 2.997833 | 3.053315 | 2732.64 | 1.323 | [process](../../../../runs/performance/1-card/single/eager-r2/readme.md) |
| inductor | 1 | 3.077908 | 4.442787 | 2661.55 | 1.161 | [process](../../../../runs/performance/1-card/single/inductor-r1/readme.md) |
| inductor | 2 | 2.931700 | 2.987988 | 2794.28 | 1.161 | [process](../../../../runs/performance/1-card/single/inductor-r2/readme.md) |

Inductor diagnostics across both repeats: 2
compile markers, 0 graph breaks,
2 recompiles, and
0 backend failures.

## Boundary

This is a profiler-off performance comparison with fixed input and no precision
acceptance. The eight-card graph runs overlap an external single-card job on
physical NPU0 and all eight-card results are diagnostic. Repeat on an idle,
healthy node before accepting a sub-5% change.
