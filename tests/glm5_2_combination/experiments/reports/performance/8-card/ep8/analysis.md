# ep8 eager versus Inductor

## Aggregate

| mode | median step | p90 step | tok/s/job | repeat drift | peak HBM |
|---|---:|---:|---:|---:|---:|
| eager | 0.791739 s | 0.805211 s | 10346.84 | 1.05% | 2.599 GiB |
| Inductor | 0.777959 s | 0.792776 s | 10530.11 | 0.33% | 2.410 GiB |

Steady-state speedup: **1.0177x** (+1.77%).
Cold-start break-even estimate: **27549 steps**.

The steady-state change is 1.77%. Treat changes below 5% as diagnostic on this node because several NPUs report Alarm and the eight-card graph batch overlapped an external NPU0 job.

## Per-run evidence

| mode | repeat | median step (s) | p90 (s) | tok/s/job | peak HBM (GiB) | record |
|---|---:|---:|---:|---:|---:|---|
| eager | 1 | 0.787602 | 0.802209 | 10401.20 | 2.599 | [process](../../../../runs/performance/8-card/ep8/eager-r1/readme.md) |
| eager | 2 | 0.795877 | 0.808212 | 10293.05 | 2.599 | [process](../../../../runs/performance/8-card/ep8/eager-r2/readme.md) |
| inductor | 1 | 0.779238 | 0.790183 | 10512.83 | 2.410 | [process](../../../../runs/performance/8-card/ep8/inductor-r1/readme.md) |
| inductor | 2 | 0.776681 | 0.795369 | 10547.45 | 2.410 | [process](../../../../runs/performance/8-card/ep8/inductor-r2/readme.md) |

Inductor diagnostics across both repeats: 2
compile markers, 0 graph breaks,
2 recompiles, and
0 backend failures.

## Boundary

This is a profiler-off performance comparison with fixed input and no precision
acceptance. The eight-card graph runs overlap an external single-card job on
physical NPU0 and all eight-card results are diagnostic. Repeat on an idle,
healthy node before accepting a sub-5% change.
