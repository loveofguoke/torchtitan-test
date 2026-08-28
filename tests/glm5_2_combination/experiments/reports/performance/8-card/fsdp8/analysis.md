# fsdp8 eager versus Inductor

## Aggregate

| mode | median step | p90 step | tok/s/job | repeat drift | peak HBM |
|---|---:|---:|---:|---:|---:|
| eager | 0.425111 s | 0.446665 s | 19270.26 | 7.07% | 1.101 GiB |
| Inductor | 0.417484 s | 0.424470 s | 19622.31 | 4.02% | 0.939 GiB |

Steady-state speedup: **1.0183x** (+1.83%).
Cold-start break-even estimate: **21 steps**.

The steady-state change is 1.83%. Treat changes below 5% as diagnostic on this node because several NPUs report Alarm and the eight-card graph batch overlapped an external NPU0 job.

## Per-run evidence

| mode | repeat | median step (s) | p90 (s) | tok/s/job | peak HBM (GiB) | record |
|---|---:|---:|---:|---:|---:|---|
| eager | 1 | 0.410603 | 0.429547 | 19951.13 | 1.101 | [process](../../../../runs/performance/8-card/fsdp8/eager-r1/readme.md) |
| eager | 2 | 0.439618 | 0.463783 | 18634.34 | 1.101 | [process](../../../../runs/performance/8-card/fsdp8/eager-r2/readme.md) |
| inductor | 1 | 0.425709 | 0.428283 | 19243.18 | 0.939 | [process](../../../../runs/performance/8-card/fsdp8/inductor-r1/readme.md) |
| inductor | 2 | 0.409259 | 0.420657 | 20016.68 | 0.939 | [process](../../../../runs/performance/8-card/fsdp8/inductor-r2/readme.md) |

Inductor diagnostics across both repeats: 2
compile markers, 0 graph breaks,
2 recompiles, and
0 backend failures.

## Boundary

This is a profiler-off performance comparison with fixed input and no precision
acceptance. The eight-card graph runs overlap an external single-card job on
physical NPU0 and all eight-card results are diagnostic. Repeat on an idle,
healthy node before accepting a sub-5% change.
