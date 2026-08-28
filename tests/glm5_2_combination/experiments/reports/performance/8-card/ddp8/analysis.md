# ddp8 eager versus Inductor

## Aggregate

| mode | median step | p90 step | tok/s/job | repeat drift | peak HBM |
|---|---:|---:|---:|---:|---:|
| eager | 0.420374 s | 0.433637 s | 19487.43 | 2.98% | 1.263 GiB |
| Inductor | 0.427276 s | 0.434478 s | 19172.60 | 4.39% | 1.101 GiB |

Steady-state speedup: **0.9838x** (-1.62%).
Cold-start break-even estimate: **not reached**.

The steady-state change is -1.62%. Treat changes below 5% as diagnostic on this node because several NPUs report Alarm and the eight-card graph batch overlapped an external NPU0 job.

## Per-run evidence

| mode | repeat | median step (s) | p90 (s) | tok/s/job | peak HBM (GiB) | record |
|---|---:|---:|---:|---:|---:|---|
| eager | 1 | 0.426555 | 0.447176 | 19205.02 | 1.263 | [process](../../../../runs/performance/8-card/ddp8/eager-r1/readme.md) |
| eager | 2 | 0.414192 | 0.420097 | 19778.27 | 1.263 | [process](../../../../runs/performance/8-card/ddp8/eager-r2/readme.md) |
| inductor | 1 | 0.418091 | 0.422361 | 19593.84 | 1.101 | [process](../../../../runs/performance/8-card/ddp8/inductor-r1/readme.md) |
| inductor | 2 | 0.436462 | 0.446595 | 18769.09 | 1.101 | [process](../../../../runs/performance/8-card/ddp8/inductor-r2/readme.md) |

Inductor diagnostics across both repeats: 2
compile markers, 0 graph breaks,
2 recompiles, and
0 backend failures.

## Boundary

This is a profiler-off performance comparison with fixed input and no precision
acceptance. The eight-card graph runs overlap an external single-card job on
physical NPU0 and all eight-card results are diagnostic. Repeat on an idle,
healthy node before accepting a sub-5% change.
