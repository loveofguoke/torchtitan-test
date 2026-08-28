# cp8 eager versus Inductor

## Aggregate

| mode | median step | p90 step | tok/s/job | repeat drift | peak HBM |
|---|---:|---:|---:|---:|---:|
| eager | 2.880988 s | 3.003082 s | 2843.47 | 1.73% | 0.204 GiB |
| Inductor | 2.813859 s | 2.832252 s | 2911.30 | 1.03% | 0.280 GiB |

Steady-state speedup: **1.0239x** (+2.39%).
Cold-start break-even estimate: **6106 steps**.

The steady-state change is 2.39%. Treat changes below 5% as diagnostic on this node because several NPUs report Alarm and the eight-card graph batch overlapped an external NPU0 job.

## Per-run evidence

| mode | repeat | median step (s) | p90 (s) | tok/s/job | peak HBM (GiB) | record |
|---|---:|---:|---:|---:|---:|---|
| eager | 1 | 2.856337 | 2.876759 | 2868.01 | 0.204 | [process](../../../../runs/performance/8-card/cp8/eager-r1/readme.md) |
| eager | 2 | 2.905639 | 3.129405 | 2819.35 | 0.204 | [process](../../../../runs/performance/8-card/cp8/eager-r2/readme.md) |
| inductor | 1 | 2.828329 | 2.848733 | 2896.41 | 0.280 | [process](../../../../runs/performance/8-card/cp8/inductor-r1/readme.md) |
| inductor | 2 | 2.799390 | 2.815770 | 2926.35 | 0.187 | [process](../../../../runs/performance/8-card/cp8/inductor-r2/readme.md) |

Inductor diagnostics across both repeats: 2
compile markers, 0 graph breaks,
2 recompiles, and
0 backend failures.

## Boundary

This is a profiler-off performance comparison with fixed input and no precision
acceptance. The eight-card graph runs overlap an external single-card job on
physical NPU0 and all eight-card results are diagnostic. Repeat on an idle,
healthy node before accepting a sub-5% change.
