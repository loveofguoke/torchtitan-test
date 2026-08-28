# ddp2 eager versus Inductor

## Aggregate

| mode | median step | p90 step | tok/s/job | repeat drift | peak HBM |
|---|---:|---:|---:|---:|---:|
| eager | 1.581147 s | 1.789337 s | 5181.05 | 6.94% | 1.323 GiB |
| Inductor | 1.524484 s | 1.536010 s | 5373.62 | 2.14% | 1.161 GiB |

Steady-state speedup: **1.0372x** (+3.72%).
Cold-start break-even estimate: **2187 steps**.

The steady-state change is 3.72%. Treat changes below 5% as diagnostic on this node because several NPUs report Alarm and the eight-card graph batch overlapped an external NPU0 job.

## Per-run evidence

| mode | repeat | median step (s) | p90 (s) | tok/s/job | peak HBM (GiB) | record |
|---|---:|---:|---:|---:|---:|---|
| eager | 1 | 1.528117 | 1.534357 | 5360.85 | 1.323 | [process](../../../../runs/performance/2-card/ddp2/eager-r1/readme.md) |
| eager | 2 | 1.634177 | 2.044316 | 5012.92 | 1.323 | [process](../../../../runs/performance/2-card/ddp2/eager-r2/readme.md) |
| inductor | 1 | 1.540654 | 1.554803 | 5317.22 | 1.161 | [process](../../../../runs/performance/2-card/ddp2/inductor-r1/readme.md) |
| inductor | 2 | 1.508314 | 1.517217 | 5431.23 | 1.161 | [process](../../../../runs/performance/2-card/ddp2/inductor-r2/readme.md) |

Inductor diagnostics across both repeats: 2
compile markers, 0 graph breaks,
2 recompiles, and
0 backend failures.

## Boundary

This is a profiler-off performance comparison with fixed input and no precision
acceptance. The eight-card graph runs overlap an external single-card job on
physical NPU0 and all eight-card results are diagnostic. Repeat on an idle,
healthy node before accepting a sub-5% change.
