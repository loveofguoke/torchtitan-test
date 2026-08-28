# fsdp2-cp4 eager versus Inductor

## Aggregate

| mode | median step | p90 step | tok/s/job | repeat drift | peak HBM |
|---|---:|---:|---:|---:|---:|
| eager | 1.477067 s | 1.499912 s | 5546.13 | 3.11% | 0.332 GiB |
| Inductor | 1.422781 s | 1.443244 s | 5757.74 | 0.22% | 0.332 GiB |

Steady-state speedup: **1.0382x** (+3.82%).
Cold-start break-even estimate: **3965 steps**.

The steady-state change is 3.82%. Treat changes below 5% as diagnostic on this node because several NPUs report Alarm and the eight-card graph batch overlapped an external NPU0 job.

## Per-run evidence

| mode | repeat | median step (s) | p90 (s) | tok/s/job | peak HBM (GiB) | record |
|---|---:|---:|---:|---:|---:|---|
| eager | 1 | 1.454427 | 1.473544 | 5632.46 | 0.332 | [process](../../../../runs/performance/8-card/fsdp2-cp4/eager-r1/readme.md) |
| eager | 2 | 1.499707 | 1.526279 | 5462.40 | 0.332 | [process](../../../../runs/performance/8-card/fsdp2-cp4/eager-r2/readme.md) |
| inductor | 1 | 1.424352 | 1.430508 | 5751.39 | 0.332 | [process](../../../../runs/performance/8-card/fsdp2-cp4/inductor-r1/readme.md) |
| inductor | 2 | 1.421210 | 1.455981 | 5764.10 | 0.295 | [process](../../../../runs/performance/8-card/fsdp2-cp4/inductor-r2/readme.md) |

Inductor diagnostics across both repeats: 2
compile markers, 0 graph breaks,
2 recompiles, and
0 backend failures.

## Boundary

This is a profiler-off performance comparison with fixed input and no precision
acceptance. The eight-card graph runs overlap an external single-card job on
physical NPU0 and all eight-card results are diagnostic. Repeat on an idle,
healthy node before accepting a sub-5% change.
