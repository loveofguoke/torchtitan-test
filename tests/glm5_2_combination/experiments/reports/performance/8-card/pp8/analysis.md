# pp8 eager versus Inductor

## Aggregate

| mode | median step | p90 step | tok/s/job | repeat drift | peak HBM |
|---|---:|---:|---:|---:|---:|
| eager | 4.185296 s | 4.313297 s | 1957.33 | 4.51% | 0.042 GiB |
| Inductor | 3.849400 s | 3.902586 s | 2128.12 | 8.28% | 0.042 GiB |

Steady-state speedup: **1.0873x** (+8.73%).
Cold-start break-even estimate: **1305 steps**.

The 8.73% change is smaller than the TP family; pipeline bubble and stage readiness remain outside model-kernel fusion.

## Per-run evidence

| mode | repeat | median step (s) | p90 (s) | tok/s/job | peak HBM (GiB) | record |
|---|---:|---:|---:|---:|---:|---|
| eager | 1 | 4.277597 | 4.475797 | 1915.09 | 0.042 | [process](../../../../runs/performance/8-card/pp8/eager-r1/readme.md) |
| eager | 2 | 4.092995 | 4.150797 | 2001.47 | 0.042 | [process](../../../../runs/performance/8-card/pp8/eager-r2/readme.md) |
| inductor | 1 | 3.696362 | 3.749177 | 2216.23 | 0.042 | [process](../../../../runs/performance/8-card/pp8/inductor-r1/readme.md) |
| inductor | 2 | 4.002438 | 4.055996 | 2046.75 | 0.042 | [process](../../../../runs/performance/8-card/pp8/inductor-r2/readme.md) |

Inductor diagnostics across both repeats: 2
compile markers, 0 graph breaks,
0 recompiles, and
0 backend failures.

## Boundary

This is a profiler-off performance comparison with fixed input and no precision
acceptance. The eight-card graph runs overlap an external single-card job on
physical NPU0 and all eight-card results are diagnostic. Repeat on an idle,
healthy node before accepting a sub-5% change.
