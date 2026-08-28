# fsdp2-tp4-ep8 eager versus Inductor

## Aggregate

| mode | median step | p90 step | tok/s/job | repeat drift | peak HBM |
|---|---:|---:|---:|---:|---:|
| eager | 3.768185 s | 4.021529 s | 2173.99 | 4.25% | 0.715 GiB |
| Inductor | 2.589503 s | 2.615316 s | 3163.54 | 1.66% | 0.662 GiB |

Steady-state speedup: **1.4552x** (+45.52%).
Cold-start break-even estimate: **407 steps**.

Inductor improves the TP-containing path by 45.52%. This is consistent with graph capture reducing model/host launch overhead, but it does not remove the topology's collective count.

## Per-run evidence

| mode | repeat | median step (s) | p90 (s) | tok/s/job | peak HBM (GiB) | record |
|---|---:|---:|---:|---:|---:|---|
| eager | 1 | 3.846671 | 4.327546 | 2129.63 | 0.715 | [process](../../../../runs/performance/8-card/fsdp2-tp4-ep8/eager-r1/readme.md) |
| eager | 2 | 3.689700 | 3.715513 | 2220.23 | 0.697 | [process](../../../../runs/performance/8-card/fsdp2-tp4-ep8/eager-r2/readme.md) |
| inductor | 1 | 2.610824 | 2.633678 | 3137.71 | 0.662 | [process](../../../../runs/performance/8-card/fsdp2-tp4-ep8/inductor-r1/readme.md) |
| inductor | 2 | 2.568182 | 2.596953 | 3189.80 | 0.646 | [process](../../../../runs/performance/8-card/fsdp2-tp4-ep8/inductor-r2/readme.md) |

Inductor diagnostics across both repeats: 2
compile markers, 0 graph breaks,
2 recompiles, and
0 backend failures.

## Boundary

This is a profiler-off performance comparison with fixed input and no precision
acceptance. The eight-card graph runs overlap an external single-card job on
physical NPU0 and all eight-card results are diagnostic. Repeat on an idle,
healthy node before accepting a sub-5% change.
