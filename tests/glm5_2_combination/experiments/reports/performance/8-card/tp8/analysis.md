# tp8 eager versus Inductor

## Aggregate

| mode | median step | p90 step | tok/s/job | repeat drift | peak HBM |
|---|---:|---:|---:|---:|---:|
| eager | 6.424063 s | 6.610020 s | 1275.21 | 0.06% | 0.262 GiB |
| Inductor | 4.240238 s | 4.339878 s | 1931.97 | 2.60% | 0.300 GiB |

Steady-state speedup: **1.5150x** (+51.50%).
Cold-start break-even estimate: **308 steps**.

Inductor improves the TP-containing path by 51.50%. This is consistent with graph capture reducing model/host launch overhead, but it does not remove the topology's collective count.

## Per-run evidence

| mode | repeat | median step (s) | p90 (s) | tok/s/job | peak HBM (GiB) | record |
|---|---:|---:|---:|---:|---:|---|
| eager | 1 | 6.422249 | 6.676168 | 1275.57 | 0.262 | [process](../../../../runs/performance/8-card/tp8/eager-r1/readme.md) |
| eager | 2 | 6.425877 | 6.543872 | 1274.85 | 0.262 | [process](../../../../runs/performance/8-card/tp8/eager-r2/readme.md) |
| inductor | 1 | 4.294679 | 4.457901 | 1907.48 | 0.300 | [process](../../../../runs/performance/8-card/tp8/inductor-r1/readme.md) |
| inductor | 2 | 4.185798 | 4.221855 | 1957.09 | 0.216 | [process](../../../../runs/performance/8-card/tp8/inductor-r2/readme.md) |

Inductor diagnostics across both repeats: 2
compile markers, 0 graph breaks,
2 recompiles, and
0 backend failures.

## Boundary

This is a profiler-off performance comparison with fixed input and no precision
acceptance. The eight-card graph runs overlap an external single-card job on
physical NPU0 and all eight-card results are diagnostic. Repeat on an idle,
healthy node before accepting a sub-5% change.
