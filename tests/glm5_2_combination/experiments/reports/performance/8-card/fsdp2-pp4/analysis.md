# fsdp2-pp4 eager versus Inductor

## Aggregate

| mode | median step | p90 step | tok/s/job | repeat drift | peak HBM |
|---|---:|---:|---:|---:|---:|
| eager | 3.488729 s | 3.556558 s | 2348.13 | 0.75% | 0.121 GiB |
| Inductor | 3.310308 s | 3.526028 s | 2474.69 | 0.57% | 0.121 GiB |

Steady-state speedup: **1.0539x** (+5.39%).
Cold-start break-even estimate: **551 steps**.

The 5.39% change is smaller than the TP family; pipeline bubble and stage readiness remain outside model-kernel fusion.

## Per-run evidence

| mode | repeat | median step (s) | p90 (s) | tok/s/job | peak HBM (GiB) | record |
|---|---:|---:|---:|---:|---:|---|
| eager | 1 | 3.501689 | 3.547332 | 2339.44 | 0.121 | [process](../../../../runs/performance/8-card/fsdp2-pp4/eager-r1/readme.md) |
| eager | 2 | 3.475770 | 3.565785 | 2356.89 | 0.121 | [process](../../../../runs/performance/8-card/fsdp2-pp4/eager-r2/readme.md) |
| inductor | 1 | 3.319747 | 3.520423 | 2467.66 | 0.121 | [process](../../../../runs/performance/8-card/fsdp2-pp4/inductor-r1/readme.md) |
| inductor | 2 | 3.300869 | 3.531632 | 2481.77 | 0.121 | [process](../../../../runs/performance/8-card/fsdp2-pp4/inductor-r2/readme.md) |

Inductor diagnostics across both repeats: 2
compile markers, 0 graph breaks,
2 recompiles, and
0 backend failures.

## Boundary

This is a profiler-off performance comparison with fixed input and no precision
acceptance. The eight-card graph runs overlap an external single-card job on
physical NPU0 and all eight-card results are diagnostic. Repeat on an idle,
healthy node before accepting a sub-5% change.
