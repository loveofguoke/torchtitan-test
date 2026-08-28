# fsdp4-tp2 eager versus Inductor

## Aggregate

| mode | median step | p90 step | tok/s/job | repeat drift | peak HBM |
|---|---:|---:|---:|---:|---:|
| eager | 1.628292 s | 1.652485 s | 5031.04 | 0.45% | 0.621 GiB |
| Inductor | 1.083835 s | 1.091692 s | 7558.35 | 0.01% | 0.524 GiB |

Steady-state speedup: **1.5023x** (+50.23%).
Cold-start break-even estimate: **359 steps**.

Inductor improves the TP-containing path by 50.23%. This is consistent with graph capture reducing model/host launch overhead, but it does not remove the topology's collective count.

## Per-run evidence

| mode | repeat | median step (s) | p90 (s) | tok/s/job | peak HBM (GiB) | record |
|---|---:|---:|---:|---:|---:|---|
| eager | 1 | 1.631961 | 1.655010 | 5019.73 | 0.621 | [process](../../../../runs/performance/8-card/fsdp4-tp2/eager-r1/readme.md) |
| eager | 2 | 1.624624 | 1.649961 | 5042.40 | 0.621 | [process](../../../../runs/performance/8-card/fsdp4-tp2/eager-r2/readme.md) |
| inductor | 1 | 1.083802 | 1.093594 | 7558.58 | 0.524 | [process](../../../../runs/performance/8-card/fsdp4-tp2/inductor-r1/readme.md) |
| inductor | 2 | 1.083867 | 1.089790 | 7558.12 | 0.524 | [process](../../../../runs/performance/8-card/fsdp4-tp2/inductor-r2/readme.md) |

Inductor diagnostics across both repeats: 2
compile markers, 0 graph breaks,
2 recompiles, and
0 backend failures.

## Boundary

This is a profiler-off performance comparison with fixed input and no precision
acceptance. The eight-card graph runs overlap an external single-card job on
physical NPU0 and all eight-card results are diagnostic. Repeat on an idle,
healthy node before accepting a sub-5% change.
