# fsdp2-tp4 eager versus Inductor

## Aggregate

| mode | median step | p90 step | tok/s/job | repeat drift | peak HBM |
|---|---:|---:|---:|---:|---:|
| eager | 3.333390 s | 3.400164 s | 2457.56 | 5.41% | 0.380 GiB |
| Inductor | 2.174089 s | 2.190727 s | 3768.02 | 0.77% | 0.349 GiB |

Steady-state speedup: **1.5332x** (+53.32%).
Cold-start break-even estimate: **338 steps**.

Inductor improves the TP-containing path by 53.32%. This is consistent with graph capture reducing model/host launch overhead, but it does not remove the topology's collective count.

## Per-run evidence

| mode | repeat | median step (s) | p90 (s) | tok/s/job | peak HBM (GiB) | record |
|---|---:|---:|---:|---:|---:|---|
| eager | 1 | 3.245549 | 3.262978 | 2524.07 | 0.380 | [process](../../../../runs/performance/8-card/fsdp2-tp4/eager-r1/readme.md) |
| eager | 2 | 3.421231 | 3.537349 | 2394.46 | 0.380 | [process](../../../../runs/performance/8-card/fsdp2-tp4/eager-r2/readme.md) |
| inductor | 1 | 2.182384 | 2.200028 | 3753.69 | 0.349 | [process](../../../../runs/performance/8-card/fsdp2-tp4/inductor-r1/readme.md) |
| inductor | 2 | 2.165793 | 2.181426 | 3782.45 | 0.317 | [process](../../../../runs/performance/8-card/fsdp2-tp4/inductor-r2/readme.md) |

Inductor diagnostics across both repeats: 2
compile markers, 0 graph breaks,
2 recompiles, and
0 backend failures.

## Boundary

This is a profiler-off performance comparison with fixed input and no precision
acceptance. The eight-card graph runs overlap an external single-card job on
physical NPU0 and all eight-card results are diagnostic. Repeat on an idle,
healthy node before accepting a sub-5% change.
