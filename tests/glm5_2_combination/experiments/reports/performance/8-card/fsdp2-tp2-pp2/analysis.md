# fsdp2-tp2-pp2 eager versus Inductor

## Aggregate

| mode | median step | p90 step | tok/s/job | repeat drift | peak HBM |
|---|---:|---:|---:|---:|---:|
| eager | 14.010456 s | 14.087240 s | 584.71 | 1.48% | 0.120 GiB |
| Inductor | 9.097924 s | 9.151451 s | 900.43 | 3.68% | 0.232 GiB |

Steady-state speedup: **1.5400x** (+54.00%).
Cold-start break-even estimate: **102 steps**.

Inductor improves the TP-containing path by 54.00%. This is consistent with graph capture reducing model/host launch overhead, but it does not remove the topology's collective count.

## Per-run evidence

| mode | repeat | median step (s) | p90 (s) | tok/s/job | peak HBM (GiB) | record |
|---|---:|---:|---:|---:|---:|---|
| eager | 1 | 14.113287 | 14.186749 | 580.45 | 0.120 | [process](../../../../runs/performance/8-card/fsdp2-tp2-pp2/eager-r1/readme.md) |
| eager | 2 | 13.907624 | 13.987731 | 589.03 | 0.120 | [process](../../../../runs/performance/8-card/fsdp2-tp2-pp2/eager-r2/readme.md) |
| inductor | 1 | 9.262351 | 9.341651 | 884.44 | 0.232 | [process](../../../../runs/performance/8-card/fsdp2-tp2-pp2/inductor-r1/readme.md) |
| inductor | 2 | 8.933497 | 8.961251 | 917.00 | 0.117 | [process](../../../../runs/performance/8-card/fsdp2-tp2-pp2/inductor-r2/readme.md) |

Inductor diagnostics across both repeats: 2
compile markers, 0 graph breaks,
2 recompiles, and
0 backend failures.

## Boundary

This is a profiler-off performance comparison with fixed input and no precision
acceptance. The eight-card graph runs overlap an external single-card job on
physical NPU0 and all eight-card results are diagnostic. Repeat on an idle,
healthy node before accepting a sub-5% change.
