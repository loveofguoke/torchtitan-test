# tp2-cp4 eager versus Inductor

## Aggregate

| mode | median step | p90 step | tok/s/job | repeat drift | peak HBM |
|---|---:|---:|---:|---:|---:|
| eager | 6.516221 s | 6.579708 s | 1257.17 | 0.45% | 0.202 GiB |
| Inductor | 4.359172 s | 4.538650 s | 1879.26 | 0.44% | 0.272 GiB |

Steady-state speedup: **1.4948x** (+49.48%).
Cold-start break-even estimate: **174 steps**.

Inductor improves the TP-containing path by 49.48%. This is consistent with graph capture reducing model/host launch overhead, but it does not remove the topology's collective count.

## Per-run evidence

| mode | repeat | median step (s) | p90 (s) | tok/s/job | peak HBM (GiB) | record |
|---|---:|---:|---:|---:|---:|---|
| eager | 1 | 6.530704 | 6.604645 | 1254.38 | 0.202 | [process](../../../../runs/performance/8-card/tp2-cp4/eager-r1/readme.md) |
| eager | 2 | 6.501738 | 6.554770 | 1259.97 | 0.202 | [process](../../../../runs/performance/8-card/tp2-cp4/eager-r2/readme.md) |
| inductor | 1 | 4.368742 | 4.442427 | 1875.14 | 0.272 | [process](../../../../runs/performance/8-card/tp2-cp4/inductor-r1/readme.md) |
| inductor | 2 | 4.349602 | 4.634874 | 1883.39 | 0.181 | [process](../../../../runs/performance/8-card/tp2-cp4/inductor-r2/readme.md) |

Inductor diagnostics across both repeats: 2
compile markers, 0 graph breaks,
2 recompiles, and
0 backend failures.

## Boundary

This is a profiler-off performance comparison with fixed input and no precision
acceptance. The eight-card graph runs overlap an external single-card job on
physical NPU0 and all eight-card results are diagnostic. Repeat on an idle,
healthy node before accepting a sub-5% change.
