# 1-card summary

The single-card exploration contains six records: three historical analyze
commands without a matching compact capture and three completed
profiler-active analyses. See [single/experiment.md](single/experiment.md) for
the exact sequence and per-run links.

| Preset | Parse mode | Median step | tok/s/device | Use |
| --- | --- | ---: | ---: | --- |
| overview | sync | 1,839.62 ms | 4,453.08 | low-overhead operator attribution |
| overview | offline | 1,739.46 ms | 4,709.52 | offline parser workflow validation |
| kernel | offline | 1,747.83 ms | 4,686.94 | shapes/kernel-family attribution |

All three completed rows are profiler-active, so none is an authoritative
throughput baseline. Their 4.45–4.71k tok/s range demonstrates that the
collection/parse/report path works, not a preset speed comparison. The next
single-card performance experiment should be a new `--profiler-off
--replicate N` run on the same physical card, followed by a separate profiler
run for attribution.

