# 4-card summary

Seven topologies completed profiler-off screening on healthy-card mappings.
The complete formulas, evidence links, and profiler attribution are in
[comparison.md](comparison.md); each topology's full command history is linked
from [index.md](index.md).

| Rank | Topology | tok/s/job | Main signal |
| ---: | --- | ---: | --- |
| 1 | DDP4 | 21,009.35 | throughput leader |
| 2 | FSDP4 | 20,390.85 | 2.94% slower, 21.7% lower active HBM |
| 3 | EP4 | 17,167.77 | promising only at this small routing load |
| 4 | CP4 | 5,069.75 | fixed CP overhead dominates seq128 |
| 5 | FSDP2-TP2 | 3,716.87 | TP collectives dominate |
| 6 | PP4 | 2,579.88 | pipeline bubble/fixed overhead |
| 7 | TP4 | 1,891.92 | strongest negative scaling signal |

DDP4 and FSDP4 compute are balanced, while exposed communication spans
42.22–223.38 ms and 19.42–294.37 ms respectively. DDP4 performs 28
AllReduces/step; FSDP4 performs 42 AllGathers, 22 ReduceScatters, and seven
tiny AllReduces. The selected slow collectives are host-bound, so launch
alignment and collective granularity are higher-priority hypotheses than raw
link bandwidth.

The successful FSDP4 profiler mapping excludes physical NPU4. Two failures
followed that card across logical ranks, so those failure records remain in
[failures.md](../failures.md) and NPU4 is excluded from formal FSDP evidence.

