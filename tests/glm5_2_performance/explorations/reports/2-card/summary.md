# 2-card summary

All two-card experiments use DDP2 on physical NPU1/2. The run inventory and
exact commands are in [ddp2/experiment.md](ddp2/experiment.md).

The clean FP32-reduction attribution run reports 741.67 ms/step,
5,522.66 tok/s/device, balanced 162.53/162.71 ms compute, but exposed
communication of 234.61/31.27 ms. Fifty AllReduce calls move about
266–267 MB/rank/step; only about 13.8 ms is physical HCCS transit. Official
analysis classifies the slow collectives as host-bound.

The BF16-reduction prototype keeps 50 calls but halves payload to about
133 MB/rank/step and lowers physical transit to 7.8–8.1 ms. Its best
profiler-off run reaches 11,234.19 tok/s/job, but the FP32 A/B/A repeats were
contended and vary from 5,676.77 to 9,049.35 tok/s/job. Therefore payload and
transit reduction are established, while end-to-end speedup is not.

The next valid comparison is at least three interleaved profiler-off FP32/BF16
repeats with stable pre/post occupancy snapshots. The broader derivation is in
[the cross-rank analysis](../summary.md).

