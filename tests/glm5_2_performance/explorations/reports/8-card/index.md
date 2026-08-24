# 8-card performance experiments

| topology | runs | best profiler-off tok/s/job | experiment document |
| --- | ---: | ---: | --- |
| cp8 | 1 | 5,037.26 | [experiment](cp8/experiment.md) |
| ddp8 | 3 | 10,960.70 | [experiment](ddp8/experiment.md) |
| ep8 | 1 | 30,677.33 | [experiment](ep8/experiment.md) |
| fsdp2-cp4 | 1 | 9,743.17 | [experiment](fsdp2-cp4/experiment.md) |
| fsdp2-pp4 | 1 | 4,697.69 | [experiment](fsdp2-pp4/experiment.md) |
| fsdp2-tp2-pp2 | 1 | 895.08 | [experiment](fsdp2-tp2-pp2/experiment.md) |
| fsdp2-tp4 | 1 | 3,671.06 | [experiment](fsdp2-tp4/experiment.md) |
| fsdp2-tp4-ep8 | 1 | 3,526.44 | [experiment](fsdp2-tp4-ep8/experiment.md) |
| fsdp4-tp2 | 1 | 7,250.90 | [experiment](fsdp4-tp2/experiment.md) |
| fsdp8 | 2 | 37,406.66 | [experiment](fsdp8/experiment.md) |
| pp8 | 1 | 4,020.93 | [experiment](pp8/experiment.md) |
| tp2-cp4 | 1 | 1,813.47 | [experiment](tp2-cp4/experiment.md) |
| tp8 | 1 | 1,852.98 | [experiment](tp8/experiment.md) |

## comparison

`fsdp8` currently leads the profiler-off screening at 37,406.66 tok/s/job. This is a run index and screening comparison; validity constraints remain in each run and analysis document.

- [scope summary](summary.md)
- [scope comparison](comparison.md)
