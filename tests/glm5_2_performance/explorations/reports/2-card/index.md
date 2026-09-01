# 2-card performance experiments

| topology | runs | best profiler-off tok/s/job | experiment | analysis |
| --- | ---: | ---: | --- | --- |
| cp2 | 8 | - | [experiment](cp2/experiment.md) | - |
| ddp2 | 21 | 11,234.19 | [experiment](ddp2/experiment.md) | - |
| ep2 | 8 | - | [experiment](ep2/experiment.md) | - |
| fsdp2 | 8 | - | [experiment](fsdp2/experiment.md) | - |
| pp2 | 8 | - | [experiment](pp2/experiment.md) | - |
| tp2 | 8 | - | [experiment](tp2/experiment.md) | - |

## comparison

`ddp2` currently leads the profiler-off screening at 11,234.19 tok/s/job. This is a run index and screening comparison; validity constraints remain in each run and analysis document.

- [scope summary](summary.md)
