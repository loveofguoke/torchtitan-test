# 4-card performance experiments

| topology | runs | best profiler-off tok/s/job | experiment | analysis |
| --- | ---: | ---: | --- | --- |
| cp4 | 9 | 5,069.75 | [experiment](cp4/experiment.md) | - |
| ddp4 | 10 | 21,009.35 | [experiment](ddp4/experiment.md) | - |
| ep4 | 9 | 17,167.77 | [experiment](ep4/experiment.md) | - |
| fsdp2-tp2 | 9 | 3,716.87 | [experiment](fsdp2-tp2/experiment.md) | - |
| fsdp4 | 12 | 20,390.85 | [experiment](fsdp4/experiment.md) | - |
| pp4 | 9 | 2,579.88 | [experiment](pp4/experiment.md) | - |
| tp4 | 9 | 1,891.92 | [experiment](tp4/experiment.md) | - |

## comparison

`ddp4` currently leads the profiler-off screening at 21,009.35 tok/s/job. This is a run index and screening comparison; validity constraints remain in each run and analysis document.

- [scope summary](summary.md)
- [scope comparison](comparison.md)
