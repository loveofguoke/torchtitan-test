# 8-card performance experiments

| topology | runs | best profiler-off tok/s/job | experiment | analysis |
| --- | ---: | ---: | --- | --- |
| cp8 | 3 | 5,037.26 | [experiment](cp8/experiment.md) | [analysis](cp8/analysis.md) |
| ddp8 | 3 | 10,960.70 | [experiment](ddp8/experiment.md) | [analysis](ddp8/analysis.md) |
| ep8 | 5 | 30,677.33 | [experiment](ep8/experiment.md) | [analysis](ep8/analysis.md) |
| fsdp2-cp4 | 3 | 9,743.17 | [experiment](fsdp2-cp4/experiment.md) | [analysis](fsdp2-cp4/analysis.md) |
| fsdp2-pp4 | 3 | 4,697.69 | [experiment](fsdp2-pp4/experiment.md) | - |
| fsdp2-tp2-pp2 | 3 | 895.08 | [experiment](fsdp2-tp2-pp2/experiment.md) | - |
| fsdp2-tp4 | 3 | 3,671.06 | [experiment](fsdp2-tp4/experiment.md) | - |
| fsdp2-tp4-ep8 | 3 | 3,526.44 | [experiment](fsdp2-tp4-ep8/experiment.md) | - |
| fsdp4-tp2 | 3 | 7,250.90 | [experiment](fsdp4-tp2/experiment.md) | - |
| fsdp8 | 2 | 37,406.66 | [experiment](fsdp8/experiment.md) | [analysis](fsdp8/analysis.md) |
| pp8 | 3 | 4,020.93 | [experiment](pp8/experiment.md) | [analysis](pp8/analysis.md) |
| tp2-cp4 | 3 | 1,813.47 | [experiment](tp2-cp4/experiment.md) | - |
| tp8 | 2 | 1,852.98 | [experiment](tp8/experiment.md) | [analysis](tp8/analysis.md) |

## comparison

`fsdp8` currently leads the profiler-off screening at 37,406.66 tok/s/job. This is a run index and screening comparison; validity constraints remain in each run and analysis document.

- [scope summary](summary.md)
- [scope comparison](comparison.md)
- [scope validation](validation.md)
