# Legacy analyze command aliases

The first history-backfill pass recorded the pre-compatibility run hash before
`analyze()` found an older compatible manifest. Those command-only directories
are retained rather than deleted. The actual artifact mappings are:

| Command-only hash | Compatible artifact hash |
| --- | --- |
| `single-overview-a7f99b62` | `single-overview-af49a08e` |
| `single-overview-offline-bf438d2a` | `single-overview-offline-b14bdd6a` |
| `single-kernel-offline-2499a8d1` | `single-kernel-offline-2ed8313b` |
| `ddp2-overview-offline-efca0dfc` | `ddp2-overview-offline-240cc2ba` |
| `ddp2-distributed-fe1e682c` | `ddp2-distributed-23bade6d` |
| `ddp2-distributed-profiler-off-5b768f16` | `ddp2-distributed-profiler-off-e51e73ea` |
| `ddp2-distributed-profiler-off-d0093fa5` | `ddp2-distributed-profiler-off-225f5586` |
| `ddp2-distributed-reduce-bf16-profiler-off-29003e6c` | `ddp2-distributed-reduce-bf16-profiler-off-051ff000` |

The workflow now records analyze-only commands after compatible-manifest
resolution, so new histories land directly beside the actual artifact. The
compatible artifact directories were re-analyzed after that correction and
contain their own current `command_history.jsonl`.

