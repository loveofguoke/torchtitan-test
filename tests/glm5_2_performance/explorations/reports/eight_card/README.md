# Eight-card NPU exploration index

- `EIGHT_CARD_TOPOLOGY_REPORT.md`: generated profiler-off matrix, stability,
  HBM, and profiler attribution table.
- `EIGHT_CARD_ANALYSIS.md`: derived topology conclusions and gated potential
  optimization plan.
- `ddp8/DDP8_ANALYSIS.md`: DDP8 all-rank Ascend Profiler and official
  `msprof-analyze` derivation.
- `fsdp8/FSDP8_ANALYSIS.md`: FSDP collective structure, throughput cross-check,
  concurrency caveat, and idle-repeat gates.
- `COMMANDS.md`: aggregate generation, derivation, validation, and test
  commands not already captured by each run directory.
- `performance_reports/eight_card/npu-eight-card-topology-comparison.html`:
  generated throughput and stability charts.

Raw traces stay under `performance_runs/`. Compact immutable evidence and all
reproduction commands are under `explorations/runs/<run-name>/`.
