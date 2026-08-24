# NPU performance exploration evidence

Each run under `runs/` has an immutable run-name directory. Keep one directory per capture,
repeat, topology, profiler preset, and reduction dtype. Raw multi-gigabyte traces
remain under `performance_runs/`; compact evidence is copied here so analysis
and reproduction do not depend on terminal scrollback.

## Per-run contract

- `command_history.jsonl`: every driver invocation, timestamp, cwd, selected
  environment, source commits, and dirty state.
- `manifest.json`: resolved experiment config and the exact generated
  `torchrun` argv.
- `metrics.jsonl`: per-step TorchTitan measurements.
- `analysis.json`: phase, rank, collective, kernel, memory, and deliverable
  summaries derived from official profiler output.
- `artifacts.json`: raw run, compact artifact, HTML report, and MindStudio
  Insight import paths.
- `tool_commands/*.json`: exact `msprof-analyze` argv, return code, stdout, and
  stderr for cluster, cluster-time, communication-bottleneck, free, advisor, or
  compare recipes.
- `failure.json` and `failed_runtime.log`: retained when a launch fails. A
  subsequent retry does not erase this evidence.

The raw `*_ascend_pt` roots and `cluster_analysis.db` are the visualization
handoff. The generated HTML and Markdown are summaries, not replacements for
MindStudio Insight.

## Reproduction order

1. Enter the `glm5-npu-dev` container and activate `torchtitan-0803` so its
   `bin` directory precedes system Python on `PATH`.
2. Replay the most recent applicable `shell_command` from
   `command_history.jsonl`.
3. Confirm the generated `torchrun` argv against `manifest.json`.
4. Replay official offline-tool commands from `tool_commands/*.json`.
5. Regenerate the cross-topology report:

   ```bash
   python tests/glm5_2_performance/explorations/tools/summarize_topologies.py \
     --steps 20 --replicate 1
   ```

   Regenerate the eight-card report, selecting the latest successful repeat
   for each topology (DDP8 repeat 1 is intentionally retained as a failure):

   ```bash
   python tests/glm5_2_performance/explorations/tools/summarize_topologies.py \
     --steps 20 --latest-replicate --world-size 8 \
     --markdown tests/glm5_2_performance/explorations/reports/eight_card/EIGHT_CARD_TOPOLOGY_REPORT.md \
     --html performance_reports/eight_card/npu-eight-card-topology-comparison.html
   ```

## Interpretation rules

- Profiler-off steady state is throughput evidence.
- Profiler-active windows are attribution evidence only.
- Compare runs only when device mapping, global token budget, model config,
  sequence length, step selection, and occupancy snapshots are compatible.
- Report both tok/s/device and tok/s/job. For model-parallel topologies, low
  per-device throughput can still reflect a different global critical path.
- Use rank min/median/max; a mean hides launch skew.
- Separate collective payload and physical transit from wait/idle time.
- Runs that include NPU0 are diagnostic topology comparisons, not
  healthy-hardware acceptance evidence, while its health warning persists.
- The eight-card summary and derived conclusions live under
  `reports/eight_card/`; DDP8 has its own attribution subdirectory.

## Directory layout

```text
explorations/
  reports/   # conclusions, topology comparisons, and per-exploration subdirs
  runs/      # one immutable evidence directory per run
  tools/     # report-generation utilities
  history/   # commands used to regenerate aggregate reports
```
