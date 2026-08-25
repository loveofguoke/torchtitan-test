# NPU performance exploration evidence

Exploration evidence is organized by card count, then topology, then immutable
run name. Start from [reports/index.md](reports/index.md) for conclusions, or
from a topology's `experiment.md` for exact run order, commands, results, and
links.

After reading the evidence, use
[`optimization_backlog.md`](optimization_backlog.md) for the numbered,
one-patch-at-a-time prototype tasks, exact source symbols, risks, and promotion
gates.

## Directory contract

```text
explorations/
  index.md
  reports/
    summary.md                 # cross-topology analysis
    1-card/
      index.md                 # topology/run navigation
      summary.md               # same-rank aggregate analysis
      single/experiment.md     # all single-card experiments
    2-card/
    4-card/
    8-card/
  runs/
    <card-count>/<topology>/<run-name>/
      readme.md                # readable, complete experiment process
      command_history.jsonl    # machine-readable driver history
      manifest.json            # resolved config and torchrun argv
      metrics.jsonl
      analysis.json
      artifacts.json
      tool_commands/*.json
  tools/
  history/
```

Large generated data uses the identical `<card-count>/<topology>/<run-name>`
hierarchy at repository root:

- `performance_runs/`: raw Ascend Profiler data and runtime logs; ignored by Git.
- `performance_artifacts/`: compact generated analysis cache; ignored by Git.
- `performance_reports/`: generated HTML reports; ignored by Git.
- `explorations/runs/`: small, reviewable evidence and per-run Markdown; committed.

## Per-run contract

Every run's `readme.md` records the experiment settings, driver invocation,
generated `torchrun` command, official offline-tool commands and return codes,
key measurements, failures, and output locations. JSON/JSONL remains the
machine-readable source of truth; the Markdown is regenerated from it.

## Reproduction order

1. Attach to `codex-glm5`, enter the `glm5-npu-dev` container, and activate
   `torchtitan-0803`.
2. Open `reports/<card-count>/<topology>/experiment.md` and select a run.
3. Replay the numbered commands from that run's `readme.md`.
4. Use its profiler root as MindStudio Insight input when visual inspection is
   required.
5. Regenerate run/topology indexes after importing historical evidence:

   ```bash
   python tests/glm5_2_performance/explorations/tools/organize.py
   ```

6. Regenerate a same-rank comparison, for example eight cards:

   ```bash
   python tests/glm5_2_performance/explorations/tools/summarize_topologies.py \
     --steps 20 --latest-replicate --world-size 8 \
     --markdown tests/glm5_2_performance/explorations/reports/8-card/comparison.md \
     --html performance_reports/8-card/comparison.html
   ```

## Interpretation rules

- Profiler-off steady state is throughput evidence; profiler-active data is
  attribution evidence.
- Compare runs only when model config, global token budget, sequence length,
  device mapping, step selection, and occupancy snapshots are compatible.
- Report both tok/s/device and tok/s/job, plus rank min/median/max.
- Separate collective payload/physical transit from exposed communication wait.
- Runs including the currently unhealthy NPU0 are diagnostic, not acceptance
  evidence, until the hardware warning is cleared.

