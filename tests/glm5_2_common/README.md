# Shared experiment primitives

`glm5_2_common` is dependency-free with respect to individual experiments. It
owns canonical accelerator selection, topology definitions, topology selection,
and conflict-checked execution feature composition.

Dependency direction is one way:

```text
glm5_2_common
  <- precision
  <- performance
  <- graph execution feature
  <- precision + graph <- checkpoint / stability
  <- graph <- smoke
  <- precision + performance + graph <- combination
```

Experiment modules must not be imported by `glm5_2_common`. Features contribute
arguments and environment variables through `TrainingFeature`; only the central
combination workflow assembles them into a training command.

Performance and graph experiments share one external dependency inventory:
[性能与图模式环境、外部工具和依赖总表](PERFORMANCE_GRAPH_DEPENDENCIES_ZH.md).
It separates training runtime requirements from optional analysis/GUI tools and
is the installation authority for performance, graph, and combination reports.

## Standard experiment lifecycle

Training experiments use one reproducible input contract whenever numerical
results or restart behavior are compared:

1. `--data` creates a step-0 model checkpoint and topology-independent fixed
   token plan.
2. `--capture ...` or the experiment's training action consumes that fixture.
3. `--compare` reads portable artifacts and writes the report without an
   accelerator.

The parity, formal precision, combination/graph, checkpoint, and stability
experiments follow this contract. A fixture can be generated on either backend
and is reused across topologies when its training settings are unchanged.

The standalone performance profiler is the deliberate exception. It measures
profiling overhead and runtime behavior and does not claim numerical
comparability. Use the combination runner when fixed inputs, graph mode,
distributed execution, precision comparison, and profiling must be enabled in
the same training process.

Graph and profiler execution are currently implemented only for Ascend NPU.
Their CUDA device values are reserved public interfaces that fail explicitly
until the corresponding CUDA policies are implemented. Device-neutral eager
precision, checkpoint, stability, and smoke experiments continue to support
both CUDA and NPU.

## Shared topology selectors

Experiment CLIs that support distributed execution use the same vocabulary:

| Name | Ranks | Parallel decomposition |
|---|---:|---|
| `single` | 1 | no distributed degree |
| `ddp2`, `ddp8` | 2, 8 | DP replicate 2 or 8 |
| `fsdp8` | 8 | DP shard 8 |
| `tp8` | 8 | TP 8 |
| `cp8` | 8 | CP 8 |
| `pp8` | 8 | PP 8 with the shared pipeline schedule |
| `ep8` | 8 | dense FSDP 8 plus EP 8 |
| `fsdp2-tp4` | 8 | FSDP 2 x TP 4 |
| `fsdp2-cp4` | 8 | FSDP 2 x CP 4 |
| `tp2-cp4` | 8 | TP 2 x CP 4 |
| `fsdp4-tp2` | 8 | FSDP 4 x TP 2 |
| `fsdp2-pp4` | 8 | FSDP 2 x PP 4 |
| `fsdp2-tp2-pp2` | 8 | FSDP 2 x TP 2 x PP 2 |
| `fsdp2-tp4-ep8` | 8 | dense FSDP 2 x TP 4; expert region EP 8 |

`--topology NAME` selects one member, `--topology all` selects every member
advertised by that experiment, and `--topologies A,B,C` selects a subset. The
singular and plural selectors are mutually exclusive. `ddp16` and `fsdp16`
exist in the common registry as future multi-node definitions but are excluded
from current single-node `all` suites.

```bash
# Single card.
python <experiment-entry.py> <action> --topology single

# One representative distributed topology.
python <experiment-entry.py> <action> --topology fsdp8

# A focused subset.
python <experiment-entry.py> <action> --topologies ddp8,fsdp8,tp8

# Every topology advertised by that experiment.
python <experiment-entry.py> <action> --topology all
```

Each experiment README states its own default: smoke/checkpoint/stability and
standalone performance default to `single`; formal precision and the central
combination suite default to `all`; graph convenience entry points default to
`single`.

## Output names

Each experiment root already identifies the artifact type, so child names do
not repeat `fixture`, `combo`, `stability`, `checkpoint`, or the model name.
Readable settings are followed by an eight-character digest of only the values
that can change capture results. Report thresholds and presentation settings
are excluded, so report-only changes never require another training run.

When a pre-digest output directory contains the same stored training contract,
the workflow renames it to the current config-digested name and continues. A
different contract is never adopted silently.

## Rerun and generation rules

Experiment output is resumed per suite member:

- Without `--force`, a completed member is reused only when its stored
  contract belongs to the current fixture generation. Failed, incomplete, or
  incompatible output is archived and that member is retried.
- `--force` starts a new generation for the complete selected range. Every
  selected member is removed before the first training process starts; this
  prevents a mid-suite failure from mixing newly captured members with
  untouched output from an older run.
- After a forced suite stops part way through, rerun the same command without
  `--force`. Completed members from that generation are skipped and execution
  continues at the first incomplete member.
- `--data --force` creates a new fixture generation. Captures tied to an older
  generation are not finalized or reused, even when their files are complete.

Long-running mutable experiments use the shared `RunAttempt` lifecycle. Each
run directory contains `run_state.json` with an attempt ID, orchestrator PID,
context, and `running`/`failed`/`completed` status. Runtime logs record the same
attempt ID where the experiment owns log creation. Before `--force` removes
anything, the complete selected range is preflighted for live orchestrators;
all selected run/artifact/report/input-contract paths and their exact-name
`.previous-*`/`.failed-*` siblings are then removed, printed, and verified
absent before the first new process starts. An interrupted command
is resumed without `--force`: completed members are retained, while incomplete
members are archived or replaced as one unit.

This policy applies to precision (including graph and combination captures),
performance/profiler, stability, checkpoint, and smoke. Parity artifacts remain
immutable and graph-debug probes create timestamped directories, so those two
workflows never overwrite an existing generation and intentionally do not add a
mutable `--force` lifecycle.

Use `--topology NAME --force` to replace only one topology. Use `--topology
all --force` only when the whole suite should start over.
