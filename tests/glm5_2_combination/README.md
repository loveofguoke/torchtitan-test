# GLM-5.2 combined experiments

This package composes independent execution features at one training boundary:

```text
fixed inputs + topology + graph policy + optional profiler
                                  -> precision and/or performance reports
```

It does not inject settings into the precision or performance implementations.
Topology comes from `glm5_2_common`, graph arguments come from `glm5_2_graph`,
formal inputs and metric artifacts come from `glm5_2_precision`, and profiler
analysis comes from `glm5_2_performance`.

Current graph and profiler execution is NPU-only. CUDA graph/profiler interfaces
are reserved and raise `NotImplementedError`; ordinary CUDA/NPU eager precision
experiments remain available from `glm5_2_precision`.

Detailed result interpretation lives in the
[performance report guide](../glm5_2_performance/REPORT_GUIDE_ZH.md) and
[graph visualization guide](../glm5_2_graph/VISUALIZATION_GUIDE_ZH.md). The
top-level combined report also starts with a Chinese reading order so precision,
performance, compiler, runtime Timeline, and TensorBoard evidence are not mixed.
Install and verify the shared runtime, Python readers, Ascend analyzers, and GUI
viewers using the
[unified dependency guide](../glm5_2_common/PERFORMANCE_GRAPH_DEPENDENCIES_ZH.md).

## Comparison semantics

The default combined experiment is NPU self-consistency:

```text
reference: NPU, single card, selectable eager/graph policy
candidate: NPU, selected single or distributed topology, selectable policy
```

The normal graph acceptance is:

```text
reference-graph=eager
candidate-graph=inductor or npugraphs
```

The runner also supports eager/eager and graph/graph diagnostics. Reference and
candidate graph modes are independent.

Topology selection:

```text
--topology single
--topology fsdp8
--topologies ddp8,fsdp8,tp8
--topology all
```

For self-consistency, the reference topology stays `single`; only the candidate
changes. One reference capture is reused by every selected candidate topology.

## Objectives and profiler

```text
--objectives precision
--objectives performance
--objectives precision,performance
```

Performance metrics do not require Profiler. `--profiler-preset off` is the
default and is appropriate for undisturbed step-time and throughput comparison.
Profiler-off reports exclude the first 10 steps by default. Use
`--performance-skip-steps` to change that boundary; the value is part of the
experiment identity, so results with different warmup windows cannot collide.
Select a preset explicitly when operator or communication diagnosis is needed:

```text
--profiler-preset overview
--profiler-preset comparison
--profiler-preset standard
--profiler-preset distributed
--profiler-preset kernel
--profiler-preset operator
--profiler-preset memory
--profiler-preset flamegraph
--profiler-preset runtime
--profiler-preset system
```

Profiler runs are diagnostic and should not replace profiler-off formal
throughput measurements.

## Complete NPU workflow

Export one card for single-card data/reference work or all devices required by
the selected candidate topology:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset CUDA_VISIBLE_DEVICES
```

On the validated graph server, run every stage through the common CANN 9.1
launcher. This gives eager reference and graph candidate the same clean CANN,
Conda, ATB, HCCL, and cache policy before TorchNPU import. The launcher injects
`reference-graph=eager` and the selected candidate backend only when those
options are absent. Replace the first `inductor` with `npugraphs` for that
backend; do not mix direct default-shell Python commands with wrapped captures.

Prepare the step-0 checkpoint and fixed token plan once:

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination \
  --data --data-device npu --topology all \
  --objectives precision,performance \
  --reference-graph eager --candidate-graph inductor \
  --profiler-preset off
```

The fixture identity contains the training input contract, not graph mode,
objectives, profiler preset, or selected candidate topology. The same fixture
can therefore be reused for eager/eager, eager/graph, graph/graph, precision,
and performance captures when model, precision, steps, batch, sequence length,
seed, and checkpoint kind are unchanged.

Capture the single-card reference. When `--topology all` is selected, duplicate
reference work is detected and reused:

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination \
  --capture reference --topology all \
  --objectives precision,performance \
  --reference-graph eager --candidate-graph inductor \
  --profiler-preset off --compiler-diagnostics
```

Capture every candidate topology:

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination \
  --capture candidate --topology all \
  --objectives precision,performance \
  --reference-graph eager --candidate-graph inductor \
  --profiler-preset off --compiler-diagnostics
```

Generate one suite report without launching training:

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination \
  --compare --topology all \
  --objectives precision,performance \
  --reference-graph eager --candidate-graph inductor \
  --profiler-preset off --compiler-diagnostics --require-all
```

The commands above are the complete all-topology delivery workflow. For a
focused first pass, keep the same four stages and change only the topology
selector. The following single-card workflow runs precision and performance
together without Profiler overhead:

Single-card candidate:

```bash
COMMON_ARGS="--topology single --objectives precision,performance --reference-graph eager --candidate-graph inductor --profiler-preset off"
tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination \
  --data --data-device npu $COMMON_ARGS
tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination \
  --capture reference $COMMON_ARGS
tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination \
  --capture candidate $COMMON_ARGS
tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination \
  --compare --require-all $COMMON_ARGS
```

Distributed candidate example (`fsdp8`; replace it with any registered
topology):

```bash
COMMON_ARGS="--topology fsdp8 --objectives precision,performance --reference-graph eager --candidate-graph inductor --profiler-preset off"
tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination \
  --data --data-device npu $COMMON_ARGS
tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination \
  --capture reference $COMMON_ARGS
tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination \
  --capture candidate $COMMON_ARGS
tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination \
  --compare --require-all $COMMON_ARGS
```

Reference/candidate capture and compare must use identical training, graph,
objective, diagnostics, and profiler selections. `--data` is independent of
those execution features: it only needs the same model, precision, steps,
batch, sequence length, seed, and checkpoint kind. Changing an execution
feature creates a different capture identity but does not require a new
fixed-input fixture.

Omitting `--repeat` runs every configured repeat. `--repeat N` selects one
repeat. Completed valid artifacts are skipped; incomplete run output is
archived and retried. `--force` replaces completed output.

## Focused examples

Eager single versus one eager distributed candidate (`fsdp8` in this example):

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination \
  --capture candidate --topology fsdp8 \
  --objectives precision \
  --reference-graph eager --candidate-graph eager
```

Eager single versus Inductor TP8 performance without Profiler:

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination \
  --capture candidate --topology tp8 \
  --objectives performance \
  --reference-graph eager --candidate-graph inductor \
  --profiler-preset off --compiler-diagnostics
```

The same comparison with all-rank distributed profiling:

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination \
  --capture candidate --topology tp8 \
  --objectives performance \
  --reference-graph eager --candidate-graph inductor \
  --profiler-preset distributed --compiler-diagnostics
```

## Command-line parameters

| Parameter | Purpose and choices | Default |
|---|---|---|
| `--data` | Generate the topology-independent step-0 checkpoint and token plan. | no action; one action is required |
| `--capture reference\|candidate` | Run and persist one endpoint. Reference is the configured single-card endpoint; candidate uses the selected topology. | no action; one action is required |
| `--compare` | Build precision/performance reports from existing artifacts without training. | no action; one action is required |
| `--list-topologies` | Print the common topology registry. | no action; one action is required |
| `--topology` | One candidate topology or `all`. | `all` |
| `--topologies` | Comma-separated subset or `all`; mutually exclusive with `--topology`. | unset |
| `--repeat` | Capture one repeat number. Omit it to run both configured repeats. | all configured repeats (`2`) |
| `--precision` | `fp32`, `bf16`, or `full-bf16`. | `bf16` from `CONFIG` |
| `--objectives` | Comma-separated `precision`, `performance`, or both. | `precision,performance` |
| `--reference-graph` | Independent reference mode: `eager`, `inductor`, or `npugraphs`. | `eager` |
| `--candidate-graph` | Independent candidate mode: `eager`, `inductor`, or `npugraphs`. | `inductor` |
| `--compile-loss` | Add `loss` to the default `model` compile component. NPUGraph accepts model-only. | disabled |
| `--compiler-diagnostics` | Enable graph-break/recompile/dynamic diagnostics on each compiled endpoint; eager endpoints have no compiler diagnostics. | disabled |
| `--profiler-preset` | `off`, `overview`, `comparison`, `standard`, `distributed`, `kernel`, `operator`, `memory`, `flamegraph`, `runtime`, or `system`. Used only when `performance` is selected. The scheduled performance runner, not one combination capture, owns the `all` multi-preset matrix. | `off` |
| `--profile-skip-steps` | Steps skipped before scheduled profiling. | `10` |
| `--profile-warmup-steps` | Profiler warmup steps. | `1` |
| `--profile-active-steps` | Profiler active collection steps. | `3` |
| `--performance-skip-steps` | Startup/compile steps excluded from profiler-off summaries. | `10` |
| `--steps` | Identified exploration override for training length; minimum 10. | maintained config (`5000`) |
| `--performance-nondeterministic` | Permit performance-only Inductor autotuning with deterministic algorithms disabled. | disabled |
| `--data-device` | Generic fixture backend override (`cuda` or `npu`). The current NPU/NPU combined config uses `npu`. | inferred from the NPU visibility variable |
| `--force` | Replace valid existing fixture/capture output. | disabled |
| `--require-all` | Require all selected topology/repeat artifacts during compare. | disabled |

Combination captures inherit the same audited lifecycle as precision and graph:
attempt/generation/PID state is recorded, live runs cannot be overwritten, and
forced execution prints and verifies deletion of every selected capture output
and exact-name failed archive before any topology starts.

The built-in combined profile uses 5000 steps, local batch 8, global batch 64,
sequence length 128, seed 61, BF16 mixed precision, and two repeats. These
values are defined in `combination_benchmark.py`; changing them defines a new
maintained experiment rather than a one-off CLI override.

For a bounded performance-only exploration, `--steps N` creates a separate
fixture and experiment identity without changing the maintained 5000-step
default. At least 10 steps are required, and the override is rejected for
precision or mixed objectives. The common graph launcher enables Turbo's
pointwise-only vetted-autotune compatibility, so deterministic Inductor is the
current default for both performance and precision. The optional
`--performance-nondeterministic` flag is retained only for diagnostic baselines
and reproducing the 2026-08-26 performance matrix. It is rejected when the
precision objective is present and is recorded as `nondet` in the storage name.
Eager and graph endpoints in an A/B comparison must use the same setting.

```bash
COMMON_ARGS="--objectives performance --profiler-preset off --steps 30 --performance-skip-steps 10"
tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination \
  --capture candidate --topology fsdp8 \
  --reference-graph eager --candidate-graph eager $COMMON_ARGS
tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination \
  --capture candidate --topology fsdp8 \
  --reference-graph eager --candidate-graph inductor $COMMON_ARGS
```

Running graph candidates through the common launcher is required: it loads the
paired Turbo compatibility profile, clean CANN environment, and isolated
compiler cache. Direct Python invocation is appropriate only for CPU-side
listing/report operations that do not import an NPU graph backend.

The self-consistency reference remains single-card by design because it is the
numerical baseline. It is not a same-topology performance baseline. Therefore
an eager-vs-graph speedup matrix captures each selected topology twice through
the candidate role, once with `candidate-graph=eager` and once with the graph
backend, as in the commands above. Pair the two experiment roots with the
tracked offline summarizer:

```bash
python tests/glm5_2_combination/experiments/tools/summarize_performance.py \
  --eager-root combination_runs/<eager-experiment-id> \
  --inductor-root combination_runs/<inductor-experiment-id> \
  --artifact-root combination_artifacts \
  --output-root tests/glm5_2_combination/experiments
```

The normal combined HTML's reference/candidate time ratio can compare a
single-card numerical reference with a distributed candidate; do not call that
ratio graph acceleration. The curated 2026-08-26 matrix follows the stricter
same-topology pairing and is indexed from `experiments/index.md`. A future
three-endpoint workflow may unify single-card precision reference,
topology-matched eager performance baseline, and graph candidate in one CLI,
but the current two-endpoint capture schema does not silently pretend those
roles are the same.

All capture and compare invocations for one experiment must repeat the same
precision, objectives, graph modes, compiled components, compiler diagnostics,
and profiler schedule. The fixture identity deliberately excludes those
execution features and is reusable when the underlying training input contract
is unchanged.

## Report contents and acceptance

The top-level combined HTML records selected objectives and both graph modes,
links the formal precision topology suite, and presents per-topology/per-repeat
reference and candidate median step time plus candidate speedup. Each
performance details link opens the endpoint report containing throughput,
active profiling window, memory and operator/kernel summaries, and official
Ascend analysis outputs when profiling was enabled. The endpoint report also
links MindStudio/Perfetto Timeline inputs, the official MindStudio Host flame
graph, portable CPU/NPU folded-stack SVGs, and the TensorBoard scalar dashboard.
The top-level report links per-rank
`tlparse`, FX graph, Inductor IR, generated code, and raw structured traces for
compiled endpoints captured with `--compiler-diagnostics`.

The linked precision suite remains the authority for numerical PASS/FAIL and
contains the full loss/grad-norm curves, standards, distributions, repeat
diagnostics, and fixed-input validation. Performance is reported independently
and does not weaken precision acceptance. With both objectives enabled, one
training capture supplies both metric families, but the two conclusions remain
separate.

## Outputs

```text
precision_fixtures/<fixture-id>/
combination_runs/<experiment-id>/<topology>/<role>-r<repeat>/
combination_runs/<experiment-id>/<topology>/<role>-r<repeat>/graph_visualization/
combination_artifacts/<experiment-id>/<topology>/<role>-r<repeat>/
combination_reports/<experiment-id>/
```

Directory dimensions have fixed meanings:

- `<experiment-id>`: one immutable fixture/training/feature contract;
- `<topology>`: selected single or distributed execution layout;
- `<role>`: reference or candidate;
- `r<repeat>`: independent repeat of that endpoint;
- `graph_visualization/rank_<rank>`: rank-local compile evidence;
- `trainer_output/profiling/traces`: optional rank-local Ascend runtime data.

The report directory contains the combined index plus linked formal precision
and endpoint performance reports. Raw profiler/compiler output stays in
`combination_runs`; portable metrics stay in `combination_artifacts`.

`combination_runs` contains runtime logs, raw metrics, trainer output, and large
Profiler data and is ignored by Git. Artifacts are compact transfer inputs but
remain ignored; lightweight `combination_reports` are tracked by Git under the
repository-wide output policy. When endpoints are on different servers, synchronize the fixture
before capture and synchronize required artifacts plus Profiler-containing run
directories before CPU-side comparison.

## Maintained graph-precision entry point

For the maintained 5000-step eager/graph precision matrix, use the wrapper below
instead of repeating the four workflow stages manually:

```bash
bash tests/glm5_2_combination/run_graph_precision_5000.sh inductor all
```

It runs data preparation, eager reference capture, every selected Inductor
candidate topology, and strict comparison in order. Set `TOPOLOGY` or `REPEAT`
only for a focused retry; omitting both preserves the maintained all-topology,
two-repeat contract. The complete command ledger, run status, smoke evidence,
failure history, and acceptance report are indexed from
[`experiments/index.md`](experiments/index.md).
