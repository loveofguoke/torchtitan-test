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
Select a preset explicitly when operator or communication diagnosis is needed:

```text
--profiler-preset overview
--profiler-preset comparison
--profiler-preset standard
--profiler-preset distributed
--profiler-preset kernel
--profiler-preset runtime
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

Prepare the step-0 checkpoint and fixed token plan once:

```bash
python tests/glm5_2_combination/combination_benchmark.py \
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
python tests/glm5_2_combination/combination_benchmark.py \
  --capture reference --topology all \
  --objectives precision,performance \
  --reference-graph eager --candidate-graph inductor \
  --profiler-preset off --compiler-diagnostics
```

Capture every candidate topology:

```bash
python tests/glm5_2_combination/combination_benchmark.py \
  --capture candidate --topology all \
  --objectives precision,performance \
  --reference-graph eager --candidate-graph inductor \
  --profiler-preset off --compiler-diagnostics
```

Generate one suite report without launching training:

```bash
python tests/glm5_2_combination/combination_benchmark.py \
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
python tests/glm5_2_combination/combination_benchmark.py --data --data-device npu $COMMON_ARGS
python tests/glm5_2_combination/combination_benchmark.py --capture reference $COMMON_ARGS
python tests/glm5_2_combination/combination_benchmark.py --capture candidate $COMMON_ARGS
python tests/glm5_2_combination/combination_benchmark.py --compare --require-all $COMMON_ARGS
```

Distributed candidate example (`fsdp8`; replace it with any registered
topology):

```bash
COMMON_ARGS="--topology fsdp8 --objectives precision,performance --reference-graph eager --candidate-graph inductor --profiler-preset off"
python tests/glm5_2_combination/combination_benchmark.py --data --data-device npu $COMMON_ARGS
python tests/glm5_2_combination/combination_benchmark.py --capture reference $COMMON_ARGS
python tests/glm5_2_combination/combination_benchmark.py --capture candidate $COMMON_ARGS
python tests/glm5_2_combination/combination_benchmark.py --compare --require-all $COMMON_ARGS
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
python tests/glm5_2_combination/combination_benchmark.py \
  --capture candidate --topology fsdp8 \
  --objectives precision \
  --reference-graph eager --candidate-graph eager
```

Eager single versus Inductor TP8 performance without Profiler:

```bash
python tests/glm5_2_combination/combination_benchmark.py \
  --capture candidate --topology tp8 \
  --objectives performance \
  --reference-graph eager --candidate-graph inductor \
  --profiler-preset off --compiler-diagnostics
```

The same comparison with all-rank distributed profiling:

```bash
python tests/glm5_2_combination/combination_benchmark.py \
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
| `--profiler-preset` | `off`, `overview`, `comparison`, `standard`, `distributed`, `kernel`, or `runtime`. Used only when `performance` is selected. | `off` |
| `--profile-skip-steps` | Steps skipped before scheduled profiling. | `10` |
| `--profile-warmup-steps` | Profiler warmup steps. | `1` |
| `--profile-active-steps` | Profiler active collection steps. | `3` |
| `--data-device` | Generic fixture backend override (`cuda` or `npu`). The current NPU/NPU combined config uses `npu`. | inferred from the NPU visibility variable |
| `--force` | Replace valid existing fixture/capture output. | disabled |
| `--require-all` | Require all selected topology/repeat artifacts during compare. | disabled |

The built-in combined profile uses 5000 steps, local batch 8, global batch 64,
sequence length 128, seed 61, BF16 mixed precision, and two repeats. These
values are defined in `combination_benchmark.py`; changing them defines a new
maintained experiment rather than a one-off CLI override.

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
Ascend analysis outputs when profiling was enabled.

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
combination_artifacts/<experiment-id>/<topology>/<role>-r<repeat>/
combination_reports/<experiment-id>/
```

`combination_runs` contains runtime logs, raw metrics, trainer output, and large
Profiler data and is ignored by Git. Artifacts and reports are the portable
deliverables. When endpoints are on different servers, synchronize the fixture
before capture and synchronize required artifacts plus Profiler-containing run
directories before CPU-side comparison.
