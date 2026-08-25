# GLM-5.2 graph mode

This package owns only the graph execution policy and its convenience entry
points. It does not implement a second precision, performance, checkpoint, or
stability framework. Graph-aware comparisons delegate to
`tests/glm5_2_combination`, which reuses the formal fixed-checkpoint and
fixed-token workflow.

Current executable support is Ascend NPU only. The `cuda` device vocabulary is
reserved, but selecting Inductor or NPUGraph for a CUDA endpoint raises
`NotImplementedError` until the CUDA `torch.compile` policy is defined.

The complete 15-topology debug history, downgrade boundary, three-repository
fix ownership, runnable commands, and remaining backend issues are summarized
in [NPU_GRAPH_DEBUG_REPORT.md](NPU_GRAPH_DEBUG_REPORT.md).

## Documentation map

This README is the primary user entry point. Read the documents below in this
order when reproducing or extending the graph work:

| Layer | Document or implementation | Purpose |
|---|---|---|
| Experiment interface | This README | Commands, options, outputs, reports, and acceptance rules. |
| Current engineering status | [NPU_GRAPH_DEBUG_REPORT.md](NPU_GRAPH_DEBUG_REPORT.md) | Complete single/multi-card bring-up process, solved and unresolved issues, downgrade boundary, and three-repository ownership. |
| Lower-layer handoff | [LOWER_LAYER_ISSUE_HANDOFF.md](LOWER_LAYER_ISSUE_HANDOFF.md) | Ticket-ready source locations, functions, confidence boundaries, patch directions, minimal bisects, and workaround-off acceptance criteria for PyTorch, torch_npu, op-plugin, CANN, and HCCL. |
| Raw debug evidence | [graph debug README](../glm5_2_graph_debug/README.md), [report index](../glm5_2_graph_debug/experiments/reports/index.md), and [failure history](../glm5_2_graph_debug/experiments/reports/failures.md) | Immutable command history, topology evidence, failed attempts, and detailed root-cause records. |
| Ascend implementation | [TorchTitanTurbo graph-mode document](https://github.com/loveofguoke/TorchTitanTurbo/blob/glm-dev/torchtitanturbo/tools/GRAPH_MODE.md) and [patch inventory](https://github.com/loveofguoke/TorchTitanTurbo/blob/glm-dev/PATCHES.md) | Opt-in NPU compatibility patches, activation variables, patched objects, and limitations. |
| Device-neutral framework | `torchtitan/distributed/compile.py` in the source-installed TorchTitan checkout | Native `torch.compile` component selection and backend invocation. It contains no NPU workaround. |

The test report is authoritative for experiment results and issue status. Turbo
documents are authoritative for the implementation of an Ascend patch. Raw
debug reports are evidence, not a second user interface.

Supported modes are:

- `eager`: no compile arguments;
- `inductor`: TorchTitan `torch.compile`, backend `inductor`;
- `npugraphs`: NPU-only NPUGraph backend, model component only.

## Direct NPU training

Eager:

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
unset CUDA_VISIBLE_DEVICES

NGPU=1 LOG_RANK=0 MODULE=glm5 CONFIG=glm5_debugmodel ./run_train.sh
```

Inductor:

```bash
NGPU=1 LOG_RANK=0 MODULE=glm5 CONFIG=glm5_debugmodel ./run_train.sh \
  --compile.enable \
  --compile.components=model \
  --compile.backend=inductor
```

NPUGraph:

```bash
NGPU=1 LOG_RANK=0 MODULE=glm5 CONFIG=glm5_debugmodel ./run_train.sh \
  --compile.enable \
  --compile.components=model \
  --compile.backend=npugraphs
```

`--training.disable_cuda_graphs` controls TorchTitan whole-step CUDA Graph and
is independent from `--compile.enable`. Keep it disabled for the current NPU
graph experiments, especially PP.

For the validated CANN environment, persistent cache, and detailed compiler
diagnostics, use:

- `tests/glm5_2_graph_debug/run_npu_inductor.sh`
- `tests/glm5_2_graph_debug/RUN_NPU_INDUCTOR.md`

For ordinary run-through validation, prefer the shared smoke runner over
manually spelling every topology argument:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_smoke/train_smoke.py \
  --device npu --topology all --graph inductor \
  --compiler-diagnostics
```

Use `--topology single`, one topology, or `--topologies` for a focused run.

## Eager versus graph acceptance

The primary graph acceptance experiment is same-device self-consistency:

```text
reference = NPU eager, single card
candidate = NPU eager/Inductor/NPUGraph, selected topology
```

The fixed reference is captured once and reused by every candidate topology.
The same fixture is reused across graph modes and objectives when the training
configuration is unchanged.

Fast single-card probe:

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
unset CUDA_VISIBLE_DEVICES

python tests/glm5_2_graph/compile_probe.py \
  --data --data-device npu --topology single \
  --objectives precision \
  --reference-graph eager --candidate-graph inductor

python tests/glm5_2_graph/compile_probe.py \
  --capture reference --topology single \
  --objectives precision \
  --reference-graph eager --candidate-graph inductor \
  --compiler-diagnostics

python tests/glm5_2_graph/compile_probe.py \
  --capture candidate --topology single \
  --objectives precision \
  --reference-graph eager --candidate-graph inductor \
  --compiler-diagnostics

python tests/glm5_2_graph/compile_probe.py \
  --compare --topology single \
  --objectives precision \
  --reference-graph eager --candidate-graph inductor \
  --compiler-diagnostics --require-all
```

Use `precision_benchmark.py` for the 5000-step configuration and
`performance_benchmark.py` as the graph-performance convenience entry point.
Both support:

```text
--topology single
--topology fsdp8
--topologies ddp8,fsdp8,tp8
--topology all
```

For a distributed candidate, export the number of devices required by that
topology and run the same complete fixture, reference, candidate, and compare
sequence. This example uses `fsdp8`; it can be replaced by any registered
topology:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset CUDA_VISIBLE_DEVICES

python tests/glm5_2_graph/precision_benchmark.py \
  --data --data-device npu --topology fsdp8 \
  --reference-graph eager --candidate-graph inductor
python tests/glm5_2_graph/precision_benchmark.py \
  --capture reference --topology fsdp8 \
  --reference-graph eager --candidate-graph inductor
python tests/glm5_2_graph/precision_benchmark.py \
  --capture candidate --topology fsdp8 \
  --reference-graph eager --candidate-graph inductor
python tests/glm5_2_graph/precision_benchmark.py \
  --compare --topology fsdp8 --require-all \
  --reference-graph eager --candidate-graph inductor
```

For the complete graph-acceptance suite, change the selector to `all` so every
registered candidate topology is captured and compared against the reusable
single-card eager reference:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset CUDA_VISIBLE_DEVICES

python tests/glm5_2_graph/precision_benchmark.py \
  --data --data-device npu --topology all \
  --reference-graph eager --candidate-graph inductor
python tests/glm5_2_graph/precision_benchmark.py \
  --capture reference --topology all \
  --reference-graph eager --candidate-graph inductor
python tests/glm5_2_graph/precision_benchmark.py \
  --capture candidate --topology all \
  --reference-graph eager --candidate-graph inductor
python tests/glm5_2_graph/precision_benchmark.py \
  --compare --topology all --require-all \
  --reference-graph eager --candidate-graph inductor
```

CUDA eager remains available in the ordinary precision/smoke suites. Compiled
CUDA graph commands are intentionally not shown as runnable examples because
selecting `--device gpu` with `inductor` or `npugraphs` currently raises
`NotImplementedError`.

The reference remains single-card even when the candidate is FSDP8, TP8, or
another distributed topology. Run `--capture reference` with at least one
visible NPU; the artifact is shared by all selected candidates.

Any pair is valid on NPU:

```text
eager       vs eager
eager       vs inductor
eager       vs npugraphs
inductor    vs inductor
inductor    vs npugraphs
npugraphs   vs npugraphs
```

The normal acceptance baseline is eager single versus the selected graph
candidate. Other pairs are diagnostic experiments.

## Command-line parameters

`compile_probe.py`, `precision_benchmark.py`, and
`performance_benchmark.py` use the combination executor, so they share the
same options. Their entry-point defaults differ only where noted below.

| Parameter | Purpose and choices | Default |
|---|---|---|
| `--data` | Generate the shared step-0 checkpoint and fixed token plan, then exit. | no action; one action is required |
| `--capture reference\|candidate` | Run one endpoint and save its portable metrics artifact. | no action; one action is required |
| `--compare` | Compare existing captures on CPU and generate the suite report. | no action; one action is required |
| `--list-topologies` | Print every selectable topology and its degrees. | no action; one action is required |
| `--topology` | Select one candidate topology or `all`. | `single` for all graph entry points |
| `--topologies` | Comma-separated candidate subset such as `ddp8,fsdp8,tp8`; mutually exclusive with `--topology`. | unset |
| `--repeat` | Capture only repeat `N`. If omitted, capture both configured repeats. | all configured repeats (`2`) |
| `--precision` | `fp32`, `bf16`, or `full-bf16`. `bf16` keeps FP32 master training with BF16 parameters; `full-bf16` changes the training dtype too. | `bf16` from the entry config |
| `--objectives` | `precision`, `performance`, or `precision,performance`. | `precision` for probe/precision; `performance` for performance entry point |
| `--reference-graph` | Reference execution: `eager`, `inductor`, or `npugraphs`. | `eager` |
| `--candidate-graph` | Candidate execution: `eager`, `inductor`, or `npugraphs`. | `inductor` |
| `--compile-loss` | Compile `loss` in addition to `model`. NPUGraph is currently model-only. | disabled |
| `--compiler-diagnostics` | Record graph-break, recompile, and dynamic-shape diagnostics for compiled endpoints. | disabled |
| `--profiler-preset` | `off`, `overview`, `comparison`, `standard`, `distributed`, `kernel`, or `runtime`. | `off` |
| `--profile-skip-steps` | Steps before the scheduled Profiler window. | `10` |
| `--profile-warmup-steps` | Profiler warmup steps. | `1` |
| `--profile-active-steps` | Recorded Profiler steps. | `3` |
| `--data-device` | Fixture-generation backend override. The generic choices are `cuda` and `npu`; current NPU graph entry points use `npu`. | inferred from the NPU visibility variable |
| `--force` | Replace an existing valid fixture/capture instead of reusing it. | disabled |
| `--require-all` | Make compare fail unless every selected topology and repeat is present. | disabled |

The fixed training profile is 5000 steps, local batch 8, global batch 64,
sequence length 128, seed 61, and two repeats. `compile_probe.py` overrides
that profile to 10 steps, batch 1, sequence length 32 for a fast functional
check.

Capture and compare commands must use identical precision, selected graph
modes, compiled components, compiler diagnostics, objectives, and profiler
settings. The `--data` phase only depends on the model/training input contract,
so it does not need graph, objective, or profiler flags.

## Report contents and acceptance

Graph entry points do not define a separate report format. They produce the
same formal precision and performance reports as the combination executor,
with reference/candidate graph mode, compiled components, diagnostics setting,
and topology recorded in the experiment identity. The precision report shows
loss and grad-norm curves plus formal error statistics; the performance report
shows each endpoint's step time/throughput and candidate speedup, with Profiler
details only when a preset is enabled.

For the primary graph acceptance experiment, precision PASS/FAIL comes from
NPU eager-single versus the selected NPU graph candidate. Performance numbers
are diagnostic unless a separate regression target is declared. A compile
that merely launches through smoke is not sufficient evidence of graph-mode
precision acceptance.

## Outputs

```text
precision_fixtures/<fixture-id>/
combination_runs/<experiment-id>/<topology>/<role>-r<repeat>/runtime.log
combination_artifacts/<experiment-id>/<topology>/<role>-r<repeat>/
combination_reports/<experiment-id>/
```

Compiler cache and `torch_compile_debug` output are not experiment reports.
Place them outside the repository or under an ignored cache root.
