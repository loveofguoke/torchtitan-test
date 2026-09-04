# Experiment framework dependency and regression contract

torchtitan-test is the external validation and launch repository for
source-installed TorchTitan and TorchTitanTurbo. It must not contain model
implementation fixes or NPU monkey patches.

## Three-repository relationship

- `../torchtitan`: device-independent GLM model, configuration, parallelism,
  and state-dict adapter.
- `../TorchTitanTurbo`: NPU patch, graph compatibility, profiler integration,
  and NPU-specific optimized paths.
- This repository: experiment orchestration, fixed inputs, captures, reports,
  lifecycle handling, and release transport.

Every artifact records source metadata, but experiment identity is determined
by its normalized configuration, fixture generation, input contract, and
artifact checksums. Git cleanliness is diagnostic, not a pass/fail condition.

## Shared-module impact map

| Changed module | Required consumers to inspect |
|---|---|
| `glm5_2_common.topology` | precision, performance, checkpoint, stability, smoke, graph, combination, MindStudio official validation |
| naming/config hashing | fixtures, runs, artifacts, reports, legacy migration, MindStudio official validation, GitHub Release |
| common execution/process lifecycle | checkpoint failure modes, smoke cleanup, all capture workflows |
| fixed token plan or Trainer data CLI | precision, checkpoint, stability, graph precision, combination precision, MindStudio official validation |
| precision fixture/capture | self-consistency suite, migration suite, graph and combination precision |
| graph mode or Turbo compile API | graph debug, graph smoke, graph precision/performance, combination |
| profiler API or output layout | performance capture, preset-all matrix, stack/flamegraph and memory-timeline rendering, TensorBoard discovery, offline analysis, curated explorations, combination reports, Release |
| Nsight Systems CLI or GPU performance output layout | `glm5_2_nsys` capture, stats/export, report paths, lifecycle tests, GPU/NPU comparison inputs, Release discovery |
| graph diagnostics output | per-rank `TORCH_TRACE`, `tlparse`, Inductor FX/IR/code inventory, combination reports, Release |
| external profiler/compiler/accuracy tool or environment dependency | common dependency inventory, performance, graph and MindStudio guides/READMEs, combination reports, Release portability |
| msOpProf or msMemScope CLI/output layout | MindStudio specialized tuning launchers, toolchain doctor/lock, operator/memory guides, lifecycle tests, Release analysis filter |
| GPU MindStudio collection/analysis environment | MindStudio README/source install, msProbe GPU capture, Nsight Systems output, calibrate_npu_gpu input contract, Release/inbound transport |
| MindStudio toolchain lock/bootstrap/doctor | `glm5_2_mindstudio` capture/compare identity, source-install guide, server validation matrix, Release provenance |
| MindStudio pre-check output layout | endpoint artifact discovery, pre-check compare, main report, README sync commands, Release analysis filter |
| MindStudio Monitor V2 hook/config | single and distributed capture, PP/optimizer ownership validation, per-rank CSV completeness, report semantics |
| MindStudio graph visualization output | L0/mix construct validation, `.vis.db` hash index, TensorBoard command, Release analysis sensitivity policy |
| parity trace/report schema | paired and offline parity, artifact reader/writer, HTML regression tests |
| output directory nesting | rerun reset, report links, docs, release discovery and restore |

## Current GLM graph/parity boundary

The current compatibility baseline is TorchTitan `59899ade`, Turbo
`a5306484`, and this repository `01f2f3e1`, plus the reviewed working-tree
changes recorded by the combination submission-readiness report.

- Deterministic pointwise autotune compatibility is Turbo-owned and activated
  by the graph common launcher. The lower-layer torch_npu root fix and exact
  symbol are tracked as G020 in
  `tests/glm5_2_graph/LOWER_LAYER_ISSUE_HANDOFF.md`.
- The stronger Ascend GLM Router contract (BF16 input, one FP32 gate call, FP32
  scores) is Turbo-owned after TorchTitan `ad17686a` removed the common
  `Linear(compute_dtype=...)` extension. CPU parity tests must not pretend to
  validate that NPU-specific dtype contract.
- An NPU parity endpoint must be explicit (`GLM5_PARITY_DEVICE=npu`) so Turbo
  patches are loaded before model construction. Auto device selection must not
  reuse an NPU merely because an earlier test imported torch_npu.

## Experiment lifecycle contract

- `--force` starts a new selected generation and removes every selected old
  run, artifact, report, state file, and archived failed/previous output before
  launching the first process.
- Rerunning the same command without `--force` skips complete members from the
  current fixture generation and retries only incomplete/failed members.
- A fixture generation ID, exact fixture digest, or equivalent immutable
  parity fixture digest must prevent old and new captures from being mixed.
- An active recorded orchestrator PID blocks both retry and force deletion.
- Finalizing an already completed training run must preserve the original
  training attempt identity in its artifact metadata.
- Every subprocess stage prints and records its exact log path and resolved
  command.

## Required change procedure

1. Read the owning experiment README and the common module dependency table
   above.
2. Search every listed consumer before changing an API, path, naming rule,
   config field, or command argument.
3. Add lifecycle tests for force reset, interrupted resume, complete skip,
   active-run refusal, and fixture-generation mismatch where applicable.
4. Run CPU unit tests, then the smallest smoke command for the affected device
   and topology. Run formal numerical/performance experiments only after smoke.
5. Keep raw runs ignored; preserve curated exploration evidence intentionally.
6. Update README commands, report descriptions, Release discovery, and this
   document whenever framework organization changes.

## MindStudio official-workflow audit

Changes below require an additional audit even when the public CLI does not
change:

- A `toolchain.lock.json` or bootstrap change must verify the external
  `toolchain.resolved.json`, doctor CLI/import coherence, capture identity, and
  source-install documentation. Floating `master` is exploratory only; formal
  runs pin a tag or full commit.
- All new GLM MindStudio, graph, performance, accuracy, and Full DSA runs use
  CANN 9.1.0. Doctor must reject an NPU shell that resolves CANN 9.0.0; historical
  9.0.0 outputs remain readable evidence but are not a new capture environment.
- API pre-check has asymmetric storage: endpoint `reference`/`candidate`
  results live under `mindstudio_artifacts/.../precision_precheck/`, while
  `api_precision_compare` and its HTML/JSON index live under
  `mindstudio_reports/.../precision_precheck/compare-rN/`. Audit all sync and
  Release documentation after changing either path.
- Monitor V2 comparison is an index of per-rank CSV captures, not an official
  cross-device numerical comparator. Do not turn an `unparsed` monitor summary
  into PASS. Revalidate sharded optimizer ownership and PP model-parts on the
  target server after changing Trainer or topology integration.
- `graph_visualize` consumes complete same-generation L0/mix captures and
  produces processed `.vis.db` files. Release `analysis` may retain reviewed
  databases while excluding raw tensors; the database can still contain model
  names, statistics, source paths, and server paths.
- Keep official module/API evidence, end-to-end training evidence, performance
  capture evidence, and compile evidence in separate classes. Tool-stage
  completion is never a numerical or performance verdict.
- A `msprof-analyze cluster` option change must update the analysis identity,
  CLI help, output inventory, Release discovery, and the official field guide.
  `cluster --force` bypasses analyzer input checks; it is not the experiment
  lifecycle `--force` and must never delete or replace a capture generation.
- Advanced msprof-analyze recipe selection is capture-, topology-, and
  baseline-sensitive. Audit the cluster stage identity, per-rank discovery,
  report inventory, Release discovery, and both cluster field guides. Recipes
  that mutate rank databases (mstx2commop, p2p_pairing, and pp_chart) must
  never be added to an automatic policy. Read-only recipes use the profiler
  root as their shared official `-o` and run both `--export_type=db` and
  `--export_type=text`; never introduce an `advanced/` delivery hierarchy.
- MindStudio standard cluster analysis keeps the official DB and text
  deliveries together under
  `trainer_output/profiling/traces/cluster_analysis_output/`. Ascend
  PyTorch Profiler captures commonly contain both input formats, so the
  orchestrator runs the official cluster analyzer once for the DB input and
  once through a disposable text-only view, then merges the analyzer-produced
  CSV/JSON files beside `cluster_analysis.db`. Do not synthesize these files
  from SQLite or restore the legacy top-level recipe directory fan-out.
- Offline parse, advisor, cluster, and compare are independently resumable
  derived stages. Adding or forcing one stage must preserve completed outputs
  from the others; only changing the capture generation invalidates every
  derived stage.
- The official `standard` preset must populate single-card Timeline, Memory,
  and Operator views; `distributed` must additionally populate Summary and
  Communication for every rank. Both therefore keep `profile_memory=True`.
  `overview` remains the explicitly lightweight topology-scan policy and does
  not promise a complete Insight Memory view.
- After cluster analysis, the profiler root is the single portable Insight
  import bundle: it contains every rank profile plus the one canonical
  `cluster_analysis_output/`. Never create a second run-level `cluster/` copy.
  Historical layouts are byte-checked and moved into this official location
  before resume; a conflict is an error rather than a silent data mix.
- Nsight Systems `.nsys-rep`, `.sqlite`, official statistics and their logs are
  run-owned under `trainer_output/profiling/nsys/`. `nsys_artifacts` contains
  only lifecycle/provenance metadata. Historical artifact-owned payloads are
  byte-checked and moved without recollection.
- msOpProf and msMemScope are independent from system profiling. Preserve their
  native `OPPROF_*` and memory DB/CSV trees below `mindstudio_runs/operator/`
  and `mindstudio_runs/memory/`; never rename or synthesize official files.
  Tool-specific metrics and hook settings are part of capture identity.
