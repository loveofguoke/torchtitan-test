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
| `glm5_2_common.topology` | precision, performance, checkpoint, stability, smoke, graph, combination |
| naming/config hashing | fixtures, runs, artifacts, reports, legacy migration, GitHub Release |
| common execution/process lifecycle | checkpoint failure modes, smoke cleanup, all capture workflows |
| fixed token plan or Trainer data CLI | precision, checkpoint, stability, graph precision, combination precision |
| precision fixture/capture | self-consistency suite, migration suite, graph and combination precision |
| graph mode or Turbo compile API | graph debug, graph smoke, graph precision/performance, combination |
| profiler API or output layout | performance capture, preset-all matrix, stack/flamegraph and memory-timeline rendering, TensorBoard discovery, offline analysis, curated explorations, combination reports, Release |
| graph diagnostics output | per-rank `TORCH_TRACE`, `tlparse`, Inductor FX/IR/code inventory, combination reports, Release |
| external profiler/compiler tool or environment dependency | common dependency inventory, performance and graph guides/READMEs, combination reports, Release portability |
| parity trace/report schema | paired and offline parity, artifact reader/writer, HTML regression tests |
| output directory nesting | rerun reset, report links, docs, release discovery and restore |

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
