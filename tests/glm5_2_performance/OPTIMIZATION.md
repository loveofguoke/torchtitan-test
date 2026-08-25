# GLM-5 Full DSA optimization experiments

This layer turns existing model/runtime capabilities into reproducible A/B
experiments. It does not modify TorchTitan Trainer code and enables no
optimization by default.

The all-topology NPU evidence and one-patch-at-a-time task cards are in
[`explorations/optimization_backlog.md`](explorations/optimization_backlog.md).
Files under either repository's `ops_candidate/` are deliberately absent from
this CLI; they are code-reviewable hypotheses, not runnable optimizations.

## Three-repository implementation map

This test document is the authoritative entry for commands, measurements,
reports, and promotion decisions. Implementation details remain with their
owning repository:

| State | TorchTitan | TorchTitanTurbo | torchtitan-test |
|---|---|---|---|
| Readable reference | `models/glm5/model.py`, `FULL_DSA.md` | no duplicated model | `--full-dsa --*-kernel reference` |
| Confirmed optional implementation | `models/glm5/ops/` for CUDA Triton | `models/glm5/ops/` for Triton-Ascend and native SparseMLA | named `--full-dsa-kernel` / `--optimization` choices |
| Isolated candidate | `models/glm5/ops_candidate/` for model/general distributed prototypes | `models/glm5/ops_candidate/` for HCCL/CANN/NPUGraph prototypes | indexed by [optimization_backlog.md](explorations/optimization_backlog.md), never executable |
| Evidence and decision | supplies the component contract | supplies the backend implementation | fixed inputs, precision, profiler-off A/B, active trace, report, and promotion gate |

The repository-level implementation documents are TorchTitan
`torchtitan/models/glm5/性能优化实践.md` and TorchTitanTurbo
[`torchtitanturbo/models/glm5/性能优化实践.md`](https://github.com/loveofguoke/TorchTitanTurbo/blob/feat/glm5-full-dsa-npu/torchtitanturbo/models/glm5/%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96%E5%AE%9E%E8%B7%B5.md).

## Full DSA as an orthogonal factor

The smoke interface supports explicit model and operator selection:

```bash
# Existing behavior remains unchanged.
python tests/glm5_2_smoke/train_smoke.py \
  --device npu --topology single

# Full DSA readable reference.
python tests/glm5_2_smoke/train_smoke.py \
  --device npu --topology single --full-dsa

# Full DSA with the device-specific optional kernel.
python tests/glm5_2_smoke/train_smoke.py \
  --device npu --topology single --full-dsa --full-dsa-kernel auto
```

`reference` is the default kernel. `auto` maps CUDA to Triton and NPU to the
same mathematical kernels through Triton-Ascend. `npu-sparse` keeps the
reference indexer and uses the Ascend-native SparseMLA. Unregistered
`ops_candidate` prototypes are not benchmark choices. Repeated
`--extra-train-arg` values become part of
the manifest and run identity.

## Precision matrix

The formal benchmark accepts endpoint implementation factors:

```bash
# Reference GPU versus reference NPU.
python tests/glm5_2_precision/full_dsa_migration_benchmark.py \
  --capture candidate --topology single

# CUDA Triton versus Triton-Ascend.
python tests/glm5_2_precision/full_dsa_migration_benchmark.py \
  --capture candidate --topology single \
  --reference-kernel triton --candidate-kernel triton
```

Use the same kernel flags for data, both captures, and compare. Operator
variants reuse the reference Full DSA fixture because checkpoint and token plan
are unchanged, but captures and reports have distinct identities.

The combination experiment can apply the same factor together with graph,
precision, and profiling:

```bash
python tests/glm5_2_combination/combination_benchmark.py \
  --data --data-device npu --topology single --full-dsa \
  --reference-full-dsa-kernel reference \
  --candidate-full-dsa-kernel npu-sparse

python tests/glm5_2_combination/combination_benchmark.py \
  --capture candidate --topology single --objectives precision,performance \
  --reference-graph eager --candidate-graph inductor \
  --profiler-preset comparison --full-dsa \
  --reference-full-dsa-kernel reference \
  --candidate-full-dsa-kernel npu-sparse
```

Full DSA `all` excludes PP because shared top-k is not transported between
pipeline stages. Explicit PP selection is rejected.

## Performance profiles

`optimization_benchmark.py` reuses the Ascend profiler workflow. Each profile
changes one main factor:

| Profile | Area | Actual change |
|---|---|---|
| `reference` | baseline | readable PyTorch Full DSA |
| `sparse-mla` | compute/memory | NPU fused SparseMLA override |
| `triton-full-dsa` | compute | Triton-Ascend index scores and SparseMLA forward/backward |
| `async-tp` | communication | compile model and enable TorchTitan async TP |
| `foreach-optimizer` | compute/runtime | foreach optimizer implementation |
| `chunked-loss` | memory | increase loss chunks to 16 |
| `cpu-offload` | memory/copy | enable FSDP CPU offload |
| `full-recompute` | compute-memory | full-block activation checkpointing |

The wrapper consumes `--optimization`; all remaining flags are the normal
performance CLI documented in `README.md`. `--topology all` is profile-aware,
`--replicate` distinguishes repeated measurements, `--profiler-off` measures
undisturbed throughput, and `--preset` selects the diagnostic depth. No profile
is enabled unless it is named explicitly; the default is `reference`.

Quick single-card A/B:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0

python tests/glm5_2_performance/optimization_benchmark.py \
  --probe --device npu --topology single --optimization reference \
  --preset comparison --profiler-off --replicate 1

python tests/glm5_2_performance/optimization_benchmark.py \
  --probe --device npu --topology single --optimization sparse-mla \
  --preset comparison --profiler-off --replicate 1

python tests/glm5_2_performance/optimization_benchmark.py \
  --probe --device npu --topology single --optimization triton-full-dsa \
  --preset comparison --profiler-off --replicate 1
```

Capture the mechanism after steady-state throughput:

```bash
python tests/glm5_2_performance/optimization_benchmark.py \
  --probe --device npu --topology single --optimization triton-full-dsa \
  --preset kernel --offline --advisor
```

Run the same profile across every applicable Full DSA topology:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_performance/optimization_benchmark.py \
  --probe --device npu --topology all --optimization reference \
  --preset comparison --profiler-off --replicate 1
```

For ordinary profiles, `all` means every non-PP Full DSA topology with at most
eight ranks. For `async-tp`, it means all supported pure or FSDP+TP layouts.
Explicit Full DSA PP selections are rejected because shared top-k is not
transported across pipeline-stage boundaries.

Communication overlap requires TP:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_performance/optimization_benchmark.py \
  --probe --device npu --topology tp8 --optimization async-tp \
  --preset distributed --offline --cluster
```

Memory/recompute examples:

```bash
python tests/glm5_2_performance/optimization_benchmark.py \
  --probe --device npu --topology single --optimization chunked-loss \
  --preset runtime --offline

python tests/glm5_2_performance/optimization_benchmark.py \
  --probe --device npu --topology single --optimization foreach-optimizer \
  --preset standard --offline

python tests/glm5_2_performance/optimization_benchmark.py \
  --probe --device npu --topology fsdp8 --optimization full-recompute \
  --preset runtime --offline

python tests/glm5_2_performance/optimization_benchmark.py \
  --probe --device npu --topology fsdp8 --optimization cpu-offload \
  --preset runtime --offline
```

The profiler currently raises `NotImplementedError` for CUDA. Its interface is
reserved so a GPU implementation can be added without changing NPU behavior.

## Acceptance

First use `--profiler-off` repeats for authoritative steady-state throughput,
then a bounded trace for attribution. Keep model, checkpoint, token plan,
topology, precision, batch, sequence length, warmup, and step window identical.
Compare:

- median/p95 step time and tokens/s;
- peak active/reserved memory;
- compute, exposed communication, and idle time;
- kernel count and top kernel shapes;
- graph breaks, recompiles, AICPU fallbacks, and host synchronization;
- formal loss and grad-norm acceptance.

An optimization passes only when numerics are acceptable and its claimed
mechanism appears in the trace. Higher overlap with slower step time, or lower
memory with excessive recompute cost, is a trade-off rather than a universal
win.
