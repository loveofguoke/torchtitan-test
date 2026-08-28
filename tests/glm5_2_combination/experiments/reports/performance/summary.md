# eager/Inductor 组合性能实验

## Executive summary

The matrix contains 15 topologies, two eager repeats, and two Inductor
repeats: 60/60 runs completed with 30 metrics each. Steps 1-10 are
excluded; steps 11-30 are the profiler-off performance authority.

The controlled result is structural: every TP-containing topology gains
45.52%-54.00%, while single/DDP/FSDP/CP/EP paths move mostly within
-1.62% to +3.82%. PP-only gains are 5.39%-8.73%. Inductor therefore
reduces a large model/host-launch component in TP compositions but does
not erase collective count, CP traffic, or pipeline bubble.

## 实验设置

| field | value |
|---|---|
| model/config | `glm5_debugmodel` |
| input | fixed random checkpoint and token plan |
| fixture | `precision_fixtures/self-npu-bf16-random-s30-b64-seq128-seed61-976dfb03` |
| training | FP32 compute, BF16 parameters, FP32 reduction |
| batch/sequence | local batch 8, global batch 64, sequence 128 |
| tokens per step | 8,192 |
| steps | 30; exclude 1-10, measure 11-30 |
| repeats | 2 per topology and mode |
| graph modes | eager vs Inductor/AOT/Triton |
| profiler | off; this matrix is undisturbed throughput, not trace analysis |
| determinism | performance-only nondeterministic autotuning, same for both modes |
| topology set | single, DDP, FSDP, TP, CP, PP, EP and composite parallelism; 15 total |

每个 repeat 的权威 step time 是 steps 11-30 的中位数；每个拓扑/模式的值是两个
repeat 中位数的中位数。`speedup = eager_step / inductor_step`，
`change = (speedup - 1) * 100%`。p90 和 peak active HBM 保留在逐拓扑报告。
两个模式复用同一 fixture，但分别作为 candidate capture 运行；离线汇总器按完整
训练契约、拓扑和 repeat 配对，不把 single reference 混入 A/B 性能值。

## 三仓身份

| repository | branch | commit |
|---|---|---|
| torchtitan | `feat/glm5-model-distributed` | `3327058390dad1f48fb61327b66d1bab3deac1a2` |
| TorchTitanTurbo | `glm-dev` | `7343c9beda9a067700030a086dc3611ce9db3f54` |
| torchtitan-test | `master` | `77f4e2eb11e4073155afe38592be9ec4505524aa` |

`graph_compat.py` content SHA256: `f677d07bdd19db4d60e611be44adeeee0357515936bf0c1bfcba4b9fc3aa5762`. Artifact manifests preserve
the complete dirty status and package versions for every run.

## Complete comparison

| topology | cards | eager step | Inductor step | speedup | change | eager drift | graph drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| [fsdp2-tp2-pp2](8-card/fsdp2-tp2-pp2/analysis.md) | 8 | 14.010456 s | 9.097924 s | 1.5400x | +54.00% | 1.48% | 3.68% |
| [fsdp2-tp4](8-card/fsdp2-tp4/analysis.md) | 8 | 3.333390 s | 2.174089 s | 1.5332x | +53.32% | 5.41% | 0.77% |
| [tp8](8-card/tp8/analysis.md) | 8 | 6.424063 s | 4.240238 s | 1.5150x | +51.50% | 0.06% | 2.60% |
| [fsdp4-tp2](8-card/fsdp4-tp2/analysis.md) | 8 | 1.628292 s | 1.083835 s | 1.5023x | +50.23% | 0.45% | 0.01% |
| [tp2-cp4](8-card/tp2-cp4/analysis.md) | 8 | 6.516221 s | 4.359172 s | 1.4948x | +49.48% | 0.45% | 0.44% |
| [fsdp2-tp4-ep8](8-card/fsdp2-tp4-ep8/analysis.md) | 8 | 3.768185 s | 2.589503 s | 1.4552x | +45.52% | 4.25% | 1.66% |
| [pp8](8-card/pp8/analysis.md) | 8 | 4.185296 s | 3.849400 s | 1.0873x | +8.73% | 4.51% | 8.28% |
| [fsdp2-pp4](8-card/fsdp2-pp4/analysis.md) | 8 | 3.488729 s | 3.310308 s | 1.0539x | +5.39% | 0.75% | 0.57% |
| [fsdp2-cp4](8-card/fsdp2-cp4/analysis.md) | 8 | 1.477067 s | 1.422781 s | 1.0382x | +3.82% | 3.11% | 0.22% |
| [ddp2](2-card/ddp2/analysis.md) | 2 | 1.581147 s | 1.524484 s | 1.0372x | +3.72% | 6.94% | 2.14% |
| [cp8](8-card/cp8/analysis.md) | 8 | 2.880988 s | 2.813859 s | 1.0239x | +2.39% | 1.73% | 1.03% |
| [fsdp8](8-card/fsdp8/analysis.md) | 8 | 0.425111 s | 0.417484 s | 1.0183x | +1.83% | 7.07% | 4.02% |
| [single](1-card/single/analysis.md) | 1 | 3.058331 s | 3.004804 s | 1.0178x | +1.78% | 4.04% | 4.99% |
| [ep8](8-card/ep8/analysis.md) | 8 | 0.791739 s | 0.777959 s | 1.0177x | +1.77% | 1.05% | 0.33% |
| [ddp8](8-card/ddp8/analysis.md) | 8 | 0.420374 s | 0.427276 s | 0.9838x | -1.62% | 2.98% | 4.39% |

## Interpretation

1. TP2/4/8 and TP composite shapes consistently gain about 1.46x-1.54x.
   This matches the prior profiler evidence that TP paths are launch-heavy.
2. DDP8 regresses 1.62% while FSDP8 gains 1.83%; both are within the
   diagnostic noise boundary on the current unhealthy/contended node.
3. CP8 and FSDP2-CP4 gain only 2.39% and 3.82%; graph compilation does
   not remove their AllGather/ReduceScatter traffic.
4. PP8 and FSDP2-PP4 gain 8.73% and 5.39%; stage readiness and bubble
   remain dominant after kernel/host improvements.
5. Inductor reports zero graph breaks and zero backend failures in every
   successful run. Every topology except PP8 records one recompilation
   marker per repeat; PP8 records none.

## Validity boundary

- CANN 9.1.0, torch 2.14 dev, torch_npu 2.14, triton_ascend 3.2.1.
- This historical matrix used the performance-only nondeterministic switch
  because torch_npu had not marked pointwise autotuning as vetted. The current
  common launcher has a pointwise-only deterministic compatibility path, but
  these recorded numbers were not rerun under it. Precision acceptance remains
  a separate deterministic task.
- Physical NPU0/1/3/4 reported Alarm. During the eight-card Inductor batch,
  an external 5000-step single-card job occupied NPU0. Eight-card speedups
  are diagnostic; large TP gains are strong candidates, while sub-5%
  changes require idle-node confirmation.
- NPUGraphs native replay was not tested here. The graph candidate is
  Inductor/AOT/Triton.

## Entry points

- [1-card](1-card/index.md)
- [2-card](2-card/index.md)
- [8-card](8-card/index.md)
- [failure and limitation record](failures.md)
- [complete command history](../../history/performance-commands.md)
- `comparison.html` for the visual comparison table
