# FSDP8 NPU performance analysis

## Status and validity

The FSDP8 all-rank profile completed the full Ascend Profiler → offline parse →
`msprof-analyze` → MindStudio handoff. Collective structure is valid, but
active-window timing attribution is contaminated by a concurrent graph-mode
TP8 compile experiment. That external experiment ran from approximately
18:26:22 to 18:37:31; the FSDP8 active window was approximately 18:26:40-42.
Do not use the captured rank wait/overlap times as an idle-system acceptance
measurement.

The independent profiler-off baseline remains the throughput authority. It is
stable at 37,406.66 tok/s/job with 1.21% p90 drift. The profiler run's
non-active segment reaches 35,261.06 tok/s/job, 5.74% lower, which supports the
order of magnitude but does not remove the concurrency caveat.

NPU0 is included and currently reports a health warning, so all eight-card
results remain diagnostic rather than healthy-hardware acceptance evidence.

## Experiment settings

The model, tokens, precision, 20-step schedule, all-rank Level1 profile, and
official analysis recipes are identical to DDP8. The topology is one eight-rank
FSDP mesh (`dp_shard=8`). Exact commands and generated `torchrun` argv are in:

[`run readme`](../../../runs/8-card/fsdp8/npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-8501ffd6/readme.md)

## Throughput cross-check

| Source | Median step | Job throughput | Use |
| --- | ---: | ---: | --- |
| profiler-off replicate 1 | 219.00 ms | 37,406.66 tok/s | screening authority |
| profiler run, non-active | 232.32 ms | 35,261.06 tok/s | cross-check only |
| profiler active window | 354.06 ms | 23,137.44 tok/s | attribution only; 52.40% overhead |

## Collective structure

| Collective | Calls/step | Payload/rank/step | Physical transit/rank/step | Key observation |
| --- | ---: | ---: | ---: | --- |
| AllGather | 21 | 60.69 MB | 4.42-4.65 ms | stable 13.04-13.72 GB/s |
| ReduceScatter | 11 | 58.33 MB | 10.16-18.00 ms | only 3.24-5.74 GB/s and larger rank spread |
| AllReduce | 7 | 0.0001-0.0015 MB | 0.013-0.017 ms | nearly zero payload, but high synchronization wait on ranks 1-7 |

FSDP reduces the large gradient collective count from DDP8's 17 AllReduce
calls to 11 ReduceScatter calls, while parameter materialization adds 21
AllGather calls. The large-payload physical transit total is roughly
14.6-22.7 ms/rank/step. The seven tiny AllReduce calls are latency/synchronism
operations rather than bandwidth operations: ranks 1-7 report 92.8%-96.0%
wait even though transit is only about 0.016 ms/step.

This yields two separate future targets:

1. identify what the seven scalar/small AllReduces represent and whether they
   can be grouped or moved off the critical path;
2. explain why ReduceScatter bandwidth is both lower and more variable than
   AllGather on the same ranks.

## Rank critical path (contaminated timing)

Compute is still balanced at 40.76-41.94 ms/rank. Exposed communication ranges
from 13.39 ms on rank 1 to 201.34 ms on rank 2, and overlap is only 0%-7.91%.
The stage times are tightly grouped at 348.43-356.38 ms because low-wait ranks
spend correspondingly more time in the official `Free` category.

The pattern resembles host launch skew, but the concurrent graph compile can
create exactly that host-side contention. Preserve these values as a
diagnostic trace and repeat FSDP8 on an idle host before attributing them to the
training implementation.

## Official-tool findings

All 160 Top-20 rows exported by per-rank `communication_bottleneck` are
Host-bound; none is network-bound. Reason text mentions `AddBackward0` in 89
rows and `FSDP::pre_forward` in 57 rows, with event record, allocation, view,
and copy operations also present. Categories can overlap within one row.

`free_analysis` again exports rank 6. Its largest gaps are 5.45-11.89 ms
event/memcpy/wait regions and approximately 4.07-4.31 ms PyTorch no-dispatch
regions. These values are consistent with a launch pipeline bottleneck but are
not clean enough for an optimization claim because of the overlapping compile.

## Potential experiments after an idle repeat

No optimization is implemented in this phase.

1. Instrument and name the seven tiny AllReduces at the source FSDP/model
   boundary. Group only operations with identical semantic synchronization
   points; do not fuse them merely because payload is small.
2. Sweep FSDP layer grouping/wrap granularity. Record how 21 AllGather and 11
   ReduceScatter calls change, together with HBM and profiler-off p90.
3. Use the FSDP2 public collective hook for a dedicated HCCL stream prototype.
   Gate on increased overlap and unchanged optimizer dependency ordering.
4. Compare ReduceScatter payload/transit on alternative rank mappings. If the
   slow ranks follow physical devices, investigate HCCS/health; if they follow
   logical ranks, investigate launch and scheduling order.
5. Repeat the existing BF16 reduction performance-only injection. Its expected
   benefit applies to large gradient payload, not the tiny synchronization
   AllReduces.

## Reproduction

```bash
cd /workspace/y50064852_yyb/torchtitan-test
unset CUDA_VISIBLE_DEVICES
TORCHTITAN_MSPROF_ANALYZE=/workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli \
python tests/glm5_2_performance/profiler_benchmark.py \
  --probe --device npu --topology fsdp8 --preset distributed \
  --visible-devices 0,1,2,3,4,5,6,7 \
  --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 \
  --replicate 2 --cluster
```

`replicate 2` is reserved for the next idle-system repeat; this report used
replicate 1. Use a new replicate number rather than overwriting evidence.
