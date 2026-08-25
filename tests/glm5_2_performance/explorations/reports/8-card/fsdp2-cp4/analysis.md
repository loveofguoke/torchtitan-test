# FSDP2-CP4 NPU performance analysis

## Status and validity

FSDP2-CP4 completed the full three-stage tool flow on all ranks. Its clean
profiler-off result is 9,743.17 tok/s/job at 840.80 ms/step with 2.13% p90
drift. This is 1.93x CP8 throughput under the same model and token budget.

The profiled run changes from 879.64 ms before profiling to 1,227.79 ms in the
active window (+39.58%) and returns to 859.52 ms afterwards. The clean run is
the throughput authority; the active window is used for collective and rank
attribution only.

## Controlled comparison with CP8

| Metric | CP8 | FSDP2-CP4 | Change |
| --- | ---: | ---: | ---: |
| profiler-off median step | 1,626.28 ms | 840.80 ms | -48.30% |
| profiler-off job throughput | 5,037.26 tok/s | 9,743.17 tok/s | 1.93x |
| AllGather calls/step | 368 | 184 | -50% |
| ReduceScatter calls/step | 216 | 108 | -50% |
| collective payload/rank/step | ~1,194.44 MB | ~579.91 MB | -51.45% |
| physical transit/rank/step | 156.27-187.76 ms | 82.77-92.44 ms | about -47% |

Splitting the eight-rank CP mesh into CP4 plus FSDP2 almost exactly halves CP
collective frequency and payload, and the clean step time nearly halves with
it. This is the strongest topology-causal result in the CP experiments. It
does not prove CP4 is optimal at long sequence length, but it shows pure CP8
is over-sharded for sequence length 128.

## Rank critical path

The active window has 1,223.74-1,229.29 ms stage time and 123.75-125.90 ms
compute, both tightly balanced. Exposed communication ranges from 53.23 to
753.87 ms, while `Free` ranges from 338.21 to 1,054.20 ms. As in CP8, ranks
trade communication wait for free time while the synchronized stage boundary
stays aligned. Compute imbalance is not the limiter.

## Collective accounting

| Collective | Calls/step | Payload/rank/step | Physical transit/rank/step | Effective bandwidth |
| --- | ---: | ---: | ---: | ---: |
| AllGather | 184 | 296.25 MB | 20.82-21.60 ms | 13.72-14.23 GB/s |
| ReduceScatter | 108 | 283.66 MB | 61.94-70.93 ms | 4.00-4.58 GB/s |
| tiny AllReduce | 7 | <=0.0015 MB | <=0.0154 ms | not bandwidth-relevant |
| Total | 299 | ~579.91 MB | 82.77-92.44 ms | - |

AllGather reaches normal HCCS bandwidth. ReduceScatter remains slower and is
the first physical-transit target, but exposed communication is still up to
8x the total transit. Both granularity and host/dependency alignment matter.

## Official-tool findings

All 160 exported slow-collective rows are Host-bound. The free analysis selects
rank 6 and reports event/memcpy wait windows up to 9.42 ms and PyTorch
no-dispatch intervals around 4.34-4.45 ms. These gaps are much smaller than
CP8's worst 91 ms no-dispatch interval, consistent with the cleaner CP4
critical path, but per-rank exposed wait remains uneven.

## Potential experiments after the measurement matrix

No optimization is implemented in this phase.

1. Repeat CP8 and FSDP2-CP4 profiler-off at longer sequences and at least three
   replicates to determine where CP8's extra sharding begins to amortize.
2. Label the 184 AllGather and 108 ReduceScatter calls at the DSA/DTensor
   boundary; group only adjacent operations with identical dependencies.
3. Inspect ReduceScatter algorithm/process-group selection through public
   HCCL/PyTorch configuration before any private HCCL call injection.
4. Add per-rank sequence IDs around CP attention exchange and FSDP hooks to
   locate the first host launch divergence behind 53-754 ms exposed wait.
5. Measure an overlap prototype with explicit communication-stream events;
   accept only if profiler-off median/p90 improves without higher HBM.

## Reproduction and evidence

```bash
cd /workspace/y50064852_yyb/torchtitan-test
unset CUDA_VISIBLE_DEVICES
TORCHTITAN_MSPROF_ANALYZE=/workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli \
TORCHTITAN_MSPROF_ANALYZE_WORKERS=2 \
HCCL_NPU_SOCKET_PORT_RANGE=auto HCCL_IF_BASE_PORT=63232 \
python tests/glm5_2_performance/profiler_benchmark.py \
  --probe --device npu --topology fsdp2-cp4 --preset distributed \
  --visible-devices 0,1,2,3,4,5,6,7 \
  --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 \
  --replicate 3 --cluster
```

- Structured analysis: [analysis.json](../../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/analysis.json)
- Exact process and tool commands: [run readme](../../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/readme.md)
- HTML report: `performance_reports/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993.html`
- Profiler-off authority: [run readme](../../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-d7029385/readme.md)
