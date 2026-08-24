# fsdp2-cp4 experiments

## topology

| field | value |
| --- | --- |
| world size | 8 |
| dp replicate / shard | 1 / 2 |
| tp / cp / pp / ep | 1 / 4 / 1 / 1 |
| recorded runs | 1 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-d7029385](../../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-d7029385/readme.md) | completed | profiler-off | distributed | 840.80 ms | 1,217.90 | 9,743.17 | 0.201 GiB |

## experiment sequence

### 1. npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-d7029385

- Status: `completed`
- Started: `2026-08-24T17:59:23.655844+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-d7029385/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-d7029385.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp2-cp4 --preset distributed --profiler-off --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

## current summary

The highest recorded job throughput is 9,743.17 tok/s from `npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-d7029385`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
