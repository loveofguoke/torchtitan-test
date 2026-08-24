# fsdp4-tp2 experiments

## topology

| field | value |
| --- | --- |
| world size | 8 |
| dp replicate / shard | 1 / 4 |
| tp / cp / pp / ep | 2 / 1 / 1 / 1 |
| recorded runs | 1 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-fsdp4-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-e3971d37](../../../runs/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-e3971d37/readme.md) | completed | profiler-off | distributed | 1,129.79 ms | 906.36 | 7,250.90 | 0.372 GiB |

## experiment sequence

### 1. npu-fsdp4-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-e3971d37

- Status: `completed`
- Started: `2026-08-24T18:02:44.721148+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-e3971d37/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp4-tp2/npu-fsdp4-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-e3971d37.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp4-tp2 --preset distributed --profiler-off --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

## current summary

The highest recorded job throughput is 7,250.90 tok/s from `npu-fsdp4-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-e3971d37`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.
