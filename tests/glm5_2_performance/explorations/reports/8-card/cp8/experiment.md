# cp8 experiments

## topology

| field | value |
| --- | --- |
| world size | 8 |
| dp replicate / shard | 1 / 1 |
| tp / cp / pp / ep | 1 / 8 / 1 / 1 |
| recorded runs | 3 |

## run results

| run | status | mode | preset | median step | tok/s/device | tok/s/job | peak hbm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| [npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-486604c0](../../../runs/8-card/cp8/npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-486604c0/readme.md) | completed | profiler-off | distributed | 1,626.28 ms | 629.66 | 5,037.26 | 0.130 GiB |
| [npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-d80e36ca](../../../runs/8-card/cp8/npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-d80e36ca/readme.md) | failed | profiler-active | - | - ms | - | - | - GiB |
| [npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-03d875ae](../../../runs/8-card/cp8/npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-03d875ae/readme.md) | completed | profiler-active | distributed | 1,723.89 ms | 594.00 | 4,752.03 | - GiB |

## experiment sequence

### 1. npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-486604c0

- Status: `completed`
- Started: `2026-08-24T17:52:46.682516+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/cp8/npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-486604c0/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/cp8/npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-486604c0.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology cp8 --preset distributed --profiler-off --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
```

### 2. npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-d80e36ca

- Status: `failed`
- Started: `2026-08-25T09:07:49.872209+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/cp8/npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-d80e36ca/readme.md)

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology cp8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1 --cluster
```

### 3. npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-03d875ae

- Status: `completed`
- Started: `2026-08-25T11:49:31.538912+08:00`
- Full process and outputs: [run readme](../../../runs/8-card/cp8/npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-03d875ae/readme.md)
- HTML report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/cp8/npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-03d875ae.html`

```bash
/root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology cp8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 2 --cluster
```

## current summary

The highest recorded job throughput is 5,037.26 tok/s from `npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-486604c0`. Profiler-off runs are throughput evidence; profiler-active runs are attribution evidence only.

## analysis

- [Detailed topology analysis](analysis.md)
