# npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-270760c1

## experiment

| field | value |
| --- | --- |
| status | failed |
| topology | ddp8 |
| world size | 8 |
| mode | profiler-off |
| device | npu |
| preset | - |
| steps | - |
| local / global batch | - / - |
| sequence length | - |
| parameter / reduction dtype | - / - |
| replicate | - |
| visible devices | - |
| started | 2026-08-24T17:40:01.329634+08:00 |

## process

1. Run the `probe` driver:

   ```bash
   /root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology ddp8 --preset distributed --profiler-off --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1
   ```

2. Produce compact evidence (`manifest.json`, `metrics.jsonl`, `analysis.json`, tool status files) and the HTML report when analysis completed.

## results

| metric | value |
| --- | ---: |
| median step | - ms |
| p90 step | - ms |
| throughput / device | - tok/s |
| throughput / job | - tok/s |
| peak active hbm | - GiB |

### failure

`CalledProcessError(1, ['/root/miniconda3/envs/torchtitan-0803/bin/torchrun', '--nnodes=1', '--node_rank=0', '--nproc_per_node=8', '--rdzv_backend=c10d', '--rdzv_endpoint=localhost:0', '--rdzv_id=npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-270760c1', '--local-ranks-filter=0', '--role=rank', '--tee=3', '-m', 'tests.glm5_2_performance.capture_metrics', '--module', 'glm5', '--config', 'glm5_debugmodel', '--dump_folder=/workspace/y50064852_yyb/torchtitan-test/performance_runs/npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-270760c1/trainer_output', '--training.steps=20', '--training.num_tokens_per_microbatch_per_dp_rank=1024', '--training.num_tokens_per_train_step=8192', '--training.max_context_length=128', '--training.dtype=float32', '--training.mixed_precision_param=bfloat16', '--training.mixed_precision_reduce=float32', '--debug.seed=61', '--debug.no-enable-structured-logging', '--metrics.log_freq=1', '--metrics.enable_tensorboard', '--metrics.disable_color_printing', '--metrics.save_tb_folder=tensorboard', '--parallelism.spmd_backend=partial_dtensor', '--parallelism.data_parallel_replicate_degree=8', '--parallelism.data_parallel_shard_degree=1', '--parallelism.context_parallel_degree=1', '--parallelism.tensor_parallel_degree=1', '--parallelism.pipeline_parallel_degree=1', '--parallelism.expert_parallel_degree=1'])`

See [failure.json](failure.json) and [failed_runtime.log](failed_runtime.log) for the complete evidence.

## outputs

| output | purpose |
| --- | --- |
| [command_history.jsonl](command_history.jsonl) | exact driver history and environment |
| [failure.json](failure.json) | failure command and exception |
| [failed_runtime.log](failed_runtime.log) | failed training stdout/stderr |


Profiler-off results are throughput evidence. Profiler-active results are used for attribution only and must not be compared as authoritative throughput.
