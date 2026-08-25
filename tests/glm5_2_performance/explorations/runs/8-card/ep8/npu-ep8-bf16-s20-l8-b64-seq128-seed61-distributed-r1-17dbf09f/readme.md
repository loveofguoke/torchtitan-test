# npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-17dbf09f

## experiment

| field | value |
| --- | --- |
| status | failed |
| topology | ep8 |
| world size | 8 |
| mode | profiler-active |
| device | npu |
| preset | - |
| steps | - |
| local / global batch | - / - |
| sequence length | - |
| parameter / reduction dtype | - / - |
| replicate | - |
| visible devices | - |
| started | 2026-08-25T09:06:17.768552+08:00 |

## process

1. Run the `probe` driver:

   ```bash
   /root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology ep8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1 --cluster
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

`CalledProcessError(1, ['/root/miniconda3/envs/torchtitan-0803/bin/torchrun', '--nnodes=1', '--node_rank=0', '--nproc_per_node=8', '--rdzv_backend=c10d', '--rdzv_endpoint=localhost:0', '--rdzv_id=npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-17dbf09f', '--local-ranks-filter=0', '--role=rank', '--tee=3', '-m', 'tests.glm5_2_performance.capture_metrics', '--module', 'glm5', '--config', 'glm5_debugmodel', '--dump_folder=/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-17dbf09f/trainer_output', '--training.steps=20', '--training.num_tokens_per_microbatch_per_dp_rank=1024', '--training.num_tokens_per_train_step=8192', '--training.max_context_length=128', '--training.dtype=float32', '--training.mixed_precision_param=bfloat16', '--training.mixed_precision_reduce=float32', '--debug.seed=61', '--debug.no-enable-structured-logging', '--metrics.log_freq=1', '--metrics.enable_tensorboard', '--metrics.disable_color_printing', '--metrics.save_tb_folder=tensorboard', '--profiler.enable_profiling', '--profiler.save_traces_folder=profiling/traces', '--profiler.profile_freq=5', '--profiler.profiler_repeat=1', '--profiler.profiler_skip_first=8', '--profiler.profiler_warmup=2', '--profiler.profiler_active=3', '--parallelism.spmd_backend=partial_dtensor', '--parallelism.data_parallel_replicate_degree=1', '--parallelism.data_parallel_shard_degree=8', '--parallelism.context_parallel_degree=1', '--parallelism.tensor_parallel_degree=1', '--parallelism.pipeline_parallel_degree=1', '--parallelism.expert_parallel_degree=8'])`

See [failure.json](failure.json) and [failed_runtime.log](failed_runtime.log) for the complete evidence.

## outputs

| output | purpose |
| --- | --- |
| [command_history.jsonl](command_history.jsonl) | exact driver history and environment |
| [failure.json](failure.json) | failure command and exception |
| [failed_runtime.log](failed_runtime.log) | failed training stdout/stderr |


Profiler-off results are throughput evidence. Profiler-active results are used for attribution only and must not be compared as authoritative throughput.
