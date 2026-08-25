# npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c

## experiment

| field | value |
| --- | --- |
| status | completed |
| topology | ep8 |
| world size | 8 |
| mode | profiler-active |
| device | npu |
| preset | distributed |
| steps | 20 |
| local / global batch | 8 / 64 |
| sequence length | 128 |
| parameter / reduction dtype | bfloat16 / float32 |
| replicate | 4 |
| visible devices | 0, 1, 2, 3, 4, 5, 6, 7 |
| started | 2026-08-25T11:40:40.938165+08:00 |

## process

1. Run the `probe` driver:

   ```bash
   /root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology ep8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 4 --cluster
   ```

2. The workflow generated and executed this training command:

   ```bash
   /root/miniconda3/envs/torchtitan-0803/bin/torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id=npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c --local-ranks-filter=0 --role=rank --tee=3 -m tests.glm5_2_performance.capture_metrics --module glm5 --config glm5_debugmodel --dump_folder=/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/trainer_output --training.steps=20 --training.num_tokens_per_microbatch_per_dp_rank=1024 --training.num_tokens_per_train_step=8192 --training.max_context_length=128 --training.dtype=float32 --training.mixed_precision_param=bfloat16 --training.mixed_precision_reduce=float32 --debug.seed=61 --debug.no-enable-structured-logging --metrics.log_freq=1 --metrics.enable_tensorboard --metrics.disable_color_printing --metrics.save_tb_folder=tensorboard --profiler.enable_profiling --profiler.save_traces_folder=profiling/traces --profiler.profile_freq=5 --profiler.profiler_repeat=1 --profiler.profiler_skip_first=8 --profiler.profiler_warmup=2 --profiler.profiler_active=3 --parallelism.spmd_backend=partial_dtensor --parallelism.data_parallel_replicate_degree=1 --parallelism.data_parallel_shard_degree=8 --parallelism.context_parallel_degree=1 --parallelism.tensor_parallel_degree=1 --parallelism.pipeline_parallel_degree=1 --parallelism.expert_parallel_degree=8
   ```

3. Parse and analyze the capture with the official tools:

   - `cluster` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli cluster -m all -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/cluster
     ```

   - `cluster_time_summary` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m cluster_time_summary -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/cluster_time_summary --export_type text
     ```

   - `communication_bottleneck_rank_0` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/communication_bottleneck_rank_0 --rank_id 0 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_1` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/communication_bottleneck_rank_1 --rank_id 1 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_2` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/communication_bottleneck_rank_2 --rank_id 2 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_3` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/communication_bottleneck_rank_3 --rank_id 3 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_4` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/communication_bottleneck_rank_4 --rank_id 4 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_5` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/communication_bottleneck_rank_5 --rank_id 5 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_6` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/communication_bottleneck_rank_6 --rank_id 6 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_7` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/communication_bottleneck_rank_7 --rank_id 7 --top_num 20 --export_type text
     ```

   - `free_analysis` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m free_analysis -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/free_analysis --top_num 20 --export_type text
     ```

4. Produce compact evidence (`manifest.json`, `metrics.jsonl`, `analysis.json`, tool status files) and the HTML report when analysis completed.

## results

| metric | value |
| --- | ---: |
| median step | 874.31 ms |
| p90 step | - ms |
| throughput / device | 1,171.21 tok/s |
| throughput / job | 9,369.64 tok/s |
| peak active hbm | - GiB |

### collectives

| rank | kind | calls/step | payload mb/step | transit ms/step | effective gb/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | alltoallv | 35.00 | 24.94 | 2.09 | 11.93 |
| 0 | allgather | 21.00 | 22.15 | 1.55 | 14.33 |
| 0 | reducescatter | 11.00 | 19.80 | 5.16 | 3.83 |
| 0 | allreduce | 8.00 | 0.00 | 0.01 | 0.01 |
| 1 | alltoallv | 35.00 | 21.94 | 1.86 | 11.82 |
| 1 | reducescatter | 11.00 | 19.80 | 4.44 | 4.45 |
| 1 | allgather | 21.00 | 22.15 | 1.51 | 14.70 |
| 1 | allreduce | 8.00 | 0.00 | 0.02 | 0.09 |
| 2 | alltoallv | 35.00 | 21.17 | 1.92 | 11.03 |
| 2 | allgather | 21.00 | 22.15 | 1.51 | 14.71 |
| 2 | reducescatter | 11.00 | 19.80 | 5.42 | 3.65 |
| 2 | allreduce | 8.00 | 0.00 | 0.02 | 0.09 |
| 3 | alltoallv | 35.00 | 24.34 | 2.04 | 11.91 |
| 3 | allgather | 21.00 | 22.15 | 1.49 | 14.88 |
| 3 | reducescatter | 11.00 | 19.80 | 4.60 | 4.30 |
| 3 | allreduce | 8.00 | 0.00 | 0.02 | 0.08 |
| 4 | alltoallv | 35.00 | 31.10 | 2.34 | 13.27 |
| 4 | allgather | 21.00 | 22.15 | 1.48 | 14.92 |
| 4 | reducescatter | 11.00 | 19.80 | 4.96 | 4.00 |
| 4 | allreduce | 8.00 | 0.00 | 0.02 | 0.08 |
| 5 | alltoallv | 35.00 | 33.11 | 2.56 | 12.96 |
| 5 | allgather | 21.00 | 22.15 | 1.50 | 14.78 |
| 5 | reducescatter | 11.00 | 19.80 | 5.32 | 3.72 |
| 5 | allreduce | 8.00 | 0.00 | 0.02 | 0.09 |
| 6 | alltoallv | 35.00 | 26.10 | 2.28 | 11.46 |
| 6 | reducescatter | 11.00 | 19.80 | 5.80 | 3.41 |
| 6 | allgather | 21.00 | 22.15 | 1.53 | 14.52 |
| 6 | allreduce | 8.00 | 0.00 | 0.02 | 0.09 |
| 7 | alltoallv | 35.00 | 22.82 | 2.02 | 11.28 |
| 7 | allgather | 21.00 | 22.15 | 1.47 | 15.03 |
| 7 | allreduce | 8.00 | 0.00 | 0.02 | 0.09 |
| 7 | reducescatter | 11.00 | 19.80 | 4.93 | 4.02 |

## outputs

| output | purpose |
| --- | --- |
| [command_history.jsonl](command_history.jsonl) | exact driver history and environment |
| [manifest.json](manifest.json) | resolved configuration and generated training command |
| [metrics.jsonl](metrics.jsonl) | raw per-step metrics |
| [analysis.json](analysis.json) | structured derived analysis |
| [artifacts.json](artifacts.json) | raw run, compact artifact, report, and MindStudio paths |
| [tool_commands](tool_commands/) | official analysis commands and return codes |

- raw run: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c`
- compact artifact: `/workspace/y50064852_yyb/torchtitan-test/performance_artifacts/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c`
- html report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c.html`
- MindStudio input: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/ep8/npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c/trainer_output/profiling/traces`

Profiler-off results are throughput evidence. Profiler-active results are used for attribution only and must not be compared as authoritative throughput.
