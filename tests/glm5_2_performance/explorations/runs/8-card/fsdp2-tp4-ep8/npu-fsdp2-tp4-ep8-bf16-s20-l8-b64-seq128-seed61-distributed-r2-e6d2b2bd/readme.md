# npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd

## experiment

| field | value |
| --- | --- |
| status | completed |
| topology | fsdp2-tp4-ep8 |
| world size | 8 |
| mode | profiler-active |
| device | npu |
| preset | distributed |
| steps | 20 |
| local / global batch | 8 / 64 |
| sequence length | 128 |
| parameter / reduction dtype | bfloat16 / float32 |
| replicate | 2 |
| visible devices | 0, 1, 2, 3, 4, 5, 6, 7 |
| started | 2026-08-25T13:57:47.973837+08:00 |

## process

1. Run the `probe` driver:

   ```bash
   /root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topologies pp8,fsdp2-tp4,fsdp2-cp4,tp2-cp4,fsdp4-tp2,fsdp2-pp4,fsdp2-tp2-pp2,fsdp2-tp4-ep8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 2 --cluster
   ```

2. The workflow generated and executed this training command:

   ```bash
   /root/miniconda3/envs/torchtitan-0803/bin/torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id=npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd --local-ranks-filter=0 --role=rank --tee=3 -m tests.glm5_2_performance.capture_metrics --module glm5 --config glm5_debugmodel --dump_folder=/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/trainer_output --training.steps=20 --training.num_tokens_per_microbatch_per_dp_rank=1024 --training.num_tokens_per_train_step=8192 --training.max_context_length=128 --training.dtype=float32 --training.mixed_precision_param=bfloat16 --training.mixed_precision_reduce=float32 --debug.seed=61 --debug.no-enable-structured-logging --metrics.log_freq=1 --metrics.enable_tensorboard --metrics.disable_color_printing --metrics.save_tb_folder=tensorboard --profiler.enable_profiling --profiler.save_traces_folder=profiling/traces --profiler.profile_freq=5 --profiler.profiler_repeat=1 --profiler.profiler_skip_first=8 --profiler.profiler_warmup=2 --profiler.profiler_active=3 --parallelism.spmd_backend=partial_dtensor --parallelism.data_parallel_replicate_degree=1 --parallelism.data_parallel_shard_degree=2 --parallelism.context_parallel_degree=1 --parallelism.tensor_parallel_degree=4 --parallelism.pipeline_parallel_degree=1 --parallelism.expert_parallel_degree=8
   ```

3. Parse and analyze the capture with the official tools:

   - `cluster` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli cluster -m all -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/cluster
     ```

   - `cluster_time_summary` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m cluster_time_summary -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/cluster_time_summary --export_type text
     ```

   - `communication_bottleneck_rank_0` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/communication_bottleneck_rank_0 --rank_id 0 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_1` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/communication_bottleneck_rank_1 --rank_id 1 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_2` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/communication_bottleneck_rank_2 --rank_id 2 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_3` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/communication_bottleneck_rank_3 --rank_id 3 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_4` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/communication_bottleneck_rank_4 --rank_id 4 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_5` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/communication_bottleneck_rank_5 --rank_id 5 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_6` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/communication_bottleneck_rank_6 --rank_id 6 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_7` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/communication_bottleneck_rank_7 --rank_id 7 --top_num 20 --export_type text
     ```

   - `free_analysis` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m free_analysis -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/free_analysis --top_num 20 --export_type text
     ```

4. Produce compact evidence (`manifest.json`, `metrics.jsonl`, `analysis.json`, tool status files) and the HTML report when analysis completed.

## results

| metric | value |
| --- | ---: |
| median step | 2,658.32 ms |
| p90 step | - ms |
| throughput / device | 385.21 tok/s |
| throughput / job | 3,081.64 tok/s |
| peak active hbm | - GiB |

### collectives

| rank | kind | calls/step | payload mb/step | transit ms/step | effective gb/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | alltoallv | 140.00 | 21.61 | 1.92 | 11.23 |
| 0 | reducescatter | 212.00 | 66.09 | 4.30 | 15.37 |
| 0 | allgather | 220.00 | 71.41 | 4.19 | 17.05 |
| 0 | allreduce | 362.00 | 0.00 | 0.00 | 0.01 |
| 1 | alltoallv | 140.00 | 23.87 | 2.06 | 11.61 |
| 1 | reducescatter | 212.00 | 66.09 | 4.55 | 14.52 |
| 1 | allreduce | 362.00 | 11.96 | 1.27 | 9.44 |
| 1 | allgather | 220.00 | 71.41 | 4.12 | 17.35 |
| 2 | alltoallv | 140.00 | 21.66 | 1.95 | 11.09 |
| 2 | reducescatter | 212.00 | 66.09 | 4.34 | 15.24 |
| 2 | allgather | 220.00 | 71.41 | 4.11 | 17.38 |
| 2 | allreduce | 362.00 | 11.96 | 1.28 | 9.31 |
| 3 | alltoallv | 140.00 | 22.35 | 1.97 | 11.33 |
| 3 | reducescatter | 212.00 | 66.09 | 4.32 | 15.30 |
| 3 | allreduce | 362.00 | 11.96 | 1.28 | 9.36 |
| 3 | allgather | 220.00 | 71.41 | 4.13 | 17.30 |
| 4 | alltoallv | 140.00 | 35.09 | 2.64 | 13.31 |
| 4 | reducescatter | 212.00 | 66.09 | 4.43 | 14.92 |
| 4 | allgather | 220.00 | 71.41 | 4.15 | 17.22 |
| 4 | allreduce | 362.00 | 0.00 | 0.02 | 0.09 |
| 5 | alltoallv | 140.00 | 28.83 | 2.31 | 12.47 |
| 5 | reducescatter | 212.00 | 66.09 | 4.47 | 14.80 |
| 5 | allreduce | 362.00 | 11.96 | 1.28 | 9.34 |
| 5 | allgather | 220.00 | 71.41 | 4.11 | 17.36 |
| 6 | alltoallv | 140.00 | 28.84 | 2.31 | 12.50 |
| 6 | reducescatter | 212.00 | 66.09 | 4.41 | 15.00 |
| 6 | allreduce | 362.00 | 11.96 | 1.30 | 9.23 |
| 6 | allgather | 220.00 | 71.41 | 4.12 | 17.32 |
| 7 | alltoallv | 140.00 | 23.42 | 1.99 | 11.78 |
| 7 | reducescatter | 212.00 | 66.09 | 4.84 | 13.66 |
| 7 | allreduce | 362.00 | 11.96 | 1.30 | 9.18 |
| 7 | allgather | 220.00 | 71.41 | 4.16 | 17.16 |

## outputs

| output | purpose |
| --- | --- |
| [command_history.jsonl](command_history.jsonl) | exact driver history and environment |
| [manifest.json](manifest.json) | resolved configuration and generated training command |
| [metrics.jsonl](metrics.jsonl) | raw per-step metrics |
| [analysis.json](analysis.json) | structured derived analysis |
| [artifacts.json](artifacts.json) | raw run, compact artifact, report, and MindStudio paths |
| [tool_commands](tool_commands/) | official analysis commands and return codes |

- raw run: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd`
- compact artifact: `/workspace/y50064852_yyb/torchtitan-test/performance_artifacts/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd`
- html report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd.html`
- MindStudio input: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd/trainer_output/profiling/traces`

Profiler-off results are throughput evidence. Profiler-active results are used for attribution only and must not be compared as authoritative throughput.
