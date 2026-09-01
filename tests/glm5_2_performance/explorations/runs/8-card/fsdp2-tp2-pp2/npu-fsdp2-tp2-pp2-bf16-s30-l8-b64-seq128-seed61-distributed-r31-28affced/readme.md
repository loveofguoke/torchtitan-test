# npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced

## experiment

| field | value |
| --- | --- |
| status | completed |
| topology | fsdp2-tp2-pp2 |
| world size | 8 |
| mode | profiler-active |
| device | npu |
| preset | distributed |
| steps | 30 |
| local / global batch | 8 / 64 |
| sequence length | 128 |
| parameter / reduction dtype | bfloat16 / float32 |
| replicate | 31 |
| visible devices | 0, 1, 2, 3, 4, 5, 6, 7 |
| started | 2026-08-31T15:42:23.767336+08:00 |

## process

1. Run the `probe` driver:

   ```bash
   /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
   ```

2. Run the `analyze` driver:

   ```bash
   /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-tp2-pp2 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
   ```

3. The workflow generated and executed this training command:

   ```bash
   /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id=npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced --local-ranks-filter=4 --role=rank --tee=3 -m tests.glm5_2_performance.capture_metrics --module glm5 --config glm5_debugmodel --dump_folder=/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/trainer_output --training.steps=30 --training.num_tokens_per_microbatch_per_dp_rank=128 --training.num_tokens_per_train_step=8192 --training.max_context_length=128 --parallelism.num_pp_microbatches=8 --training.dtype=float32 --training.mixed_precision_param=bfloat16 --training.mixed_precision_reduce=float32 --debug.seed=61 --debug.no-enable-structured-logging --metrics.log_freq=1 --metrics.enable_tensorboard --metrics.disable_color_printing --metrics.save_tb_folder=tensorboard --profiler.enable_profiling --profiler.save_traces_folder=profiling/traces --profiler.profile_freq=7 --profiler.profiler_repeat=1 --profiler.profiler_skip_first=10 --profiler.profiler_warmup=2 --profiler.profiler_active=5 --parallelism.spmd_backend=partial_dtensor --parallelism.data_parallel_replicate_degree=1 --parallelism.data_parallel_shard_degree=2 --parallelism.context_parallel_degree=1 --parallelism.tensor_parallel_degree=2 --parallelism.pipeline_parallel_degree=2 --parallelism.expert_parallel_degree=1 --parallelism.no-enable-sequence-parallel --parallelism.pipeline_parallel_schedule=1F1B
   ```

4. Parse and analyze the capture with the official tools:

   - `advisor` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli advisor all -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/advisor
     ```

   - `cluster` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli cluster -m all -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/cluster
     ```

   - `cluster_time_summary` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m cluster_time_summary -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/cluster_time_summary --export_type text
     ```

   - `communication_bottleneck_rank_0` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/communication_bottleneck_rank_0 --rank_id 0 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_1` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/communication_bottleneck_rank_1 --rank_id 1 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_2` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/communication_bottleneck_rank_2 --rank_id 2 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_3` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/communication_bottleneck_rank_3 --rank_id 3 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_4` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/communication_bottleneck_rank_4 --rank_id 4 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_5` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/communication_bottleneck_rank_5 --rank_id 5 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_6` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/communication_bottleneck_rank_6 --rank_id 6 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_7` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/communication_bottleneck_rank_7 --rank_id 7 --top_num 20 --export_type text
     ```

   - `free_analysis` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m free_analysis -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/free_analysis --top_num 20 --export_type text
     ```

   - `run_state` (return code `None`)

     ```bash
     -
     ```

5. Produce compact evidence (`manifest.json`, `metrics.jsonl`, `analysis.json`, tool status files) and the HTML report when analysis completed.

## results

| metric | value |
| --- | ---: |
| median step | 13,350.39 ms |
| p90 step | - ms |
| throughput / device | 76.70 tok/s |
| throughput / job | 613.62 tok/s |
| peak active hbm | - GiB |

### collectives

| rank | kind | calls/step | payload mb/step | transit ms/step | effective gb/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | batchsendrecv | 29.60 | 0.92 | 0.07 | 13.02 |
| 0 | receive | 4.00 | 0.13 | 0.01 | 13.46 |
| 0 | reducescatter | 404.00 | 40.18 | 2.73 | 14.74 |
| 0 | allgather | 308.00 | 26.40 | 1.61 | 16.43 |
| 0 | allreduce | 617.00 | 0.00 | 0.00 | 0.01 |
| 0 | send | 4.00 | 0.00 | 0.00 | - |
| 1 | batchsendrecv | 29.60 | 0.92 | 0.07 | 13.05 |
| 1 | allreduce | 617.00 | 47.47 | 3.46 | 13.74 |
| 1 | receive | 4.00 | 0.13 | 0.01 | 13.57 |
| 1 | reducescatter | 404.00 | 40.18 | 2.74 | 14.66 |
| 1 | allgather | 308.00 | 26.40 | 1.60 | 16.47 |
| 1 | send | 4.00 | 0.00 | 0.00 | - |
| 2 | batchsendrecv | 29.60 | 0.92 | 0.07 | 12.77 |
| 2 | receive | 4.00 | 0.13 | 0.01 | 13.41 |
| 2 | allreduce | 617.00 | 0.00 | 0.01 | 0.06 |
| 2 | reducescatter | 404.00 | 40.18 | 2.64 | 15.23 |
| 2 | allgather | 308.00 | 26.40 | 1.61 | 16.41 |
| 2 | send | 4.00 | 0.00 | 0.00 | - |
| 3 | batchsendrecv | 29.60 | 0.92 | 0.07 | 12.98 |
| 3 | receive | 4.00 | 0.13 | 0.01 | 13.39 |
| 3 | allreduce | 617.00 | 47.47 | 3.46 | 13.73 |
| 3 | reducescatter | 404.00 | 40.18 | 2.64 | 15.20 |
| 3 | allgather | 308.00 | 26.40 | 1.61 | 16.37 |
| 3 | send | 4.00 | 0.00 | 0.00 | - |
| 4 | receive | 4.00 | 0.26 | 0.02 | 16.32 |
| 4 | allreduce | 1,514.00 | 0.00 | 0.01 | 0.01 |
| 4 | reducescatter | 436.00 | 58.82 | 3.79 | 15.53 |
| 4 | allgather | 276.00 | 42.02 | 2.38 | 17.65 |
| 4 | batchsendrecv | 29.60 | 1.84 | 0.11 | 16.14 |
| 4 | send | 4.00 | 0.00 | 0.00 | - |
| 5 | allreduce | 1,514.00 | 56.09 | 5.70 | 9.85 |
| 5 | reducescatter | 436.00 | 58.82 | 3.79 | 15.53 |
| 5 | receive | 4.00 | 0.26 | 0.02 | 16.40 |
| 5 | allgather | 276.00 | 42.02 | 2.38 | 17.65 |
| 5 | batchsendrecv | 29.60 | 1.84 | 0.11 | 16.22 |
| 5 | send | 4.00 | 0.00 | 0.00 | - |
| 6 | allreduce | 1,514.00 | 0.00 | 0.02 | 0.05 |
| 6 | reducescatter | 436.00 | 58.82 | 3.68 | 15.97 |
| 6 | receive | 4.00 | 0.26 | 0.02 | 16.37 |
| 6 | allgather | 276.00 | 42.02 | 2.37 | 17.70 |
| 6 | batchsendrecv | 29.60 | 1.84 | 0.11 | 16.19 |
| 6 | send | 4.00 | 0.00 | 0.00 | - |
| 7 | receive | 4.00 | 0.26 | 0.02 | 16.36 |
| 7 | allreduce | 1,514.00 | 56.09 | 5.71 | 9.83 |
| 7 | reducescatter | 436.00 | 58.82 | 3.67 | 16.02 |
| 7 | allgather | 276.00 | 42.02 | 2.37 | 17.70 |
| 7 | batchsendrecv | 29.60 | 1.84 | 0.11 | 16.21 |
| 7 | send | 4.00 | 0.00 | 0.00 | - |

## outputs

| output | purpose |
| --- | --- |
| [command_history.jsonl](command_history.jsonl) | exact driver history and environment |
| [manifest.json](manifest.json) | resolved configuration and generated training command |
| [metrics.jsonl](metrics.jsonl) | raw per-step metrics |
| [analysis.json](analysis.json) | structured derived analysis |
| [artifacts.json](artifacts.json) | raw run, compact artifact, report, and MindStudio paths |
| [tool_commands](tool_commands/) | official analysis commands and return codes |

- raw run: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced`
- compact artifact: `/workspace/y50064852_yyb/torchtitan-test/performance_artifacts/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced`
- html report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced.html`
- MindStudio input: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/trainer_output/profiling/traces`
- TensorBoard logdir: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/trainer_output/tensorboard`
- flame graph stacks/SVG root: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/trainer_output/profiling/traces`
- torch.compile visualization: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-distributed-r31-28affced/graph_visualization`

Profiler-off results are throughput evidence. Profiler-active results are used for attribution only and must not be compared as authoritative throughput.
