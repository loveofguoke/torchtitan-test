# npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600

## experiment

| field | value |
| --- | --- |
| status | completed |
| topology | fsdp2-tp2-pp2 |
| world size | 8 |
| mode | profiler-active |
| device | npu |
| preset | system |
| steps | 30 |
| local / global batch | 8 / 64 |
| sequence length | 128 |
| parameter / reduction dtype | bfloat16 / float32 |
| replicate | 31 |
| visible devices | 0, 1, 2, 3, 4, 5, 6, 7 |
| started | 2026-09-01T13:35:20.581936+08:00 |

## process

1. Run the `capture` driver:

   ```bash
   /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-tp2-pp2 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
   ```

2. Run the `analyze` driver:

   ```bash
   /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-tp2-pp2 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
   ```

3. The workflow generated and executed this training command:

   ```bash
   /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id=npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600 --local-ranks-filter=4 --role=rank --tee=3 -m tests.glm5_2_performance.capture_metrics --module glm5 --config glm5_debugmodel --dump_folder=/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/trainer_output --training.steps=30 --training.num_tokens_per_microbatch_per_dp_rank=128 --training.num_tokens_per_train_step=8192 --training.max_context_length=128 --parallelism.num_pp_microbatches=8 --training.dtype=float32 --training.mixed_precision_param=bfloat16 --training.mixed_precision_reduce=float32 --debug.seed=61 --debug.no-enable-structured-logging --metrics.log_freq=1 --metrics.enable_tensorboard --metrics.disable_color_printing --metrics.save_tb_folder=tensorboard --profiler.enable_profiling --profiler.save_traces_folder=profiling/traces --profiler.profile_freq=7 --profiler.profiler_repeat=1 --profiler.profiler_skip_first=10 --profiler.profiler_warmup=2 --profiler.profiler_active=5 --parallelism.spmd_backend=partial_dtensor --parallelism.data_parallel_replicate_degree=1 --parallelism.data_parallel_shard_degree=2 --parallelism.context_parallel_degree=1 --parallelism.tensor_parallel_degree=2 --parallelism.pipeline_parallel_degree=2 --parallelism.expert_parallel_degree=1 --parallelism.no-enable-sequence-parallel --parallelism.pipeline_parallel_schedule=1F1B
   ```

4. Parse and analyze the capture with the official tools:

   - `advisor` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli advisor all -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/advisor
     ```

   - `cluster` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli cluster -m all -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/cluster
     ```

   - `cluster_time_summary` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m cluster_time_summary -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/cluster_time_summary --export_type text
     ```

   - `communication_bottleneck_rank_0` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/communication_bottleneck_rank_0 --rank_id 0 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_1` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/communication_bottleneck_rank_1 --rank_id 1 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_2` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/communication_bottleneck_rank_2 --rank_id 2 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_3` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/communication_bottleneck_rank_3 --rank_id 3 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_4` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/communication_bottleneck_rank_4 --rank_id 4 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_5` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/communication_bottleneck_rank_5 --rank_id 5 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_6` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/communication_bottleneck_rank_6 --rank_id 6 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_7` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/communication_bottleneck_rank_7 --rank_id 7 --top_num 20 --export_type text
     ```

   - `free_analysis` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m free_analysis -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/free_analysis --top_num 20 --export_type text
     ```

   - `run_state` (return code `None`)

     ```bash
     -
     ```

5. Produce compact evidence (`manifest.json`, `metrics.jsonl`, `analysis.json`, tool status files) and the HTML report when analysis completed.

## results

| metric | value |
| --- | ---: |
| median step | 13,187.91 ms |
| p90 step | - ms |
| throughput / device | 77.65 tok/s |
| throughput / job | 621.17 tok/s |
| peak active hbm | - GiB |

### collectives

| rank | kind | calls/step | payload mb/step | transit ms/step | effective gb/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | batchsendrecv | 29.60 | 0.92 | 0.07 | 12.85 |
| 0 | receive | 4.00 | 0.13 | 0.01 | 13.47 |
| 0 | reducescatter | 404.00 | 40.18 | 2.73 | 14.72 |
| 0 | allgather | 308.00 | 26.40 | 1.61 | 16.44 |
| 0 | allreduce | 617.00 | 0.00 | 0.00 | 0.01 |
| 0 | send | 4.00 | 0.00 | 0.00 | - |
| 1 | batchsendrecv | 29.60 | 0.92 | 0.07 | 12.92 |
| 1 | allreduce | 617.00 | 47.47 | 3.47 | 13.67 |
| 1 | receive | 4.00 | 0.13 | 0.01 | 13.50 |
| 1 | reducescatter | 404.00 | 40.18 | 2.74 | 14.66 |
| 1 | allgather | 308.00 | 26.40 | 1.60 | 16.47 |
| 1 | send | 4.00 | 0.00 | 0.00 | - |
| 2 | batchsendrecv | 29.60 | 0.92 | 0.07 | 12.92 |
| 2 | receive | 4.00 | 0.13 | 0.01 | 13.44 |
| 2 | allgather | 308.00 | 26.40 | 1.61 | 16.37 |
| 2 | allreduce | 617.00 | 0.00 | 0.01 | 0.06 |
| 2 | reducescatter | 404.00 | 40.18 | 2.64 | 15.25 |
| 2 | send | 4.00 | 0.00 | 0.00 | - |
| 3 | batchsendrecv | 29.60 | 0.92 | 0.07 | 12.81 |
| 3 | receive | 4.00 | 0.13 | 0.01 | 13.41 |
| 3 | allreduce | 617.00 | 47.47 | 3.47 | 13.67 |
| 3 | reducescatter | 404.00 | 40.18 | 2.65 | 15.19 |
| 3 | allgather | 308.00 | 26.40 | 1.61 | 16.41 |
| 3 | send | 4.00 | 0.00 | 0.00 | - |
| 4 | receive | 4.00 | 0.26 | 0.02 | 16.33 |
| 4 | allreduce | 1,514.00 | 0.00 | 0.01 | 0.01 |
| 4 | reducescatter | 436.00 | 58.82 | 3.79 | 15.54 |
| 4 | allgather | 276.00 | 42.02 | 2.38 | 17.65 |
| 4 | batchsendrecv | 29.60 | 1.84 | 0.11 | 16.16 |
| 4 | send | 4.00 | 0.00 | 0.00 | - |
| 5 | allreduce | 1,514.00 | 56.09 | 5.71 | 9.83 |
| 5 | receive | 4.00 | 0.26 | 0.02 | 16.41 |
| 5 | reducescatter | 436.00 | 58.82 | 3.78 | 15.55 |
| 5 | allgather | 276.00 | 42.02 | 2.38 | 17.66 |
| 5 | batchsendrecv | 29.60 | 1.84 | 0.11 | 16.17 |
| 5 | send | 4.00 | 0.00 | 0.00 | - |
| 6 | allreduce | 1,514.00 | 0.00 | 0.02 | 0.05 |
| 6 | reducescatter | 436.00 | 58.82 | 3.68 | 15.97 |
| 6 | receive | 4.00 | 0.26 | 0.02 | 16.36 |
| 6 | allgather | 276.00 | 42.02 | 2.38 | 17.67 |
| 6 | batchsendrecv | 29.60 | 1.84 | 0.11 | 16.19 |
| 6 | send | 4.00 | 0.00 | 0.00 | - |
| 7 | receive | 4.00 | 0.26 | 0.02 | 16.36 |
| 7 | allreduce | 1,514.00 | 56.09 | 5.73 | 9.79 |
| 7 | reducescatter | 436.00 | 58.82 | 3.67 | 16.03 |
| 7 | allgather | 276.00 | 42.02 | 2.38 | 17.68 |
| 7 | batchsendrecv | 29.60 | 1.84 | 0.11 | 16.16 |
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

- raw run: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600`
- compact artifact: `/workspace/y50064852_yyb/torchtitan-test/performance_artifacts/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600`
- html report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600.html`
- MindStudio input: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/trainer_output/profiling/traces`
- TensorBoard logdir: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/trainer_output/tensorboard`
- flame graph stacks/SVG root: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/trainer_output/profiling/traces`
- torch.compile visualization: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp2-pp2/npu-fsdp2-tp2-pp2-bf16-s30-l8-b64-seq128-seed61-system-r31-0d2fd600/graph_visualization`

Profiler-off results are throughput evidence. Profiler-active results are used for attribution only and must not be compared as authoritative throughput.
