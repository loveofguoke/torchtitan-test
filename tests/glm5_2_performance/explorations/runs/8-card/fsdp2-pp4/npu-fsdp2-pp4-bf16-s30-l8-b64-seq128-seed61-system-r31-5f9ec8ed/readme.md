# npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed

## experiment

| field | value |
| --- | --- |
| status | completed |
| topology | fsdp2-pp4 |
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
| started | 2026-09-01T13:32:37.255259+08:00 |

## process

1. Run the `capture` driver:

   ```bash
   /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-pp4 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
   ```

2. Run the `analyze` driver:

   ```bash
   /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-pp4 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
   ```

3. The workflow generated and executed this training command:

   ```bash
   /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id=npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed --local-ranks-filter=6 --role=rank --tee=3 -m tests.glm5_2_performance.capture_metrics --module glm5 --config glm5_debugmodel --dump_folder=/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/trainer_output --training.steps=30 --training.num_tokens_per_microbatch_per_dp_rank=128 --training.num_tokens_per_train_step=8192 --training.max_context_length=128 --parallelism.num_pp_microbatches=8 --training.dtype=float32 --training.mixed_precision_param=bfloat16 --training.mixed_precision_reduce=float32 --debug.seed=61 --debug.no-enable-structured-logging --metrics.log_freq=1 --metrics.enable_tensorboard --metrics.disable_color_printing --metrics.save_tb_folder=tensorboard --profiler.enable_profiling --profiler.save_traces_folder=profiling/traces --profiler.profile_freq=7 --profiler.profiler_repeat=1 --profiler.profiler_skip_first=10 --profiler.profiler_warmup=2 --profiler.profiler_active=5 --parallelism.spmd_backend=partial_dtensor --parallelism.data_parallel_replicate_degree=1 --parallelism.data_parallel_shard_degree=2 --parallelism.context_parallel_degree=1 --parallelism.tensor_parallel_degree=1 --parallelism.pipeline_parallel_degree=4 --parallelism.expert_parallel_degree=1 --parallelism.pipeline_parallel_schedule=1F1B
   ```

4. Parse and analyze the capture with the official tools:

   - `advisor` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli advisor all -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/advisor
     ```

   - `cluster` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli cluster -m all -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/cluster
     ```

   - `cluster_time_summary` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m cluster_time_summary -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/cluster_time_summary --export_type text
     ```

   - `communication_bottleneck_rank_0` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/communication_bottleneck_rank_0 --rank_id 0 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_1` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/communication_bottleneck_rank_1 --rank_id 1 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_2` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/communication_bottleneck_rank_2 --rank_id 2 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_3` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/communication_bottleneck_rank_3 --rank_id 3 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_4` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/communication_bottleneck_rank_4 --rank_id 4 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_5` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/communication_bottleneck_rank_5 --rank_id 5 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_6` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/communication_bottleneck_rank_6 --rank_id 6 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_7` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/communication_bottleneck_rank_7 --rank_id 7 --top_num 20 --export_type text
     ```

   - `free_analysis` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m free_analysis -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/free_analysis --top_num 20 --export_type text
     ```

   - `run_state` (return code `None`)

     ```bash
     -
     ```

5. Produce compact evidence (`manifest.json`, `metrics.jsonl`, `analysis.json`, tool status files) and the HTML report when analysis completed.

## results

| metric | value |
| --- | ---: |
| median step | 3,248.10 ms |
| p90 step | - ms |
| throughput / device | 315.26 tok/s |
| throughput / job | 2,522.09 tok/s |
| peak active hbm | - GiB |

### collectives

| rank | kind | calls/step | payload mb/step | transit ms/step | effective gb/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | batchsendrecv | 21.60 | 1.31 | 0.08 | 16.00 |
| 0 | receive | 12.00 | 0.79 | 0.05 | 16.43 |
| 0 | send | 12.00 | 0.00 | 0.00 | - |
| 0 | reducescatter | 12.00 | 29.52 | 1.72 | 17.14 |
| 0 | allgather | 12.00 | 15.16 | 0.79 | 19.27 |
| 0 | allreduce | 8.00 | 0.00 | 0.00 | 0.01 |
| 1 | batchsendrecv | 21.60 | 1.31 | 0.08 | 16.04 |
| 1 | receive | 12.00 | 0.79 | 0.05 | 16.34 |
| 1 | send | 12.00 | 0.00 | 0.00 | - |
| 1 | reducescatter | 12.00 | 29.52 | 1.64 | 18.01 |
| 1 | allgather | 12.00 | 15.16 | 0.79 | 19.23 |
| 1 | allreduce | 8.00 | 0.00 | 0.01 | 0.03 |
| 2 | receive | 20.00 | 1.31 | 0.08 | 16.36 |
| 2 | batchsendrecv | 45.60 | 2.88 | 0.18 | 16.07 |
| 2 | allreduce | 8.00 | 0.00 | 0.01 | 0.01 |
| 2 | reducescatter | 12.00 | 49.81 | 2.94 | 16.97 |
| 2 | allgather | 12.00 | 25.51 | 1.32 | 19.33 |
| 2 | send | 20.00 | 0.00 | 0.00 | - |
| 3 | receive | 20.00 | 1.31 | 0.08 | 16.30 |
| 3 | batchsendrecv | 45.60 | 2.88 | 0.18 | 15.97 |
| 3 | reducescatter | 12.00 | 49.81 | 2.77 | 17.97 |
| 3 | allreduce | 8.00 | 0.00 | 0.02 | 0.04 |
| 3 | allgather | 12.00 | 25.51 | 1.32 | 19.32 |
| 3 | send | 20.00 | 0.00 | 0.00 | - |
| 4 | receive | 12.00 | 0.79 | 0.05 | 16.35 |
| 4 | batchsendrecv | 53.60 | 3.41 | 0.21 | 16.09 |
| 4 | allreduce | 8.00 | 0.00 | 0.01 | 0.01 |
| 4 | reducescatter | 8.00 | 33.21 | 1.97 | 16.88 |
| 4 | allgather | 8.00 | 17.01 | 0.88 | 19.32 |
| 4 | send | 12.00 | 0.00 | 0.00 | - |
| 5 | receive | 12.00 | 0.79 | 0.05 | 16.33 |
| 5 | batchsendrecv | 53.60 | 3.41 | 0.21 | 16.10 |
| 5 | allreduce | 8.00 | 0.00 | 0.01 | 0.03 |
| 5 | reducescatter | 8.00 | 33.21 | 1.86 | 17.88 |
| 5 | allgather | 8.00 | 17.01 | 0.88 | 19.31 |
| 5 | send | 12.00 | 0.00 | 0.00 | - |
| 6 | batchsendrecv | 29.60 | 1.84 | 0.11 | 15.97 |
| 6 | receive | 4.00 | 0.26 | 0.02 | 16.33 |
| 6 | allreduce | 9.00 | 0.00 | 0.01 | 0.01 |
| 6 | allgather | 72.00 | 44.17 | 2.32 | 19.03 |
| 6 | reducescatter | 40.00 | 50.17 | 2.81 | 17.86 |
| 6 | send | 4.00 | 0.00 | 0.00 | - |
| 7 | receive | 4.00 | 0.26 | 0.02 | 16.37 |
| 7 | batchsendrecv | 29.60 | 1.84 | 0.11 | 15.97 |
| 7 | allgather | 72.00 | 44.17 | 2.32 | 19.02 |
| 7 | reducescatter | 40.00 | 50.17 | 2.75 | 18.22 |
| 7 | allreduce | 9.00 | 0.00 | 0.02 | 0.02 |
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

- raw run: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed`
- compact artifact: `/workspace/y50064852_yyb/torchtitan-test/performance_artifacts/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed`
- html report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed.html`
- MindStudio input: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/trainer_output/profiling/traces`
- TensorBoard logdir: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/trainer_output/tensorboard`
- flame graph stacks/SVG root: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/trainer_output/profiling/traces`
- torch.compile visualization: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-pp4/npu-fsdp2-pp4-bf16-s30-l8-b64-seq128-seed61-system-r31-5f9ec8ed/graph_visualization`

Profiler-off results are throughput evidence. Profiler-active results are used for attribution only and must not be compared as authoritative throughput.
