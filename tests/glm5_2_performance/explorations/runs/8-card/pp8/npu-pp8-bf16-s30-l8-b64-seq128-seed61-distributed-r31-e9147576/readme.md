# npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576

## experiment

| field | value |
| --- | --- |
| status | completed |
| topology | pp8 |
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
| started | 2026-08-31T14:10:38.597515+08:00 |

## process

1. Run the `probe` driver:

   ```bash
   /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
   ```

2. The workflow generated and executed this training command:

   ```bash
   /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id=npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576 --local-ranks-filter=7 --role=rank --tee=3 -m tests.glm5_2_performance.capture_metrics --module glm5 --config glm5_debugmodel --dump_folder=/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/trainer_output --training.steps=30 --training.num_tokens_per_microbatch_per_dp_rank=128 --training.num_tokens_per_train_step=8192 --training.max_context_length=128 --parallelism.num_pp_microbatches=8 --training.dtype=float32 --training.mixed_precision_param=bfloat16 --training.mixed_precision_reduce=float32 --debug.seed=61 --debug.no-enable-structured-logging --metrics.log_freq=1 --metrics.enable_tensorboard --metrics.disable_color_printing --metrics.save_tb_folder=tensorboard --profiler.enable_profiling --profiler.save_traces_folder=profiling/traces --profiler.profile_freq=7 --profiler.profiler_repeat=1 --profiler.profiler_skip_first=10 --profiler.profiler_warmup=2 --profiler.profiler_active=5 --parallelism.spmd_backend=partial_dtensor --parallelism.data_parallel_replicate_degree=1 --parallelism.data_parallel_shard_degree=1 --parallelism.context_parallel_degree=1 --parallelism.tensor_parallel_degree=1 --parallelism.pipeline_parallel_degree=8 --parallelism.expert_parallel_degree=1 --parallelism.pipeline_parallel_schedule=1F1B
   ```

3. Parse and analyze the capture with the official tools:

   - `advisor` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli advisor all -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/advisor
     ```

   - `cluster` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli cluster -m all -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/cluster
     ```

   - `cluster_time_summary` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m cluster_time_summary -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/cluster_time_summary --export_type text
     ```

   - `communication_bottleneck_rank_0` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/communication_bottleneck_rank_0 --rank_id 0 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_1` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/communication_bottleneck_rank_1 --rank_id 1 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_2` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/communication_bottleneck_rank_2 --rank_id 2 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_3` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/communication_bottleneck_rank_3 --rank_id 3 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_4` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/communication_bottleneck_rank_4 --rank_id 4 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_5` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/communication_bottleneck_rank_5 --rank_id 5 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_6` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/communication_bottleneck_rank_6 --rank_id 6 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_7` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/communication_bottleneck_rank_7 --rank_id 7 --top_num 20 --export_type text
     ```

   - `free_analysis` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m free_analysis -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/free_analysis --top_num 20 --export_type text
     ```

   - `run_state` (return code `None`)

     ```bash
     -
     ```

4. Produce compact evidence (`manifest.json`, `metrics.jsonl`, `analysis.json`, tool status files) and the HTML report when analysis completed.

## results

| metric | value |
| --- | ---: |
| median step | 3,982.32 ms |
| p90 step | - ms |
| throughput / device | 257.14 tok/s |
| throughput / job | 2,057.09 tok/s |
| peak active hbm | - GiB |

### collectives

| rank | kind | calls/step | payload mb/step | transit ms/step | effective gb/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | receive | 56.00 | 3.67 | 0.22 | 16.38 |
| 0 | send | 56.00 | 0.00 | 0.00 | - |
| 0 | batchsendrecv | 9.60 | 0.52 | 0.03 | 15.98 |
| 0 | allreduce | 2.00 | 0.00 | 0.00 | - |
| 1 | receive | 104.00 | 6.82 | 0.42 | 16.39 |
| 1 | batchsendrecv | 25.60 | 1.57 | 0.10 | 16.05 |
| 1 | send | 104.00 | 0.00 | 0.00 | - |
| 1 | allreduce | 2.00 | 0.00 | 0.00 | 0.00 |
| 2 | receive | 88.00 | 5.77 | 0.35 | 16.40 |
| 2 | batchsendrecv | 41.60 | 2.62 | 0.16 | 16.06 |
| 2 | send | 88.00 | 0.00 | 0.00 | - |
| 2 | allreduce | 2.00 | 0.00 | 0.00 | 0.00 |
| 3 | receive | 72.00 | 4.72 | 0.29 | 16.41 |
| 3 | batchsendrecv | 57.60 | 3.67 | 0.23 | 15.98 |
| 3 | allreduce | 2.00 | 0.00 | 0.00 | 0.00 |
| 3 | send | 72.00 | 0.00 | 0.00 | - |
| 4 | receive | 56.00 | 3.67 | 0.22 | 16.37 |
| 4 | batchsendrecv | 73.60 | 4.72 | 0.29 | 16.07 |
| 4 | allreduce | 2.00 | 0.00 | 0.00 | 0.00 |
| 4 | send | 56.00 | 0.00 | 0.00 | - |
| 5 | receive | 40.00 | 2.62 | 0.16 | 16.40 |
| 5 | batchsendrecv | 89.60 | 5.77 | 0.36 | 16.04 |
| 5 | allreduce | 2.00 | 0.00 | 0.00 | 0.00 |
| 5 | send | 40.00 | 0.00 | 0.00 | - |
| 6 | receive | 24.00 | 1.57 | 0.10 | 16.41 |
| 6 | batchsendrecv | 105.60 | 6.82 | 0.42 | 16.05 |
| 6 | allreduce | 2.00 | 0.00 | 0.00 | 0.00 |
| 6 | send | 24.00 | 0.00 | 0.00 | - |
| 7 | batchsendrecv | 57.60 | 3.67 | 0.23 | 16.13 |
| 7 | receive | 8.00 | 0.52 | 0.03 | 16.36 |
| 7 | allreduce | 2.00 | 0.00 | 0.00 | 0.00 |
| 7 | send | 8.00 | 0.00 | 0.00 | - |

## outputs

| output | purpose |
| --- | --- |
| [command_history.jsonl](command_history.jsonl) | exact driver history and environment |
| [manifest.json](manifest.json) | resolved configuration and generated training command |
| [metrics.jsonl](metrics.jsonl) | raw per-step metrics |
| [analysis.json](analysis.json) | structured derived analysis |
| [artifacts.json](artifacts.json) | raw run, compact artifact, report, and MindStudio paths |
| [tool_commands](tool_commands/) | official analysis commands and return codes |

- raw run: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576`
- compact artifact: `/workspace/y50064852_yyb/torchtitan-test/performance_artifacts/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576`
- html report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576.html`
- MindStudio input: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/trainer_output/profiling/traces`
- TensorBoard logdir: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/trainer_output/tensorboard`
- flame graph stacks/SVG root: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/trainer_output/profiling/traces`
- torch.compile visualization: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/pp8/npu-pp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e9147576/graph_visualization`

Profiler-off results are throughput evidence. Profiler-active results are used for attribution only and must not be compared as authoritative throughput.
