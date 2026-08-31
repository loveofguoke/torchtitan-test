# npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4

## experiment

| field | value |
| --- | --- |
| status | completed |
| topology | cp8 |
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
| started | 2026-08-31T13:50:13.747507+08:00 |

## process

1. Run the `probe` driver:

   ```bash
   /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
   ```

2. The workflow generated and executed this training command:

   ```bash
   /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id=npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4 --local-ranks-filter=0 --role=rank --tee=3 -m tests.glm5_2_performance.capture_metrics --module glm5 --config glm5_debugmodel --dump_folder=/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/trainer_output --training.steps=30 --training.num_tokens_per_microbatch_per_dp_rank=1024 --training.num_tokens_per_train_step=8192 --training.max_context_length=128 --training.dtype=float32 --training.mixed_precision_param=bfloat16 --training.mixed_precision_reduce=float32 --debug.seed=61 --debug.no-enable-structured-logging --metrics.log_freq=1 --metrics.enable_tensorboard --metrics.disable_color_printing --metrics.save_tb_folder=tensorboard --profiler.enable_profiling --profiler.save_traces_folder=profiling/traces --profiler.profile_freq=7 --profiler.profiler_repeat=1 --profiler.profiler_skip_first=10 --profiler.profiler_warmup=2 --profiler.profiler_active=5 --parallelism.spmd_backend=partial_dtensor --parallelism.data_parallel_replicate_degree=1 --parallelism.data_parallel_shard_degree=1 --parallelism.context_parallel_degree=8 --parallelism.tensor_parallel_degree=1 --parallelism.pipeline_parallel_degree=1 --parallelism.expert_parallel_degree=1
   ```

3. Parse and analyze the capture with the official tools:

   - `advisor` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli advisor all -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/advisor
     ```

   - `cluster` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli cluster -m all -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/cluster
     ```

   - `cluster_time_summary` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m cluster_time_summary -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/cluster_time_summary --export_type text
     ```

   - `communication_bottleneck_rank_0` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/communication_bottleneck_rank_0 --rank_id 0 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_1` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/communication_bottleneck_rank_1 --rank_id 1 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_2` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/communication_bottleneck_rank_2 --rank_id 2 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_3` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/communication_bottleneck_rank_3 --rank_id 3 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_4` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/communication_bottleneck_rank_4 --rank_id 4 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_5` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/communication_bottleneck_rank_5 --rank_id 5 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_6` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/communication_bottleneck_rank_6 --rank_id 6 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_7` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/communication_bottleneck_rank_7 --rank_id 7 --top_num 20 --export_type text
     ```

   - `free_analysis` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m free_analysis -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/free_analysis --top_num 20 --export_type text
     ```

   - `run_state` (return code `None`)

     ```bash
     -
     ```

4. Produce compact evidence (`manifest.json`, `metrics.jsonl`, `analysis.json`, tool status files) and the HTML report when analysis completed.

## results

| metric | value |
| --- | ---: |
| median step | 2,774.08 ms |
| p90 step | - ms |
| throughput / device | 369.13 tok/s |
| throughput / job | 2,953.06 tok/s |
| peak active hbm | - GiB |

### collectives

| rank | kind | calls/step | payload mb/step | transit ms/step | effective gb/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | allgather | 368.00 | 610.34 | 44.05 | 13.85 |
| 0 | reducescatter | 216.00 | 584.10 | 138.70 | 4.21 |
| 0 | allreduce | 6.00 | 0.00 | 0.05 | 0.00 |
| 1 | allgather | 368.00 | 610.34 | 44.03 | 13.86 |
| 1 | reducescatter | 216.00 | 584.10 | 110.96 | 5.26 |
| 1 | allreduce | 6.00 | 0.00 | 0.01 | 0.11 |
| 2 | allgather | 368.00 | 610.34 | 43.92 | 13.90 |
| 2 | reducescatter | 216.00 | 584.10 | 131.24 | 4.45 |
| 2 | allreduce | 6.00 | 0.00 | 0.01 | 0.10 |
| 3 | allgather | 368.00 | 610.34 | 42.58 | 14.33 |
| 3 | reducescatter | 216.00 | 584.10 | 121.70 | 4.80 |
| 3 | allreduce | 6.00 | 0.00 | 0.01 | 0.10 |
| 4 | reducescatter | 216.00 | 584.10 | 127.71 | 4.57 |
| 4 | allgather | 368.00 | 610.34 | 43.20 | 14.13 |
| 4 | allreduce | 6.00 | 0.00 | 0.01 | 0.10 |
| 5 | reducescatter | 216.00 | 584.10 | 130.26 | 4.48 |
| 5 | allgather | 368.00 | 610.34 | 43.18 | 14.13 |
| 5 | allreduce | 6.00 | 0.00 | 0.01 | 0.10 |
| 6 | reducescatter | 216.00 | 584.10 | 140.33 | 4.16 |
| 6 | allgather | 368.00 | 610.34 | 44.17 | 13.82 |
| 6 | allreduce | 6.00 | 0.00 | 0.01 | 0.10 |
| 7 | allgather | 368.00 | 610.34 | 42.73 | 14.28 |
| 7 | reducescatter | 216.00 | 584.10 | 121.36 | 4.81 |
| 7 | allreduce | 6.00 | 0.00 | 0.01 | 0.10 |

## outputs

| output | purpose |
| --- | --- |
| [command_history.jsonl](command_history.jsonl) | exact driver history and environment |
| [manifest.json](manifest.json) | resolved configuration and generated training command |
| [metrics.jsonl](metrics.jsonl) | raw per-step metrics |
| [analysis.json](analysis.json) | structured derived analysis |
| [artifacts.json](artifacts.json) | raw run, compact artifact, report, and MindStudio paths |
| [tool_commands](tool_commands/) | official analysis commands and return codes |

- raw run: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4`
- compact artifact: `/workspace/y50064852_yyb/torchtitan-test/performance_artifacts/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4`
- html report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4.html`
- MindStudio input: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/trainer_output/profiling/traces`
- TensorBoard logdir: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/trainer_output/tensorboard`
- flame graph stacks/SVG root: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/trainer_output/profiling/traces`
- torch.compile visualization: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/cp8/npu-cp8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-ba7e0fe4/graph_visualization`

Profiler-off results are throughput evidence. Profiler-active results are used for attribution only and must not be compared as authoritative throughput.
