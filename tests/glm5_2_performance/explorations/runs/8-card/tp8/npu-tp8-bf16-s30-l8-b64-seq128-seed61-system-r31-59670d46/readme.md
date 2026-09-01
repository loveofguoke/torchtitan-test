# npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46

## experiment

| field | value |
| --- | --- |
| status | completed |
| topology | tp8 |
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
| started | 2026-09-01T13:11:57.266861+08:00 |

## process

1. Run the `capture` driver:

   ```bash
   /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology tp8 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
   ```

2. Run the `analyze` driver:

   ```bash
   /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology tp8 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
   ```

3. The workflow generated and executed this training command:

   ```bash
   /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id=npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46 --local-ranks-filter=0 --role=rank --tee=3 -m tests.glm5_2_performance.capture_metrics --module glm5 --config glm5_debugmodel --dump_folder=/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/trainer_output --training.steps=30 --training.num_tokens_per_microbatch_per_dp_rank=1024 --training.num_tokens_per_train_step=8192 --training.max_context_length=128 --training.dtype=float32 --training.mixed_precision_param=bfloat16 --training.mixed_precision_reduce=float32 --debug.seed=61 --debug.no-enable-structured-logging --metrics.log_freq=1 --metrics.enable_tensorboard --metrics.disable_color_printing --metrics.save_tb_folder=tensorboard --profiler.enable_profiling --profiler.save_traces_folder=profiling/traces --profiler.profile_freq=7 --profiler.profiler_repeat=1 --profiler.profiler_skip_first=10 --profiler.profiler_warmup=2 --profiler.profiler_active=5 --parallelism.spmd_backend=partial_dtensor --parallelism.data_parallel_replicate_degree=1 --parallelism.data_parallel_shard_degree=1 --parallelism.context_parallel_degree=1 --parallelism.tensor_parallel_degree=8 --parallelism.pipeline_parallel_degree=1 --parallelism.expert_parallel_degree=1 --parallelism.no-enable-sequence-parallel
   ```

4. Parse and analyze the capture with the official tools:

   - `advisor` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli advisor all -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/advisor
     ```

   - `cluster` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli cluster -m all -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/cluster
     ```

   - `cluster_time_summary` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m cluster_time_summary -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/cluster_time_summary --export_type text
     ```

   - `communication_bottleneck_rank_0` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/communication_bottleneck_rank_0 --rank_id 0 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_1` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/communication_bottleneck_rank_1 --rank_id 1 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_2` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/communication_bottleneck_rank_2 --rank_id 2 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_3` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/communication_bottleneck_rank_3 --rank_id 3 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_4` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/communication_bottleneck_rank_4 --rank_id 4 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_5` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/communication_bottleneck_rank_5 --rank_id 5 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_6` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/communication_bottleneck_rank_6 --rank_id 6 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_7` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/communication_bottleneck_rank_7 --rank_id 7 --top_num 20 --export_type text
     ```

   - `free_analysis` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m free_analysis -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/free_analysis --top_num 20 --export_type text
     ```

   - `run_state` (return code `None`)

     ```bash
     -
     ```

5. Produce compact evidence (`manifest.json`, `metrics.jsonl`, `analysis.json`, tool status files) and the HTML report when analysis completed.

## results

| metric | value |
| --- | ---: |
| median step | 6,100.42 ms |
| p90 step | - ms |
| throughput / device | 167.86 tok/s |
| throughput / job | 1,342.86 tok/s |
| peak active hbm | - GiB |

### collectives

| rank | kind | calls/step | payload mb/step | transit ms/step | effective gb/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | allreduce | 865.00 | 146.80 | 10.83 | 13.56 |
| 0 | reducescatter | 192.00 | 55.05 | 4.10 | 13.44 |
| 0 | allgather | 120.00 | 55.05 | 3.42 | 16.09 |
| 1 | allreduce | 865.00 | 171.17 | 13.73 | 12.47 |
| 1 | reducescatter | 192.00 | 55.05 | 4.28 | 12.85 |
| 1 | allgather | 120.00 | 55.05 | 3.54 | 15.55 |
| 2 | allreduce | 865.00 | 171.17 | 14.10 | 12.14 |
| 2 | reducescatter | 192.00 | 55.05 | 4.31 | 12.76 |
| 2 | allgather | 120.00 | 55.05 | 3.58 | 15.38 |
| 3 | allreduce | 865.00 | 171.17 | 13.92 | 12.30 |
| 3 | reducescatter | 192.00 | 55.05 | 4.23 | 13.02 |
| 3 | allgather | 120.00 | 55.05 | 3.59 | 15.32 |
| 4 | allreduce | 865.00 | 171.17 | 13.90 | 12.31 |
| 4 | reducescatter | 192.00 | 55.05 | 4.23 | 13.03 |
| 4 | allgather | 120.00 | 55.05 | 3.57 | 15.42 |
| 5 | allreduce | 865.00 | 171.17 | 13.86 | 12.35 |
| 5 | reducescatter | 192.00 | 55.05 | 4.19 | 13.13 |
| 5 | allgather | 120.00 | 55.05 | 3.54 | 15.56 |
| 6 | allreduce | 865.00 | 171.17 | 13.85 | 12.36 |
| 6 | reducescatter | 192.00 | 55.05 | 4.19 | 13.13 |
| 6 | allgather | 120.00 | 55.05 | 3.59 | 15.34 |
| 7 | allreduce | 865.00 | 171.17 | 13.66 | 12.53 |
| 7 | reducescatter | 192.00 | 55.05 | 4.10 | 13.44 |
| 7 | allgather | 120.00 | 55.05 | 3.58 | 15.39 |

## outputs

| output | purpose |
| --- | --- |
| [command_history.jsonl](command_history.jsonl) | exact driver history and environment |
| [manifest.json](manifest.json) | resolved configuration and generated training command |
| [metrics.jsonl](metrics.jsonl) | raw per-step metrics |
| [analysis.json](analysis.json) | structured derived analysis |
| [artifacts.json](artifacts.json) | raw run, compact artifact, report, and MindStudio paths |
| [tool_commands](tool_commands/) | official analysis commands and return codes |

- raw run: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46`
- compact artifact: `/workspace/y50064852_yyb/torchtitan-test/performance_artifacts/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46`
- html report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46.html`
- MindStudio input: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/trainer_output/profiling/traces`
- TensorBoard logdir: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/trainer_output/tensorboard`
- flame graph stacks/SVG root: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/trainer_output/profiling/traces`
- torch.compile visualization: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s30-l8-b64-seq128-seed61-system-r31-59670d46/graph_visualization`

Profiler-off results are throughput evidence. Profiler-active results are used for attribution only and must not be compared as authoritative throughput.
