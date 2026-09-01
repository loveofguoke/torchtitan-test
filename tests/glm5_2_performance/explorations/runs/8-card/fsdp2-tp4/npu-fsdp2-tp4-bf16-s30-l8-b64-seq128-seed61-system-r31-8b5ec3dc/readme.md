# npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc

## experiment

| field | value |
| --- | --- |
| status | completed |
| topology | fsdp2-tp4 |
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
| started | 2026-09-01T13:22:33.589156+08:00 |

## process

1. Run the `capture` driver:

   ```bash
   /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp2-tp4 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools none
   ```

2. Run the `analyze` driver:

   ```bash
   /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp2-tp4 --preset system --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all --parse-workers 8
   ```

3. The workflow generated and executed this training command:

   ```bash
   /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id=npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc --local-ranks-filter=0 --role=rank --tee=3 -m tests.glm5_2_performance.capture_metrics --module glm5 --config glm5_debugmodel --dump_folder=/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/trainer_output --training.steps=30 --training.num_tokens_per_microbatch_per_dp_rank=1024 --training.num_tokens_per_train_step=8192 --training.max_context_length=128 --training.dtype=float32 --training.mixed_precision_param=bfloat16 --training.mixed_precision_reduce=float32 --debug.seed=61 --debug.no-enable-structured-logging --metrics.log_freq=1 --metrics.enable_tensorboard --metrics.disable_color_printing --metrics.save_tb_folder=tensorboard --profiler.enable_profiling --profiler.save_traces_folder=profiling/traces --profiler.profile_freq=7 --profiler.profiler_repeat=1 --profiler.profiler_skip_first=10 --profiler.profiler_warmup=2 --profiler.profiler_active=5 --parallelism.spmd_backend=partial_dtensor --parallelism.data_parallel_replicate_degree=1 --parallelism.data_parallel_shard_degree=2 --parallelism.context_parallel_degree=1 --parallelism.tensor_parallel_degree=4 --parallelism.pipeline_parallel_degree=1 --parallelism.expert_parallel_degree=1 --parallelism.no-enable-sequence-parallel
   ```

4. Parse and analyze the capture with the official tools:

   - `advisor` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli advisor all -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/advisor
     ```

   - `cluster` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli cluster -m all -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/cluster
     ```

   - `cluster_time_summary` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m cluster_time_summary -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/cluster_time_summary --export_type text
     ```

   - `communication_bottleneck_rank_0` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/communication_bottleneck_rank_0 --rank_id 0 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_1` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/communication_bottleneck_rank_1 --rank_id 1 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_2` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/communication_bottleneck_rank_2 --rank_id 2 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_3` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/communication_bottleneck_rank_3 --rank_id 3 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_4` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/communication_bottleneck_rank_4 --rank_id 4 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_5` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/communication_bottleneck_rank_5 --rank_id 5 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_6` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/communication_bottleneck_rank_6 --rank_id 6 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_7` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/communication_bottleneck_rank_7 --rank_id 7 --top_num 20 --export_type text
     ```

   - `free_analysis` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m free_analysis -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/free_analysis --top_num 20 --export_type text
     ```

   - `run_state` (return code `None`)

     ```bash
     -
     ```

5. Produce compact evidence (`manifest.json`, `metrics.jsonl`, `analysis.json`, tool status files) and the HTML report when analysis completed.

## results

| metric | value |
| --- | ---: |
| median step | 3,017.31 ms |
| p90 step | - ms |
| throughput / device | 339.38 tok/s |
| throughput / job | 2,715.00 tok/s |
| peak active hbm | - GiB |

### collectives

| rank | kind | calls/step | payload mb/step | transit ms/step | effective gb/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | allreduce | 440.00 | 62.91 | 4.01 | 15.70 |
| 0 | reducescatter | 140.00 | 59.80 | 3.77 | 15.85 |
| 0 | allgather | 144.00 | 63.54 | 3.52 | 18.04 |
| 1 | allreduce | 440.00 | 75.10 | 5.36 | 14.00 |
| 1 | reducescatter | 140.00 | 59.80 | 3.84 | 15.59 |
| 1 | allgather | 144.00 | 63.54 | 3.49 | 18.18 |
| 2 | allreduce | 440.00 | 75.10 | 5.22 | 14.39 |
| 2 | reducescatter | 140.00 | 59.80 | 3.79 | 15.78 |
| 2 | allgather | 144.00 | 63.54 | 3.49 | 18.19 |
| 3 | allreduce | 440.00 | 75.10 | 5.18 | 14.49 |
| 3 | reducescatter | 140.00 | 59.80 | 3.77 | 15.86 |
| 3 | allgather | 144.00 | 63.54 | 3.51 | 18.12 |
| 4 | allreduce | 440.00 | 62.92 | 4.01 | 15.69 |
| 4 | reducescatter | 140.00 | 59.80 | 3.71 | 16.12 |
| 4 | allgather | 144.00 | 63.54 | 3.50 | 18.16 |
| 5 | allreduce | 440.00 | 75.10 | 5.33 | 14.10 |
| 5 | reducescatter | 140.00 | 59.80 | 3.69 | 16.19 |
| 5 | allgather | 144.00 | 63.54 | 3.50 | 18.14 |
| 6 | allreduce | 440.00 | 75.10 | 5.27 | 14.24 |
| 6 | reducescatter | 140.00 | 59.80 | 3.69 | 16.21 |
| 6 | allgather | 144.00 | 63.54 | 3.50 | 18.15 |
| 7 | allreduce | 440.00 | 75.10 | 5.57 | 13.49 |
| 7 | reducescatter | 140.00 | 59.80 | 3.91 | 15.31 |
| 7 | allgather | 144.00 | 63.54 | 3.50 | 18.15 |

## outputs

| output | purpose |
| --- | --- |
| [command_history.jsonl](command_history.jsonl) | exact driver history and environment |
| [manifest.json](manifest.json) | resolved configuration and generated training command |
| [metrics.jsonl](metrics.jsonl) | raw per-step metrics |
| [analysis.json](analysis.json) | structured derived analysis |
| [artifacts.json](artifacts.json) | raw run, compact artifact, report, and MindStudio paths |
| [tool_commands](tool_commands/) | official analysis commands and return codes |

- raw run: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc`
- compact artifact: `/workspace/y50064852_yyb/torchtitan-test/performance_artifacts/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc`
- html report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc.html`
- MindStudio input: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/trainer_output/profiling/traces`
- TensorBoard logdir: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/trainer_output/tensorboard`
- flame graph stacks/SVG root: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/trainer_output/profiling/traces`
- torch.compile visualization: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s30-l8-b64-seq128-seed61-system-r31-8b5ec3dc/graph_visualization`

Profiler-off results are throughput evidence. Profiler-active results are used for attribution only and must not be compared as authoritative throughput.
