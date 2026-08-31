# npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e

## experiment

| field | value |
| --- | --- |
| status | completed |
| topology | fsdp2-cp4 |
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
| started | 2026-08-31T14:44:05.958361+08:00 |

## process

1. Run the `probe` driver:

   ```bash
   /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology all --preset all --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
   ```

2. The workflow generated and executed this training command:

   ```bash
   /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id=npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e --local-ranks-filter=0 --role=rank --tee=3 -m tests.glm5_2_performance.capture_metrics --module glm5 --config glm5_debugmodel --dump_folder=/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/trainer_output --training.steps=30 --training.num_tokens_per_microbatch_per_dp_rank=1024 --training.num_tokens_per_train_step=8192 --training.max_context_length=128 --training.dtype=float32 --training.mixed_precision_param=bfloat16 --training.mixed_precision_reduce=float32 --debug.seed=61 --debug.no-enable-structured-logging --metrics.log_freq=1 --metrics.enable_tensorboard --metrics.disable_color_printing --metrics.save_tb_folder=tensorboard --profiler.enable_profiling --profiler.save_traces_folder=profiling/traces --profiler.profile_freq=7 --profiler.profiler_repeat=1 --profiler.profiler_skip_first=10 --profiler.profiler_warmup=2 --profiler.profiler_active=5 --parallelism.spmd_backend=partial_dtensor --parallelism.data_parallel_replicate_degree=1 --parallelism.data_parallel_shard_degree=2 --parallelism.context_parallel_degree=4 --parallelism.tensor_parallel_degree=1 --parallelism.pipeline_parallel_degree=1 --parallelism.expert_parallel_degree=1
   ```

3. Parse and analyze the capture with the official tools:

   - `advisor` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli advisor all -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/advisor
     ```

   - `cluster` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli cluster -m all -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/cluster
     ```

   - `cluster_time_summary` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m cluster_time_summary -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/cluster_time_summary --export_type text
     ```

   - `communication_bottleneck_rank_0` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/communication_bottleneck_rank_0 --rank_id 0 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_1` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/communication_bottleneck_rank_1 --rank_id 1 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_2` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/communication_bottleneck_rank_2 --rank_id 2 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_3` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/communication_bottleneck_rank_3 --rank_id 3 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_4` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/communication_bottleneck_rank_4 --rank_id 4 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_5` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/communication_bottleneck_rank_5 --rank_id 5 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_6` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/communication_bottleneck_rank_6 --rank_id 6 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_7` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/communication_bottleneck_rank_7 --rank_id 7 --top_num 20 --export_type text
     ```

   - `free_analysis` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m free_analysis -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/free_analysis --top_num 20 --export_type text
     ```

   - `run_state` (return code `None`)

     ```bash
     -
     ```

4. Produce compact evidence (`manifest.json`, `metrics.jsonl`, `analysis.json`, tool status files) and the HTML report when analysis completed.

## results

| metric | value |
| --- | ---: |
| median step | 1,380.72 ms |
| p90 step | - ms |
| throughput / device | 741.64 tok/s |
| throughput / job | 5,933.13 tok/s |
| peak active hbm | - GiB |

### collectives

| rank | kind | calls/step | payload mb/step | transit ms/step | effective gb/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | reducescatter | 108.00 | 283.66 | 67.92 | 4.18 |
| 0 | allgather | 184.00 | 296.25 | 21.68 | 13.67 |
| 0 | allreduce | 7.00 | 0.00 | 0.01 | 0.01 |
| 1 | reducescatter | 108.00 | 283.66 | 46.92 | 6.05 |
| 1 | allgather | 184.00 | 296.25 | 21.07 | 14.06 |
| 1 | allreduce | 7.00 | 0.00 | 0.02 | 0.10 |
| 2 | reducescatter | 108.00 | 283.66 | 71.60 | 3.96 |
| 2 | allgather | 184.00 | 296.25 | 21.61 | 13.71 |
| 2 | allreduce | 7.00 | 0.00 | 0.02 | 0.10 |
| 3 | reducescatter | 108.00 | 283.66 | 60.14 | 4.72 |
| 3 | allgather | 184.00 | 296.25 | 20.72 | 14.30 |
| 3 | allreduce | 7.00 | 0.00 | 0.02 | 0.09 |
| 4 | reducescatter | 108.00 | 283.66 | 65.99 | 4.30 |
| 4 | allgather | 184.00 | 296.25 | 21.21 | 13.96 |
| 4 | allreduce | 7.00 | 0.00 | 0.02 | 0.09 |
| 5 | reducescatter | 108.00 | 283.66 | 68.76 | 4.13 |
| 5 | allgather | 184.00 | 296.25 | 21.32 | 13.89 |
| 5 | allreduce | 7.00 | 0.00 | 0.02 | 0.10 |
| 6 | reducescatter | 108.00 | 283.66 | 72.96 | 3.89 |
| 6 | allgather | 184.00 | 296.25 | 21.64 | 13.69 |
| 6 | allreduce | 7.00 | 0.00 | 0.02 | 0.09 |
| 7 | reducescatter | 108.00 | 283.66 | 70.10 | 4.05 |
| 7 | allgather | 184.00 | 296.25 | 21.26 | 13.93 |
| 7 | allreduce | 7.00 | 0.00 | 0.02 | 0.10 |

## outputs

| output | purpose |
| --- | --- |
| [command_history.jsonl](command_history.jsonl) | exact driver history and environment |
| [manifest.json](manifest.json) | resolved configuration and generated training command |
| [metrics.jsonl](metrics.jsonl) | raw per-step metrics |
| [analysis.json](analysis.json) | structured derived analysis |
| [artifacts.json](artifacts.json) | raw run, compact artifact, report, and MindStudio paths |
| [tool_commands](tool_commands/) | official analysis commands and return codes |

- raw run: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e`
- compact artifact: `/workspace/y50064852_yyb/torchtitan-test/performance_artifacts/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e`
- html report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e.html`
- MindStudio input: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/trainer_output/profiling/traces`
- TensorBoard logdir: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/trainer_output/tensorboard`
- flame graph stacks/SVG root: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/trainer_output/profiling/traces`
- torch.compile visualization: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-7929572e/graph_visualization`

Profiler-off results are throughput evidence. Profiler-active results are used for attribution only and must not be compared as authoritative throughput.
