# npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7

## experiment

| field | value |
| --- | --- |
| status | completed |
| topology | fsdp2-tp4-ep8 |
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
| started | 2026-08-31T16:50:46.742996+08:00 |

## process

1. Run the `probe` driver:

   ```bash
   /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topologies fsdp2-tp4-ep8,ddp4,fsdp2,fsdp4,tp2,tp4,cp2,cp4,pp2,pp4,ep2,ep4,fsdp2-tp2 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
   ```

2. The workflow generated and executed this training command:

   ```bash
   /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id=npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7 --local-ranks-filter=0 --role=rank --tee=3 -m tests.glm5_2_performance.capture_metrics --module glm5 --config glm5_debugmodel --dump_folder=/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/trainer_output --training.steps=30 --training.num_tokens_per_microbatch_per_dp_rank=1024 --training.num_tokens_per_train_step=8192 --training.max_context_length=128 --training.dtype=float32 --training.mixed_precision_param=bfloat16 --training.mixed_precision_reduce=float32 --debug.seed=61 --debug.no-enable-structured-logging --metrics.log_freq=1 --metrics.enable_tensorboard --metrics.disable_color_printing --metrics.save_tb_folder=tensorboard --profiler.enable_profiling --profiler.save_traces_folder=profiling/traces --profiler.profile_freq=7 --profiler.profiler_repeat=1 --profiler.profiler_skip_first=10 --profiler.profiler_warmup=2 --profiler.profiler_active=5 --parallelism.spmd_backend=partial_dtensor --parallelism.data_parallel_replicate_degree=1 --parallelism.data_parallel_shard_degree=2 --parallelism.context_parallel_degree=1 --parallelism.tensor_parallel_degree=4 --parallelism.pipeline_parallel_degree=1 --parallelism.expert_parallel_degree=8
   ```

3. Parse and analyze the capture with the official tools:

   - `advisor` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli advisor all -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/advisor
     ```

   - `cluster` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli cluster -m all -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/cluster
     ```

   - `cluster_time_summary` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m cluster_time_summary -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/cluster_time_summary --export_type text
     ```

   - `communication_bottleneck_rank_0` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/communication_bottleneck_rank_0 --rank_id 0 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_1` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/communication_bottleneck_rank_1 --rank_id 1 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_2` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/communication_bottleneck_rank_2 --rank_id 2 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_3` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/communication_bottleneck_rank_3 --rank_id 3 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_4` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/communication_bottleneck_rank_4 --rank_id 4 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_5` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/communication_bottleneck_rank_5 --rank_id 5 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_6` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/communication_bottleneck_rank_6 --rank_id 6 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_7` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/communication_bottleneck_rank_7 --rank_id 7 --top_num 20 --export_type text
     ```

   - `free_analysis` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m free_analysis -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/free_analysis --top_num 20 --export_type text
     ```

   - `run_state` (return code `None`)

     ```bash
     -
     ```

4. Produce compact evidence (`manifest.json`, `metrics.jsonl`, `analysis.json`, tool status files) and the HTML report when analysis completed.

## results

| metric | value |
| --- | ---: |
| median step | 3,690.06 ms |
| p90 step | - ms |
| throughput / device | 277.50 tok/s |
| throughput / job | 2,220.02 tok/s |
| peak active hbm | - GiB |

### collectives

| rank | kind | calls/step | payload mb/step | transit ms/step | effective gb/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | alltoallv | 140.00 | 20.99 | 1.79 | 11.70 |
| 0 | reducescatter | 212.00 | 66.09 | 4.32 | 15.32 |
| 0 | allgather | 220.00 | 71.41 | 4.17 | 17.11 |
| 0 | allreduce | 362.00 | 0.00 | 0.00 | 0.01 |
| 1 | alltoallv | 140.00 | 25.35 | 2.09 | 12.12 |
| 1 | reducescatter | 212.00 | 66.09 | 4.43 | 14.90 |
| 1 | allreduce | 362.00 | 11.96 | 1.26 | 9.48 |
| 1 | allgather | 220.00 | 71.41 | 4.13 | 17.30 |
| 2 | alltoallv | 140.00 | 19.46 | 1.75 | 11.13 |
| 2 | reducescatter | 212.00 | 66.09 | 4.33 | 15.26 |
| 2 | allreduce | 362.00 | 11.96 | 1.28 | 9.34 |
| 2 | allgather | 220.00 | 71.41 | 4.10 | 17.42 |
| 3 | alltoallv | 140.00 | 22.68 | 1.95 | 11.65 |
| 3 | reducescatter | 212.00 | 66.09 | 4.34 | 15.24 |
| 3 | allreduce | 362.00 | 11.96 | 1.27 | 9.39 |
| 3 | allgather | 220.00 | 71.41 | 4.10 | 17.40 |
| 4 | alltoallv | 140.00 | 28.95 | 2.20 | 13.16 |
| 4 | allreduce | 362.00 | 0.00 | 0.02 | 0.09 |
| 4 | reducescatter | 212.00 | 66.09 | 4.37 | 15.14 |
| 4 | allgather | 220.00 | 71.41 | 4.17 | 17.13 |
| 5 | allreduce | 362.00 | 11.96 | 1.27 | 9.38 |
| 5 | alltoallv | 140.00 | 29.64 | 2.27 | 13.05 |
| 5 | reducescatter | 212.00 | 66.09 | 4.40 | 15.01 |
| 5 | allgather | 220.00 | 71.41 | 4.06 | 17.57 |
| 6 | allreduce | 362.00 | 11.96 | 1.28 | 9.35 |
| 6 | alltoallv | 140.00 | 39.00 | 2.82 | 13.85 |
| 6 | reducescatter | 212.00 | 66.09 | 4.35 | 15.19 |
| 6 | allgather | 220.00 | 71.41 | 4.07 | 17.54 |
| 7 | allreduce | 362.00 | 11.96 | 1.29 | 9.24 |
| 7 | alltoallv | 140.00 | 19.44 | 1.74 | 11.16 |
| 7 | reducescatter | 212.00 | 66.09 | 4.84 | 13.66 |
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

- raw run: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7`
- compact artifact: `/workspace/y50064852_yyb/torchtitan-test/performance_artifacts/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7`
- html report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7.html`
- MindStudio input: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/trainer_output/profiling/traces`
- TensorBoard logdir: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/trainer_output/tensorboard`
- flame graph stacks/SVG root: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/trainer_output/profiling/traces`
- torch.compile visualization: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4-ep8/npu-fsdp2-tp4-ep8-bf16-s30-l8-b64-seq128-seed61-distributed-r31-e45371d7/graph_visualization`

Profiler-off results are throughput evidence. Profiler-active results are used for attribution only and must not be compared as authoritative throughput.
