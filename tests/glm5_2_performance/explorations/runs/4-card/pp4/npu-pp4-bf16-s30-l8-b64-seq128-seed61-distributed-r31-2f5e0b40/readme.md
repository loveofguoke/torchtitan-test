# npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40

## experiment

| field | value |
| --- | --- |
| status | completed |
| topology | pp4 |
| world size | 4 |
| mode | profiler-active |
| device | npu |
| preset | distributed |
| steps | 30 |
| local / global batch | 8 / 64 |
| sequence length | 128 |
| parameter / reduction dtype | bfloat16 / float32 |
| replicate | 31 |
| visible devices | 0, 1, 2, 3, 4, 5, 6, 7 |
| started | 2026-08-31T18:54:28.775057+08:00 |

## process

1. Run the `probe` driver:

   ```bash
   /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topologies fsdp2-tp4-ep8,ddp4,fsdp2,fsdp4,tp2,tp4,cp2,cp4,pp2,pp4,ep2,ep4,fsdp2-tp2 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 --analysis-tools all
   ```

2. The workflow generated and executed this training command:

   ```bash
   /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/torchrun --nnodes=1 --node_rank=0 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id=npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40 --local-ranks-filter=3 --role=rank --tee=3 -m tests.glm5_2_performance.capture_metrics --module glm5 --config glm5_debugmodel --dump_folder=/workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40/trainer_output --training.steps=30 --training.num_tokens_per_microbatch_per_dp_rank=128 --training.num_tokens_per_train_step=8192 --training.max_context_length=128 --parallelism.num_pp_microbatches=8 --training.dtype=float32 --training.mixed_precision_param=bfloat16 --training.mixed_precision_reduce=float32 --debug.seed=61 --debug.no-enable-structured-logging --metrics.log_freq=1 --metrics.enable_tensorboard --metrics.disable_color_printing --metrics.save_tb_folder=tensorboard --profiler.enable_profiling --profiler.save_traces_folder=profiling/traces --profiler.profile_freq=7 --profiler.profiler_repeat=1 --profiler.profiler_skip_first=10 --profiler.profiler_warmup=2 --profiler.profiler_active=5 --parallelism.spmd_backend=partial_dtensor --parallelism.data_parallel_replicate_degree=1 --parallelism.data_parallel_shard_degree=1 --parallelism.context_parallel_degree=1 --parallelism.tensor_parallel_degree=1 --parallelism.pipeline_parallel_degree=4 --parallelism.expert_parallel_degree=1 --parallelism.pipeline_parallel_schedule=1F1B
   ```

3. Parse and analyze the capture with the official tools:

   - `advisor` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli advisor all -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40/advisor
     ```

   - `cluster` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli cluster -m all -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40/cluster
     ```

   - `cluster_time_summary` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m cluster_time_summary -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40/cluster_time_summary --export_type text
     ```

   - `communication_bottleneck_rank_0` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40/communication_bottleneck_rank_0 --rank_id 0 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_1` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40/communication_bottleneck_rank_1 --rank_id 1 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_2` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40/communication_bottleneck_rank_2 --rank_id 2 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_3` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40/communication_bottleneck_rank_3 --rank_id 3 --top_num 20 --export_type text
     ```

   - `free_analysis` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m free_analysis -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40/free_analysis --top_num 20 --export_type text
     ```

   - `run_state` (return code `None`)

     ```bash
     -
     ```

4. Produce compact evidence (`manifest.json`, `metrics.jsonl`, `analysis.json`, tool status files) and the HTML report when analysis completed.

## results

| metric | value |
| --- | ---: |
| median step | 6,210.33 ms |
| p90 step | - ms |
| throughput / device | 329.77 tok/s |
| throughput / job | 1,319.09 tok/s |
| peak active hbm | - GiB |

### collectives

| rank | kind | calls/step | payload mb/step | transit ms/step | effective gb/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | batchsendrecv | 41.60 | 2.62 | 0.17 | 15.85 |
| 0 | receive | 24.00 | 1.57 | 0.10 | 16.39 |
| 0 | send | 24.00 | 0.00 | 0.00 | - |
| 0 | allreduce | 2.00 | 0.00 | 0.00 | - |
| 1 | receive | 40.00 | 2.62 | 0.16 | 16.37 |
| 1 | batchsendrecv | 89.60 | 5.77 | 0.36 | 16.14 |
| 1 | allreduce | 2.00 | 0.00 | 0.00 | 0.00 |
| 1 | send | 40.00 | 0.00 | 0.00 | - |
| 2 | batchsendrecv | 105.60 | 6.82 | 0.43 | 15.97 |
| 2 | receive | 24.00 | 1.57 | 0.10 | 16.39 |
| 2 | allreduce | 2.00 | 0.00 | 0.00 | 0.00 |
| 2 | send | 24.00 | 0.00 | 0.00 | - |
| 3 | batchsendrecv | 57.60 | 3.67 | 0.23 | 15.93 |
| 3 | receive | 8.00 | 0.52 | 0.03 | 16.45 |
| 3 | allreduce | 2.00 | 0.00 | 0.00 | 0.00 |
| 3 | send | 8.00 | 0.00 | 0.00 | - |

## outputs

| output | purpose |
| --- | --- |
| [command_history.jsonl](command_history.jsonl) | exact driver history and environment |
| [manifest.json](manifest.json) | resolved configuration and generated training command |
| [metrics.jsonl](metrics.jsonl) | raw per-step metrics |
| [analysis.json](analysis.json) | structured derived analysis |
| [artifacts.json](artifacts.json) | raw run, compact artifact, report, and MindStudio paths |
| [tool_commands](tool_commands/) | official analysis commands and return codes |

- raw run: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40`
- compact artifact: `/workspace/y50064852_yyb/torchtitan-test/performance_artifacts/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40`
- html report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40.html`
- MindStudio input: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40/trainer_output/profiling/traces`
- TensorBoard logdir: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40/trainer_output/tensorboard`
- flame graph stacks/SVG root: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40/trainer_output/profiling/traces`
- torch.compile visualization: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/pp4/npu-pp4-bf16-s30-l8-b64-seq128-seed61-distributed-r31-2f5e0b40/graph_visualization`

Profiler-off results are throughput evidence. Profiler-active results are used for attribution only and must not be compared as authoritative throughput.
