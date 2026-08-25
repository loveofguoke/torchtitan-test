# npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c

## experiment

| field | value |
| --- | --- |
| status | completed |
| topology | tp8 |
| world size | 8 |
| mode | profiler-active |
| device | npu |
| preset | distributed |
| steps | 20 |
| local / global batch | 8 / 64 |
| sequence length | 128 |
| parameter / reduction dtype | bfloat16 / float32 |
| replicate | 1 |
| visible devices | 0, 1, 2, 3, 4, 5, 6, 7 |
| started | 2026-08-24T20:42:29.508722+08:00 |

## process

1. Run the `probe` driver:

   ```bash
   /root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology tp8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 1 --cluster
   ```

2. The workflow generated and executed this training command:

   ```bash
   /root/miniconda3/envs/torchtitan-0803/bin/torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id=npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c --local-ranks-filter=0 --role=rank --tee=3 -m tests.glm5_2_performance.capture_metrics --module glm5 --config glm5_debugmodel --dump_folder=/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/trainer_output --training.steps=20 --training.num_tokens_per_microbatch_per_dp_rank=1024 --training.num_tokens_per_train_step=8192 --training.max_context_length=128 --training.dtype=float32 --training.mixed_precision_param=bfloat16 --training.mixed_precision_reduce=float32 --debug.seed=61 --debug.no-enable-structured-logging --metrics.log_freq=1 --metrics.enable_tensorboard --metrics.disable_color_printing --metrics.save_tb_folder=tensorboard --profiler.enable_profiling --profiler.save_traces_folder=profiling/traces --profiler.profile_freq=5 --profiler.profiler_repeat=1 --profiler.profiler_skip_first=8 --profiler.profiler_warmup=2 --profiler.profiler_active=3 --parallelism.spmd_backend=partial_dtensor --parallelism.data_parallel_replicate_degree=1 --parallelism.data_parallel_shard_degree=1 --parallelism.context_parallel_degree=1 --parallelism.tensor_parallel_degree=8 --parallelism.pipeline_parallel_degree=1 --parallelism.expert_parallel_degree=1 --parallelism.no-enable-sequence-parallel
   ```

3. Parse and analyze the capture with the official tools:

   - `cluster` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli cluster -m all -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/cluster
     ```

   - `cluster_time_summary` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m cluster_time_summary -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/cluster_time_summary --export_type text
     ```

   - `communication_bottleneck_rank_0` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/communication_bottleneck_rank_0 --rank_id 0 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_1` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/communication_bottleneck_rank_1 --rank_id 1 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_2` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/communication_bottleneck_rank_2 --rank_id 2 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_3` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/communication_bottleneck_rank_3 --rank_id 3 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_4` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/communication_bottleneck_rank_4 --rank_id 4 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_5` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/communication_bottleneck_rank_5 --rank_id 5 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_6` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/communication_bottleneck_rank_6 --rank_id 6 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_7` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/communication_bottleneck_rank_7 --rank_id 7 --top_num 20 --export_type text
     ```

   - `free_analysis` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m free_analysis -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/free_analysis --top_num 20 --export_type text
     ```

4. Produce compact evidence (`manifest.json`, `metrics.jsonl`, `analysis.json`, tool status files) and the HTML report when analysis completed.

## results

| metric | value |
| --- | ---: |
| median step | 9,097.91 ms |
| p90 step | - ms |
| throughput / device | 112.55 tok/s |
| throughput / job | 900.43 tok/s |
| peak active hbm | - GiB |

### collectives

| rank | kind | calls/step | payload mb/step | transit ms/step | effective gb/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | allreduce | 865.00 | 146.80 | 11.24 | 13.06 |
| 0 | reducescatter | 192.00 | 55.05 | 4.19 | 13.14 |
| 0 | allgather | 120.00 | 55.05 | 3.65 | 15.09 |
| 1 | allreduce | 865.00 | 171.17 | 14.00 | 12.22 |
| 1 | reducescatter | 192.00 | 55.05 | 4.27 | 12.89 |
| 1 | allgather | 120.00 | 55.05 | 3.63 | 15.16 |
| 2 | allreduce | 865.00 | 171.17 | 14.08 | 12.16 |
| 2 | reducescatter | 192.00 | 55.05 | 4.28 | 12.87 |
| 2 | allgather | 120.00 | 55.05 | 3.61 | 15.24 |
| 3 | allreduce | 865.00 | 171.17 | 13.72 | 12.47 |
| 3 | reducescatter | 192.00 | 55.05 | 4.25 | 12.95 |
| 3 | allgather | 120.00 | 55.05 | 3.61 | 15.25 |
| 4 | allreduce | 865.00 | 171.17 | 14.08 | 12.15 |
| 4 | reducescatter | 192.00 | 55.05 | 4.30 | 12.80 |
| 4 | allgather | 120.00 | 55.05 | 3.66 | 15.04 |
| 5 | allreduce | 865.00 | 171.17 | 14.02 | 12.21 |
| 5 | reducescatter | 192.00 | 55.05 | 4.25 | 12.95 |
| 5 | allgather | 120.00 | 55.05 | 3.66 | 15.05 |
| 6 | allreduce | 865.00 | 171.17 | 13.93 | 12.29 |
| 6 | reducescatter | 192.00 | 55.05 | 4.23 | 13.02 |
| 6 | allgather | 120.00 | 55.05 | 3.67 | 14.98 |
| 7 | allreduce | 865.00 | 171.17 | 13.65 | 12.54 |
| 7 | reducescatter | 192.00 | 55.05 | 4.11 | 13.40 |
| 7 | allgather | 120.00 | 55.05 | 3.63 | 15.15 |

## outputs

| output | purpose |
| --- | --- |
| [command_history.jsonl](command_history.jsonl) | exact driver history and environment |
| [manifest.json](manifest.json) | resolved configuration and generated training command |
| [metrics.jsonl](metrics.jsonl) | raw per-step metrics |
| [analysis.json](analysis.json) | structured derived analysis |
| [artifacts.json](artifacts.json) | raw run, compact artifact, report, and MindStudio paths |
| [tool_commands](tool_commands/) | official analysis commands and return codes |

- raw run: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c`
- compact artifact: `/workspace/y50064852_yyb/torchtitan-test/performance_artifacts/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c`
- html report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c.html`
- MindStudio input: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/tp8/npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c/trainer_output/profiling/traces`

Profiler-off results are throughput evidence. Profiler-active results are used for attribution only and must not be compared as authoritative throughput.
