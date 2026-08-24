# npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684

## experiment

| field | value |
| --- | --- |
| status | completed |
| topology | fsdp4 |
| world size | 4 |
| mode | profiler-active |
| device | npu |
| preset | distributed |
| steps | 20 |
| local / global batch | 8 / 64 |
| sequence length | 128 |
| parameter / reduction dtype | bfloat16 / float32 |
| replicate | 3 |
| visible devices | - |
| started | 2026-08-24T17:02:27.528676+08:00 |

## process

1. Run the `probe` driver:

   ```bash
   /root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topology fsdp4 --preset distributed --visible-devices 1,2,3,5 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 3 --cluster
   ```

2. The workflow generated and executed this training command:

   ```bash
   /root/miniconda3/envs/torchtitan-0803/bin/torchrun --nnodes=1 --node_rank=0 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id=npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684 --local-ranks-filter=0 --role=rank --tee=3 -m tests.glm5_2_performance.capture_metrics --module glm5 --config glm5_debugmodel --dump_folder=/workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684/trainer_output --training.steps=20 --training.num_tokens_per_microbatch_per_dp_rank=1024 --training.num_tokens_per_train_step=8192 --training.max_context_length=128 --training.dtype=float32 --training.mixed_precision_param=bfloat16 --training.mixed_precision_reduce=float32 --debug.seed=61 --debug.no-enable-structured-logging --metrics.log_freq=1 --metrics.enable_tensorboard --metrics.disable_color_printing --metrics.save_tb_folder=tensorboard --profiler.enable_profiling --profiler.save_traces_folder=profiling/traces --profiler.profile_freq=5 --profiler.profiler_repeat=1 --profiler.profiler_skip_first=8 --profiler.profiler_warmup=2 --profiler.profiler_active=3 --parallelism.spmd_backend=partial_dtensor --parallelism.data_parallel_replicate_degree=1 --parallelism.data_parallel_shard_degree=4 --parallelism.context_parallel_degree=1 --parallelism.tensor_parallel_degree=1 --parallelism.pipeline_parallel_degree=1 --parallelism.expert_parallel_degree=1
   ```

3. Parse and analyze the capture with the official tools:

   - `cluster` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli cluster -m all -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684/cluster
     ```

   - `cluster_time_summary` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m cluster_time_summary -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684/cluster_time_summary --export_type text
     ```

   - `communication_bottleneck_rank_0` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684/communication_bottleneck_rank_0 --rank_id 0 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_1` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684/communication_bottleneck_rank_1 --rank_id 1 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_2` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684/communication_bottleneck_rank_2 --rank_id 2 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_3` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684/communication_bottleneck_rank_3 --rank_id 3 --top_num 20 --export_type text
     ```

   - `free_analysis` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m free_analysis -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684/free_analysis --top_num 20 --export_type text
     ```

4. Produce compact evidence (`manifest.json`, `metrics.jsonl`, `analysis.json`, tool status files) and the HTML report when analysis completed.

## results

| metric | value |
| --- | ---: |
| median step | 409.04 ms |
| p90 step | - ms |
| throughput / device | 5,006.86 tok/s |
| throughput / job | 20,027.44 tok/s |
| peak active hbm | - GiB |

### collectives

| rank | kind | calls/step | payload mb/step | transit ms/step | effective gb/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | allgather | 42.00 | 103.99 | 5.80 | 17.92 |
| 0 | allreduce | 7.00 | 0.00 | 0.01 | 0.01 |
| 0 | reducescatter | 22.00 | 100.00 | 6.88 | 14.53 |
| 1 | reducescatter | 22.00 | 100.00 | 9.29 | 10.77 |
| 1 | allgather | 42.00 | 103.99 | 5.71 | 18.22 |
| 1 | allreduce | 7.00 | 0.00 | 0.02 | 0.10 |
| 2 | reducescatter | 22.00 | 100.00 | 8.51 | 11.75 |
| 2 | allgather | 42.00 | 103.99 | 5.66 | 18.37 |
| 2 | allreduce | 7.00 | 0.00 | 0.02 | 0.10 |
| 3 | reducescatter | 22.00 | 100.00 | 8.99 | 11.12 |
| 3 | allgather | 42.00 | 103.99 | 6.05 | 17.18 |
| 3 | allreduce | 7.00 | 0.00 | 0.01 | 0.10 |

## outputs

| output | purpose |
| --- | --- |
| [command_history.jsonl](command_history.jsonl) | exact driver history and environment |
| [manifest.json](manifest.json) | resolved configuration and generated training command |
| [metrics.jsonl](metrics.jsonl) | raw per-step metrics |
| [analysis.json](analysis.json) | structured derived analysis |
| [artifacts.json](artifacts.json) | raw run, compact artifact, report, and MindStudio paths |
| [tool_commands](tool_commands/) | official analysis commands and return codes |

- raw run: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684`
- compact artifact: `/workspace/y50064852_yyb/torchtitan-test/performance_artifacts/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684`
- html report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684.html`
- MindStudio input: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/fsdp4/npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684/trainer_output/profiling/traces`

Profiler-off results are throughput evidence. Profiler-active results are used for attribution only and must not be compared as authoritative throughput.
