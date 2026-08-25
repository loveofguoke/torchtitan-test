# npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5

## experiment

| field | value |
| --- | --- |
| status | completed |
| topology | fsdp2-tp4 |
| world size | 8 |
| mode | profiler-active |
| device | npu |
| preset | distributed |
| steps | 20 |
| local / global batch | 8 / 64 |
| sequence length | 128 |
| parameter / reduction dtype | bfloat16 / float32 |
| replicate | 2 |
| visible devices | 0, 1, 2, 3, 4, 5, 6, 7 |
| started | 2026-08-25T12:26:31.693143+08:00 |

## process

1. Run the `probe` driver:

   ```bash
   /root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topologies pp8,fsdp2-tp4,fsdp2-cp4,tp2-cp4,fsdp4-tp2,fsdp2-pp4,fsdp2-tp2-pp2,fsdp2-tp4-ep8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 2 --cluster
   ```

2. The workflow generated and executed this training command:

   ```bash
   /root/miniconda3/envs/torchtitan-0803/bin/torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id=npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5 --local-ranks-filter=0 --role=rank --tee=3 -m tests.glm5_2_performance.capture_metrics --module glm5 --config glm5_debugmodel --dump_folder=/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5/trainer_output --training.steps=20 --training.num_tokens_per_microbatch_per_dp_rank=1024 --training.num_tokens_per_train_step=8192 --training.max_context_length=128 --training.dtype=float32 --training.mixed_precision_param=bfloat16 --training.mixed_precision_reduce=float32 --debug.seed=61 --debug.no-enable-structured-logging --metrics.log_freq=1 --metrics.enable_tensorboard --metrics.disable_color_printing --metrics.save_tb_folder=tensorboard --profiler.enable_profiling --profiler.save_traces_folder=profiling/traces --profiler.profile_freq=5 --profiler.profiler_repeat=1 --profiler.profiler_skip_first=8 --profiler.profiler_warmup=2 --profiler.profiler_active=3 --parallelism.spmd_backend=partial_dtensor --parallelism.data_parallel_replicate_degree=1 --parallelism.data_parallel_shard_degree=2 --parallelism.context_parallel_degree=1 --parallelism.tensor_parallel_degree=4 --parallelism.pipeline_parallel_degree=1 --parallelism.expert_parallel_degree=1 --parallelism.no-enable-sequence-parallel
   ```

3. Parse and analyze the capture with the official tools:

   - `cluster` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli cluster -m all -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5/cluster
     ```

   - `cluster_time_summary` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m cluster_time_summary -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5/cluster_time_summary --export_type text
     ```

   - `communication_bottleneck_rank_0` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5/communication_bottleneck_rank_0 --rank_id 0 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_1` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5/communication_bottleneck_rank_1 --rank_id 1 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_2` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5/communication_bottleneck_rank_2 --rank_id 2 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_3` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5/communication_bottleneck_rank_3 --rank_id 3 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_4` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5/communication_bottleneck_rank_4 --rank_id 4 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_5` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5/communication_bottleneck_rank_5 --rank_id 5 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_6` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5/communication_bottleneck_rank_6 --rank_id 6 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_7` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5/communication_bottleneck_rank_7 --rank_id 7 --top_num 20 --export_type text
     ```

   - `free_analysis` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m free_analysis -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5/free_analysis --top_num 20 --export_type text
     ```

4. Produce compact evidence (`manifest.json`, `metrics.jsonl`, `analysis.json`, tool status files) and the HTML report when analysis completed.

## results

| metric | value |
| --- | ---: |
| median step | 2,365.60 ms |
| p90 step | - ms |
| throughput / device | 432.87 tok/s |
| throughput / job | 3,462.97 tok/s |
| peak active hbm | - GiB |

### collectives

| rank | kind | calls/step | payload mb/step | transit ms/step | effective gb/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | allreduce | 440.00 | 62.91 | 4.08 | 15.44 |
| 0 | reducescatter | 140.00 | 59.80 | 3.77 | 15.88 |
| 0 | allgather | 144.00 | 63.54 | 3.53 | 18.02 |
| 1 | allreduce | 440.00 | 75.10 | 5.40 | 13.90 |
| 1 | reducescatter | 140.00 | 59.80 | 3.84 | 15.59 |
| 1 | allgather | 144.00 | 63.54 | 3.49 | 18.21 |
| 2 | allreduce | 440.00 | 75.10 | 5.20 | 14.46 |
| 2 | reducescatter | 140.00 | 59.80 | 3.77 | 15.85 |
| 2 | allgather | 144.00 | 63.54 | 3.49 | 18.22 |
| 3 | allreduce | 440.00 | 75.10 | 5.17 | 14.52 |
| 3 | reducescatter | 140.00 | 59.80 | 3.79 | 15.77 |
| 3 | allgather | 144.00 | 63.54 | 3.48 | 18.25 |
| 4 | reducescatter | 140.00 | 59.80 | 3.71 | 16.14 |
| 4 | allgather | 144.00 | 63.54 | 3.55 | 17.89 |
| 4 | allreduce | 440.00 | 62.92 | 4.09 | 15.39 |
| 5 | reducescatter | 140.00 | 59.80 | 3.76 | 15.91 |
| 5 | allreduce | 440.00 | 75.10 | 5.41 | 13.89 |
| 5 | allgather | 144.00 | 63.54 | 3.46 | 18.35 |
| 6 | reducescatter | 140.00 | 59.80 | 3.72 | 16.07 |
| 6 | allreduce | 440.00 | 75.10 | 5.36 | 14.00 |
| 6 | allgather | 144.00 | 63.54 | 3.48 | 18.28 |
| 7 | allreduce | 440.00 | 75.10 | 5.63 | 13.34 |
| 7 | reducescatter | 140.00 | 59.80 | 3.89 | 15.35 |
| 7 | allgather | 144.00 | 63.54 | 3.55 | 17.92 |

## outputs

| output | purpose |
| --- | --- |
| [command_history.jsonl](command_history.jsonl) | exact driver history and environment |
| [manifest.json](manifest.json) | resolved configuration and generated training command |
| [metrics.jsonl](metrics.jsonl) | raw per-step metrics |
| [analysis.json](analysis.json) | structured derived analysis |
| [artifacts.json](artifacts.json) | raw run, compact artifact, report, and MindStudio paths |
| [tool_commands](tool_commands/) | official analysis commands and return codes |

- raw run: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5`
- compact artifact: `/workspace/y50064852_yyb/torchtitan-test/performance_artifacts/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5`
- html report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5.html`
- MindStudio input: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-tp4/npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5/trainer_output/profiling/traces`

Profiler-off results are throughput evidence. Profiler-active results are used for attribution only and must not be compared as authoritative throughput.
