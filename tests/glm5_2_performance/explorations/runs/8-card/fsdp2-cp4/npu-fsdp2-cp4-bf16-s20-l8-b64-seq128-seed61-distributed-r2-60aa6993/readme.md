# npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993

## experiment

| field | value |
| --- | --- |
| status | completed |
| topology | fsdp2-cp4 |
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
| started | 2026-08-25T12:37:58.515192+08:00 |

## process

1. Run the `probe` driver:

   ```bash
   /root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --probe --device npu --topologies pp8,fsdp2-tp4,fsdp2-cp4,tp2-cp4,fsdp4-tp2,fsdp2-pp4,fsdp2-tp2-pp2,fsdp2-tp4-ep8 --preset distributed --visible-devices 0,1,2,3,4,5,6,7 --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 --replicate 2 --cluster
   ```

2. The workflow generated and executed this training command:

   ```bash
   /root/miniconda3/envs/torchtitan-0803/bin/torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id=npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993 --local-ranks-filter=0 --role=rank --tee=3 -m tests.glm5_2_performance.capture_metrics --module glm5 --config glm5_debugmodel --dump_folder=/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/trainer_output --training.steps=20 --training.num_tokens_per_microbatch_per_dp_rank=1024 --training.num_tokens_per_train_step=8192 --training.max_context_length=128 --training.dtype=float32 --training.mixed_precision_param=bfloat16 --training.mixed_precision_reduce=float32 --debug.seed=61 --debug.no-enable-structured-logging --metrics.log_freq=1 --metrics.enable_tensorboard --metrics.disable_color_printing --metrics.save_tb_folder=tensorboard --profiler.enable_profiling --profiler.save_traces_folder=profiling/traces --profiler.profile_freq=5 --profiler.profiler_repeat=1 --profiler.profiler_skip_first=8 --profiler.profiler_warmup=2 --profiler.profiler_active=3 --parallelism.spmd_backend=partial_dtensor --parallelism.data_parallel_replicate_degree=1 --parallelism.data_parallel_shard_degree=2 --parallelism.context_parallel_degree=4 --parallelism.tensor_parallel_degree=1 --parallelism.pipeline_parallel_degree=1 --parallelism.expert_parallel_degree=1
   ```

3. Parse and analyze the capture with the official tools:

   - `cluster` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli cluster -m all -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/cluster
     ```

   - `cluster_time_summary` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m cluster_time_summary -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/cluster_time_summary --export_type text
     ```

   - `communication_bottleneck_rank_0` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/communication_bottleneck_rank_0 --rank_id 0 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_1` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/communication_bottleneck_rank_1 --rank_id 1 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_2` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/communication_bottleneck_rank_2 --rank_id 2 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_3` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/communication_bottleneck_rank_3 --rank_id 3 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_4` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/communication_bottleneck_rank_4 --rank_id 4 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_5` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/communication_bottleneck_rank_5 --rank_id 5 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_6` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/communication_bottleneck_rank_6 --rank_id 6 --top_num 20 --export_type text
     ```

   - `communication_bottleneck_rank_7` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m communication_bottleneck -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/communication_bottleneck_rank_7 --rank_id 7 --top_num 20 --export_type text
     ```

   - `free_analysis` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli -m free_analysis -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/free_analysis --top_num 20 --export_type text
     ```

4. Produce compact evidence (`manifest.json`, `metrics.jsonl`, `analysis.json`, tool status files) and the HTML report when analysis completed.

## results

| metric | value |
| --- | ---: |
| median step | 879.64 ms |
| p90 step | - ms |
| throughput / device | 1,164.11 tok/s |
| throughput / job | 9,312.91 tok/s |
| peak active hbm | - GiB |

### collectives

| rank | kind | calls/step | payload mb/step | transit ms/step | effective gb/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | reducescatter | 108.00 | 283.66 | 70.93 | 4.00 |
| 0 | allgather | 184.00 | 296.25 | 21.50 | 13.78 |
| 0 | allreduce | 7.00 | 0.00 | 0.01 | 0.01 |
| 1 | reducescatter | 108.00 | 283.66 | 61.94 | 4.58 |
| 1 | allgather | 184.00 | 296.25 | 21.57 | 13.73 |
| 1 | allreduce | 7.00 | 0.00 | 0.02 | 0.10 |
| 2 | reducescatter | 108.00 | 283.66 | 65.48 | 4.33 |
| 2 | allgather | 184.00 | 296.25 | 21.60 | 13.72 |
| 2 | allreduce | 7.00 | 0.00 | 0.02 | 0.10 |
| 3 | reducescatter | 108.00 | 283.66 | 65.36 | 4.34 |
| 3 | allgather | 184.00 | 296.25 | 20.82 | 14.23 |
| 3 | allreduce | 7.00 | 0.00 | 0.02 | 0.10 |
| 4 | reducescatter | 108.00 | 283.66 | 66.20 | 4.28 |
| 4 | allgather | 184.00 | 296.25 | 21.17 | 13.99 |
| 4 | allreduce | 7.00 | 0.00 | 0.02 | 0.10 |
| 5 | allgather | 184.00 | 296.25 | 21.17 | 13.99 |
| 5 | reducescatter | 108.00 | 283.66 | 65.23 | 4.35 |
| 5 | allreduce | 7.00 | 0.00 | 0.02 | 0.10 |
| 6 | reducescatter | 108.00 | 283.66 | 70.93 | 4.00 |
| 6 | allgather | 184.00 | 296.25 | 21.28 | 13.92 |
| 6 | allreduce | 7.00 | 0.00 | 0.02 | 0.10 |
| 7 | reducescatter | 108.00 | 283.66 | 63.19 | 4.49 |
| 7 | allgather | 184.00 | 296.25 | 21.27 | 13.93 |
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

- raw run: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993`
- compact artifact: `/workspace/y50064852_yyb/torchtitan-test/performance_artifacts/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993`
- html report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993.html`
- MindStudio input: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/8-card/fsdp2-cp4/npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993/trainer_output/profiling/traces`

Profiler-off results are throughput evidence. Profiler-active results are used for attribution only and must not be compared as authoritative throughput.
