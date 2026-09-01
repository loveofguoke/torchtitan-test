# npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-c03a3c30

## experiment

| field | value |
| --- | --- |
| status | completed |
| topology | fsdp4 |
| world size | 4 |
| mode | profiler-active |
| device | npu |
| preset | operator |
| steps | 30 |
| local / global batch | 8 / 64 |
| sequence length | 128 |
| parameter / reduction dtype | bfloat16 / float32 |
| replicate | 32 |
| visible devices | 0, 1, 2, 3, 4, 5, 6, 7 |
| started | 2026-09-01T00:03:08.761420+08:00 |

## process

1. Run the `capture` driver:

   ```bash
   /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --capture --device npu --topology fsdp4 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools none
   ```

2. Run the `analyze` driver:

   ```bash
   /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology fsdp4 --preset operator --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 --analysis-tools all --parse-workers 8
   ```

3. The workflow generated and executed this training command:

   ```bash
   /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/torchrun --nnodes=1 --node_rank=0 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id=npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-c03a3c30 --local-ranks-filter=0 --role=rank --tee=3 -m tests.glm5_2_performance.capture_metrics --module glm5 --config glm5_debugmodel --dump_folder=/workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-c03a3c30/trainer_output --training.steps=30 --training.num_tokens_per_microbatch_per_dp_rank=1024 --training.num_tokens_per_train_step=8192 --training.max_context_length=128 --training.dtype=float32 --training.mixed_precision_param=bfloat16 --training.mixed_precision_reduce=float32 --debug.seed=61 --debug.no-enable-structured-logging --metrics.log_freq=1 --metrics.enable_tensorboard --metrics.disable_color_printing --metrics.save_tb_folder=tensorboard --profiler.enable_profiling --profiler.save_traces_folder=profiling/traces --profiler.profile_freq=7 --profiler.profiler_repeat=1 --profiler.profiler_skip_first=10 --profiler.profiler_warmup=2 --profiler.profiler_active=5 --parallelism.spmd_backend=partial_dtensor --parallelism.data_parallel_replicate_degree=1 --parallelism.data_parallel_shard_degree=4 --parallelism.context_parallel_degree=1 --parallelism.tensor_parallel_degree=1 --parallelism.pipeline_parallel_degree=1 --parallelism.expert_parallel_degree=1
   ```

4. Parse and analyze the capture with the official tools:

   - `advisor` (return code `0`)

     ```bash
     /workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli advisor all -d /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-c03a3c30/trainer_output/profiling/traces -o /workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-c03a3c30/advisor
     ```

   - `run_state` (return code `None`)

     ```bash
     -
     ```

5. Produce compact evidence (`manifest.json`, `metrics.jsonl`, `analysis.json`, tool status files) and the HTML report when analysis completed.

## results

| metric | value |
| --- | ---: |
| median step | 713.76 ms |
| p90 step | - ms |
| throughput / device | 2,869.30 tok/s |
| throughput / job | 11,477.19 tok/s |
| peak active hbm | - GiB |

### collectives

| rank | kind | calls/step | payload mb/step | transit ms/step | effective gb/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | reducescatter | 22.00 | 100.00 | 6.75 | 14.82 |
| 0 | allgather | 42.00 | 103.99 | 5.73 | 18.16 |
| 0 | allreduce | 7.00 | 0.00 | 0.01 | 0.01 |

## outputs

| output | purpose |
| --- | --- |
| [command_history.jsonl](command_history.jsonl) | exact driver history and environment |
| [manifest.json](manifest.json) | resolved configuration and generated training command |
| [metrics.jsonl](metrics.jsonl) | raw per-step metrics |
| [analysis.json](analysis.json) | structured derived analysis |
| [artifacts.json](artifacts.json) | raw run, compact artifact, report, and MindStudio paths |
| [tool_commands](tool_commands/) | official analysis commands and return codes |

- raw run: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-c03a3c30`
- compact artifact: `/workspace/y50064852_yyb/torchtitan-test/performance_artifacts/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-c03a3c30`
- html report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-c03a3c30.html`
- MindStudio input: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-c03a3c30/trainer_output/profiling/traces`
- TensorBoard logdir: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-c03a3c30/trainer_output/tensorboard`
- flame graph stacks/SVG root: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-c03a3c30/trainer_output/profiling/traces`
- torch.compile visualization: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/4-card/fsdp4/npu-fsdp4-bf16-s30-l8-b64-seq128-seed61-operator-r32-offline-c03a3c30/graph_visualization`

Profiler-off results are throughput evidence. Profiler-active results are used for attribution only and must not be compared as authoritative throughput.
