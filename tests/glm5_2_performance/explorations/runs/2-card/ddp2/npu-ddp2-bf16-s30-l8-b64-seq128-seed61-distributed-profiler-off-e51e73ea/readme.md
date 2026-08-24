# npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-profiler-off-e51e73ea

## experiment

| field | value |
| --- | --- |
| status | completed |
| topology | ddp2 |
| world size | 2 |
| mode | profiler-off |
| device | npu |
| preset | distributed |
| steps | 30 |
| local / global batch | 8 / 64 |
| sequence length | 128 |
| parameter / reduction dtype | bfloat16 / float32 |
| replicate | - |
| visible devices | - |
| started | 2026-08-24T17:14:28.439011+08:00 |

## process

1. Run the `analyze` driver:

   ```bash
   /root/miniconda3/envs/torchtitan-0803/bin/python tests/glm5_2_performance/profiler_benchmark.py --analyze --device npu --topology ddp2 --preset distributed --profiler-off --visible-devices 1,2
   ```

2. The workflow generated and executed this training command:

   ```bash
   /root/miniconda3/envs/torchtitan-0803/bin/torchrun --nnodes=1 --node_rank=0 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id=npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-profiler-off-e51e73ea --local-ranks-filter=0 --role=rank --tee=3 -m tests.glm5_2_performance.capture_metrics --module glm5 --config glm5_debugmodel --dump_folder=/workspace/y50064852_yyb/torchtitan-test/performance_runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-profiler-off-e51e73ea/trainer_output --training.steps=30 --training.num_tokens_per_microbatch_per_dp_rank=1024 --training.num_tokens_per_train_step=8192 --training.max_context_length=128 --training.dtype=float32 --training.mixed_precision_param=bfloat16 --training.mixed_precision_reduce=float32 --debug.seed=61 --debug.no-enable-structured-logging --metrics.log_freq=1 --metrics.enable_tensorboard --metrics.disable_color_printing --metrics.save_tb_folder=tensorboard --parallelism.spmd_backend=partial_dtensor --parallelism.data_parallel_replicate_degree=2 --parallelism.data_parallel_shard_degree=1 --parallelism.context_parallel_degree=1 --parallelism.tensor_parallel_degree=1 --parallelism.pipeline_parallel_degree=1 --parallelism.expert_parallel_degree=1
   ```

3. Produce compact evidence (`manifest.json`, `metrics.jsonl`, `analysis.json`, tool status files) and the HTML report when analysis completed.

## results

| metric | value |
| --- | ---: |
| median step | 1,443.12 ms |
| p90 step | 1,852.21 ms |
| throughput / device | 2,838.38 tok/s |
| throughput / job | 5,676.77 tok/s |
| peak active hbm | 0.843 GiB |

## outputs

| output | purpose |
| --- | --- |
| [command_history.jsonl](command_history.jsonl) | exact driver history and environment |
| [manifest.json](manifest.json) | resolved configuration and generated training command |
| [metrics.jsonl](metrics.jsonl) | raw per-step metrics |
| [analysis.json](analysis.json) | structured derived analysis |
| [artifacts.json](artifacts.json) | raw run, compact artifact, report, and MindStudio paths |

- raw run: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-profiler-off-e51e73ea`
- compact artifact: `/workspace/y50064852_yyb/torchtitan-test/performance_artifacts/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-profiler-off-e51e73ea`
- html report: `/workspace/y50064852_yyb/torchtitan-test/performance_reports/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-profiler-off-e51e73ea.html`
- MindStudio input: `/workspace/y50064852_yyb/torchtitan-test/performance_runs/2-card/ddp2/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-profiler-off-e51e73ea/trainer_output/profiling/traces`

Profiler-off results are throughput evidence. Profiler-active results are used for attribution only and must not be compared as authoritative throughput.
