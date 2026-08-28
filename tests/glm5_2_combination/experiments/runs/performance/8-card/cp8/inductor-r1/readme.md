# cp8 inductor repeat 1

## Result

| field | value |
|---|---:|
| status | PASS |
| cards/ranks | 8 |
| metrics | 30/30 |
| measured steps | 11-30 |
| median step | 2.828329 s |
| p90 step | 2.848733 s |
| job throughput | 2896.41 tok/s |
| peak active HBM | 0.280 GiB |
| compile markers | 1 |
| graph breaks | 0 |
| recompiles | 1 |

## Executed process

The run was emitted by this actual batch driver:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination --capture candidate --topology all --objectives performance --reference-graph eager --candidate-graph inductor --profiler-preset off --steps 30 --performance-skip-steps 10 --performance-nondeterministic --compiler-diagnostics
```

The combination runner generated this exact command:

```bash
/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id=self-npu-bf16-random-s30-b64-seq128-seed61-eager-inductor-performance-no-prof-skip10-nondet-9d360eea --local-ranks-filter=0 --role=rank --tee=3 -m tests.glm5_2_combination.capture_metrics --module glm5 --config glm5_debugmodel --dump_folder=/workspace/y50064852_yyb/torchtitan-test/combination_runs/self-npu-bf16-random-s30-b64-seq128-seed61-eager-inductor-performance-no-prof-skip10-nondet-9d360eea/cp8/cp8-r1/trainer_output --training.steps=30 --training.num_tokens_per_microbatch_per_dp_rank=1024 --training.num_tokens_per_train_step=8192 --training.max_context_length=128 --training.dtype=float32 --training.mixed_precision_param=bfloat16 --training.mixed_precision_reduce=float32 --debug.seed=61 --metrics.log_freq=1 --metrics.enable_tensorboard --metrics.disable_color_printing --metrics.save_tb_folder=tensorboard --debug.no-enable-structured-logging --parallelism.spmd_backend=partial_dtensor --parallelism.data_parallel_replicate_degree=1 --parallelism.data_parallel_shard_degree=1 --parallelism.context_parallel_degree=8 --parallelism.tensor_parallel_degree=1 --parallelism.pipeline_parallel_degree=1 --parallelism.expert_parallel_degree=1 --compile.enable --compile.components=model --compile.backend=inductor --checkpoint.enable --checkpoint.load_only --checkpoint.initial_load_path=/workspace/y50064852_yyb/torchtitan-test/precision_fixtures/self-npu-bf16-random-s30-b64-seq128-seed61-976dfb03/checkpoint
```

Execution order was `fixture -> candidate capture -> metrics/artifact validation ->`
`summarize_performance.py`. Profiler was disabled; steps 1-10 are startup and
compile warmup, and steps 11-30 are the performance authority.

## Outputs

- Runtime: `combination_runs/self-npu-bf16-random-s30-b64-seq128-seed61-eager-inductor-performance-no-prof-skip10-nondet-9d360eea/cp8/cp8-r1`
- Artifact: `combination_artifacts/self-npu-bf16-random-s30-b64-seq128-seed61-eager-inductor-performance-no-prof-skip10-nondet-9d360eea/cp8/cp8-r1`
- Metrics: `combination_runs/self-npu-bf16-random-s30-b64-seq128-seed61-eager-inductor-performance-no-prof-skip10-nondet-9d360eea/cp8/cp8-r1/raw_metrics.jsonl`
- Trainer log: `combination_runs/self-npu-bf16-random-s30-b64-seq128-seed61-eager-inductor-performance-no-prof-skip10-nondet-9d360eea/cp8/cp8-r1/runtime.log`
- Input contract: `combination_artifacts/self-npu-bf16-random-s30-b64-seq128-seed61-eager-inductor-performance-no-prof-skip10-nondet-9d360eea/cp8/cp8-r1/attachments/input_contract.json`

## Source identity

- TorchTitan commit: `3327058390dad1f48fb61327b66d1bab3deac1a2`
- TorchTitanTurbo commit: `7343c9beda9a067700030a086dc3611ce9db3f54`
- TorchTitan test commit: `77f4e2eb11e4073155afe38592be9ec4505524aa`
- TorchTitanTurbo `graph_compat.py` SHA256: `f677d07bdd19db4d60e611be44adeeee0357515936bf0c1bfcba4b9fc3aa5762`

The source checkout was dirty because graph compatibility and experiment work
were under development. The content hash above pins the compatibility file used
by this matrix.
