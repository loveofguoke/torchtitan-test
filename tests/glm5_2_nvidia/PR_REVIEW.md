# GLM upstream review workload

Run from torchtitan-test with the source-installed TorchTitan review branch.
This reuses performance_benchmark.py, fixed checkpoints/token plans, normal
training, and standard nvidia_{runs,artifacts,reports}/performance/system paths.
No new launch entry point, model patch, or output hierarchy is introduced.

## Preparation and execution

The existing scaled-debug recipe supports context length 4096. Keep the global
batch and context fixed across CP1/2/4; this measures strong scaling rather than
increasing the global work with rank count. The recipe retains default remat.

```bash
export TORCHTITAN_CONFIG_FIRST_CLI=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python tests/glm5_2_nvidia/performance_benchmark.py --data \
  --config glm5_2_scaled_debugmodel --topologies single,cp2,cp4 \
  --sequence-length 4096 --local-batch-size 1 --global-batch-size 8 \
  --steps 30 --skip-steps 10 --workload indexer-review

for repeat in 1 2 3; do
  python tests/glm5_2_nvidia/performance_benchmark.py --probe --profiler-off \
    --config glm5_2_scaled_debugmodel --topologies single,cp2,cp4 \
    --sequence-length 4096 --local-batch-size 1 --global-batch-size 8 \
    --steps 30 --skip-steps 10 --workload indexer-review --replicate "$repeat" || break
done

python tests/glm5_2_nvidia/performance_benchmark.py --probe \
  --config glm5_2_scaled_debugmodel --topologies single,cp2,cp4 \
  --sequence-length 4096 --local-batch-size 1 --global-batch-size 8 \
  --steps 30 --skip-steps 10 --workload indexer-review --profile standard
```

Preparation is an independent action; it creates one topology-independent
checkpoint/token plan shared by the selected captures. Check its recorded
fixture path before capture. Existing complete runs skip; use another repeat
number for independent repetitions, not force over previous evidence. The GPU
must be available before launching these commands. An OOM or unsupported kernel
is a failure to investigate, not a reason to silently shrink the workload.

## What the result does and does not establish

- Profiler-off runs supply steady-state step time, throughput and reported peak
  memory; record repeat dispersion, device and exact software/source versions.
- Nsight supplies attribution. Inspect indexer sorting/ranking/gather work and
  CP collectives in the official timeline. Whole-step time alone is not indexer
  time, and overlapping kernel durations must not simply be summed as elapsed
  time. Profiling adds overhead; do not substitute its step time for the
  uninstrumented baseline.
- CP rank count changes neither global batch nor global context here. The
  current full-document indexer may still repeat query work across ranks.
- This is the current FP32 reference-indexer implementation, not a fused A/B
  switch. A BF16 fused comparison requires an explicitly implemented algorithm
  alternative and capability checks; do not monkey-patch the production model
  from this experiment or relabel a dtype-only experiment as fused.
- Random checkpoint/token-plan results are reduced-workload evidence. Real
  checkpoint top-k overlap, representative-data convergence and CI goldens need
  their own inputs and GPU-specific measurements; this workload does not
  manufacture those results.

Regression checks (on the test server):

```bash
python -m pytest tests/unit_tests/test_parity_native_interface.py \
  tests/unit_tests/test_parity_cli_config.py \
  tests/unit_tests/test_glm5_2_parity_workflow.py \
  tests/unit_tests/test_glm5_2_performance.py \
  tests/unit_tests/test_glm5_2_nvidia.py -q
```
