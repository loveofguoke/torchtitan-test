# Retained failed performance attempts

Failures are evidence. They are not deleted or silently replaced by a later
successful run.

## DDP4 launcher environment mismatch

The first DDP4 attempt invoked the conda Python executable without putting the
conda environment's `bin` directory first on `PATH`. The workflow consequently
resolved `/usr/local/python3.11.15/bin/torchrun`, loaded the system TorchTitan,
and failed while TorchTitanTurbo imported `RoutedExperts`.

Failed raw run:

`performance_runs/npu-ddp4-bf16-s20-l8-b64-seq128-seed61-distributed-profiler-off-r1-5da7b9c5.failed-20260824-163342`

The successful retry used the same driver arguments with:

```bash
PATH=/root/miniconda3/envs/torchtitan-0803/bin:$PATH \
python tests/glm5_2_performance/profiler_benchmark.py \
  --probe --device npu --topology ddp4 --preset distributed \
  --profiler-off --visible-devices 1,2,3,4 \
  --steps 20 --skip-steps 8 --warmup-steps 2 --active-steps 3 \
  --replicate 1
```

This failure led to recording `PATH` and `PYTHONPATH` in every subsequent
`command_history.jsonl`.

## FSDP4 profiler communicator and physical NPU4

Two all-rank profiler captures failed before the first training step. The
profiler-off FSDP4 run and the DDP4 profiler run had already succeeded.

Attempt 1 mapped logical ranks 0–3 to physical NPU1,2,3,4. Physical NPU4 was
rank3 and was the only rank not connected to the communicator. HCCL reported
`ERR02200`, `EI0015`, and `List of unconnected ranks: "[3,]"`.

- Evidence:
  `npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r1-965e89d3/`
- Command environment: `ASCEND_RT_VISIBLE_DEVICES=1,2,3,4`

Attempt 2 mapped ranks to physical NPU4,5,6,7. Physical NPU4 moved to rank0
and again failed to enter the communicator; ranks 1–3 timed out waiting for
rank0's unique ID after 300 seconds.

- Evidence:
  `npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-49400b80/`
- Command environment: `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7`

Attempt 3 excluded physical NPU4 and mapped ranks to NPU1,2,3,5. It completed
training, offline parsing, cluster analysis, all-rank bottleneck analysis, and
free analysis.

- Evidence:
  `npu-fsdp4-bf16-s20-l8-b64-seq128-seed61-distributed-r3-d3b92684/`
- Command environment: `ASCEND_RT_VISIBLE_DEVICES=1,2,3,5`

The failure therefore follows physical NPU4, not logical rank. This establishes
a topology-specific HCCL initialization problem involving that card; it does
not yet establish the exact failing HCCS lane. Do not increase the timeout and
call the issue fixed. Inspect HCCL plog and link diagnostics before including
NPU4 in formal FSDP measurements.

## Earlier BF16 reduction configuration rejection

The initial performance-only BF16 reduction attempt was rejected by the public
TorchTitan config metadata before training started. It is retained at:

`performance_runs/npu-ddp2-bf16-s30-l8-b64-seq128-seed61-distributed-reduce-bf16-profiler-off-051ff000.failed-20260824-155758`

The later harness change widens only the capture process's runtime dataclass
metadata and leaves the generic Trainer and installed TorchTitan source
unchanged.

