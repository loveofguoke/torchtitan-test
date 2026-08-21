# GLM-5.2 training smoke tests

This suite verifies that the current source-installed TorchTitan and
TorchTitanTurbo versions can launch short single-device and distributed GLM
training jobs. It reuses the canonical topology definitions in
`tests/glm5_2_common`.

Export device visibility once. Device detection is automatic:

```bash
# GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# NPU, in the NPU environment
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

Run the default single-device check:

```bash
python tests/glm5_2_smoke/train_smoke.py
```

Run one distributed topology or a subset:

```bash
python tests/glm5_2_smoke/train_smoke.py --topology fsdp8
python tests/glm5_2_smoke/train_smoke.py --topologies ddp8,fsdp8,tp8
```

Run every topology of at most eight ranks:

```bash
python tests/glm5_2_smoke/train_smoke.py --topology all
```

Successful topologies are skipped on the next invocation. Incomplete output is
preserved with a `.failed-<timestamp>` suffix and retried. Use `--force` to
replace successful output as well.

Every run writes `runtime.log`, `manifest.json`, and TorchTitan output below:

```text
runs/smoke/<device-and-training-config>/<topology>/
```

Direct `run_train.sh` invocations also tee complete terminal output to:

```text
runs/train/<module-config-timestamp-pid>/runtime.log
```

Set `TORCHTITAN_RUN_LOG` when a caller needs an exact log path.
