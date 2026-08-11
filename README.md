# torchtitan-test

This repository keeps execution and parity testing outside the two runtime
packages:

- `torchtitan` contains device-independent model and trainer code.
- `torchtitanturbo` depends on `torchtitan` and owns NPU patches, acceleration,
  and profiling.
- `torchtitan-test` owns launch scripts, fixtures, captures, comparisons, and
  reports.

Install the repositories from source into the same environment. A GPU-only
environment needs TorchTitan and the test requirements:

```bash
pip install -e ../torchtitan
pip install -r requirements-test.txt
```

An NPU environment also needs the matching TorchTitanTurbo checkout and its
NPU runtime dependencies:

```bash
pip install -e ../torchtitan
pip install -e ../TorchTitanTurbo
pip install -r requirements-test.txt
```

## Training

`run_train.sh` selects NPU when `ASCEND_RT_VISIBLE_DEVICES` is set and otherwise
selects GPU. The existing commands therefore remain unchanged.

GPU:

```bash
export CUDA_VISIBLE_DEVICES=4
NGPU=1 LOG_RANK=0 MODULE=glm5 CONFIG=glm5_debugmodel ./run_train.sh
```

NPU:

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
# export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
NGPU=1 LOG_RANK=0 MODULE=glm5 CONFIG=glm5_debugmodel ./run_train.sh
```

The NPU entry imports `torchtitanturbo` before `torchtitan.train`, so Turbo's
NPU integration is active before the trainer is constructed. Use
`run_train_gpu.sh` or `run_train_npu.sh` when an explicit backend is preferable,
or set `TORCHTITAN_DEVICE=gpu|npu` when calling `run_train.sh`.

## GLM-5.2 parity

Run all parity commands from this repository root. Existing pytest commands and
environment variables are unchanged because this repository preserves the
`tests/unit_tests/test_glm5_parity.py` path. The framework imports the
source-installed `torchtitan` package; an NPU capture additionally imports
`torchtitanturbo` before selecting the device backend.

The offline workflow records both this test repository's commit and the
installed TorchTitan source commit. Comparison requires both commits to match,
while also recording the optional TorchTitanTurbo source and all package
versions in each artifact.

See [tests/glm5_2_parity/README.md](tests/glm5_2_parity/README.md) for paired,
fixture, capture, and CPU comparison commands.
