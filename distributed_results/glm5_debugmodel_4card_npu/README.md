# Four-card GLM-5 debugmodel NPU bundle

This directory contains the fixed inputs and completed NPU captures for the
15-scenario distributed comparison matrix. It is deduplicated for Git: the
identical seed checkpoint, dataset, and tokenizer assets are stored once, while
each scenario keeps its own DCP metadata, fixture manifest, and prepare log.

From the `torchtitan-test` repository root on the GPU machine, restore the
normal workflow layout:

```bash
python distributed_results/glm5_debugmodel_4card_npu/materialize.py \
  --output distributed_runs/glm5_debugmodel_4card
```

Materialization refuses to overwrite an existing output directory. It uses
hard links when supported (falling back to copies) and validates every fixture
and successful capture against the original SHA-256 digests.

Then run the GPU matrix:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m distributed.suite \
  --suite distributed/scenarios/glm5_debugmodel_4card_suite.json \
  --output-root distributed_runs/glm5_debugmodel_4card \
  --skip-existing \
  --continue-on-error \
  capture --backend gpu
```

The two repaired TP+EP captures are installed under their standard `_npu.json`
names. Their original failure and retry records remain under
`captures/history/`. Runtime directories are intentionally omitted because all
metrics needed for offline comparison are embedded in the capture JSON files.
