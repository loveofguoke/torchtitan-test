# Ascend GLM on TorchTitan 0.4.0

`worker.py` calls the Ascend GLM recipe and unmodified Trainer under torchrun.
It reuses the canonical topology registry; the additional two/four-card cases
are minimal diagnostic compositions, not changes to existing experiment presets.

Use the fixed TorchTitan 0.4.0 source and the Ascend integration branch through
`PYTHONPATH`. Set `GLM_TOPOLOGY`, `GLM_OUTPUT`, `GLM_REMAT=selective|full`, and
optionally `GLM_MTP=1`. Launch from the fixed TorchTitan root so its tokenizer
fixture is resolved. Every rank writes JSON with the explicit degrees, remat
policy, step count, full-precision metrics and verdict. A run passes only after
all expected rank files report two completed steps and finite metrics.

The worker does not disable remat, install dependencies, select graph backends,
or modify the production recipe. PP+MTP is rejected. Runs are short functional
checks, not cross-device accuracy, deterministic equivalence or performance tests.
