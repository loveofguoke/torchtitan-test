#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Run TorchTitan with fixed inputs and an optional checkpoint stop step.

The controlled/graceful split does not exit at an arbitrary optimizer step. It
keeps Trainer alive until TorchTitan has committed the requested checkpoint,
then returns through the normal training loop. Signal and rank-failure modes are
injected separately by the parent benchmark.
"""

from __future__ import annotations

import os


STOP_AFTER_ENV = "GLM5_CHECKPOINT_STOP_AFTER"


def _install_stop_after_step() -> None:
    value = os.environ.get(STOP_AFTER_ENV)
    if not value:
        return
    stop_after = int(value)
    if stop_after < 1:
        raise ValueError(f"{STOP_AFTER_ENV} must be positive")

    from torchtitan.trainer import Trainer

    from tests.glm5_2_checkpoint.fault_injection import (
        wait_for_completed_checkpoint,
    )

    original_should_continue = Trainer.should_continue_training

    def should_continue_until_checkpoint(self: Trainer) -> bool:
        return wait_for_completed_checkpoint(
            self, stop_after
        ) and original_should_continue(self)

    Trainer.should_continue_training = should_continue_until_checkpoint


def main() -> None:
    device_type = os.environ.get("TORCHTITAN_DEVICE", "gpu")
    if device_type == "npu":
        import torchtitanturbo  # noqa: F401
    elif device_type != "gpu":
        raise ValueError(f"unsupported TorchTitan device: {device_type!r}")

    from tests.glm5_2_precision.capture_metrics import (
        _install_jsonl_metrics_capture,
    )
    from tests.glm5_2_precision.fixed_token_dataloader import (
        install_fixed_token_dataloader,
    )

    install_fixed_token_dataloader()
    _install_jsonl_metrics_capture()
    from tests.glm5_2_checkpoint.fault_injection import (
        install_checkpoint_fault_injection,
    )

    install_checkpoint_fault_injection()
    _install_stop_after_step()

    from torchtitan.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
