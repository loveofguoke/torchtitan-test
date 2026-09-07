#!/usr/bin/env python3
"""Run the formal precision entry with diagnostic msProbe instrumentation."""

from __future__ import annotations

import os


def main() -> None:
    device_type = os.environ.get("TORCHTITAN_DEVICE", "gpu")
    if device_type == "npu":
        # Install NPU compatibility before msProbe imports torch_npu internals.
        import torchtitanturbo  # noqa: F401

    from tests.glm5_2_precision.msprobe_tensorboard import (
        install_final_norm_ungroup_fsdp_ablation,
        install_trainer_capture,
    )

    install_final_norm_ungroup_fsdp_ablation()
    install_trainer_capture()

    from tests.glm5_2_precision.capture_metrics import main as capture_main

    capture_main()


if __name__ == "__main__":
    main()
