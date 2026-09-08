#!/usr/bin/env python3
"""Run the formal precision entry with diagnostic msProbe instrumentation."""

from __future__ import annotations

import os


def _synchronize_npu_isend_after_msprobe_hook() -> None:
    """Keep torch_npu's P2P send identity check valid after msProbe wrapping.

    torch_npu's batched P2P implementation distinguishes ``isend`` from
    ``irecv`` by function identity.  The statistics hook replaces PyTorch's
    public ``isend`` with an API wrapper, so torch_npu must observe that same
    wrapper or it selects the receive-only ``group_src`` keyword for a send.
    """

    import torch.distributed.distributed_c10d as torch_c10d
    import torch_npu.distributed.distributed_c10d as npu_c10d

    npu_c10d.isend = torch_c10d.isend


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
    if device_type == "npu":
        _synchronize_npu_isend_after_msprobe_hook()

    from tests.glm5_2_precision.capture_metrics import main as capture_main

    capture_main()


if __name__ == "__main__":
    main()
