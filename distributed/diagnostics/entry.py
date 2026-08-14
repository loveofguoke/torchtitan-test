#!/usr/bin/env python3
"""TorchTitan entry point with DDP step diagnostics installed."""

from __future__ import annotations

import os


device = os.environ.get("TORCHTITAN_DEVICE", "gpu")
if device == "npu":
    import train_npu  # noqa: F401
elif device != "gpu":
    raise ValueError(f"unsupported diagnostic backend: {device}")

from distributed.fixed_batches import (  # noqa: E402
    install_fixed_batch_dataloader,
)

from distributed.diagnostics.runtime import install

install_fixed_batch_dataloader()
install()

from torchtitan.train import main  # noqa: E402


if __name__ == "__main__":
    main()
