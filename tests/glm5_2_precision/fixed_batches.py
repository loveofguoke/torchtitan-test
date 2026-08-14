# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Compatibility imports for the shared distributed fixed-batch implementation."""

from distributed.fixed_batches import (
    FIXED_BATCHES_ENV,
    TENSOR_KEYS,
    FixedGlobalBatchDataLoader,
    install_fixed_batch_dataloader,
    materialize_fixed_batches,
)


__all__ = [
    "FIXED_BATCHES_ENV",
    "TENSOR_KEYS",
    "FixedGlobalBatchDataLoader",
    "install_fixed_batch_dataloader",
    "materialize_fixed_batches",
]
