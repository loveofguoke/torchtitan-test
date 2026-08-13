#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Apply TorchTitanTurbo NPU integration before creating a seed checkpoint."""

import torchtitanturbo  # noqa: F401
from torchtitan.train import main


if __name__ == "__main__":
    main()

