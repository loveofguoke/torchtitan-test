#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Apply TorchTitanTurbo NPU integration before creating a seed checkpoint."""

import os
import sys

if os.environ.get("TORCHTITAN_DEVICE") == "npu":
    import torchtitanturbo  # noqa: F401
from tests.glm5_2_common.config_first import translate_config_first_cli
sys.argv = translate_config_first_cli(sys.argv)
from torchtitan.train import main


if __name__ == "__main__":
    main()

