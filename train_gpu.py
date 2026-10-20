#!/usr/bin/env python3
"""Adapt the experiment CLI before starting TorchTitan on CUDA."""

import sys

from tests.glm5_2_common.config_first import translate_config_first_cli


sys.argv = translate_config_first_cli(sys.argv)

from torchtitan.train import main  # noqa: E402


if __name__ == "__main__":
    main()
