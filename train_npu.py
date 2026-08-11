#!/usr/bin/env python3
"""Apply TorchTitanTurbo's NPU integration before starting TorchTitan."""

import torchtitanturbo  # noqa: F401
from torchtitan.train import main


if __name__ == "__main__":
    main()
