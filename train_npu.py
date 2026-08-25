#!/usr/bin/env python3
"""Apply TorchTitanTurbo's NPU integration before starting TorchTitan."""

import os
import sys

import torchtitanturbo  # noqa: F401


# Cold graph compilation can delay a pipeline peer beyond the ordinary
# process-group timeout. This remains a launcher concern: append the isolated
# experiment override only when the user did not provide an explicit value.
comm_timeout = os.environ.get("TORCHTITAN_COMM_INIT_TIMEOUT_SECONDS")
if comm_timeout is not None:
    try:
        timeout_value = int(comm_timeout)
    except ValueError as error:
        raise ValueError(
            "TORCHTITAN_COMM_INIT_TIMEOUT_SECONDS must be a positive integer; "
            f"got {comm_timeout!r}"
        ) from error
    if timeout_value <= 0:
        raise ValueError(
            "TORCHTITAN_COMM_INIT_TIMEOUT_SECONDS must be a positive integer; "
            f"got {comm_timeout!r}"
        )
    if not any(
        argument.startswith("--comm.init_timeout_seconds")
        for argument in sys.argv
    ):
        sys.argv.append(f"--comm.init_timeout_seconds={timeout_value}")

from torchtitan.train import main  # noqa: E402


if __name__ == "__main__":
    main()
