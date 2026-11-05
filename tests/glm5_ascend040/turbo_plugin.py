"""Import Turbo before collecting process-isolated A5 patch A/B tests."""

import json

import torch_npu  # noqa: F401
import torchtitanturbo  # noqa: F401
from torchtitanturbo.patch import get_patch_status

print("TURBO_PATCH_STATUS=" + json.dumps(get_patch_status(), sort_keys=True))
