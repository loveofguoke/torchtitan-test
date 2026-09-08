# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""TorchTitan GPU BF16 versus TorchTitan NPU BF16 with random data."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_parity.workflow import (  # noqa: E402
    OfflineEndpointConfig,
    OfflineParityConfig,
    ParityModelConfig,
    run_offline_cli,
)


CONFIG = OfflineParityConfig(
    actual=OfflineEndpointConfig(
        name="npu",
        endpoint="titan:bf16",
        device_type="npu",
        visible_device="4",
        visible_devices_env="ASCEND_RT_VISIBLE_DEVICES",
        artifact_name="npu_capture",
    ),
    expected=OfflineEndpointConfig(
        name="gpu",
        endpoint="titan:bf16",
        device_type="cuda",
        visible_device="7",
        visible_devices_env="CUDA_VISIBLE_DEVICES",
        artifact_name="gpu_capture",
    ),
    data_case="random",
    data_seed=61,
    model_seed=61,
    batch_size=2,
    # sequence_length=16,
    sequence_length=64,
    layers="all",
    components="all",
    component_execution="independent",
    titan_routed_expert_compute="model",
    model=ParityModelConfig(),
    fixture_root="parity_fixtures",
    artifact_root="parity_artifacts",
    report_root="parity_reports",
    log_root="parity_reports/logs",
    fixture_name="fixture",
    report_name="gpu_vs_npu_bf16_seqlen64_offline.html",
)


if __name__ == "__main__":
    run_offline_cli(CONFIG, __file__)
