# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""TorchTitan GPU FP32 versus TorchTitan NPU FP32 with random data."""

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
        endpoint="titan:fp32",
        device_type="npu",
        visible_device="4",
        visible_devices_env="ASCEND_RT_VISIBLE_DEVICES",
        artifact_name="npu_capture",
    ),
    expected=OfflineEndpointConfig(
        name="gpu",
        endpoint="titan:fp32",
        device_type="cuda",
        visible_device="7",
        visible_devices_env="CUDA_VISIBLE_DEVICES",
        artifact_name="gpu_capture",
    ),
    data_case="random",
    data_seed=61,
    model_seed=61,
    batch_size=2,
    sequence_length=16,
    layers="all",
    components="all",
    component_execution="independent",
    titan_routed_expert_compute="model",
    allow_dirty=False,
    model=ParityModelConfig(
        vocab_size=154880,
        dim=2048,
        layers=6,
        dense_layers=1,
        attention_heads=32,
        dense_hidden_dim=4096,
        moe_hidden_dim=768,
        experts=32,
    ),
    fixture_root="parity_fixtures",
    artifact_root="parity_artifacts",
    report_root="parity_reports",
    log_root="parity_reports/logs",
    fixture_name="fixture",
    report_name="gpu_vs_npu_fp32_offline.html",
)


if __name__ == "__main__":
    run_offline_cli(CONFIG, __file__)
