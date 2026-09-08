# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""TorchTitan FP32 versus HF FP32 as two offline GPU captures."""

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
        name="titan-gpu",
        endpoint="titan:fp32",
        device_type="cuda",
        visible_device="7",
        visible_devices_env="CUDA_VISIBLE_DEVICES",
        artifact_name="titan_gpu_capture",
    ),
    expected=OfflineEndpointConfig(
        name="hf-gpu",
        endpoint="hf:fp32",
        device_type="cuda",
        visible_device="7",
        visible_devices_env="CUDA_VISIBLE_DEVICES",
        artifact_name="hf_gpu_capture",
    ),
    data_case="random",
    data_seed=61,
    model_seed=61,
    batch_size=2,
    sequence_length=128,
    layers="all",
    components="all",
    component_execution="independent",
    titan_routed_expert_compute="fp32",
    model=ParityModelConfig(),
    fixture_root="parity_fixtures",
    artifact_root="parity_artifacts",
    report_root="parity_reports",
    run_root="parity_runs",
    fixture_name="fixture",
    report_name="titan_vs_hf_gpu_fp32_offline.html",
)


if __name__ == "__main__":
    run_offline_cli(CONFIG, __file__)
