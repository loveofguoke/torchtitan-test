#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""TorchTitan GLM-5 BF16 parity across two healthy NPU devices."""

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
        name="npu1",
        endpoint="titan:bf16",
        device_type="npu",
        visible_device="1",
        visible_devices_env="ASCEND_RT_VISIBLE_DEVICES",
        artifact_name="npu1_capture",
    ),
    expected=OfflineEndpointConfig(
        name="npu2",
        endpoint="titan:bf16",
        device_type="npu",
        visible_device="2",
        visible_devices_env="ASCEND_RT_VISIBLE_DEVICES",
        artifact_name="npu2_capture",
    ),
    data_case="random",
    data_seed=61,
    model_seed=61,
    batch_size=2,
    sequence_length=128,
    layers="all",
    components="all",
    component_execution="independent",
    titan_routed_expert_compute="model",
    # Exploratory captures must record the exact dirty source state instead of
    # requiring experiment-runner fixes to be committed first.
    model=ParityModelConfig(),
    fixture_root="parity_fixtures",
    artifact_root="parity_artifacts",
    report_root="parity_reports",
    run_root="parity_runs",
    fixture_name="fixture",
    report_name="npu1_vs_npu2_bf16_offline.html",
)


if __name__ == "__main__":
    run_offline_cli(CONFIG, __file__)
