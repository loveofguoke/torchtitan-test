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
    sequence_length=16,
    layers="all",
    components="all",
    component_execution="independent",
    titan_routed_expert_compute="model",
    # Exploratory captures must record the exact dirty source state instead of
    # requiring experiment-runner fixes to be committed first.
    allow_dirty=True,
    model=ParityModelConfig(
        vocab_size=2048,
        dim=256,
        layers=2,
        dense_layers=1,
        attention_heads=8,
        q_lora_rank=128,
        kv_lora_rank=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=64,
        dense_hidden_dim=1024,
        moe_hidden_dim=256,
        experts=8,
        shared_experts=1,
        router_top_k=2,
        expert_groups=1,
        limited_groups=1,
        route_scale=2.5,
        index_heads=4,
        index_head_dim=64,
        index_top_k=8,
        max_position_embeddings=128,
        rope_theta=1_000_000.0,
        rope_cache_max_seq_len=128,
    ),
    fixture_root="parity_fixtures",
    artifact_root="parity_artifacts",
    report_root="parity_reports",
    log_root="parity_reports/logs",
    fixture_name="fixture",
    report_name="npu1_vs_npu2_bf16_offline.html",
)


if __name__ == "__main__":
    run_offline_cli(CONFIG, __file__)
