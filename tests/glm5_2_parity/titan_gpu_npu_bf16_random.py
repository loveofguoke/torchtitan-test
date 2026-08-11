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
    sequence_length=16,
    layers="all",
    components="all",
    component_execution="independent",
    titan_routed_expert_compute="model",
    allow_dirty=False,
    model=ParityModelConfig(
        # Match TorchTitan's glm5_debugmodel; previous parity values are kept
        # beside each setting so the larger profile can be restored easily.
        vocab_size=2048,  # previous: 154880
        dim=256,  # previous: 2048
        layers=4,  # previous: 6
        dense_layers=1,  # previous: 1
        attention_heads=8,  # previous: 32
        q_lora_rank=128,  # previous/default: 768
        kv_lora_rank=64,  # previous/default: 192
        qk_nope_head_dim=32,  # previous/default: 192
        qk_rope_head_dim=32,  # previous/default: 64
        v_head_dim=64,  # previous/default: 256
        dense_hidden_dim=1024,  # previous: 4096
        moe_hidden_dim=256,  # previous: 768
        experts=8,  # previous: 32
        shared_experts=1,  # previous/default: 1
        router_top_k=2,  # previous/default: 8
        expert_groups=1,  # previous/default: 1
        limited_groups=1,  # previous/default: 1
        route_scale=2.5,  # previous/default: 2.5
        index_heads=4,  # previous/default: 32
        index_head_dim=64,  # previous/default: 128
        index_top_k=8,  # previous/default: 2048
        max_position_embeddings=128,  # previous/default: 1048576
        rope_theta=1_000_000.0,  # previous/default: 8_000_000.0
        rope_cache_max_seq_len=128,  # previous/default: 128
    ),
    fixture_root="parity_fixtures",
    artifact_root="parity_artifacts",
    report_root="parity_reports",
    log_root="parity_reports/logs",
    fixture_name="fixture",
    report_name="gpu_vs_npu_bf16_offline.html",
)


if __name__ == "__main__":
    run_offline_cli(CONFIG, __file__)
