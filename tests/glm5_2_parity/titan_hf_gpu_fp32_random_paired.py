# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""TorchTitan FP32 versus HF FP32 paired execution on one GPU."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_parity.workflow import (  # noqa: E402
    PairedParityConfig,
    ParityModelConfig,
    run_paired_cli,
)


CONFIG = PairedParityConfig(
    actual_endpoint="titan:fp32",
    expected_endpoint="hf:fp32",
    device="cuda",
    visible_device="7",
    data_case="random",
    data_seed=61,
    model_seed=61,
    batch_size=2,
    sequence_length=16,
    layers="all",
    components="all",
    component_execution="independent",
    hf_routed_expert_compute="model",
    titan_routed_expert_compute="fp32",
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
    report_root="parity_reports",
    log_root="parity_reports/logs",
    report_name="paired.html",
)


if __name__ == "__main__":
    run_paired_cli(CONFIG, __file__)
