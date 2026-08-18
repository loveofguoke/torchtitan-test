# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
from pathlib import Path

from tests.glm5_2_checkpoint.checkpoint_benchmark import (
    CheckpointFixtureConfig,
    _checkpoint_command,
    _combine_contracts,
    _device_from_environment,
    _precision_training,
)
from tests.glm5_2_precision.workflow import (
    TrainingEndpoint,
    standard_topologies,
    training_topology_plan,
)


def test_checkpoint_default_batch_supports_single_and_pp8() -> None:
    training = _precision_training(
        "bf16",
        steps=20,
        local_batch_size=8,
        global_batch_size=64,
        sequence_length=128,
        seed=61,
    )

    assert training_topology_plan(
        training, standard_topologies()["single"]
    ).gradient_accumulation_steps == 8
    assert training_topology_plan(
        training, standard_topologies()["pp8"]
    ).pipeline_microbatches == 8


def test_checkpoint_device_is_inferred_from_visibility(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)

    assert _device_from_environment(None) == (
        "cuda",
        "CUDA_VISIBLE_DEVICES",
        "0,1",
    )


def test_resume_contract_combines_process_phases(tmp_path: Path) -> None:
    phase_1 = tmp_path / "phase_1"
    phase_2 = tmp_path / "phase_2"
    phase_1.mkdir()
    phase_2.mkdir()
    (phase_1 / "rank-0.jsonl").write_text(
        json.dumps({"step": 1, "global_slots": [0]}) + "\n",
        encoding="utf-8",
    )
    (phase_2 / "rank-0.jsonl").write_text(
        json.dumps({"step": 2, "global_slots": [0]}) + "\n",
        encoding="utf-8",
    )

    records = _combine_contracts(phase_1, phase_2)["rank-0.jsonl"]
    assert [record["step"] for record in records] == [1, 2]


def test_checkpoint_phase_keeps_total_training_steps() -> None:
    training = _precision_training(
        "bf16",
        steps=20,
        local_batch_size=8,
        global_batch_size=64,
        sequence_length=128,
        seed=61,
    )
    command = _checkpoint_command(
        training=training,
        topology=standard_topologies()["single"],
        dump_folder=Path("run"),
        initial_checkpoint=Path("seed"),
        checkpoint_interval=10,
        async_mode="disabled",
    )

    assert "--training.steps=20" in command
    assert "--checkpoint.interval=10" in command
    assert "--checkpoint.no-last-save-model-only" in command


def test_checkpoint_fixture_name_is_compact() -> None:
    topology = standard_topologies()["single"]
    endpoint = TrainingEndpoint(
        name="gpu",
        device_type="cuda",
        visible_devices="7",
        topology=topology,
        repeats=1,
    )
    training = _precision_training(
        "bf16",
        steps=20,
        local_batch_size=8,
        global_batch_size=64,
        sequence_length=128,
        seed=61,
    )
    config = CheckpointFixtureConfig(
        name="checkpoint",
        kind="self_consistency",
        reference=endpoint,
        candidate=endpoint,
        training=training,
    )

    assert config.storage_name == "ckpt-cuda-bf16-s20-b64-q128-seed61"
