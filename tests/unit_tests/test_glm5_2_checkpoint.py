# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
from pathlib import Path

import pytest
import torch

from tests.glm5_2_common.cli import _replace_topology
from tests.glm5_2_checkpoint.checkpoint_benchmark import (
    CheckpointFixtureConfig,
    _checkpoint_command,
    _combine_contracts,
    _combine_recovered_contracts,
    _combine_recovered_metrics,
    _device_from_environment,
    _precision_training,
    _selected_failure_modes,
)
from tests.glm5_2_checkpoint.fault_injection import (
    _missing_optimizer_state,
)
from tests.glm5_2_precision.workflow import (
    TrainingEndpoint,
    standard_topologies,
    training_topology_plan,
)


def test_all_topology_runner_replaces_both_cli_forms() -> None:
    assert _replace_topology(
        ["--device", "cuda", "--topology", "all"], "fsdp8"
    ) == ["--device", "cuda", "--topology=fsdp8"]
    assert _replace_topology(
        ["--topology=all", "--force"], "cp8"
    ) == ["--force", "--topology=cp8"]


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


def test_recovered_contract_discards_work_after_loaded_checkpoint(
    tmp_path: Path,
) -> None:
    interrupted = tmp_path / "interrupted"
    resumed = tmp_path / "resumed"
    interrupted.mkdir()
    resumed.mkdir()
    (interrupted / "rank-0.jsonl").write_text(
        "".join(json.dumps({"step": step}) + "\n" for step in (1, 2, 3)),
        encoding="utf-8",
    )
    (resumed / "rank-0.jsonl").write_text(
        "".join(json.dumps({"step": step}) + "\n" for step in (3, 4)),
        encoding="utf-8",
    )

    records = _combine_recovered_contracts(
        interrupted, resumed, restored_step=2
    )["rank-0.jsonl"]

    assert [record["step"] for record in records] == [1, 2, 3, 4]


def test_recovered_metrics_discard_uncommitted_work_before_replay(
    tmp_path: Path,
) -> None:
    interrupted = tmp_path / "interrupted.jsonl"
    resumed = tmp_path / "resumed.jsonl"
    combined = tmp_path / "combined.jsonl"
    interrupted.write_text(
        "".join(json.dumps({"step": step}) + "\n" for step in (1, 2, 3)),
        encoding="utf-8",
    )
    resumed.write_text(
        "".join(json.dumps({"step": step}) + "\n" for step in (3, 4)),
        encoding="utf-8",
    )

    _combine_recovered_metrics(
        interrupted,
        resumed,
        combined,
        restored_step=2,
    )

    assert [
        json.loads(line)["step"]
        for line in combined.read_text(encoding="utf-8").splitlines()
    ] == [1, 2, 3, 4]


def test_recovered_metrics_reject_duplicate_steps_within_one_phase(
    tmp_path: Path,
) -> None:
    interrupted = tmp_path / "interrupted.jsonl"
    resumed = tmp_path / "resumed.jsonl"
    interrupted.write_text(
        "".join(json.dumps({"step": step}) + "\n" for step in (1, 2)),
        encoding="utf-8",
    )
    resumed.write_text(
        "".join(json.dumps({"step": step}) + "\n" for step in (3, 3, 4)),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="not strictly increasing"):
        _combine_recovered_metrics(
            interrupted,
            resumed,
            tmp_path / "combined.jsonl",
            restored_step=2,
        )


def test_all_failure_modes_add_rank_failures_only_for_distributed() -> None:
    assert _selected_failure_modes(["all"], world_size=1) == [
        "graceful",
        "sigint",
        "sigterm",
        "sigkill",
        "incomplete",
    ]
    assert _selected_failure_modes(["all"], world_size=8)[-2:] == [
        "rank-error",
        "rank-sigkill",
    ]


def test_checkpoint_wrapper_reports_only_missing_trainable_optimizer_state() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4, bias=False),
        torch.nn.Linear(4, 4, bias=False),
        torch.nn.Linear(4, 4, bias=False),
    )
    named_parameters = list(model.named_parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    optimizer.param_groups[0]["param_names"] = [
        name for name, _ in named_parameters
    ]
    model[2].weight.requires_grad_(False)
    model[0].weight.grad = torch.ones_like(model[0].weight)
    optimizer.step()

    container = type("Container", (), {"optimizers": [optimizer]})()
    checkpointer = type("Checkpointer", (), {"states": {"optimizer": container}})()

    assert _missing_optimizer_state(checkpointer) == ["1.weight"]


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


def test_checkpoint_phase_forwards_graph_arguments() -> None:
    training = _precision_training(
        "bf16",
        steps=20,
        local_batch_size=8,
        global_batch_size=64,
        sequence_length=128,
        seed=61,
        extra_args=("--compile.enable", "--compile.backend=inductor"),
    )
    command = _checkpoint_command(
        training=training,
        topology=standard_topologies()["single"],
        dump_folder=Path("run"),
        initial_checkpoint=Path("seed"),
        checkpoint_interval=10,
        async_mode="disabled",
    )

    assert "--compile.enable" in command
    assert "--compile.backend=inductor" in command


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
