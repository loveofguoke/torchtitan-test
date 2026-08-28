# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
import os
from pathlib import Path
import signal
import sys

import pytest
import torch

from tests.glm5_2_checkpoint.checkpoint_benchmark import (
    CheckpointFixtureConfig,
    _boundary_memory_comparison,
    _checkpoint_command,
    _combine_contracts,
    _combine_recovered_contracts,
    _combine_recovered_metrics,
    _device_from_environment,
    _metrics_comparison,
    _precision_training,
    _resume_failure_location,
    _run_process,
    _selected_failure_modes,
)
from tests.glm5_2_checkpoint.fault_injection import (
    _missing_optimizer_state,
    _state_fingerprint,
)
from tests.glm5_2_checkpoint.topology_suite import (
    checkpoint_member_complete,
    checkpoint_member_paths,
    checkpoint_output_names,
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


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux process groups")
def test_faulted_process_group_is_killed_and_reaped(tmp_path: Path) -> None:
    ready_path = tmp_path / "ready"
    child_pid_path = tmp_path / "child-pid"
    script = (
        "import pathlib, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(60)'], start_new_session=True); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid)); "
        f"pathlib.Path({str(ready_path)!r}).touch(); "
        "time.sleep(60)"
    )

    outcome = _run_process(
        [sys.executable, "-c", script],
        root=tmp_path,
        environment=os.environ.copy(),
        log_path=tmp_path / "runtime.log",
        expected_failure=True,
        launcher_signal=signal.SIGKILL,
        ready_path=ready_path,
        timeout=10,
    )

    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    assert not Path(f"/proc/{child_pid}").exists()
    assert not outcome["cleanup"]["group_remaining"]
    assert outcome["cleanup"]["reaped_children"] >= 1


def test_state_fingerprint_handles_scalar_tensor() -> None:
    fingerprint = _state_fingerprint(
        {
            "float_step": torch.tensor(1.0),
            "integer_step": torch.tensor(2, dtype=torch.int64),
        }
    )

    assert fingerprint["tensor_leaf_count"] == 2
    assert fingerprint["tensor_elements"] == 2
    assert fingerprint["leaves"]["float_step"]["shape"] == []
    assert fingerprint["leaves"]["integer_step"]["shape"] == []


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


def test_checkpoint_state_fingerprint_localizes_changed_optimizer_leaf() -> None:
    saved = _state_fingerprint(
        {"layers.0.weight": {"exp_avg": torch.tensor([1.0, 2.0])}}
    )
    loaded = _state_fingerprint(
        {"layers.0.weight": {"exp_avg": torch.tensor([1.0, 3.0])}}
    )

    assert saved["sha256"] != loaded["sha256"]
    assert (
        saved["leaves"]["layers.0.weight/exp_avg"]["sha256"]
        != loaded["leaves"]["layers.0.weight/exp_avg"]["sha256"]
    )


def test_boundary_memory_comparison_reports_scope_and_leaf(tmp_path: Path) -> None:
    saved_directory = tmp_path / "saved"
    loaded_directory = tmp_path / "loaded"
    saved_directory.mkdir()
    loaded_directory.mkdir()
    saved_scope = _state_fingerprint({"step": torch.tensor(50.0)})
    loaded_scope = _state_fingerprint({"step": torch.tensor(0.0)})
    common = {"event": "state-fingerprint", "rank": 0, "step": 50}
    (saved_directory / "rank-0.jsonl").write_text(
        json.dumps({**common, "phase": "save", "scopes": {"optimizer": saved_scope}})
        + "\n",
        encoding="utf-8",
    )
    (loaded_directory / "rank-0.jsonl").write_text(
        json.dumps(
            {**common, "phase": "load", "scopes": {"optimizer": loaded_scope}}
        )
        + "\n",
        encoding="utf-8",
    )

    result = _boundary_memory_comparison(
        saved_directory,
        loaded_directory,
        step=50,
        expected_ranks={0},
    )

    assert not result["passed"]
    optimizer = result["ranks"][0]["scopes"]["optimizer"]
    assert optimizer["changed_leaves"] == ["step"]


def test_boundary_memory_comparison_accepts_identical_rank_local_state(
    tmp_path: Path,
) -> None:
    saved_directory = tmp_path / "saved"
    loaded_directory = tmp_path / "loaded"
    saved_directory.mkdir()
    loaded_directory.mkdir()
    scope = _state_fingerprint(
        {"state.layers.0.weight.exp_avg": torch.tensor([1.0, 2.0])}
    )
    common = {
        "event": "state-fingerprint",
        "rank": 3,
        "step": 50,
        "scopes": {"optimizer": scope},
    }
    (saved_directory / "rank-3.jsonl").write_text(
        json.dumps({**common, "phase": "save"}) + "\n",
        encoding="utf-8",
    )
    (loaded_directory / "rank-3.jsonl").write_text(
        json.dumps({**common, "phase": "load"}) + "\n",
        encoding="utf-8",
    )

    result = _boundary_memory_comparison(
        saved_directory,
        loaded_directory,
        step=50,
        expected_ranks={3},
    )

    assert result["passed"]
    assert result["ranks"][3]["scopes"]["optimizer"]["passed"]


def test_metrics_comparison_identifies_first_post_restart_difference(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.jsonl"
    candidate = tmp_path / "candidate.jsonl"

    def write_metrics(path: Path, losses: list[float]) -> None:
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "step": step,
                        "metrics": {
                            "loss_metrics/global_avg_loss": loss,
                            "loss_metrics/global_max_loss": loss,
                            "grad_norm": loss / 2,
                        },
                    }
                )
                + "\n"
                for step, loss in enumerate(losses, start=1)
            ),
            encoding="utf-8",
        )

    write_metrics(reference, [4.0, 3.0, 2.0, 1.0])
    write_metrics(candidate, [4.0, 3.0, 2.25, 1.0])

    result = _metrics_comparison(
        reference,
        candidate,
        exact=True,
        loss_absolute_limit=0.01,
        loss_relative_limit=0.01,
        grad_relative_limit=0.05,
        restored_step=2,
    )

    assert result["loss"]["first_bitwise_mismatch_step"] == 3
    assert result["loss"]["through_checkpoint"]["bitwise_equal"]
    assert not result["loss"]["after_restart"]["bitwise_equal"]


def test_resume_failure_location_prioritizes_immediate_load_mismatch() -> None:
    location = _resume_failure_location(
        split_state={"passed": True},
        boundary_memory={"passed": False},
        metrics={
            "loss": {"first_bitwise_mismatch_step": 51},
            "grad_norm": {"first_bitwise_mismatch_step": 51},
        },
        final_state={"passed": False},
        restored_step=50,
    )

    assert location["stage"] == "checkpoint-load"


def test_resume_failure_location_reports_first_post_load_step() -> None:
    location = _resume_failure_location(
        split_state={"passed": True},
        boundary_memory={"passed": True},
        metrics={
            "loss": {"first_bitwise_mismatch_step": 52},
            "grad_norm": {"first_bitwise_mismatch_step": 51},
        },
        final_state={"passed": False},
        restored_step=50,
    )

    assert location == {
        "stage": "post-load-training",
        "first_differing_step": 51,
        "description": (
            "All persisted state is exact immediately after load, but the "
            "training trajectory first differs at step 51. Investigate "
            "non-checkpointed runtime state or TP execution."
        ),
    }


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

    base = "cuda-bf16-s20-b64-seq128-seed61"
    assert config.storage_name.startswith(base + "-")
    assert len(config.storage_name.removeprefix(base + "-")) == 8


def test_checkpoint_topologies_share_one_suite_directory(tmp_path: Path) -> None:
    common = {
        "device": "npu",
        "precision": "bf16",
        "total_steps": 100,
        "split_step": 50,
        "local_batch_size": 8,
        "global_batch_size": 64,
        "sequence_length": 128,
        "seed": 61,
        "async_mode": "disabled",
        "comparison": "exact",
        "requested_failure_modes": ("all",),
        "run_tag": None,
    }
    single_suite, single_member = checkpoint_output_names(
        topology_slug="single", **common
    )
    fsdp_suite, fsdp_member = checkpoint_output_names(
        topology_slug="fsdp8", **common
    )

    assert single_suite == fsdp_suite
    assert single_member != fsdp_member
    assert not single_suite.startswith("ckpt-")
    single_run, _, _, _ = checkpoint_member_paths(
        tmp_path,
        suite_name=single_suite,
        topology_slug="single",
        member_name=single_member,
    )
    fsdp_run, _, _, _ = checkpoint_member_paths(
        tmp_path,
        suite_name=fsdp_suite,
        topology_slug="fsdp8",
        member_name=fsdp_member,
    )
    assert single_run.parent == fsdp_run.parent
    assert single_run.name == "single"
    assert fsdp_run.name == "fsdp8"


def test_checkpoint_member_is_complete_only_for_matching_pass_summary(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema": "torchtitan.glm5_2.checkpoint_fault_recovery",
                "schema_version": 2,
                "run_name": "expected",
                "fixture_generation_id": "generation-a",
                "passed": True,
            }
        ),
        encoding="utf-8",
    )

    assert checkpoint_member_complete(summary, member_name="expected")
    assert checkpoint_member_complete(
        summary,
        member_name="expected",
        fixture_generation_id="generation-a",
    )
    assert not checkpoint_member_complete(
        summary,
        member_name="expected",
        fixture_generation_id="generation-b",
    )
    assert not checkpoint_member_complete(summary, member_name="different")
    summary.write_text(
        json.dumps(
            {
                "schema": "torchtitan.glm5_2.checkpoint_fault_recovery",
                "schema_version": 2,
                "run_name": "expected",
                "passed": False,
            }
        ),
        encoding="utf-8",
    )
    assert not checkpoint_member_complete(summary, member_name="expected")
