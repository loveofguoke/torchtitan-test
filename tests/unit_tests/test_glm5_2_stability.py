# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
from pathlib import Path

import pytest

from tests.glm5_2_precision.workflow import standard_topologies, training_topology_plan
from tests.glm5_2_stability.stability_benchmark import (
    _command,
    _completed_stability_member,
    _device_from_environment,
    _precision_training,
    _stability_output_names,
    _write_report,
)


def test_default_stability_batch_supports_single_and_pp8() -> None:
    training = _precision_training(
        "bf16",
        steps=20_000,
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


def test_stability_device_is_inferred_from_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert _device_from_environment(None) == (
        "npu",
        "ASCEND_RT_VISIBLE_DEVICES",
        "4",
    )


def test_stability_report_uses_portable_run_name(tmp_path: Path) -> None:
    report = tmp_path / "stability-npu-single-bf16-s20000.html"
    summary = tmp_path / "stability-npu-single-bf16-s20000.json"
    payload = {
        "run_name": "stability-npu-single-bf16-s20000",
        "status": "PASS",
        "device": "npu",
        "metric_ranges": {
            "loss": {"minimum": 1.0, "maximum": 2.0, "final": 1.1}
        },
        "command": ["torchrun", "--nproc_per_node=1"],
    }

    _write_report(report_path=report, summary_path=summary, summary=payload)

    assert report.is_file()
    assert json.loads(summary.read_text(encoding="utf-8"))["status"] == "PASS"


def test_stability_command_forwards_graph_arguments() -> None:
    training = _precision_training(
        "bf16",
        steps=20,
        local_batch_size=8,
        global_batch_size=64,
        sequence_length=128,
        seed=61,
        extra_args=("--compile.enable", "--compile.backend=inductor"),
    )
    command = _command(
        training=training,
        topology=standard_topologies()["single"],
        dump_folder=Path("run"),
        initial_checkpoint=Path("fixture/checkpoint"),
    )

    assert "--compile.enable" in command
    assert "--compile.backend=inductor" in command
    assert "--checkpoint.load_only" in command
    assert "--checkpoint.initial_load_path=fixture/checkpoint" in command


def test_stability_topologies_share_one_suite_name() -> None:
    common = {
        "device": "cuda",
        "precision": "bf16",
        "steps": 20_000,
        "local_batch_size": 8,
        "global_batch_size": 64,
        "sequence_length": 128,
        "seed": 61,
        "run_tag": None,
    }
    single_suite, single_member = _stability_output_names(
        topology_slug="single", **common
    )
    fsdp_suite, fsdp_member = _stability_output_names(
        topology_slug="fsdp8", **common
    )

    assert single_suite == fsdp_suite
    assert single_member != fsdp_member
    assert not single_suite.startswith("stability-")


def test_only_passed_stability_summary_is_complete(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    payload = {
        "schema": "torchtitan.glm5_2.stability",
        "schema_version": 1,
        "run_name": "stability-member",
        "status": "PASS",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _completed_stability_member(path, member_name="stability-member")

    payload["status"] = "INSUFFICIENT_DURATION"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert not _completed_stability_member(path, member_name="stability-member")
