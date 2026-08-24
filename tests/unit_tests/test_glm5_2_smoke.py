# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
from types import SimpleNamespace

import pytest

from tests.glm5_2_common.topology import ParallelTopology
from tests.glm5_2_smoke.train_smoke import _completed, _contract, _run_topology


def test_completed_smoke_contract_survives_json_round_trip(tmp_path) -> None:
    contract = _contract(
        device="npu",
        topology=ParallelTopology(
            "tp2", 2, tensor_parallel_degree=2, extra_args=("--example",)
        ),
        steps=10,
        local_batch_size=8,
        global_batch_size=64,
        sequence_length=128,
        seed=61,
        module="glm5",
        config="glm5_debugmodel",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"status": "passed", "contract": contract}),
        encoding="utf-8",
    )

    assert _completed(tmp_path, contract)


def test_smoke_disables_trainer_cuda_graphs(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("tests.glm5_2_smoke.train_smoke.subprocess.run", run)
    _run_topology(
        root=tmp_path,
        suite_root=tmp_path / "smoke_runs",
        device="gpu",
        visible_devices="0,1,2,3,4,5,6,7",
        topology=ParallelTopology("pp8", 8, pipeline_parallel_degree=8),
        steps=1,
        local_batch_size=8,
        global_batch_size=64,
        sequence_length=8,
        seed=61,
        module="glm5",
        config="glm5_debugmodel",
        force=False,
    )

    assert len(commands) == 1
    assert "--training.disable_cuda_graphs" in commands[0]
    assert "--parallelism.num_pp_microbatches=8" in commands[0]
