# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
from types import SimpleNamespace

import pytest

from tests.glm5_2_common.cli import LoggedProcessError
from tests.glm5_2_common.topology import ParallelTopology
from tests.glm5_2_graph.config import GraphFeatureConfig
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
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
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
    manifest = json.loads((tmp_path / "smoke_runs/pp8/manifest.json").read_text())
    assert manifest["elapsed_seconds"] >= 0
    assert manifest["started_at"] <= manifest["finished_at"]
    assert manifest["visible_devices"] == "0,1,2,3,4,5,6,7"
    assert "--training.disable_cuda_graphs" in commands[0]
    assert "--parallelism.num_pp_microbatches=8" in commands[0]
    assert capsys.readouterr().out.strip().splitlines()[-1] == (
        f"Runtime log: {(tmp_path / 'smoke_runs/pp8/runtime.log').resolve()}"
    )


def test_failed_smoke_exception_identifies_runtime_log(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "tests.glm5_2_smoke.train_smoke.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=9),
    )
    runtime_log = tmp_path / "smoke_runs/single/runtime.log"

    with pytest.raises(LoggedProcessError) as error:
        _run_topology(
            root=tmp_path,
            suite_root=tmp_path / "smoke_runs",
            device="gpu",
            visible_devices="0",
            topology=ParallelTopology("single", 1),
            steps=1,
            local_batch_size=1,
            global_batch_size=1,
            sequence_length=8,
            seed=61,
            module="glm5",
            config="glm5_debugmodel",
            force=False,
        )
    assert str(error.value).endswith(
        f"runtime log: {runtime_log.resolve()}"
    )


def test_suite_report_preserves_historical_unknown_time(tmp_path) -> None:
    from tests.glm5_2_smoke.train_smoke import _write_suite_report

    results = {"single": {"status": "passed"}, "tp2": {"status": "not_run"}}
    _write_suite_report(tmp_path, results)
    payload = json.loads((tmp_path / "summary.json").read_text())
    assert payload["results"] == results
    report = (tmp_path / "README.md").read_text()
    assert "unknown" in report
    assert "not_run" in report
    assert "single/runtime.log" in report


def test_npu_smoke_can_compile_each_topology(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    environments: list[dict[str, str]] = []

    def run(command, **kwargs):
        commands.append(command)
        environments.append(kwargs["env"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("tests.glm5_2_smoke.train_smoke.subprocess.run", run)
    _run_topology(
        root=tmp_path,
        suite_root=tmp_path / "smoke_runs",
        device="npu",
        visible_devices="0,1",
        topology=ParallelTopology("fsdp2", 2, data_parallel_shard_degree=2),
        steps=1,
        local_batch_size=1,
        global_batch_size=2,
        sequence_length=8,
        seed=61,
        module="glm5",
        config="glm5_debugmodel",
        graph=GraphFeatureConfig(mode="inductor", diagnostics=True),
        force=False,
    )

    assert "--compile.enable" in commands[0]
    assert "--compile.components=model" in commands[0]
    assert "--compile.backend=inductor" in commands[0]
    assert environments[0]["TORCH_LOGS"] == "graph_breaks,recompiles,dynamic"


def test_gpu_smoke_reserves_compiled_graph_interface(tmp_path) -> None:
    with pytest.raises(NotImplementedError, match="only NPU endpoints"):
        _run_topology(
            root=tmp_path,
            suite_root=tmp_path / "smoke_runs",
            device="gpu",
            visible_devices="0",
            topology=ParallelTopology("single", 1),
            steps=1,
            local_batch_size=1,
            global_batch_size=1,
            sequence_length=8,
            seed=61,
            module="glm5",
            config="glm5_debugmodel",
            graph=GraphFeatureConfig(mode="inductor"),
            force=False,
        )
