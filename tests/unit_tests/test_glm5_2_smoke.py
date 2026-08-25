# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
from types import SimpleNamespace

import pytest

from tests.glm5_2_common.full_dsa import (
    FULL_DSA_CONFIG,
    full_dsa_override_targets,
    merge_override_imports,
    resolve_full_dsa_config,
)
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


def test_smoke_passes_explicit_component_override(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("tests.glm5_2_smoke.train_smoke.subprocess.run", run)
    override = "torchtitanturbo.models.glm5.ops.sparse_mla.npu_sparse_mla"
    output = _run_topology(
        root=tmp_path,
        suite_root=tmp_path / "smoke_runs",
        device="npu",
        visible_devices="0",
        topology=ParallelTopology("single", 1),
        steps=1,
        local_batch_size=1,
        global_batch_size=1,
        sequence_length=8,
        seed=61,
        module="glm5",
        config="glm5_full_dsa_debugmodel",
        override_imports=override,
        force=False,
    )

    assert f"--override.imports={override}" in commands[0]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["contract"]["override_imports"] == override


def test_full_dsa_selection_is_explicit_and_device_specific() -> None:
    assert resolve_full_dsa_config(enabled=False, requested_config=None) == (
        "glm5_debugmodel"
    )
    assert resolve_full_dsa_config(enabled=True, requested_config=None) == (
        FULL_DSA_CONFIG
    )
    npu_targets = full_dsa_override_targets(device_type="npu", kernel="auto")
    assert "npu_triton_dsa_indexer" in npu_targets[0]
    assert "npu_triton_sparse_mla" in npu_targets[1]
    merged = merge_override_imports(
        "example.existing",
        full_dsa_override_targets(device_type="gpu", kernel="auto"),
    )
    assert merged is not None
    assert merged.startswith("example.existing,")
    assert "triton_dsa_indexer" in merged
    assert "triton_sparse_mla" in merged


def test_smoke_records_and_passes_extra_training_arguments(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("tests.glm5_2_smoke.train_smoke.subprocess.run", run)
    output = _run_topology(
        root=tmp_path,
        suite_root=tmp_path / "smoke_runs",
        device="npu",
        visible_devices="0",
        topology=ParallelTopology("single", 1),
        steps=1,
        local_batch_size=1,
        global_batch_size=1,
        sequence_length=8,
        seed=61,
        module="glm5",
        config=FULL_DSA_CONFIG,
        extra_args=("--loss.num_chunks=16",),
        force=False,
    )

    assert "--loss.num_chunks=16" in commands[0]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["contract"]["extra_args"] == ["--loss.num_chunks=16"]


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
