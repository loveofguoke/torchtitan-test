# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
from types import SimpleNamespace

import pytest

from tests.glm5_2_common.cli import LoggedProcessError
from tests.glm5_2_common.topology import ParallelTopology
from tests.glm5_2_graph.config import GraphFeatureConfig
from tests.glm5_2_smoke.analyze_flex_compiler_artifacts import analyze
from tests.glm5_2_smoke.train_smoke import (
    _automatic_replay_captures,
    _completed,
    _contract,
    _default_log_rank,
    _run_topology,
)


def test_automatic_replay_selects_largest_qk_gradient_per_rank(tmp_path) -> None:
    capture_root = tmp_path / "nonfinite_replay"
    expected = []
    for rank, values in ((0, (0.1, 9.0)), (1, (float("inf"), 2.0))):
        rank_directory = capture_root / f"rank{rank}"
        for call, value in enumerate(values):
            call_directory = rank_directory / f"call{call:03d}"
            call_directory.mkdir(parents=True)
            finite_count = 0 if value == float("inf") else 1
            (call_directory / "actual_gradients.json").write_text(
                json.dumps(
                    {
                        "dq_QNH": {
                            "finite_count": finite_count,
                            "numel": 1,
                            "max_abs": None if value == float("inf") else value,
                        },
                        "dk_KNH": {
                            "finite_count": 1,
                            "numel": 1,
                            "max_abs": 0.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
        expected.append(rank_directory / ("call001" if rank == 0 else "call000"))

    assert _automatic_replay_captures(capture_root) == expected


def test_compiler_artifact_analysis_separates_ranks_and_replay(tmp_path) -> None:
    capture_root = tmp_path / "nonfinite_replay"
    rank_file = capture_root / "compiler/rank6/debug/output_code.py"
    replay_file = capture_root / "compiler/replay/trace/dedicated_log.log"
    rank_file.parent.mkdir(parents=True)
    replay_file.parent.mkdir(parents=True)
    rank_file.write_text(
        "kernel_name = 'triton_flex_attention_backward'\ngrid=(24,)\n",
        encoding="utf-8",
    )
    replay_file.write_text(
        "autotune flex_attention triton_flex_attention_backward\n",
        encoding="utf-8",
    )

    report = analyze(capture_root)

    assert report["capture"]["rank6"]["file_count"] == 1
    assert report["replay"]["file_count"] == 1
    assert "triton_flex_attention_backward" in report["kernel_name_sets"]["rank6"]
    assert "triton_flex_attention_backward" in report["kernel_name_sets"]["replay"]


@pytest.mark.parametrize(
    ("topology", "expected_rank"),
    (
        (ParallelTopology("pp8", 8, pipeline_parallel_degree=8), 7),
        (
            ParallelTopology(
                "fsdp2-pp4",
                8,
                data_parallel_shard_degree=2,
                pipeline_parallel_degree=4,
            ),
            6,
        ),
        (
            ParallelTopology(
                "fsdp2-tp2-pp2",
                8,
                data_parallel_shard_degree=2,
                tensor_parallel_degree=2,
                pipeline_parallel_degree=2,
            ),
            4,
        ),
    ),
)
def test_default_log_rank_owns_pipeline_loss(
    topology: ParallelTopology, expected_rank: int
) -> None:
    assert _default_log_rank(topology) == expected_rank


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
    environments: list[dict[str, str]] = []

    def run(command, **kwargs):
        commands.append(command)
        environments.append(kwargs["env"])
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
    assert environments[0]["LOG_RANK"] == "7"
    assert manifest["contract"]["log_rank"] == 7
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


def test_failed_smoke_runs_nonfinite_replay_before_raising(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        if len(commands) == 1:
            capture = (
                tmp_path
                / "smoke_runs/cp8/nonfinite_replay/rank0/call000"
            )
            capture.mkdir(parents=True)
            (capture / "actual_gradients.json").write_text(
                json.dumps(
                    {
                        name: {"finite_count": 1, "numel": 1, "max_abs": 0.1}
                        for name in ("dq_QNH", "dk_KNH")
                    }
                ),
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=9)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("tests.glm5_2_smoke.train_smoke.subprocess.run", run)
    with pytest.raises(LoggedProcessError):
        _run_topology(
            root=tmp_path,
            suite_root=tmp_path / "smoke_runs",
            device="npu",
            visible_devices="0,1,2,3,4,5,6,7",
            topology=ParallelTopology("cp8", 8, context_parallel_degree=8),
            steps=1,
            local_batch_size=8,
            global_batch_size=64,
            sequence_length=128,
            seed=61,
            module="glm5",
            config="glm5_debugmodel",
            nonfinite_diagnostics=True,
            diagnostic_rank="all",
            force=False,
        )

    replay_root = tmp_path / "smoke_runs/cp8/nonfinite_replay"
    assert len(commands) == 3
    assert commands[1][-1] == "--compact"
    assert commands[2][-1] == str(replay_root)
    assert (replay_root / "replay.log").is_file()
    status = json.loads((replay_root / "replay_status.json").read_text())
    assert status["status"] == "completed"


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


def test_npu_nonfinite_diagnostics_are_recorded_and_routed_to_run(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environments: list[dict[str, str]] = []

    def run(_command, **kwargs):
        environments.append(kwargs["env"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("tests.glm5_2_smoke.train_smoke.subprocess.run", run)
    run_directory = _run_topology(
        root=tmp_path,
        suite_root=tmp_path / "smoke_runs",
        device="npu",
        visible_devices="0,1,2,3,4,5,6,7",
        topology=ParallelTopology("cp8", 8, context_parallel_degree=8),
        steps=1,
        local_batch_size=8,
        global_batch_size=64,
        sequence_length=128,
        seed=61,
        module="glm5",
        config="glm5_debugmodel",
        nonfinite_diagnostics=True,
        diagnostic_compiler_cache="per-rank",
        diagnostic_rank=6,
        diagnostic_layer="layers.6.attention.inner_attention",
        force=False,
    )

    environment = environments[0]
    assert environment["TORCHTITAN_DIAGNOSE_NONFINITE"] == "1"
    assert environment["TORCHTITAN_NONFINITE_CAPTURE_RANK"] == "6"
    assert environment["TORCHTITAN_NONFINITE_CAPTURE_LAYER"] == (
        "layers.6.attention.inner_attention"
    )
    assert environment["TORCHTITAN_NONFINITE_DUMP_DIR"] == str(
        run_directory / "nonfinite_replay"
    )
    assert environment["TORCHTITAN_NONFINITE_COMPILER_CACHE"] == "per-rank"
    manifest = json.loads((run_directory / "manifest.json").read_text())
    assert manifest["contract"]["nonfinite_diagnostics"] == {
        "rank": 6,
        "layer": "layers.6.attention.inner_attention",
        "capture_schema_version": 4,
        "compiler_cache": "per-rank",
    }


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
