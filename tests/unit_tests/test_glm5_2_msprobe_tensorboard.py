from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from tests.glm5_2_precision import msprobe_tensorboard
from tests.glm5_2_precision.msprobe_tensorboard import (
    MSPROBE_CONFIG_PATH_ENV,
    MsprobeCaptureConfig,
    MsprobeParallelSpec,
    build_tensorboard_assets,
    install_trainer_capture,
    tensorboard_command,
    validate_debug_dump_directory,
    write_capture_config,
)


def _dump(directory: Path) -> Path:
    rank = directory / "step0" / "rank0"
    rank.mkdir(parents=True)
    (rank / "dump.json").write_text("{}\n", encoding="utf-8")
    (rank / "construct.json").write_text("{}\n", encoding="utf-8")
    (rank / "stack.json").write_text("{}\n", encoding="utf-8")
    return directory


def test_capture_config_supports_hierarchy_and_trend_views(tmp_path: Path) -> None:
    path = write_capture_config(
        tmp_path / "config.json",
        dump_path=tmp_path / "dump",
        config=MsprobeCaptureConfig(steps=(0, 3), ranks=(1,), level="mix"),
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["task"] == "statistics"
    assert payload["level"] == "mix"
    assert payload["step"] == [0, 3]
    assert payload["rank"] == [1]
    assert payload["statistics"]["summary_mode"] == "statistics"


def test_block_boundary_capture_uses_public_debug_tensor_mode(tmp_path: Path) -> None:
    config = MsprobeCaptureConfig(
        task="tensor",
        level="debug",
        ranks=(0,),
        block_boundaries=True,
        block_global_step=True,
        block_backward=True,
        parameter_state=True,
        router_state=True,
    )
    payload = config.payload(tmp_path / "dump")

    assert payload["task"] == "tensor"
    assert payload["level"] == "debug"
    assert "block_boundaries" not in payload
    assert "block_global_step" not in payload
    assert "block_backward" not in payload
    assert "parameter_state" not in payload
    assert "router_state" not in payload

    with pytest.raises(ValueError, match="task=tensor and level=debug"):
        MsprobeCaptureConfig(block_boundaries=True)
    with pytest.raises(ValueError, match="requires block-boundary capture"):
        MsprobeCaptureConfig(block_global_step=True)
    with pytest.raises(ValueError, match="requires global-step block capture"):
        MsprobeCaptureConfig(block_backward=True)
    with pytest.raises(ValueError, match="task=tensor and level=debug"):
        MsprobeCaptureConfig(parameter_state=True)
    with pytest.raises(ValueError, match="task=tensor and level=debug"):
        MsprobeCaptureConfig(router_state=True)


def test_debug_dump_validation_requires_saved_data(tmp_path: Path) -> None:
    rank = tmp_path / "step0" / "rank0"
    rank.mkdir(parents=True)
    (rank / "debug.json").write_text('{"data": {}}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="no saved msProbe debug tensors"):
        validate_debug_dump_directory(tmp_path)

    (rank / "debug.json").write_text(
        '{"data": {"block_00_output.forward.0": {}}}\n', encoding="utf-8"
    )
    assert validate_debug_dump_directory(tmp_path) == tmp_path


def test_trainer_capture_wraps_every_step_and_finalizes_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    events: list[object] = []

    class FakeDebugger:
        def __init__(self, *, config_path: str) -> None:
            events.append(("init", config_path))

        def start(self, *, model: object) -> None:
            events.append(("start", model))

        def stop(self) -> None:
            events.append("stop")

        def step(self) -> None:
            events.append("step")

    class FakeTrainer:
        model_parts = ["part-0", "part-1"]

        def train_step(self, *, fail: bool = False) -> str:
            events.append("train")
            if fail:
                raise ValueError("boom")
            return "result"

    msprobe_package = ModuleType("msprobe")
    msprobe_package.__path__ = []  # type: ignore[attr-defined]
    msprobe_pytorch = ModuleType("msprobe.pytorch")
    msprobe_pytorch.PrecisionDebugger = FakeDebugger  # type: ignore[attr-defined]
    torchtitan_package = ModuleType("torchtitan")
    torchtitan_package.__path__ = []  # type: ignore[attr-defined]
    torchtitan_trainer = ModuleType("torchtitan.trainer")
    torchtitan_trainer.Trainer = FakeTrainer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "msprobe", msprobe_package)
    monkeypatch.setitem(sys.modules, "msprobe.pytorch", msprobe_pytorch)
    monkeypatch.setitem(sys.modules, "torchtitan", torchtitan_package)
    monkeypatch.setitem(sys.modules, "torchtitan.trainer", torchtitan_trainer)

    install_trainer_capture(config)
    trainer = FakeTrainer()
    assert trainer.train_step() == "result"
    with pytest.raises(ValueError, match="boom"):
        trainer.train_step(fail=True)

    assert events == [
        ("init", str(config.resolve())),
        ("start", trainer.model_parts),
        "train",
        "stop",
        "step",
        ("start", trainer.model_parts),
        "train",
        "stop",
        "step",
    ]


def test_build_tensorboard_assets_runs_official_msprobe_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = _dump(tmp_path / "reference")
    candidate = _dump(tmp_path / "candidate")
    executable = tmp_path / "msprobe"
    executable.write_text("", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> None:
        commands.append(command)
        if command[1] == "graph_visualize":
            output = Path(command[command.index("-o") + 1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "compare.vis.db").write_bytes(b"graph")
        elif command[1] == "data2db":
            output = Path(command[command.index("--db") + 1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "dump_data.trend.db").write_bytes(b"trend")

    monkeypatch.setattr(msprobe_tensorboard, "_run", fake_run)
    output = build_tensorboard_assets(
        reference_dump=reference,
        candidate_dump=candidate,
        output=tmp_path / "tensorboard",
        reference_parallel=MsprobeParallelSpec(rank_size=1),
        candidate_parallel=MsprobeParallelSpec(rank_size=1),
        msprobe_executable=executable,
    )

    assert (output / "compare.vis.db").read_bytes() == b"graph"
    assert (output / "reference.trend.db").read_bytes() == b"trend"
    assert (output / "candidate.trend.db").read_bytes() == b"trend"
    assert commands[0][1:] == [
        "graph_visualize",
        "-tp",
        str((candidate / "step0").resolve()),
        "-gp",
        str((reference / "step0").resolve()),
        "-o",
        str(output.resolve()),
        "--rank_size",
        "1",
        "1",
        "--tp",
        "1",
        "1",
        "--pp",
        "1",
        "1",
        "--vpp",
        "1",
        "1",
    ]
    assert [command[1] for command in commands] == [
        "graph_visualize",
        "data2db",
        "data2db",
    ]
    manifest = json.loads(
        (output / "msprobe_tensorboard.json").read_text(encoding="utf-8")
    )
    assert manifest["tensorboard"]["tabs"] == [
        "GRAPH_ASCEND",
        "TREND ANALYZER",
    ]
    assert manifest["comparison"]["kind"] == "msprobe_parallel_merge"


def test_parallel_merge_rejects_single_vs_data_parallel(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="identical Data Parallelism"):
        build_tensorboard_assets(
            reference_dump=_dump(tmp_path / "reference"),
            candidate_dump=_dump(tmp_path / "candidate"),
            output=tmp_path / "tensorboard",
            reference_parallel=MsprobeParallelSpec(rank_size=1),
            candidate_parallel=MsprobeParallelSpec(rank_size=4, data_parallel=4),
        )


def test_tensorboard_command_is_local_by_default(tmp_path: Path) -> None:
    executable = tmp_path / "tensorboard"
    executable.write_text("", encoding="utf-8")

    local = tensorboard_command(
        tmp_path, tensorboard_executable=executable, port=7007
    )
    exposed = tensorboard_command(
        tmp_path, tensorboard_executable=executable, bind_all=True
    )

    assert local[-2:] == ["--port", "7007"]
    assert "--bind_all" not in local
    assert exposed[-1] == "--bind_all"


def test_msprobe_capture_is_separate_from_formal_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.glm5_2_precision import workflow

    topology = workflow.ParallelTopology("single", 1)
    endpoint = workflow.TrainingEndpoint("endpoint", "cuda", "0", topology)
    config = workflow.FormalExperimentConfig(
        name="msprobe",
        kind="self_consistency",
        reference=endpoint,
        candidate=endpoint,
        training=workflow.FormalTrainingConfig(steps=2, global_batch_size=2),
    )
    fixture = workflow._fixture_directory(tmp_path, config)
    checkpoint = fixture / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.bin").write_bytes(b"checkpoint")
    (fixture / "fixture.json").write_text(
        json.dumps(
            {
                "checkpoint_relative_path": "checkpoint",
                "checkpoint_sha256": workflow._directory_digest(checkpoint),
                "fixed_batches_relative_path": None,
                "fixed_batches_sha256": None,
            }
        ),
        encoding="utf-8",
    )

    def fake_run_process(command, *, root, environment, log_path) -> None:
        del root
        log_path.write_text("completed\n", encoding="utf-8")
        config_path = Path(environment[MSPROBE_CONFIG_PATH_ENV])
        dump_path = Path(
            json.loads(config_path.read_text(encoding="utf-8"))["dump_path"]
        )
        _dump(dump_path)
        assert "tests.glm5_2_precision.capture_metrics_with_msprobe" in command
        assert "--training.steps=1" in command

    monkeypatch.setattr(workflow, "_run_process", fake_run_process)
    monkeypatch.setattr(workflow, "_source_metadata", lambda root: {"root": str(root)})

    result = workflow.capture_msprobe_endpoint(
        tmp_path,
        config,
        role="reference",
        repeat=1,
        capture_config=MsprobeCaptureConfig(),
        force=False,
    )

    assert result.name.endswith("-msprobe")
    assert (result / "msprobe_capture.json").is_file()
    assert not (tmp_path / config.artifact_root).exists()
