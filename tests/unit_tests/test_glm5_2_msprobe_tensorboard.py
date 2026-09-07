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
    adamw_update_components,
    build_tensorboard_assets,
    install_trainer_capture,
    reconstruct_rmsnorm_weight_gradient,
    tensorboard_command,
    validate_debug_dump_directory,
    write_capture_config,
)


def _dump(directory: Path, rank_size: int = 1) -> Path:
    for rank_index in range(rank_size):
        rank = directory / "step0" / f"rank{rank_index}"
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
        optimizer_state=True,
        final_norm_state=True,
    )
    payload = config.payload(tmp_path / "dump")

    assert payload["task"] == "tensor"
    assert payload["level"] == "debug"
    assert "block_boundaries" not in payload
    assert "block_global_step" not in payload
    assert "block_backward" not in payload
    assert "parameter_state" not in payload
    assert "router_state" not in payload
    assert "optimizer_state" not in payload
    assert "final_norm_state" not in payload

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
    with pytest.raises(ValueError, match="requires parameter-state capture"):
        MsprobeCaptureConfig(task="tensor", level="debug", optimizer_state=True)
    with pytest.raises(ValueError, match="task=tensor and level=debug"):
        MsprobeCaptureConfig(final_norm_state=True)
    with pytest.raises(ValueError, match="requires final-norm capture"):
        MsprobeCaptureConfig(
            task="tensor",
            level="debug",
            final_norm_reduce_transition=True,
        )
    with pytest.raises(ValueError, match="requires reduce-transition capture"):
        MsprobeCaptureConfig(
            task="tensor",
            level="debug",
            final_norm_state=True,
            final_norm_pre_reduce_sync=True,
        )
    with pytest.raises(ValueError, match="requires reduce-transition capture"):
        MsprobeCaptureConfig(
            task="tensor",
            level="debug",
            final_norm_state=True,
            final_norm_sharded_grad_all_reduce=True,
        )
    with pytest.raises(ValueError, match="requires reduce-transition capture"):
        MsprobeCaptureConfig(
            task="tensor",
            level="debug",
            final_norm_state=True,
            final_norm_native_last_backward_sync=True,
        )
    with pytest.raises(ValueError, match="requires reduce-transition capture"):
        MsprobeCaptureConfig(
            task="tensor",
            level="debug",
            final_norm_state=True,
            final_norm_reset_group_forward_state=True,
        )
    with pytest.raises(ValueError, match="requires reduce-transition capture"):
        MsprobeCaptureConfig(
            task="tensor",
            level="debug",
            final_norm_state=True,
            final_norm_ungroup_fsdp_unit=True,
        )
    transition = MsprobeCaptureConfig(
        task="tensor",
        level="debug",
        final_norm_state=True,
        final_norm_reduce_transition=True,
        final_norm_pre_reduce_sync=True,
        final_norm_sharded_grad_all_reduce=True,
        final_norm_native_last_backward_sync=True,
        final_norm_reset_group_forward_state=True,
        final_norm_ungroup_fsdp_unit=True,
    )
    assert transition.final_norm_reduce_transition
    assert transition.final_norm_pre_reduce_sync
    assert transition.final_norm_sharded_grad_all_reduce
    assert transition.final_norm_native_last_backward_sync
    assert transition.final_norm_reset_group_forward_state
    assert transition.final_norm_ungroup_fsdp_unit


def test_final_norm_fsdp_ungroup_ablation_splits_only_target_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    norm = object()
    lm_head = object()
    other = object()
    calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    class FakeModel:
        enable_weight_tying = False

        def __init__(self) -> None:
            self.norm = norm
            self.lm_head = lm_head

    fsdp_module = ModuleType("torchtitan.distributed.fsdp")

    def fully_shard(
        module: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        calls.append((module, args, kwargs))
        return module

    fsdp_module.fully_shard = fully_shard  # type: ignore[attr-defined]
    parallelize_module = ModuleType("torchtitan.models.glm5.parallelize")

    def apply_fsdp_to_decoder(model: FakeModel) -> str:
        fsdp_module.fully_shard(  # type: ignore[attr-defined]
            [model.norm, model.lm_head], mesh="dp"
        )
        fsdp_module.fully_shard(other, mesh="dp")  # type: ignore[attr-defined]
        return "applied"

    parallelize_module.apply_fsdp_to_decoder = (  # type: ignore[attr-defined]
        apply_fsdp_to_decoder
    )
    torchtitan_package = ModuleType("torchtitan")
    torchtitan_package.__path__ = []  # type: ignore[attr-defined]
    distributed_package = ModuleType("torchtitan.distributed")
    distributed_package.__path__ = []  # type: ignore[attr-defined]
    models_package = ModuleType("torchtitan.models")
    models_package.__path__ = []  # type: ignore[attr-defined]
    glm5_package = ModuleType("torchtitan.models.glm5")
    glm5_package.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torchtitan", torchtitan_package)
    monkeypatch.setitem(sys.modules, "torchtitan.distributed", distributed_package)
    monkeypatch.setitem(sys.modules, "torchtitan.distributed.fsdp", fsdp_module)
    monkeypatch.setitem(sys.modules, "torchtitan.models", models_package)
    monkeypatch.setitem(sys.modules, "torchtitan.models.glm5", glm5_package)
    monkeypatch.setitem(
        sys.modules,
        "torchtitan.models.glm5.parallelize",
        parallelize_module,
    )
    monkeypatch.setenv(
        msprobe_tensorboard.MSPROBE_FINAL_NORM_UNGROUP_FSDP_UNIT_ENV,
        "1",
    )

    assert msprobe_tensorboard.install_final_norm_ungroup_fsdp_ablation()
    assert parallelize_module.apply_fsdp_to_decoder(FakeModel()) == "applied"  # type: ignore[attr-defined]
    assert [(call[0], call[2]) for call in calls] == [
        (norm, {"mesh": "dp"}),
        (lm_head, {"mesh": "dp"}),
        (other, {"mesh": "dp"}),
    ]
    assert fsdp_module.fully_shard is fully_shard  # type: ignore[attr-defined]
    assert not msprobe_tensorboard.install_final_norm_ungroup_fsdp_ablation()


def test_adamw_update_components_reconstruct_first_step() -> None:
    torch = pytest.importorskip("torch")
    initial = torch.tensor([2.0, -4.0])
    gradient = torch.tensor([0.25, -0.5])
    exp_avg = 0.1 * gradient
    exp_avg_sq = 0.001 * gradient.square()

    adaptive, decay, combined = adamw_update_components(
        initial,
        exp_avg,
        exp_avg_sq,
        step=1.0,
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.1,
    )

    torch.testing.assert_close(adaptive, torch.tensor([-0.01, 0.01]))
    torch.testing.assert_close(decay, torch.tensor([-0.002, 0.004]))
    torch.testing.assert_close(combined, torch.tensor([-0.012, 0.014]))


def test_reconstruct_rmsnorm_weight_gradient_from_output_boundary() -> None:
    torch = pytest.importorskip("torch")
    normalized = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    )
    weight = torch.tensor([2.0, 4.0])
    norm_output = normalized * weight
    grad_output = torch.tensor(
        [[[0.5, 1.0], [1.5, 2.0]], [[2.5, 3.0], [3.5, 4.0]]]
    )

    actual = reconstruct_rmsnorm_weight_gradient(
        norm_output,
        grad_output,
        weight,
    )

    expected = (normalized * grad_output).sum(dim=(0, 1))
    torch.testing.assert_close(actual, expected)


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
    assert manifest["schema_version"] == 3
    assert manifest["tensorboard"]["tabs"] == [
        "GRAPH_ASCEND",
        "TREND ANALYZER",
    ]
    assert manifest["comparison"]["kind"] == "msprobe_parallel_merge_compare"


def test_data_parallel_mismatch_builds_standalone_merged_graphs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = _dump(tmp_path / "reference")
    candidate = _dump(tmp_path / "candidate", rank_size=4)
    executable = tmp_path / "msprobe"
    executable.write_text("", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> None:
        commands.append(command)
        if command[1] == "graph_visualize":
            output = Path(command[command.index("-o") + 1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "build.vis.db").write_bytes(b"graph")
        else:
            output = Path(command[command.index("--db") + 1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "dump.trend.db").write_bytes(b"trend")

    monkeypatch.setattr(msprobe_tensorboard, "_run", fake_run)
    output = build_tensorboard_assets(
        reference_dump=reference,
        candidate_dump=candidate,
        output=tmp_path / "tensorboard",
        reference_parallel=MsprobeParallelSpec(rank_size=1),
        candidate_parallel=MsprobeParallelSpec(rank_size=4, data_parallel=4),
        msprobe_executable=executable,
    )

    assert (output / "reference.vis.db").is_file()
    assert (output / "candidate.vis.db").is_file()
    graph_commands = [
        command for command in commands if command[1] == "graph_visualize"
    ]
    assert len(graph_commands) == 2
    assert all("-gp" not in command for command in graph_commands)
    assert graph_commands[0][graph_commands[0].index("--rank_size") + 1] == "1"
    assert graph_commands[1][graph_commands[1].index("--rank_size") + 1] == "4"
    manifest = json.loads(
        (output / "msprobe_tensorboard.json").read_text(encoding="utf-8")
    )
    assert manifest["comparison"]["kind"] == "msprobe_parallel_merge_standalone"
    assert (
        "identical Data Parallelism"
        in manifest["comparison"]["fallback_reason"]
    )


def test_expert_parallel_builds_every_rank_as_standalone_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = _dump(tmp_path / "reference")
    candidate = _dump(tmp_path / "candidate", rank_size=4)
    executable = tmp_path / "msprobe"
    executable.write_text("", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> None:
        commands.append(command)
        if command[1] == "graph_visualize":
            output = Path(command[command.index("-o") + 1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "build.vis.db").write_bytes(b"graph")
        else:
            output = Path(command[command.index("--db") + 1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "dump.trend.db").write_bytes(b"trend")

    monkeypatch.setattr(msprobe_tensorboard, "_run", fake_run)
    output = build_tensorboard_assets(
        reference_dump=reference,
        candidate_dump=candidate,
        output=tmp_path / "tensorboard",
        reference_parallel=MsprobeParallelSpec(rank_size=1),
        candidate_parallel=MsprobeParallelSpec(
            rank_size=4, data_parallel=4, expert_parallel=4
        ),
        msprobe_executable=executable,
    )

    graph_commands = [
        command for command in commands if command[1] == "graph_visualize"
    ]
    assert len(graph_commands) == 5
    assert all("--rank_size" not in command for command in graph_commands)
    assert len(list(output.glob("*.vis.db"))) == 5
    manifest = json.loads(
        (output / "msprobe_tensorboard.json").read_text(encoding="utf-8")
    )
    assert manifest["comparison"]["kind"] == "msprobe_per_rank_standalone"
    assert "Expert Parallelism" in manifest["comparison"]["fallback_reason"]


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

    debug_directory = workflow._msprobe_run_directory(
        tmp_path,
        config,
        "reference",
        endpoint,
        1,
        MsprobeCaptureConfig(task="tensor", level="debug"),
    )
    assert debug_directory.name.endswith("-msprobe-tensor-debug")
    assert debug_directory != result

    legacy_directory = workflow._msprobe_run_directory(
        tmp_path, config, "candidate", endpoint, 1
    )
    legacy_directory.mkdir(parents=True)
    (legacy_directory / "msprobe_capture.json").write_text(
        json.dumps({"msprobe": {"task": "tensor", "level": "debug"}}),
        encoding="utf-8",
    )
    migrated_statistics_directory = workflow._msprobe_run_directory(
        tmp_path,
        config,
        "candidate",
        endpoint,
        1,
        MsprobeCaptureConfig(),
    )
    assert migrated_statistics_directory.name.endswith(
        "-msprobe-statistics-mix"
    )
