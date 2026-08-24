# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from dataclasses import asdict, replace
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest
import torch

from tests.glm5_2_common.topology import training_command_args
from tests.glm5_2_precision.artifacts import (
    PrecisionArtifactError,
    PrecisionArtifactReader,
    PrecisionArtifactWriter,
    TrainingMetric,
)
from tests.glm5_2_precision.generate_token_plan import (
    _build_source_dataloader,
    _reshape_source_batch,
)
from tests.glm5_2_precision.migrate_legacy_outputs import compact_scenario_name
from tests.glm5_2_precision.report import (
    compare_and_write_report,
    precision_report_filename,
)
from tests.glm5_2_precision.self_consistency_suite import compare_suite
from tests.glm5_2_precision.standards import (
    AnyOfErrorLimit,
    MigrationStandard,
    PrecisionStandard,
    QuantileLimit,
    SelfConsistencyStandard,
    compute_error_metrics,
    evaluate_exact_pair,
    evaluate_migration_pair,
)
from tests.glm5_2_precision.token_data import (
    INPUT_MAPPING,
    TOKEN_PLAN_SCHEMA,
    TOKEN_PLAN_VERSION,
    batches_per_optimizer_step,
    global_slots_for_batch,
    load_token_plan,
    sample_digest,
    sample_digest_v2,
    sha256_file,
    step_digest,
    step_series_digest,
    validate_runtime_input_contract,
)
from tests.glm5_2_precision.workflow import (
    FormalExperimentConfig,
    FormalTrainingConfig,
    ParallelTopology,
    TrainingEndpoint,
    _completed_capture_can_be_finalized,
    _endpoint_from_process_environment,
    _fixture_endpoint_from_environment,
    standard_topologies,
    training_topology_plan,
)


def test_completed_capture_can_be_finalized_without_training(tmp_path: Path) -> None:
    metrics_path = tmp_path / "raw_metrics.jsonl"
    contract = tmp_path / "input-contract"
    contract.mkdir()
    (contract / "summary.json").write_text(
        json.dumps({"valid": True}) + "\n", encoding="utf-8"
    )
    records = [
        {
            "step": step,
            "metrics": {
                "loss_metrics/global_avg_loss": 1.0,
                "loss_metrics/global_max_loss": 1.0,
                "grad_norm": 0.5,
            },
        }
        for step in (1, 2)
    ]
    metrics_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    assert _completed_capture_can_be_finalized(
        metrics_path,
        contract,
        expected_steps=2,
    )
    assert not _completed_capture_can_be_finalized(
        metrics_path,
        contract,
        expected_steps=3,
    )


def test_token_plan_source_uses_current_torchtitan_dataloader_contract() -> None:
    class RecordingDataLoaderConfig:
        def __init__(self) -> None:
            self.kwargs = {}

        def build(self, **kwargs):
            self.kwargs = kwargs
            return "dataloader"

    dataloader_config = RecordingDataLoaderConfig()
    tokenizer = object()
    result = _build_source_dataloader(
        SimpleNamespace(dataloader=dataloader_config),
        tokenizer=tokenizer,
        global_batch_size=64,
        sequence_length=128,
    )

    assert result == "dataloader"
    assert dataloader_config.kwargs == {
        "dp_world_size": 1,
        "dp_rank": 0,
        "tokenizer": tokenizer,
        "max_context_length": 128,
        "num_tokens_per_batch": 8192,
    }


def test_token_plan_source_reshapes_flat_torchtitan_batches() -> None:
    inputs_T = torch.arange(12)
    labels_T = torch.arange(1, 13)
    positions_T = torch.arange(12) % 4

    inputs_BL, labels_BL, positions_BL = _reshape_source_batch(
        {"input": inputs_T, "positions": positions_T},
        labels_T,
        global_batch_size=3,
        sequence_length=4,
    )

    assert tuple(inputs_BL.shape) == (3, 4)
    assert tuple(labels_BL.shape) == (3, 4)
    assert tuple(positions_BL.shape) == (3, 4)
    assert torch.equal(inputs_BL.reshape(-1), inputs_T)
    assert torch.equal(labels_BL.reshape(-1), labels_T)
    assert torch.equal(positions_BL.reshape(-1), positions_T)


def test_documented_error_formulas() -> None:
    metrics = compute_error_metrics([1.0, 2.0], [0.5, 3.0])

    assert metrics.normal == pytest.approx(-0.25)
    assert metrics.absolute == pytest.approx(0.75)
    assert metrics.relative_normal == pytest.approx(0.0)
    assert metrics.relative_absolute == pytest.approx(0.5)
    assert metrics.max_absolute == pytest.approx(1.0)


def test_first_loss_is_evaluated_before_warmup() -> None:
    evaluation = evaluate_migration_pair(
        reference_loss={1: 1.0, 2: 1.0, 3: 1.0},
        candidate_loss={1: 1.1, 2: 1.0, 3: 1.0},
        reference_grad_norm={1: 1.0, 2: 1.0, 3: 1.0},
        candidate_grad_norm={1: 1.0, 2: 1.0, 3: 1.0},
        standard=MigrationStandard(
            first_loss=AnyOfErrorLimit(absolute=0.05),
            all_loss=AnyOfErrorLimit(absolute=0.0),
            warmup_steps=1,
            minimum_observations=3,
        ),
    )

    first_loss = next(item for item in evaluation.criteria if item.name == "First loss")
    assert not first_loss.passed
    assert evaluation.loss.metrics.absolute == 0.0


def test_loss_quantiles_are_diagnostic_only() -> None:
    evaluation = evaluate_migration_pair(
        reference_loss={1: 1.0, 2: 1.0},
        candidate_loss={1: 1.001, 2: 1.001},
        reference_grad_norm={1: 1.0, 2: 1.0},
        candidate_grad_norm={1: 1.0, 2: 1.0},
        standard=MigrationStandard(
            loss_quantiles=(QuantileLimit(0.99, 0.0),),
            minimum_observations=2,
        ),
    )

    assert evaluation.passed
    assert all("P99" not in item.name for item in evaluation.criteria)


def test_exact_self_consistency_uses_float_bits() -> None:
    exact = evaluate_exact_pair(
        reference_loss={1: 1.0},
        candidate_loss={1: 1.0},
        reference_grad_norm={1: 2.0},
        candidate_grad_norm={1: 2.0},
    )
    changed = evaluate_exact_pair(
        reference_loss={1: 1.0},
        candidate_loss={1: 1.0 + 1e-12},
        reference_grad_norm={1: 2.0},
        candidate_grad_norm={1: 2.0},
    )

    assert exact.passed
    assert not changed.passed


def _metrics(loss_offset: float = 0.0) -> tuple[TrainingMetric, ...]:
    return tuple(
        TrainingMetric(
            step=step,
            loss=2.0 - step * 0.1 + loss_offset,
            global_max_loss=2.0 - step * 0.1 + loss_offset,
            grad_norm=1.0 + step * 0.01,
            extras={"lr": 0.001},
        )
        for step in range(1, 5)
    )


def test_artifact_checksums(tmp_path: Path) -> None:
    path = PrecisionArtifactWriter(tmp_path / "artifact").write(
        metadata={"role": "reference"},
        training_contract={"seed": 61},
        metrics=_metrics(),
    )
    reader = PrecisionArtifactReader(path)
    assert reader.loss_series()[1] == pytest.approx(1.9)

    with (path / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("corruption\n")
    with pytest.raises(PrecisionArtifactError, match="checksum mismatch"):
        PrecisionArtifactReader(path)


def test_topology_validates_rank_product() -> None:
    topology = ParallelTopology(
        "fsdp2-tp4", 8, data_parallel_shard_degree=2, tensor_parallel_degree=4
    )
    assert "--parallelism.data_parallel_shard_degree=2" in topology.command_args()
    assert "--parallelism.tensor_parallel_degree=4" in topology.command_args()
    assert "--parallelism.spmd_backend=partial_dtensor" in topology.command_args()
    with pytest.raises(ValueError, match="dense ranks"):
        ParallelTopology("invalid", 8, data_parallel_shard_degree=2)


def test_tp_ep_topology_keeps_sequence_parallel_enabled() -> None:
    pure_tp = ParallelTopology("tp8", 8, tensor_parallel_degree=8)
    tp_ep = ParallelTopology(
        "fsdp2-tp4-ep8",
        8,
        data_parallel_shard_degree=2,
        tensor_parallel_degree=4,
        expert_parallel_degree=8,
    )

    assert "--parallelism.no-enable-sequence-parallel" in pure_tp.command_args()
    assert "--parallelism.no-enable-sequence-parallel" not in tp_ep.command_args()


def test_training_arguments_translate_sample_batches_to_token_budgets() -> None:
    topology = ParallelTopology(
        "pp4", 4, pipeline_parallel_degree=4,
        pipeline_parallel_microbatch_size=1,
    )
    args = training_command_args(
        local_batch_size=8,
        global_batch_size=64,
        sequence_length=128,
        topology=topology,
    )
    assert "--training.num_tokens_per_microbatch_per_dp_rank=128" in args
    assert "--training.num_tokens_per_train_step=8192" in args
    assert "--training.max_context_length=128" in args
    assert "--parallelism.num_pp_microbatches=8" in args


def test_training_arguments_reject_an_undersized_pipeline_batch() -> None:
    topology = ParallelTopology("pp8", 8, pipeline_parallel_degree=8)

    with pytest.raises(ValueError, match="1 pipeline microbatches but 8 stages"):
        training_command_args(
            local_batch_size=1,
            global_batch_size=8,
            sequence_length=128,
            topology=topology,
        )


def test_common_eight_card_batch_profile_supports_every_builtin_topology() -> None:
    training = FormalTrainingConfig(local_batch_size=8, global_batch_size=64)

    for topology in standard_topologies().values():
        if topology.world_size > 8:
            continue
        plan = training_topology_plan(training, topology)
        assert plan.gradient_accumulation_steps >= 1
        assert plan.pipeline_microbatches >= topology.pipeline_parallel_degree


def test_common_eight_card_profile_preserves_every_global_sample_slot() -> None:
    training = FormalTrainingConfig(local_batch_size=8, global_batch_size=64)

    for topology in standard_topologies().values():
        if topology.world_size > 8:
            continue
        dataloader_batch_size = (
            topology.pipeline_parallel_microbatch_size
            if topology.pipeline_parallel_degree > 1
            else training.local_batch_size
        )
        num_batches = batches_per_optimizer_step(
            global_batch_size=training.global_batch_size,
            training_local_batch_size=training.local_batch_size,
            dataloader_batch_size=dataloader_batch_size,
            dp_world_size=topology.data_parallel_degree,
        )
        slots = sorted(
            slot
            for dp_rank in range(topology.data_parallel_degree)
            for batch in range(num_batches)
            for slot in global_slots_for_batch(
                batch_in_step=batch,
                dp_rank=dp_rank,
                dp_world_size=topology.data_parallel_degree,
                global_batch_size=training.global_batch_size,
                training_local_batch_size=training.local_batch_size,
                dataloader_batch_size=dataloader_batch_size,
            )
        )
        assert slots == list(range(training.global_batch_size)), topology.name


def test_pipeline_batch_validation_happens_before_launch() -> None:
    pp8 = standard_topologies()["pp8"]

    with pytest.raises(ValueError, match="2 pipeline microbatches but 8 stages"):
        training_topology_plan(
            FormalTrainingConfig(local_batch_size=2, global_batch_size=16),
            pp8,
        )


def test_eight_way_dp_rejects_an_undersized_global_batch() -> None:
    ddp8 = standard_topologies()["ddp8"]

    with pytest.raises(ValueError, match="global_batch_size"):
        training_topology_plan(
            FormalTrainingConfig(local_batch_size=8, global_batch_size=16),
            ddp8,
        )


def test_fixed_token_mapping_preserves_one_global_batch_across_dp_degrees() -> None:
    single = [
        slot
        for batch in range(8)
        for slot in global_slots_for_batch(
            batch_in_step=batch,
            dp_rank=0,
            dp_world_size=1,
            global_batch_size=16,
            training_local_batch_size=2,
            dataloader_batch_size=2,
        )
    ]
    distributed = [
        slot
        for dp_rank in range(8)
        for slot in global_slots_for_batch(
            batch_in_step=0,
            dp_rank=dp_rank,
            dp_world_size=8,
            global_batch_size=16,
            training_local_batch_size=2,
            dataloader_batch_size=2,
        )
    ]
    pipeline = [
        slot
        for batch in range(16)
        for slot in global_slots_for_batch(
            batch_in_step=batch,
            dp_rank=0,
            dp_world_size=1,
            global_batch_size=16,
            training_local_batch_size=2,
            dataloader_batch_size=1,
        )
    ]

    assert single == list(range(16))
    assert distributed == list(range(16))
    assert pipeline == list(range(16))


def _write_synthetic_token_plan(path: Path) -> tuple[str, ...]:
    path.mkdir()
    inputs = []
    labels = []
    positions = []
    sample_hashes = []
    for slot in range(4):
        input_values = (slot, slot + 1)
        # Deliberately do not require labels to be a shifted view of inputs.
        # Packed datasets can cross a document boundary inside one sequence.
        label_values = (slot + 1, 100 + slot)
        position_values = (0, 1)
        input_bytes = struct.pack("<2i", *input_values)
        label_bytes = struct.pack("<2i", *label_values)
        position_bytes = struct.pack("<2i", *position_values)
        inputs.append(input_bytes)
        labels.append(label_bytes)
        positions.append(position_bytes)
        sample_hashes.append(
            sample_digest_v2(input_bytes, label_bytes, position_bytes)
        )
    (path / "inputs.i32").write_bytes(b"".join(inputs))
    (path / "labels.i32").write_bytes(b"".join(labels))
    (path / "positions.i32").write_bytes(b"".join(positions))
    step_hashes = (step_digest(sample_hashes),)
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": TOKEN_PLAN_SCHEMA,
                "schema_version": TOKEN_PLAN_VERSION,
                "steps": 1,
                "global_batch_size": 4,
                "sequence_length": 2,
                "sample_sha256": sample_hashes,
                "step_sha256": step_hashes,
                "files": {
                    "inputs": {
                        "name": "inputs.i32",
                        "sha256": sha256_file(path / "inputs.i32"),
                    },
                    "labels": {
                        "name": "labels.i32",
                        "sha256": sha256_file(path / "labels.i32"),
                    },
                    "positions": {
                        "name": "positions.i32",
                        "sha256": sha256_file(path / "positions.i32"),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return tuple(sample_hashes)


def test_load_token_plan_accepts_legacy_shifted_label_format(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-token-plan"
    path.mkdir()
    token_bytes = struct.pack("<3i", 10, 11, 12)
    position_bytes = struct.pack("<2i", 0, 1)
    (path / "tokens.i32").write_bytes(token_bytes)
    (path / "positions.i32").write_bytes(position_bytes)
    sample_hash = sample_digest(token_bytes, position_bytes)
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": TOKEN_PLAN_SCHEMA,
                "schema_version": 1,
                "steps": 1,
                "global_batch_size": 1,
                "sequence_length": 2,
                "sample_sha256": [sample_hash],
                "step_sha256": [step_digest([sample_hash])],
                "files": {
                    "tokens": {
                        "name": "tokens.i32",
                        "sha256": sha256_file(path / "tokens.i32"),
                    },
                    "positions": {
                        "name": "positions.i32",
                        "sha256": sha256_file(path / "positions.i32"),
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    plan = load_token_plan(path)

    assert plan.schema_version == 1
    assert plan.input_file == path / "tokens.i32"
    assert plan.label_file is None


@pytest.mark.parametrize(
    ("context_parallel_degree", "tensor_parallel_degree"),
    ((1, 2), (2, 1)),
)
def test_runtime_input_contract_validates_model_parallel_replicas(
    tmp_path: Path,
    context_parallel_degree: int,
    tensor_parallel_degree: int,
) -> None:
    token_plan_path = tmp_path / "token-plan"
    sample_hashes = _write_synthetic_token_plan(token_plan_path)
    plan = load_token_plan(token_plan_path)
    contract_path = tmp_path / "contract"
    contract_path.mkdir()
    for global_rank in range(4):
        dp_rank = (global_rank // 2) % 2
        slots = (dp_rank * 2, dp_rank * 2 + 1)
        record = {
            "global_rank": global_rank,
            "dp_rank": dp_rank,
            "step": 1,
            "dataloader_batch_size": 2,
            "global_slots": slots,
            "sample_sha256": [sample_hashes[slot] for slot in slots],
        }
        (contract_path / f"rank-{global_rank}.jsonl").write_text(
            json.dumps(record) + "\n", encoding="utf-8"
        )

    summary = validate_runtime_input_contract(
        contract_directory=contract_path,
        plan=plan,
        steps=1,
        global_batch_size=4,
        training_local_batch_size=2,
        dp_world_size=2,
        context_parallel_degree=context_parallel_degree,
        tensor_parallel_degree=tensor_parallel_degree,
        pipeline_parallel_degree=1,
        node_rank=0,
        num_processes_per_node=4,
    )

    assert summary["valid"]
    assert summary["mapping"] == INPUT_MAPPING
    assert summary["context_parallel_degree"] == context_parallel_degree
    assert summary["token_plan_step_series_sha256"] == step_series_digest(
        plan.step_sha256
    )


def test_endpoint_uses_exported_visible_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    endpoint = TrainingEndpoint(
        "npu", "npu", "0,1,2,3,4,5,6,7", ParallelTopology("single", 1)
    )
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4")

    resolved = _endpoint_from_process_environment(endpoint)

    assert resolved.visible_devices == "4"


def test_fixture_backend_is_inferred_without_endpoint_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = ParallelTopology("single", 1)
    config = FormalExperimentConfig(
        name="fixture-backend",
        kind="migration",
        reference=TrainingEndpoint("gpu", "cuda", "0", topology, repeats=2),
        candidate=TrainingEndpoint("npu", "npu", "0", topology, repeats=2),
    )
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    endpoint = _fixture_endpoint_from_environment(config, None)

    assert endpoint.device_type == "npu"


def test_fixture_backend_requires_unambiguous_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = ParallelTopology("single", 1)
    config = FormalExperimentConfig(
        name="fixture-backend",
        kind="migration",
        reference=TrainingEndpoint("gpu", "cuda", "0", topology, repeats=2),
        candidate=TrainingEndpoint("npu", "npu", "0", topology, repeats=2),
    )
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    with pytest.raises(ValueError, match="fixture backend is ambiguous"):
        _fixture_endpoint_from_environment(config, None)


def test_compare_writes_html_and_json_report(tmp_path: Path) -> None:
    topology = ParallelTopology("single", 1)
    reference = TrainingEndpoint(
        "gpu", "cuda", "0", topology, repeats=2
    )
    candidate = TrainingEndpoint(
        "npu", "npu", "0", topology, repeats=2
    )
    standard = PrecisionStandard(
        migration=MigrationStandard(
            warmup_steps=0,
            minimum_observations=4,
            required_candidate_repeats=2,
        ),
        self_consistency=SelfConsistencyStandard(
            randomness_impacted=True,
            required_candidate_repeats=2,
        ),
    )
    config = FormalExperimentConfig(
        name="synthetic",
        kind="migration",
        reference=reference,
        candidate=candidate,
        training=FormalTrainingConfig(steps=4, global_batch_size=2),
        standard=standard,
        artifact_root="artifacts",
        report_root="reports",
    )
    from tests.glm5_2_precision.workflow import _artifact_directory

    contract = {
        "scenario_name": config.scenario_name,
        "training": {
            **asdict(config.training),
            "converged_checkpoint": None,
        },
        "checkpoint_sha256": "checkpoint",
        "data_sha256": {"data": "digest"},
        "topology": {
            "name": topology.name,
            "world_size": topology.world_size,
            "input_contract": {
                "valid": True,
                "mapping": INPUT_MAPPING,
                "steps": 4,
                "global_batch_size": 2,
                "training_local_batch_size": 2,
                "dp_world_size": 1,
                "token_plan_step_series_sha256": "token-plan",
            },
        },
    }
    for repeat in (1, 2):
        PrecisionArtifactWriter(
            _artifact_directory(tmp_path, config, "reference", reference, repeat)
        ).write(
            metadata={"role": "reference", "repeat": repeat},
            training_contract=contract,
            metrics=_metrics(),
        )
    for repeat in (1, 2):
        PrecisionArtifactWriter(
            _artifact_directory(tmp_path, config, "candidate", candidate, repeat)
        ).write(
            metadata={"role": "candidate", "repeat": repeat},
            training_contract=contract,
            metrics=_metrics(loss_offset=0.001 if repeat == 1 else 0.0011),
        )

    report = compare_and_write_report(tmp_path, config)
    assert report.is_file()
    assert report.name == "migration-cuda-single-vs-npu-single-bf16.html"
    report_text = report.read_text(encoding="utf-8")
    assert "Formal GLM 5.2 precision benchmark" in report_text
    assert "Repeat-run diagnostics" in report_text
    assert "Historical customer standards (reference only)" in report_text
    assert "step 1" in report_text
    summary = json.loads(
        (report.parent / "precision_summary.json").read_text(encoding="utf-8")
    )
    assert summary["passed"]
    assert not summary["candidate_reproducible_loss"]
    assert not summary["strict_reproducibility_required"]


def test_storage_name_and_capture_paths_are_compact(tmp_path: Path) -> None:
    topology = ParallelTopology("single", 1)
    config = FormalExperimentConfig(
        name="glm5-2-gpu-npu-training",
        kind="migration",
        reference=TrainingEndpoint("gpu-reference", "cuda", "0", topology),
        candidate=TrainingEndpoint("npu-candidate", "npu", "0", topology),
    )

    base = "migration-cuda-npu-single-bf16-random-s1000-b16-seq128-seed61"
    assert config.storage_name.startswith(base + "-")
    assert len(config.storage_name.removeprefix(base + "-")) == 8
    assert config.fixture_storage_name == config.storage_name
    from tests.glm5_2_precision.workflow import _artifact_directory

    assert _artifact_directory(
        tmp_path, config, "candidate", config.candidate, 1
    ).name == "candidate-r1"


def test_legacy_scenario_name_maps_to_current_storage_name() -> None:
    legacy = (
        "glm5-2-gpu-npu-training-migration-cuda-single-vs-npu-single-"
        "mixed-bfloat16-random-seed-steps1000-global-batch16-seq128-seed61"
    )

    assert compact_scenario_name(legacy) == (
        "migration-cuda-npu-single-bf16-random-s1000-b16-seq128-seed61"
    )


def test_self_consistency_suite_reuses_reference_and_reports_partial_results(
    tmp_path: Path,
) -> None:
    topologies = {
        "single": ParallelTopology("single", 1),
        "ddp8": ParallelTopology(
            "ddp8", 8, data_parallel_replicate_degree=8
        ),
        "fsdp8": ParallelTopology("fsdp8", 8, data_parallel_shard_degree=8),
    }
    reference = TrainingEndpoint(
        "single", "cuda", "0,1,2,3,4,5,6,7", topologies["single"], repeats=2
    )
    candidate = TrainingEndpoint(
        "distributed", "cuda", "0,1,2,3,4,5,6,7", topologies["ddp8"], repeats=2
    )
    standard = PrecisionStandard(
        self_consistency=SelfConsistencyStandard(
            randomness_impacted=True,
            migration_fallback=MigrationStandard(
                minimum_observations=4,
                required_reference_repeats=2,
                required_candidate_repeats=2,
            ),
        )
    )
    config = FormalExperimentConfig(
        name="suite",
        kind="self_consistency",
        reference=reference,
        candidate=candidate,
        training=FormalTrainingConfig(steps=4, global_batch_size=16),
        standard=standard,
        artifact_root="artifacts",
        report_root="reports",
    )
    from tests.glm5_2_precision.workflow import _artifact_directory

    base = "self-cuda-bf16-random-s4-b16-seq128-seed61"
    assert config.storage_name.startswith(base + "-")
    assert len(config.storage_name.removeprefix(base + "-")) == 8
    assert _artifact_directory(
        tmp_path, config, "candidate", config.candidate, 1
    ).name == "ddp8-r1"

    base_contract = {
        "scenario_name": config.storage_name,
        "training": {**asdict(config.training), "converged_checkpoint": None},
        "checkpoint_sha256": "checkpoint",
        "data_sha256": {"data": "digest"},
    }
    input_contract = {
        "valid": True,
        "mapping": INPUT_MAPPING,
        "steps": 4,
        "global_batch_size": 16,
        "training_local_batch_size": 2,
        "token_plan_step_series_sha256": "token-plan",
    }
    for repeat in (1, 2):
        PrecisionArtifactWriter(
            _artifact_directory(
                tmp_path, config, "reference", config.reference, repeat
            )
        ).write(
            metadata={"role": "reference", "repeat": repeat},
            training_contract={
                **base_contract,
                "topology": {
                    **asdict(topologies["single"]),
                    "input_contract": {**input_contract, "dp_world_size": 1},
                },
            },
            metrics=_metrics(),
        )
        PrecisionArtifactWriter(
            _artifact_directory(
                tmp_path, config, "candidate", config.candidate, repeat
            )
        ).write(
            metadata={"role": "candidate", "repeat": repeat},
            training_contract={
                **base_contract,
                "topology": {
                    **asdict(topologies["ddp8"]),
                    "input_contract": {**input_contract, "dp_world_size": 8},
                },
            },
            metrics=_metrics(loss_offset=0.001),
        )

    report = compare_suite(
        tmp_path,
        config,
        topologies=topologies,
        selected=("ddp8", "fsdp8"),
        precision=None,
        require_all=False,
    )
    text = report.read_text(encoding="utf-8")
    assert report.name == "cuda-vs-cuda-bf16-topology-suite.html"
    assert "ddp8" in text
    assert "fsdp8" in text
    assert "NOT RUN" in text
    assert "PARTIAL PASS" in text
    assert precision_report_filename(config) == (
        "self-cuda-single-vs-cuda-ddp8-bf16.html"
    )
    assert (
        report.parent
        / "ddp8"
        / precision_report_filename(config)
    ).is_file()


def test_migration_topologies_share_one_fixture_identity() -> None:
    topologies = standard_topologies()
    endpoint = TrainingEndpoint(
        "reference",
        "cuda",
        "0,1,2,3,4,5,6,7",
        topologies["single"],
        repeats=1,
    )
    base = FormalExperimentConfig(
        name="migration-suite",
        kind="migration",
        reference=endpoint,
        candidate=endpoint,
        training=FormalTrainingConfig(
            steps=8,
            local_batch_size=8,
            global_batch_size=64,
        ),
    )
    from tests.glm5_2_precision.topology_suite import topology_config

    single = topology_config(base, topologies["single"], precision=None)
    fsdp = topology_config(base, topologies["fsdp8"], precision=None)

    assert single.fixture_storage_name == fsdp.fixture_storage_name
    assert single.storage_name != fsdp.storage_name
    assert (
        single.fixture_storage_name.rsplit("-", 1)[1]
        == single.storage_name.rsplit("-", 1)[1]
        == fsdp.storage_name.rsplit("-", 1)[1]
    )
    assert fsdp.reference.topology == fsdp.candidate.topology


def test_report_standard_does_not_change_capture_storage_name() -> None:
    topology = ParallelTopology("single", 1)
    config = FormalExperimentConfig(
        name="migration",
        kind="migration",
        reference=TrainingEndpoint("gpu", "cuda", "0", topology),
        candidate=TrainingEndpoint("npu", "npu", "0", topology),
    )
    changed_standard = replace(
        config,
        standard=PrecisionStandard(
            migration=MigrationStandard(
                all_loss=AnyOfErrorLimit(absolute=0.123)
            )
        ),
    )

    assert changed_standard.storage_name == config.storage_name


def test_matching_pre_digest_artifact_directory_is_adopted(tmp_path: Path) -> None:
    topology = ParallelTopology("single", 1)
    config = FormalExperimentConfig(
        name="migration",
        kind="migration",
        reference=TrainingEndpoint("gpu", "cuda", "0", topology),
        candidate=TrainingEndpoint("npu", "npu", "0", topology),
        artifact_root="artifacts",
    )
    legacy_capture = (
        tmp_path
        / "artifacts"
        / config.storage_base_name
        / "candidate-r1"
    )
    legacy_capture.mkdir(parents=True)
    (legacy_capture / "training_contract.json").write_text(
        json.dumps(
            {
                "scenario_name": config.storage_base_name,
                "training": {
                    **asdict(config.training),
                    "converged_checkpoint": None,
                },
                "endpoint_args": [],
            }
        ),
        encoding="utf-8",
    )

    from tests.glm5_2_precision.workflow import _artifact_directory

    adopted = _artifact_directory(
        tmp_path, config, "candidate", config.candidate, 1
    )

    assert adopted.parent.name == config.storage_name
    assert adopted.is_dir()
    assert not (tmp_path / "artifacts" / config.storage_base_name).exists()
    adopted_contract = json.loads(
        (adopted / "training_contract.json").read_text(encoding="utf-8")
    )
    assert adopted_contract["scenario_name"] == config.storage_name
