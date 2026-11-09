import json
import os
from pathlib import Path

import pytest

from tests.glm5_2_mindstudio.recorded_training import compare, read_evidence, sha256, validate_reference, validate_resolved_configs


def evidence(runtime="turbo"):
    return {
        "runtime": runtime, "status": "passed", "world_size": 1, "mtp": False,
        "expected_steps": 500, "completed_steps": 500,
        "precision": {"dtype": "float32", "param_dtype": "bfloat16", "reduce_dtype": "float32"},
        "deterministic": True, "remat": "selective", "local_compile_regions": [],
        "topology": {"name": "single"}, "sequence_length": 128,
        "microbatch_tokens_per_dp_rank": 1024, "seed": 61,
        "fixture_sha256": "fixture", "input_manifest_sha256": "inputs",
        "input_microbatches": 500, "replayed_microbatches": 500,
        "metrics": [{"step": i, "loss": 1.0, "max_loss": 1.0, "grad_norm": 2.0} for i in range(1, 501)],
    }


def test_exact_and_changed_metrics():
    reference, gpu = evidence(), evidence("cuda")
    assert compare(reference, gpu)["metrics"]["loss"]["max_absolute_error"] == 0
    reference["metrics"][22]["loss"] += 0.25
    result = compare(reference, gpu)
    assert result["metrics"]["loss"]["worst_step"] == 23
    assert result["metrics"]["loss"]["max_absolute_error"] == 0.25
    assert result["verdict"] == "observation_only_single_repeat"


@pytest.mark.parametrize("key,value", [("seed", 1), ("fixture_sha256", "other"),
                                      ("input_manifest_sha256", "other"),
                                      ("replayed_microbatches", 499)])
def test_contract_mismatch_rejected(key, value):
    gpu = evidence("cuda")
    gpu[key] = value
    with pytest.raises(ValueError):
        compare(evidence(), gpu)


@pytest.mark.parametrize("mutation", ["native", "missing", "nan", "precision"])
def test_invalid_capture_rejected(tmp_path, mutation):
    record = evidence()
    if mutation == "native":
        record["runtime"] = "ascend"
    elif mutation == "missing":
        record["metrics"].pop()
    elif mutation == "nan":
        record["metrics"][0]["loss"] = float("nan")
    else:
        record["precision"]["dtype"] = "bfloat16"
    path = tmp_path / "rank.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        read_evidence(path, "turbo")


def test_reference_checks_every_input_and_fixture(tmp_path):
    root = tmp_path
    inputs = root / "steps-500/inputs"
    inputs.mkdir(parents=True)
    fixture = root / "initial-model.pt"
    fixture.write_bytes(b"fixture")
    rows = []
    for index in range(1, 501):
        name = f"microbatch-{index:06d}.pt"
        (inputs / name).write_bytes(b"input")
        rows.append({"microbatch": index, "file": name, "sha256": sha256(inputs / name)})
    manifest = inputs / "manifest.jsonl"
    manifest.write_text("\n".join(map(json.dumps, rows)) + "\n")
    record = evidence()
    record.update(fixture_sha256=sha256(fixture), input_manifest_sha256=sha256(manifest))
    (root / "steps-500/rank-0.json").write_text(json.dumps(record))
    (root / "steps-500/rank-0-config.txt").write_text("resolved config")
    assert validate_reference(root)[0]["completed_steps"] == 500
    (inputs / "microbatch-000500.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_reference(root)


def test_config_drift_rejected(tmp_path):
    original = "optim=A(), training=B(steps=500), garbage_collector=C(), activation_remat=D(), profiler=E(), loss=F(), dist_moe_runtime=None, precision=G(), hf_assets_path='x', model=H(eps=1e-5, index_sources=(0, 1, 1))"
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_text(original)
    second.write_text(original)
    validate_resolved_configs(first, second)
    second.write_text(original.replace("1e-5", "1e-6"))
    with pytest.raises(ValueError, match="normalization"):
        validate_resolved_configs(first, second)
    second.write_text(original.replace("steps=500", "steps=2"))
    with pytest.raises(ValueError, match="training="):
        validate_resolved_configs(first, second)
    second.write_text(original.replace("(0, 1, 1)", "(0, 1, 2)"))
    with pytest.raises(ValueError, match="source plan"):
        validate_resolved_configs(first, second)


@pytest.mark.skipif(not os.environ.get("GLM_TEST_REFERENCE_CONFIG"), reason="Recorded server config not supplied")
def test_current_recipe_against_recorded_config(tmp_path):
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan_recipes.tests.models.glm5 import glm5_debugmodel
    config = glm5_debugmodel(seq_len=128, parallelism=ParallelismConfig(enable_sequence_parallel=True))
    config.training.steps = 500
    config.training.num_tokens_per_microbatch_per_dp_rank = 1024
    config.precision.param_dtype = "bfloat16"
    config.model.local_compile.regions = []
    path = tmp_path / "config.txt"
    path.write_text(repr(config), encoding="utf-8")
    validate_resolved_configs(Path(os.environ["GLM_TEST_REFERENCE_CONFIG"]), path)


@pytest.mark.skipif(not os.environ.get("GLM_TEST_REFERENCE_ROOT"), reason="Recorded fixture not supplied")
def test_real_recorded_fixture_matches_current_model():
    import torch
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan_recipes.tests.models.glm5 import glm5_debugmodel
    root = Path(os.environ["GLM_TEST_REFERENCE_ROOT"])
    _, fixture, _ = validate_reference(root)
    config = glm5_debugmodel(seq_len=128, parallelism=ParallelismConfig(enable_sequence_parallel=True))
    model = config.model.build()
    model.init_states()
    state = torch.load(fixture, map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
