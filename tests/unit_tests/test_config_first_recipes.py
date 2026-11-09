"""Build real recipes from every GPU smoke topology's generated CLI."""

import pytest

from tests.glm5_2_common import config_first
from tests.glm5_2_common.topology import standard_topologies, training_command_args


_CASES = [
    (recipe, topology)
    for recipe in ("glm5_debugmodel", "glm5_2_debugmodel_mtp")
    for topology in standard_topologies().values()
    if topology.world_size <= 8
    and not (recipe.endswith("_mtp") and topology.pp > 1)
]


@pytest.mark.parametrize("seed_checkpoint", [False, True])
def test_ms_precision_recipe_cli_uses_current_fields(monkeypatch, tmp_path, seed_checkpoint):
    from tests.glm5_2_precision.workflow import FormalTrainingConfig, _base_training_args
    from tests.glm5_2_common.topology import ParallelTopology
    monkeypatch.setenv("TORCHTITAN_CONFIG_FIRST_CLI", "1")
    monkeypatch.setattr(config_first, "_LEGACY_CONFIG_STATE", {})
    arguments = ["capture", "--module", "glm5", "--config", "glm5_debugmodel",
                 *_base_training_args(FormalTrainingConfig(steps=500),
                                      dump_folder=tmp_path, topology=ParallelTopology("single", 1)),
                 "--checkpoint.enable", "--lr_scheduler.total_steps=500"]
    arguments += (["--checkpoint.create_seed_checkpoint"] if seed_checkpoint else
                  ["--checkpoint.load_only", f"--checkpoint.initial_load_path={tmp_path.resolve()}"])
    config_first.translate_config_first_cli(arguments)
    config = config_first.legacy_config()
    assert config.precision.dtype == "float32"
    assert config.precision.param_dtype == "bfloat16"
    assert config.precision.reduce_dtype == "float32"
    assert config.optim.lr_scheduler.total_steps == 500
    assert config.create_seed_checkpoint is seed_checkpoint
    assert config.checkpointer is not None
    assert config.activation_remat is not None
    if not seed_checkpoint:
        assert config.checkpointer.initial_load_path == str(tmp_path.resolve())
        assert config.checkpointer.load_only is True


@pytest.mark.parametrize("recipe,topology", _CASES)
def test_real_smoke_recipe_cli(monkeypatch, recipe, topology):
    monkeypatch.setenv("TORCHTITAN_CONFIG_FIRST_CLI", "1")
    monkeypatch.setattr(config_first, "_LEGACY_CONFIG_STATE", {})
    config_first.translate_config_first_cli(
        [
            "train_gpu.py", "--module", "glm5", "--config", recipe,
            "--training.steps=2", "--training.disable_cuda_graphs",
            *training_command_args(
                local_batch_size=8, global_batch_size=64,
                sequence_length=128, topology=topology,
            ),
            "--debug.seed=61", "--metrics.log_freq=1",
            *topology.command_args(),
        ]
    )
    config = config_first.legacy_config()
    assert config.model.max_context_length == 128
    assert config.training.steps == 2
    assert config.training.disable_accelerator_graphs is True
    assert config.training.num_tokens_per_train_step == 8192
    assert config.debug.seed == 61
    assert config.metrics.log_freq == 1
    assert config.activation_remat is not None
    assert config.parallelism.pipeline_parallel_degree == topology.pp
    assert config.parallelism.expert_parallel_degree == topology.ep
    if topology.pp > 1:
        assert config.parallelism.num_pp_microbatches == 8
    assert bool(config.model.mtp_layers) == recipe.endswith("_mtp")
