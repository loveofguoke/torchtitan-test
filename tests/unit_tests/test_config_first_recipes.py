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
