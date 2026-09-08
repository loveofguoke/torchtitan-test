import argparse
import ast
from dataclasses import asdict, make_dataclass
from pathlib import Path
import unittest
from unittest.mock import patch

from tests.glm5_2_parity.model_config import SIZE_ALIASES, model_dimensions
from tests.glm5_2_parity.workflow import (
    PairedParityConfig, _add_config_arguments, _apply_config_arguments, _config_digest,
)


def config(**values):
    return make_dataclass("Config", values.keys())(**values)


def native_model(dim=96):
    # Deliberately different from GLM debug dimensions: detect copied defaults.
    rope = config(dim=12, max_context_length=144, theta=123456, scaling="none")
    indexer = config(n_heads=3, head_dim=24, index_topk=5, rope=rope)
    attention = config(n_heads=6, q_lora_rank=48, kv_lora_rank=24,
                       qk_nope_head_dim=12, qk_rope_head_dim=12, v_head_dim=24,
                       indexer=indexer, rope=rope, sentinel="preserve-me")
    dense = config(w1=config(out_features=192))
    moe = config(num_experts=6, routed_experts=config(inner_experts=config(hidden_dim=48)),
                 shared_experts=config(w1=config(out_features=48)),
                 router=config(top_k=2, num_expert_groups=1, num_limited_groups=1, route_scale=1.75))
    return config(vocab_size=512, dim=dim, index_sources=(),
                  layers=[config(attention=attention, feed_forward=dense, moe=None),
                          config(attention=attention, feed_forward=None, moe=moe)])


class ParityCliConfigTest(unittest.TestCase):
    def resolve(self, *args, native=None):
        scenario = PairedParityConfig()
        parser = argparse.ArgumentParser()
        _add_config_arguments(parser, scenario)
        with patch("tests.glm5_2_parity.workflow.load_model_config", return_value=native or native_model()):
            return _apply_config_arguments(scenario, parser.parse_args(args))

    def test_defaults_are_read_from_factory(self):
        resolved = self.resolve()
        self.assertEqual(asdict(resolved.model), model_dimensions(native_model()))
        self.assertEqual(self.resolve(native=native_model(192)).model.dim, 192)
        self.assertEqual(self.resolve('--dim', '96'), resolved)

    def test_factory_selection(self):
        scenario = PairedParityConfig()
        parser = argparse.ArgumentParser()
        _add_config_arguments(parser, scenario)
        with patch("tests.glm5_2_parity.workflow.load_model_config", return_value=native_model()) as loader:
            _apply_config_arguments(scenario, parser.parse_args(['--model-config', 'another']))
        loader.assert_called_once_with('another')

    def test_long_sequence_and_model_overrides(self):
        resolved = self.resolve('--sequence-length', '256', '--num-layers', '4', '--layers', '0,2')
        self.assertEqual(resolved.model.layers, 4)
        self.assertEqual(resolved.layers, '0,2')
        self.assertEqual(resolved.model.rope_cache_max_seq_len, 256)
        self.assertNotEqual(_config_digest(resolved, 'test'), _config_digest(self.resolve(), 'test'))

    def test_native_configuration_changes_identity(self):
        native = native_model()
        native.layers[0].attention.sentinel = "new-setting"
        self.assertNotEqual(_config_digest(self.resolve(native=native), 'test'),
                            _config_digest(self.resolve(), 'test'))

    def test_native_layers_are_used_not_rebuilt(self):
        from dataclasses import replace
        from types import SimpleNamespace
        import os
        source = Path(__file__).parents[1] / 'glm5_2_parity' / 'suite.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_titan_config')
        native = native_model()
        namespace = dict(Any=object, os=os, replace=replace, glm5_configs={'debugmodel': lambda: native},
                         ParityModelSize=object)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
        size = SimpleNamespace(**{SIZE_ALIASES.get(k, k): v for k, v in model_dimensions(native).items()})
        with patch.dict(os.environ, {'GLM5_PARITY_MODEL_CONFIG': 'debugmodel'}):
            self.assertIs(namespace['_titan_config'](size), native)
        self.assertEqual(native.layers[0].attention.sentinel, 'preserve-me')
