"""Guard parity against the current native GLM config and fused state layout."""

from dataclasses import replace

import pytest
import torch

from tests.glm5_2_parity.model_config import load_model_config, model_dimensions
from tests.glm5_2_parity.suite import (
    FP32, ModelEndpoint, ParityDataFactory, ParityModelSize, _build_pair,
    _hf_config, _run_causal_lm_endpoint, _set_titan_routed_expert_compute_dtype,
    _titan_config,
)


@pytest.mark.parametrize("flavor", ["debugmodel", "glm5_2_debugmodel"])
def test_native_configuration_and_hf_layer_patterns(monkeypatch, flavor):
    monkeypatch.setenv("GLM5_PARITY_MODEL_CONFIG", flavor)
    native = load_model_config(flavor)
    dimensions = model_dimensions(native)
    size = ParityModelSize()
    config = _titan_config(size)
    reference = _hf_config(size)
    assert dimensions["index_top_k"] == config.layers[0].attention.indexer.topk.index_topk
    assert config.index_sources == native.index_sources
    assert reference.indexer_types == [
        "full" if source == layer else "shared"
        for layer, source in enumerate(native.index_sources)
    ]
    assert reference.first_k_dense_replace == dimensions["dense_layers"]


def test_strict_pair_and_token_first_forward(monkeypatch):
    monkeypatch.setenv("GLM5_PARITY_MODEL_CONFIG", "debugmodel")
    size = replace(ParityModelSize(), num_layers=2, num_dense_layers=1)
    pair = _build_pair(torch.device("cpu"), model_size=size)
    _set_titan_routed_expert_compute_dtype(pair.titan, torch.float32)
    batch = ParityDataFactory.make(
        device=pair.device, dtype=torch.float32, batch_size=2, sequence_length=8,
        vocab_size=size.vocab_size, hidden_size=size.dim, make_tokens=True,
    )
    with torch.no_grad():
        titan, titan_loss = _run_causal_lm_endpoint(pair.titan, ModelEndpoint("titan", FP32), batch)
        hf, hf_loss = _run_causal_lm_endpoint(pair.hf, ModelEndpoint("hf", FP32), batch)
    assert titan.shape == hf.shape == (2, 8, size.vocab_size)
    assert torch.isfinite(titan).all() and torch.isfinite(hf).all()
    assert torch.isfinite(titan_loss) and torch.isfinite(hf_loss)
    for module in pair.titan.modules():
        for buffer in module.buffers(recurse=False):
            if buffer.is_complex():
                assert buffer.dtype == torch.complex64
