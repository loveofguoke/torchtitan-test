"""Guard parity against the current native GLM config and fused state layout."""

from dataclasses import replace

import pytest
import torch

from tests.glm5_2_parity.model_config import load_model_config, model_dimensions
from tests.glm5_2_parity import suite as parity


def test_cuda_gather_attention_graph_matches_eager_output_and_gradients():
    """Guard device offsets and fused shared latent pool under capture."""
    from torchtitan.models.glm5.dsa import DsaMetadata, Glm5SparseAttention

    if not torch.cuda.is_available() or torch.version.hip:
        pytest.skip("requires NVIDIA CUDA")
    generator = torch.Generator(device="cuda").manual_seed(42)
    tensors = [
        torch.randn(32, 2, 64, device="cuda", dtype=torch.bfloat16,
                    generator=generator, requires_grad=True),
        torch.randn(32, 64, device="cuda", dtype=torch.bfloat16,
                    generator=generator, requires_grad=True),
    ]
    metadata = DsaMetadata.from_positions(torch.arange(32, device="cuda"))
    rows = torch.arange(32, device="cuda", dtype=torch.int32)
    indices = torch.stack((rows, rows - 1), dim=-1)
    attention = Glm5SparseAttention.Config().build()

    def execute():
        output = attention(
            *tensors, attention_metadata=metadata,
            topk_indices_QS=indices, scale=0.125,
        )
        gradients = torch.autograd.grad(output.float().square().sum(), tensors)
        return output, gradients

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            eager_output, eager_gradients = execute()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output, graph_gradients = execute()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_output, eager_output, rtol=0, atol=0)
    for actual, expected in zip(graph_gradients, eager_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print("fused gather_attn: shared latent pool, window=0, CUDA graph output/gradients match eager exactly")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_glm_indexer_selects_native_backend(dtype):
    """Exercise production dispatch with FP32 weights cast at the kernel boundary."""
    from unittest.mock import patch

    from torchtitan.models.common.attn_gym import lightning_indexer
    from torchtitan.models.glm5.dsa import DsaMetadata, DSAIndexerTopK
    from torchtitan.models.glm5.indexer import select_local_topk

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.version.hip or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("fused indexer requires NVIDIA SM90 or newer")
    generator = torch.Generator(device="cuda").manual_seed(61)
    q = torch.randn(128, 32, 128, device="cuda", dtype=dtype, generator=generator)
    k = torch.randn(128, 128, device="cuda", dtype=dtype, generator=generator)
    w = torch.randn(128, 32, device="cuda", generator=generator)
    metadata = DsaMetadata.from_positions(torch.arange(128, device="cuda"))
    indexer = DSAIndexerTopK.Config(index_topk=8).build()
    target, implementation = (
        ("lightning_indexer", lightning_indexer)
        if dtype == torch.bfloat16 else ("select_local_topk", select_local_topk)
    )
    with patch(f"torchtitan.models.glm5.dsa.{target}", wraps=implementation) as kernel:
        result = indexer(q, k, w, metadata=metadata)
    torch.cuda.synchronize()
    if dtype == torch.bfloat16:
        assert kernel.call_args.kwargs["impl"] == "fused"
    assert kernel.call_args.args[0].dtype == dtype
    assert kernel.call_args.args[2].dtype == dtype
    assert w.dtype == torch.float32
    assert result.shape == (128, 8)
    assert not result.requires_grad
    backend = "fused" if target == "lightning_indexer" else "bounded_reference"
    print(f"production indexer: input={dtype} impl={backend} kernel_weights={dtype} source_weights={w.dtype}")


@pytest.mark.parametrize("sequence_length", [128, 512])
def test_cuda_bf16_fused_indexer_overlap(sequence_length, tmp_path):
    """Measure random-input selection overlap without asserting exact BF16 ranking."""
    import json

    from torchtitan.models.common.attn_gym import lightning_indexer

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.version.hip or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("fused indexer requires NVIDIA SM90 or newer")
    generator = torch.Generator(device="cuda").manual_seed(61)
    q = torch.randn(1, sequence_length, 32, 128, device="cuda", generator=generator)
    k = torch.randn(1, sequence_length, 128, device="cuda", generator=generator)
    w = torch.randn(1, sequence_length, 32, device="cuda", generator=generator)
    topk = 8
    with torch.no_grad():
        reference = lightning_indexer(q, k, w, topk, causal=True, impl="reference")
        fused = lightning_indexer(
            q.bfloat16(), k.bfloat16(), w.bfloat16(), topk,
            causal=True, impl="fused",
        )
    torch.cuda.synchronize()
    selections = []
    for indices in (reference, fused):
        indices = indices.reshape(sequence_length, topk).long()
        queries = torch.arange(sequence_length, device="cuda")[:, None]
        valid = (indices >= 0) & (indices <= queries)
        assert not ((indices >= 0) & ~valid).any().item()
        selected = torch.zeros(sequence_length, sequence_length, device="cuda", dtype=torch.int32)
        selected.scatter_add_(1, indices.clamp(0, sequence_length - 1), valid.int())
        selected = selected.bool()
        assert torch.equal(selected.sum(1), (queries.flatten() + 1).clamp_max(topk))
        selections.append(selected)
    overlap = (selections[0] & selections[1]).sum(1) / selections[0].sum(1)
    result = {
        "seed": 61, "sequence_length": sequence_length, "heads": 32,
        "head_dim": 128, "topk": topk, "fixture": "random_kernel_inputs",
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "reference": "fp32/reference", "candidate": "bf16/fused",
        "mean_topk_overlap": overlap.mean().item(),
        "minimum_query_overlap": overlap.min().item(),
        "exact_query_fraction": (overlap == 1).float().mean().item(),
    }
    path = tmp_path / f"bf16-indexer-overlap-{sequence_length}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    print(path)


def test_terminal_summary_includes_failures_and_effective_tolerance():
    recorder = parity.ParityRecorder(parity.FP32)
    recorder.tensor(scope="e2e", component="loss", layer="output",
                    actual=torch.tensor([2.0]), expected=torch.tensor([1.0]),
                    atol=0.01, rtol=0.02)
    report = parity.ParitySuiteReport("summary regression")
    report.add("e2e", recorder)
    summary = report.terminal_summary()
    assert "checkpoints=1 passed=0 failed=1" in summary
    assert "atol=0.01 rtol=0.02" in summary
    assert "actual=[2] expected=[1]" in summary
    assert "e2e/loss" in summary


def test_terminal_summary_exposes_successful_indexer_checkpoints():
    recorder = parity.ParityRecorder(parity.FP32)
    indices = torch.tensor([[[0, -1], [0, 1], [1, 2]]])
    recorder.discrete(
        scope="configured", component="indexer", layer=0,
        actual=indices, expected=indices,
        positions=torch.arange(3).unsqueeze(0),
    )
    report = parity.ParitySuiteReport("indexer summary")
    report.add("end-to-end", recorder)
    summary = report.terminal_summary()
    assert "Indexer and router checkpoints:" in summary
    assert "configured/indexer layer=0 | status=PASS" in summary


def test_hf_mla_norms_follow_native_debug_configuration(monkeypatch):
    monkeypatch.setenv("GLM5_PARITY_MODEL_CONFIG", "debugmodel")
    size = ParityModelSize()
    pair = _build_pair(torch.device("cpu"), model_size=size)
    values = torch.randn(16, size.dim)
    for hf, native in zip(pair.hf.model.layers, pair.titan.layers.values(), strict=True):
        attention = native.attention
        assert hf.self_attn.q_a_layernorm.variance_epsilon == attention.q_norm.eps
        assert hf.self_attn.kv_a_layernorm.variance_epsilon == attention.kv_norm.eps
        torch.testing.assert_close(
            attention.q_norm(attention.wq_a(values)),
            hf.self_attn.q_a_layernorm(hf.self_attn.q_a_proj(values)),
            rtol=1e-6, atol=1e-7,
        )


@pytest.mark.parametrize("flavor", ["debugmodel", "glm5_2_scaled_debugmodel"])
def test_parity_uses_moved_experiment_factory_without_changing_snapshot(flavor):
    from tests.glm5_2_parity.model_config import config_snapshot, load_model_config
    from torchtitan_recipes.tests.models.glm5_configs import resolve_model_config

    native = load_model_config(flavor)
    assert config_snapshot(native) == config_snapshot(resolve_model_config(flavor))
    expected_context = 2048 if flavor == "debugmodel" else 4096
    assert native.max_context_length == expected_context
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


def test_component_paths_emit_records_on_current_models(monkeypatch):
    monkeypatch.setenv("GLM5_PARITY_MODEL_CONFIG", "debugmodel")
    monkeypatch.setenv("GLM5_PARITY_MODEL_LAYERS", "2")
    monkeypatch.setattr(parity, "_parity_device", lambda: (torch.device("cpu"), torch.cpu, "cpu"))

    class Probe(parity.Glm5ParitySuite):
        SEQUENCE_LENGTH = 16
        BATCH_SIZE = 2
        TITAN_ROUTED_EXPERT_COMPUTE = "fp32"

    Probe.setUpClass()
    probe = Probe("test_configured_precision_suite")
    for component in ("indexer", "router", "attention", "block"):
        recorder = probe._configured_component_recorder(component)
        # This test guards execution/trace interfaces, not numerical acceptance.
        # The real suite retains all original numerical/discrete thresholds.
        assert recorder.results, component
    end_to_end = probe.compare_end_to_end(probe._configured_spec())
    assert any(row.scope == "gradient" for row in end_to_end.results)
    assert any(row.component == "logits" for row in end_to_end.results)


def test_shared_indexer_model_emits_end_to_end_gradient_records(monkeypatch):
    monkeypatch.setenv("GLM5_PARITY_MODEL_CONFIG", "glm5_2_debugmodel")
    monkeypatch.setenv("GLM5_PARITY_MODEL_LAYERS", "4")
    monkeypatch.setattr(parity, "_parity_device", lambda: (torch.device("cpu"), torch.cpu, "cpu"))

    class Probe(parity.Glm5ParitySuite):
        SEQUENCE_LENGTH = 16
        BATCH_SIZE = 2
        TITAN_ROUTED_EXPERT_COMPUTE = "fp32"

    Probe.setUpClass()
    probe = Probe("test_configured_precision_suite")
    assert probe.pair.titan_layer(3).attention.indexer is None
    with pytest.raises(ValueError, match="producer context"):
        probe._configured_component_recorder("block")
    recorder = probe.compare_end_to_end(probe._configured_spec())
    assert any(row.scope == "gradient" for row in recorder.results)
    assert not any(row.component == "indexer" and row.layer == 3 for row in recorder.results)


def test_missing_both_producer_traces_cannot_silently_pass(monkeypatch):
    from contextlib import contextmanager

    monkeypatch.setenv("GLM5_PARITY_MODEL_CONFIG", "debugmodel")
    monkeypatch.setenv("GLM5_PARITY_MODEL_LAYERS", "1")
    monkeypatch.setattr(
        parity, "_parity_device",
        lambda: (torch.device("cpu"), torch.cpu, "cpu"),
    )

    class Probe(parity.Glm5ParitySuite):
        SEQUENCE_LENGTH = 16
        BATCH_SIZE = 1
        TITAN_ROUTED_EXPERT_COMPUTE = "fp32"

    install = parity.EndpointTrace.install

    @contextmanager
    def without_indexer_trace(endpoints, layer_indices):
        with install(endpoints, layer_indices) as trace:
            yield trace
            for captured in trace.indexer.values():
                captured.clear()

    monkeypatch.setattr(parity.EndpointTrace, "install", without_indexer_trace)
    Probe.setUpClass()
    probe = Probe("test_configured_precision_suite")
    recorder = probe.compare_end_to_end(probe._configured_spec())
    missing = [row for row in recorder.results if row.component == "indexer"]
    assert missing
    assert all(not row.passed for row in missing)
