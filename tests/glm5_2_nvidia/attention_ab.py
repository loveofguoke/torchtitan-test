"""Single-GPU, profiler-off optimizer-step comparison for TorchTitan PR 5578."""

import gc
import json
import math
import statistics
import time
from dataclasses import fields

import torch
from torchtitan.distributed.activation_remat import DefaultActivationRematConfig
from torchtitan.models.common.attention import FlexInnerAttention, MLAAttention
from torchtitan.models.deepseek_v3.attention import DeepSeekMLAAttention
from torchtitan.models.deepseek_v3.model import DeepSeekV3Model, DeepSeekV3TransformerBlock
from torchtitan.tools.utils import set_default_dtype

from tests.glm5_2_common.experiment_configs import glm5_scaled_model_config


T, WARMUP, STEPS = 8192, 3, 10


def base(cls, source, **overrides):
    return cls(**{f.name: overrides.get(f.name, getattr(source, f.name))
                  for f in fields(cls) if f.init})


def dense_config(config):
    layers = []
    for layer in config.layers:
        attn = layer.attention
        mla = base(MLAAttention.Config, attn.inner_attention,
                   inner_attention=FlexInnerAttention.Config())
        layers.append(base(DeepSeekV3TransformerBlock.Config, layer,
                           attention=base(DeepSeekMLAAttention.Config, attn,
                                          inner_attention=mla)))
    return base(DeepSeekV3Model.Config, config, layers=layers)


def measure(config, sparse, state, tokens, positions):
    torch.manual_seed(61)
    print("phase=build model and move to CUDA", flush=True)
    with set_default_dtype(torch.bfloat16):
        model = config.build().to("cuda")
    print("phase=initialize weights", flush=True)
    model.init_states()
    print("phase=copy shared initial state", flush=True)
    if sparse:
        state.update({k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                      if ".indexer." not in k})
    else:
        model.load_state_dict(state, strict=True)
    print("phase=configure remat and optimizer", flush=True)
    DefaultActivationRematConfig().apply(model)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=1e-4, fused=True)
    denominator = torch.tensor([T], device="cuda", dtype=torch.float32)
    inner = config.layers[0].attention.inner_attention
    elapsed = []
    for i in range(WARMUP + STEPS):
        print(f"step={i+1}/{WARMUP+STEPS} begin warmup={i < WARMUP}", flush=True)
        torch.cuda.synchronize()
        if i == WARMUP:
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        metadata = ({"attention_masks": model.get_attention_masks(positions)} if sparse
                    else {"attention_metadata": {
                        MLAAttention: inner.build_attention_metadata(positions)}})
        if i == 0:
            print("phase=first forward (may compile)", flush=True)
        logits = model(tokens[:-1], positions=positions,
                       aux_loss_denominators=denominator, **metadata)
        loss = torch.nn.functional.cross_entropy(logits.float(), tokens[1:])
        if i == 0:
            print("phase=first backward (may compile)", flush=True)
        loss.backward()
        if i == 0:
            print("phase=first AdamW step and GPU synchronization", flush=True)
        optimizer.step()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        if not math.isfinite(loss.item()):
            raise ValueError(f"non-finite loss at step {i + 1}")
        print(f"step={i+1} warmup={i < WARMUP} seconds={seconds:.4f} loss={loss.item():.6f}",
              flush=True)
        if i >= WARMUP:
            elapsed.append(seconds)
        del logits, loss, metadata
    return dict(tokens_per_second=T * STEPS / sum(elapsed),
                median_step_ms=1000 * statistics.median(elapsed),
                peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30)


def main():
    config = glm5_scaled_model_config(seq_len=T)
    torch.manual_seed(61)
    tokens = torch.randint(config.vocab_size, (T + 1,)).cuda()
    positions = torch.arange(T, device="cuda")
    print(f"{torch.cuda.get_device_name()} torch={torch.__version__} BF16 T={T} "
          f"topk=512 layers={len(config.layers)} remat=default seed=61 "
          f"warmup={WARMUP} measured={STEPS}", flush=True)
    state, results = {}, {}
    for name, cfg in (("sparse_frozen_indexer", config), ("dense_causal_flex", dense_config(config))):
        print(name, flush=True)
        results[name] = measure(cfg, cfg is config, state, tokens, positions)
        gc.collect()
        torch.cuda.empty_cache()
    results["sparse_over_dense_tokens_per_second"] = (
        results["sparse_frozen_indexer"]["tokens_per_second"]
        / results["dense_causal_flex"]["tokens_per_second"])
    print("ATTENTION_AB " + json.dumps(results), flush=True)
