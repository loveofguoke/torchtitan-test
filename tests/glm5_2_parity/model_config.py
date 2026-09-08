"""Read GLM configuration factories without owning a second model definition."""

from dataclasses import fields, is_dataclass
from enum import Enum
from functools import partial


def config_snapshot(value):
    """Portable native configuration provenance, including non-shape settings."""
    if is_dataclass(value):
        return {f.name: config_snapshot(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): config_snapshot(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [config_snapshot(v) for v in value]
    if isinstance(value, partial):
        return dict(function=config_snapshot(value.func), args=config_snapshot(value.args),
                    keywords=config_snapshot(value.keywords))
    if isinstance(value, Enum):
        return config_snapshot(value.value)
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

def load_model_config(flavor="debugmodel"):
    from torchtitan.models.glm5 import glm5_configs

    return glm5_configs[flavor]()


def model_dimensions(config):
    """Project the native configuration into the HF adapter's scalar schema."""
    attention = config.layers[0].attention
    indexer = attention.indexer
    dense = next(layer.feed_forward for layer in config.layers if layer.feed_forward is not None)
    moe = next(layer.moe for layer in config.layers if layer.moe is not None)
    hidden = moe.routed_experts.inner_experts.hidden_dim
    router = moe.router
    return dict(
        vocab_size=config.vocab_size, dim=config.dim, layers=len(config.layers),
        dense_layers=sum(layer.moe is None for layer in config.layers),
        attention_heads=attention.n_heads, q_lora_rank=attention.q_lora_rank,
        kv_lora_rank=attention.kv_lora_rank,
        qk_nope_head_dim=attention.qk_nope_head_dim,
        qk_rope_head_dim=attention.qk_rope_head_dim, v_head_dim=attention.v_head_dim,
        dense_hidden_dim=dense.w1.out_features, moe_hidden_dim=hidden,
        experts=moe.num_experts,
        shared_experts=moe.shared_experts.w1.out_features // hidden,
        router_top_k=router.top_k, expert_groups=router.num_expert_groups,
        limited_groups=router.num_limited_groups, route_scale=router.route_scale,
        index_heads=indexer.n_heads, index_head_dim=indexer.head_dim,
        index_top_k=indexer.index_topk,
        max_position_embeddings=attention.rope.max_context_length,
        rope_theta=attention.rope.theta,
        rope_cache_max_seq_len=attention.rope.max_context_length,
    )


SIZE_ALIASES = dict(
    layers="num_layers", dense_layers="num_dense_layers",
    attention_heads="num_attention_heads", experts="num_experts",
    shared_experts="num_shared_experts", expert_groups="router_num_expert_groups",
    limited_groups="router_num_limited_groups", route_scale="router_route_scale",
    index_heads="index_num_heads",
)
