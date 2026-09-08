from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import spmd_types as spmd

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.decoder_sharding import dense_activation_placement
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig, SpmdLayout
from torchtitanturbo.distributed.fsdp import (
    _cast_mixed_gradients_for_reduce,
    _patch_distributed_set_timeout,
)
from torchtitanturbo.models.glm5.final_norm_fsdp import (
    apply_patch as apply_glm5_final_norm_fsdp_patch,
)
from torchtitanturbo.models.glm5.patch import (
    _fix_tp_only_routed_expert_grad_layout,
    apply_patch as apply_glm5_patch,
)
from torchtitanturbo.non_npu_compat.runtime import _step_pre_split_microbatches
from torchtitanturbo.models.common.npu_rmsnorm import NpuRMSNorm
from torchtitanturbo.models.glm5.npu_router import (
    NpuGlm5TokenChoiceTopKRouter,
    _local_routing_map,
    _npu_moe_forward,
)


def test_mixed_fsdp_gradients_are_cast_to_reduce_dtype() -> None:
    gradients = [
        torch.tensor([1.25], dtype=torch.bfloat16),
        torch.tensor([2.5], dtype=torch.float32),
    ]

    actual = _cast_mixed_gradients_for_reduce(gradients, torch.float32)

    assert [gradient.dtype for gradient in actual] == [torch.float32, torch.float32]
    assert torch.equal(actual[0], gradients[0].float())
    assert torch.equal(actual[1], gradients[1])


def test_uniform_fsdp_gradients_are_not_copied() -> None:
    gradients = [torch.ones(1), torch.zeros(1)]

    assert _cast_mixed_gradients_for_reduce(gradients, torch.float32) is gradients


@patch("torch_npu.npu_rms_norm")
def test_npu_rmsnorm_keeps_independent_weight_storage_for_backward(mock_op) -> None:
    mock_op.side_effect = lambda value, weight, epsilon: (value * weight,)
    norm = NpuRMSNorm.Config(normalized_shape=4).build()
    value = torch.randn(2, 4)

    norm(value)

    passed_weight = mock_op.call_args.args[1]
    assert passed_weight.data_ptr() != norm.weight.data_ptr()
    assert torch.equal(passed_weight, norm.weight)
    assert passed_weight.grad_fn is not None


@patch("torchtitanturbo.models.common.npu_rmsnorm.torch_npu.npu_rms_norm")
def test_npu_rmsnorm_unwraps_dtensor_weight_for_local_pipeline_input(mock_op) -> None:
    class FakeDTensor:
        def __init__(self, local):
            self.local = local

        def full_tensor(self):
            return self.local

    mock_op.side_effect = lambda value, weight, epsilon: (value * weight,)
    local_weight = torch.ones(4, requires_grad=True)
    norm = SimpleNamespace(weight=FakeDTensor(local_weight), eps=1e-5)

    with patch(
        "torchtitanturbo.models.common.npu_rmsnorm.DTensor", FakeDTensor
    ):
        NpuRMSNorm.forward(norm, torch.randn(2, 4))

    passed_weight = mock_op.call_args.args[1]
    assert isinstance(passed_weight, torch.Tensor)
    assert passed_weight.data_ptr() != local_weight.data_ptr()
    assert passed_weight.grad_fn is not None


@patch("torchtitanturbo.models.common.npu_rmsnorm.torch_npu.npu_rms_norm")
def test_npu_rmsnorm_moves_weight_to_pipeline_activation_mesh(mock_op) -> None:
    seen = {}

    class FakeDTensor:
        def __init__(self, local, mesh, placements=("replicate",)):
            self.local = local
            self.device_mesh = mesh
            self.placements = placements

        def full_tensor(self):
            return self.local

        def clone(self):
            return self

        @classmethod
        def from_local(
            cls, local, *, device_mesh, placements, run_check
        ):
            seen.update(
                mesh=device_mesh,
                placements=placements,
                run_check=run_check,
            )
            return cls(local, device_mesh, placements)

    mock_op.side_effect = lambda value, weight, epsilon: (value,)
    norm = SimpleNamespace(
        weight=FakeDTensor(torch.ones(4), "fsdp-tp"),
        eps=1e-5,
    )
    value = FakeDTensor(torch.randn(2, 4), "tp")

    with (
        patch("torchtitanturbo.models.common.npu_rmsnorm.DTensor", FakeDTensor),
        patch(
            "torchtitanturbo.models.common.npu_rmsnorm.Replicate",
            lambda: "replicate",
        ),
    ):
        NpuRMSNorm.forward(norm, value)

    assert seen == {
        "mesh": "tp",
        "placements": ("replicate",),
        "run_check": False,
    }


def test_npu_router_uses_out_of_place_routing_scatter() -> None:
    scores = torch.randn(2, 3, 8)
    expert_ids = torch.tensor(
        [[[0, 2], [1, 3], [4, 7]], [[2, 5], [0, 6], [3, 4]]]
    )

    routing_map = _local_routing_map(scores, expert_ids)

    assert routing_map.sum(dim=-1).eq(2).all()
    assert routing_map[..., 0].sum() == 2


def test_npu_moe_forward_supports_token_first_tensors() -> None:
    expert_ids = torch.tensor([[0, 1], [1, 2], [2, 3]])
    scores = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    topk_scores = scores.gather(-1, expert_ids)
    seen = {}

    class Router:
        def __call__(self, value, expert_bias):
            seen["router_input"] = value
            seen["expert_bias"] = expert_bias
            return topk_scores, expert_ids, scores

    class RoutedExperts:
        def __call__(self, value, routed_scores, routed_ids, counts):
            seen["counts"] = counts
            assert routed_scores is topk_scores
            assert routed_ids is expert_ids
            return value * 2

    class SharedExperts:
        def __call__(self, value):
            return torch.ones_like(value)

    moe = SimpleNamespace(
        router=Router(),
        routed_experts=RoutedExperts(),
        shared_experts=SharedExperts(),
        expert_bias_E=None,
        tokens_per_expert_E=torch.zeros(4),
        training=True,
    )
    value = torch.randn(3, 5)

    actual = _npu_moe_forward(moe, value)

    assert seen["router_input"] is value
    assert seen["expert_bias"] is None
    assert torch.equal(seen["counts"], torch.tensor([1, 2, 2, 1]))
    assert torch.equal(moe.tokens_per_expert_E, seen["counts"].float())
    assert torch.equal(actual, value * 2 + 1)


def test_npu_router_group_limiting_matches_base_semantics() -> None:
    router = SimpleNamespace(
        num_limited_groups=1,
        num_expert_groups=2,
        num_experts=8,
    )
    scores = torch.arange(16, dtype=torch.float32).reshape(2, 8)

    actual = NpuGlm5TokenChoiceTopKRouter._get_node_limited_routing_scores(
        router, scores
    )

    assert torch.isneginf(actual[:, :4]).all()
    assert torch.equal(actual[:, 4:], scores[:, 4:])


def test_pre_split_pipeline_bridge_forwards_loss_kwargs_and_npu_rng_device() -> None:
    seen = {}

    class Stage:
        device = torch.device("npu:0")

        def clear_runtime_states(self):
            seen["cleared"] = True

    class Schedule:
        _has_backward = False
        _stage = Stage()

        @staticmethod
        def _loss_fn(output, target, *, global_valid_tokens):
            seen["global_valid_tokens"] = global_valid_tokens
            return output + target

        def _step_microbatches(self, **kwargs):
            with torch.random.fork_rng():
                pass
            self._loss_fn(torch.tensor(1), torch.tensor(2))
            seen["kwargs"] = kwargs

    schedule = Schedule()
    original_loss_fn = schedule._loss_fn
    with patch("torch.random.fork_rng") as fork_rng:
        _step_pre_split_microbatches(
            schedule,
            arg_mbs=[(torch.tensor(1),)],
            loss_kwargs={"global_valid_tokens": 7},
        )

    assert seen["cleared"]
    assert seen["global_valid_tokens"] == 7
    assert seen["kwargs"] == {"arg_mbs": [(torch.tensor(1),)]}
    fork_rng.assert_called_once_with(device_type="npu")
    assert schedule._loss_fn == original_loss_fn


def test_glm5_uses_npu_safe_rmsnorm_after_turbo_patch() -> None:
    import torchtitan.models.common.moe as moe
    import torchtitan.models.glm5 as glm5

    apply_glm5_patch()
    assert glm5.RMSNorm is NpuRMSNorm
    assert moe.MoE.forward is _npu_moe_forward


def test_glm5_final_norm_and_lm_head_use_independent_fsdp_units() -> None:
    import torchtitan.distributed.fsdp as fsdp
    import torchtitan.models.glm5.parallelize as glm5_parallelize

    calls = []
    norm = object()
    lm_head = object()

    def fake_fully_shard(module_or_modules, *args, **kwargs):
        calls.append((module_or_modules, args, kwargs))
        return module_or_modules

    def fake_apply(model, *args, **kwargs):
        return fsdp.fully_shard(
            [model.norm, model.lm_head],
            "dp-mesh",
            reshard_after_forward=False,
        )

    with (
        patch.object(fsdp, "fully_shard", fake_fully_shard),
        patch.object(glm5_parallelize, "apply_fsdp_to_decoder", fake_apply),
    ):
        apply_glm5_final_norm_fsdp_patch()
        wrapped = glm5_parallelize.apply_fsdp_to_decoder
        model = SimpleNamespace(
            enable_weight_tying=False,
            norm=norm,
            lm_head=lm_head,
        )

        assert wrapped(model) is lm_head
        assert calls == [
            (
                norm,
                ("dp-mesh",),
                {"reshard_after_forward": False},
            ),
            (
                lm_head,
                ("dp-mesh",),
                {"reshard_after_forward": False},
            ),
        ]
        assert fsdp.fully_shard is fake_fully_shard

        calls.clear()
        model.enable_weight_tying = True
        wrapped(model)
        assert calls == [
            (
                [norm, lm_head],
                ("dp-mesh",),
                {"reshard_after_forward": False},
            )
        ]

        apply_glm5_final_norm_fsdp_patch()
        assert glm5_parallelize.apply_fsdp_to_decoder is wrapped


def test_tp_only_routed_expert_gradients_are_partial_on_tp() -> None:
    routing_counts = SpmdLayout(
        {
            MeshAxisName.DP: spmd.P,
            MeshAxisName.CP: spmd.P,
            MeshAxisName.TP: spmd.R,
        }
    )
    replicated_activation = dense_activation_placement(tp=spmd.R, cp=spmd.R)
    routed_cfg = ShardingConfig(
        in_src_shardings={
            "x_BLD": replicated_activation,
            "topk_scores_BLK": replicated_activation,
            "topk_expert_ids_BLK": replicated_activation,
            "num_local_tokens_per_expert_E": routing_counts,
        },
        in_dst_shardings={
            "x_BLD": replicated_activation,
            "topk_scores_BLK": replicated_activation,
            "topk_expert_ids_BLK": replicated_activation,
            "num_local_tokens_per_expert_E": routing_counts,
        },
        out_src_shardings=dense_activation_placement(tp=spmd.P, cp=spmd.R),
        local_map=LocalMapConfig(in_grad_placements=None),
    )
    moe_cfg = SimpleNamespace(
        routed_experts=SimpleNamespace(sharding_config=routed_cfg)
    )

    _fix_tp_only_routed_expert_grad_layout(moe_cfg)

    assert routed_cfg.local_map is not None
    assert routed_cfg.local_map.in_grad_placements == (
        dense_activation_placement(tp=spmd.P, cp=spmd.R),
        dense_activation_placement(tp=spmd.P, cp=spmd.R),
        dense_activation_placement(tp=spmd.P, cp=spmd.R),
        routing_counts,
    )


def test_timeout_compat_ignores_degree_one_fake_backend() -> None:
    class FakeGroup:
        def set_timeout(self, _timeout) -> None:
            raise RuntimeError("Backend fake does not support setting timeout")

    _patch_distributed_set_timeout()
    torch.distributed.set_timeout(timedelta(seconds=1), FakeGroup())
