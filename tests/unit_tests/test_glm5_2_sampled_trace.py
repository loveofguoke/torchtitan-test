# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import pytest
import torch

from tests.glm5_2_precision.compare_sampled_trace import _metric
from tests.glm5_2_precision.sampled_trace import (
    _parse_detail,
    _summary,
    _trace_module,
)
from tests.glm5_2_precision.workflow import FormalTrainingConfig


def test_integer_trace_distinguishes_order_from_set_membership() -> None:
    gpu = _summary(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32))
    npu = _summary(torch.tensor([[3, 1, 2], [6, 4, 5]], dtype=torch.int32))

    metric = _metric(gpu, npu)

    assert not metric["exact"]
    assert metric["unordered_last_dim_exact"]
    assert metric["sampled_row_ordered_match_fraction"] == 0.0
    assert metric["sampled_row_set_match_fraction"] == 1.0
    assert metric["sampled_row_mean_jaccard"] == 1.0


def test_integer_trace_reports_changed_topk_membership() -> None:
    gpu = _summary(torch.tensor([[1, 2, 3]], dtype=torch.int32))
    npu = _summary(torch.tensor([[1, 2, 4]], dtype=torch.int32))

    metric = _metric(gpu, npu)

    assert not metric["unordered_last_dim_exact"]
    assert metric["sampled_row_set_match_fraction"] == 0.0
    assert metric["sampled_row_mean_jaccard"] == 0.5


def test_trace_detail_selects_indexer_and_router_children() -> None:
    detail = _parse_detail("indexer,router")

    assert _trace_module("layers.0.attention.indexer.topk", detail)
    assert _trace_module("layers.4.moe.router.gate", detail)
    assert not _trace_module("layers.4.moe.routed_experts", detail)


def test_trace_detail_requires_trace_steps() -> None:
    with pytest.raises(ValueError, match="requires exploratory_steps"):
        FormalTrainingConfig(exploratory_trace_detail=("indexer",))


def test_trace_detail_rejects_unknown_groups() -> None:
    with pytest.raises(ValueError, match="unsupported exploratory_trace_detail"):
        FormalTrainingConfig(
            exploratory_steps=(1,),
            exploratory_trace_detail=("attention",),
        )
