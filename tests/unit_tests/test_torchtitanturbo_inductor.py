# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Regression tests for NPU Inductor compatibility owned by TorchTitanTurbo."""

from __future__ import annotations

from torchtitanturbo.compiler.inductor import (
    _block_mask_unpack_arity,
    _legacy_block_mask_lowering_wrapper,
    _legacy_npu_block_mask,
    _patch_npu_compile_option_serialization,
    _sorted_npu_compile_options,
)


def _legacy_lowering(block_mask):
    (
        field_0,
        field_1,
        field_2,
        field_3,
        field_4,
        field_5,
        field_6,
        field_7,
        field_8,
        field_9,
        field_10,
        field_11,
        field_12,
    ) = block_mask
    return field_0


def _current_lowering(block_mask):
    (
        field_0,
        field_1,
        field_2,
        field_3,
        field_4,
        field_5,
        field_6,
        field_7,
        field_8,
        field_9,
        field_10,
        field_11,
        field_12,
        field_13,
        field_14,
        field_15,
        field_16,
    ) = block_mask
    return field_0


def test_block_mask_protocol_drops_only_dq_scheduling_metadata() -> None:
    current = tuple(range(17))

    legacy = _legacy_npu_block_mask(current)

    assert legacy == tuple(range(10)) + (14, 15, 16)
    assert _legacy_npu_block_mask(legacy) is legacy


def test_block_mask_unpack_arity_detects_legacy_lowering_only() -> None:
    assert _block_mask_unpack_arity(_legacy_lowering) == 13
    assert _block_mask_unpack_arity(_current_lowering) == 17


def test_forward_lowering_wrapper_normalizes_only_lowering_argument() -> None:
    captured = None

    def handler(*args):
        nonlocal captured
        captured = args
        return "lowered"

    wrapped = _legacy_block_mask_lowering_wrapper(handler, mask_arg_index=4)
    current = tuple(range(17))

    result = wrapped("q", "k", "v", "score", current, "scale")

    assert result == "lowered"
    assert captured is not None
    assert captured[:4] == ("q", "k", "v", "score")
    assert captured[4] == tuple(range(10)) + (14, 15, 16)
    assert current == tuple(range(17))


def test_backward_lowering_wrapper_preserves_other_arguments() -> None:
    captured = None

    def handler(*args):
        nonlocal captured
        captured = args
        return "lowered"

    wrapped = _legacy_block_mask_lowering_wrapper(handler, mask_arg_index=9)
    current = tuple(range(17))
    arguments = tuple(range(9)) + (current, "scale", "options")

    result = wrapped(*arguments)

    assert result == "lowered"
    assert captured is not None
    assert captured[:9] == tuple(range(9))
    assert captured[9] == tuple(range(10)) + (14, 15, 16)
    assert captured[10:] == ("scale", "options")


def test_npu_compile_options_are_serialized_in_stable_key_order() -> None:
    metadata = {"z_option": 3, "a_option": 1, "ignored": 2}

    options = _sorted_npu_compile_options(
        metadata,
        frozenset(("z_option", "missing", "a_option")),
    )

    assert list(options) == ["a_option", "z_option"]
    assert options == {"a_option": 1, "z_option": 3}


def test_npu_compile_option_patch_is_safe_when_hook_is_unavailable(
    monkeypatch,
) -> None:
    import torch_npu._inductor.select_algorithm as select_algorithm

    monkeypatch.delattr(
        select_algorithm,
        "_add_npu_template_compile_options_to_triton_meta",
        raising=False,
    )

    assert not _patch_npu_compile_option_serialization()


def test_npu_compile_option_patch_is_stable_and_idempotent(monkeypatch) -> None:
    import torch_npu._inductor.select_algorithm as select_algorithm

    def original(triton_meta, metadata, compile_option_keys):
        raise AssertionError("the unstable serializer must be replaced")

    monkeypatch.setattr(
        select_algorithm,
        "_add_npu_template_compile_options_to_triton_meta",
        original,
        raising=False,
    )

    assert _patch_npu_compile_option_serialization()
    patched = (
        select_algorithm._add_npu_template_compile_options_to_triton_meta
    )
    assert not _patch_npu_compile_option_serialization()

    triton_meta = {}
    patched(
        triton_meta,
        {"z_option": 3, "a_option": 1, "ignored": 2},
        frozenset(("z_option", "missing", "a_option")),
    )

    assert list(triton_meta["npu_compile_options"]) == [
        "a_option",
        "z_option",
    ]
    assert triton_meta["npu_compile_options"] == {
        "a_option": 1,
        "z_option": 3,
    }
