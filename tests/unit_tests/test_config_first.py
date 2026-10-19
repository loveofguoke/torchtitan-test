from tests.glm5_2_common.config_first import parse_legacy_dotted_override


def test_parse_legacy_dotted_override_normalizes_boolean_negation() -> None:
    assert parse_legacy_dotted_override(
        "--parallelism.no-enable-sequence-parallel"
    ) == ("parallelism.enable_sequence_parallel", False)


def test_parse_legacy_dotted_override_normalizes_positive_flag() -> None:
    assert parse_legacy_dotted_override(
        "--metrics.disable-color-printing"
    ) == ("metrics.disable_color_printing", True)


def test_parse_legacy_dotted_override_decodes_assigned_value() -> None:
    assert parse_legacy_dotted_override("--training.steps=2") == (
        "training.steps",
        2,
    )
