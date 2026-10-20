from tests.glm5_2_common.config_first import (
    parse_legacy_dotted_override,
    translate_config_first_cli,
)


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


def test_translate_config_first_cli_preserves_legacy_when_disabled(
    monkeypatch,
) -> None:
    monkeypatch.delenv("TORCHTITAN_CONFIG_FIRST_CLI", raising=False)
    argv = ["train_gpu.py", "--module", "glm5", "--config", "debugmodel"]
    assert translate_config_first_cli(argv) is argv


def test_translate_config_first_cli_handles_output_and_overrides(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TORCHTITAN_CONFIG_FIRST_CLI", "1")
    translated = translate_config_first_cli(
        [
            "train_gpu.py",
            "--module",
            "glm5",
            "--config",
            "glm5_debugmodel",
            "--dump_folder=/tmp/output",
            "--training.steps=2",
            "--training.disable_cuda_graphs",
        ]
    )
    assert translated == [
        "train_gpu.py",
        "--output-dir",
        "/tmp/output",
        "--module",
        "torchtitan_test_config_first",
        "--config",
        "legacy_config",
    ]
