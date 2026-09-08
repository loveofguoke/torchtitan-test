# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Scenario-driven entry points for exploratory GLM-5.2 parity workflows.

Unlike formal multi-step precision, parity captures selected forward/backward
intermediates, parameters, gradients, router/indexer decisions, and traces for
root-cause exploration. Paired mode executes both endpoints in one process;
offline mode prepares shared data, captures each endpoint independently, then
compares portable artifacts on CPU. Generation-safe force/resume rules mirror
the formal framework so stale tensor shards cannot enter a new report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from tests.glm5_2_parity.model_config import config_snapshot, load_model_config, model_dimensions

from tests.glm5_2_common.cli import (
    RunAttempt,
    archive_previous_output,
    assert_run_not_active,
    reset_output_generation,
)
from tests.glm5_2_parity.contracts import (
    GLM5_PARITY_CAPTURE_SUITE,
    GLM5_PARITY_FIXTURE_SUITE,
    GLM5_PARITY_SUITE_VERSION,
)


TEST_TARGET = (
    "tests/unit_tests/test_glm5_parity.py::"
    "TestGlm5Parity::test_configured_precision_suite"
)


@dataclass(frozen=True)
class ParityModelConfig:
    vocab_size: int | None = None
    dim: int | None = None
    layers: int | None = None
    dense_layers: int | None = None
    attention_heads: int | None = None
    q_lora_rank: int | None = None
    kv_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int | None = None
    v_head_dim: int | None = None
    dense_hidden_dim: int | None = None
    moe_hidden_dim: int | None = None
    experts: int | None = None
    shared_experts: int | None = None
    router_top_k: int | None = None
    expert_groups: int | None = None
    limited_groups: int | None = None
    route_scale: float | None = None
    index_heads: int | None = None
    index_head_dim: int | None = None
    index_top_k: int | None = None
    max_position_embeddings: int | None = None
    rope_theta: float | None = None
    rope_cache_max_seq_len: int | None = None

    def environment(self) -> dict[str, str]:
        return {
            f"GLM5_PARITY_MODEL_{name.upper()}": str(value)
            for name, value in asdict(self).items()
        }


@dataclass(frozen=True)
class CommonParityConfig:
    model_config: str = "debugmodel"
    native_model_config: dict = field(default_factory=dict)
    data_case: str = "random"
    data_seed: int = 61
    model_seed: int = 61
    batch_size: int = 2
    sequence_length: int = 128
    layers: str = "all"
    components: str = "all"
    component_execution: str = "independent"
    hf_routed_expert_compute: str = "model"
    titan_routed_expert_compute: str = "model"
    model: ParityModelConfig = field(default_factory=ParityModelConfig)
    report_root: str = "parity_reports"
    run_root: str = "parity_runs"


@dataclass(frozen=True)
class OfflineEndpointConfig:
    name: str
    endpoint: str
    device_type: str
    visible_device: str
    visible_devices_env: str
    artifact_name: str


@dataclass(frozen=True, kw_only=True)
class OfflineParityConfig(CommonParityConfig):
    actual: OfflineEndpointConfig
    expected: OfflineEndpointConfig
    fixture_root: str = "parity_fixtures"
    artifact_root: str = "parity_artifacts"
    fixture_name: str = "fixture"
    report_name: str = "actual_vs_expected.html"


@dataclass(frozen=True)
class PairedParityConfig(CommonParityConfig):
    actual_endpoint: str = "titan:fp32"
    expected_endpoint: str = "hf:fp32"
    device: str = "cuda"
    visible_device: str = "7"
    report_name: str = "paired.html"


def _add_config_arguments(parser: argparse.ArgumentParser, config: CommonParityConfig) -> None:
    """Expose the same effective configuration in paired and offline runs."""
    group = parser.add_argument_group("experiment configuration")
    for name, value in asdict(config).items():
        if isinstance(value, (str, int, float)):
            group.add_argument(
                "--" + name.replace("_", "-"), dest=name, type=type(value),
                default=argparse.SUPPRESS, help=f"Override {name}; default: {value}",
            )
    group = parser.add_argument_group("model dimensions (GLM debug defaults)")
    for name, value in asdict(config.model).items():
        option = "--num-layers" if name == "layers" else "--" + name.replace("_", "-")
        group.add_argument(
            option, dest="model_" + name,
            type=float if name in {"route_scale", "rope_theta"} else int,
            default=argparse.SUPPRESS,
            help=f"Override model {name}; default: selected TorchTitan configuration",
        )
    if isinstance(config, OfflineParityConfig):
        for role in ("actual", "expected"):
            group.add_argument(
                f"--{role}-visible-device", default=argparse.SUPPRESS,
                help=f"Physical device for {role}; default: {getattr(config, role).visible_device}",
            )


def _apply_config_arguments(config: CommonParityConfig, arguments: argparse.Namespace):
    values = vars(arguments)
    updates = {name: values[name] for name in asdict(config) if name in values}
    native = load_model_config(updates.get("model_config", config.model_config))
    updates["native_model_config"] = config_snapshot(native)
    defaults = model_dimensions(native)
    defaults.update({name: value for name, value in asdict(config.model).items() if value is not None})
    model = ParityModelConfig(**defaults)
    model_updates = {
        name: values["model_" + name]
        for name in asdict(config.model) if "model_" + name in values
    }
    # A longer test sequence also needs a sufficiently long RoPE cache.
    sequence_length = updates.get("sequence_length", config.sequence_length)
    for name in ("rope_cache_max_seq_len", "max_position_embeddings"):
        if name not in model_updates:
            model_updates[name] = max(getattr(model, name), sequence_length)
    updates["model"] = replace(model, **model_updates)
    if isinstance(config, OfflineParityConfig):
        for role in ("actual", "expected"):
            key = role + "_visible_device"
            if key in values:
                updates[role] = replace(getattr(config, role), visible_device=values[key])
    return replace(config, **updates)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _scenario_id(script_path: str | os.PathLike[str]) -> str:
    return Path(script_path).stem


def _config_json(config: CommonParityConfig, scenario_id: str) -> str:
    payload = {"scenario_id": scenario_id, "configuration": asdict(config)}
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _config_digest(config: CommonParityConfig, scenario_id: str) -> str:
    return hashlib.sha256(
        _config_json(config, scenario_id).encode("utf-8")
    ).hexdigest()


def _path(root: Path, configured_root: str, *parts: str) -> Path:
    base = Path(configured_root)
    if not base.is_absolute():
        base = root / base
    return base.joinpath(*parts)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _completed_artifact_manifest(path: Path) -> dict[str, Any] | None:
    """Read a complete parity artifact without importing the tensor runtime."""

    manifest_path = path / "manifest.json"
    complete_path = path / "complete.json"
    if not manifest_path.is_file() or not complete_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    if complete.get("manifest_sha256") != _file_digest(manifest_path):
        return None
    return manifest


def _artifact_matches_scenario(
    path: Path,
    *,
    suite: str,
    scenario_config_digest: str,
    fixture_digest: str | None = None,
) -> bool:
    manifest = _completed_artifact_manifest(path)
    if manifest is None or manifest.get("status") != "success":
        return False
    if manifest.get("suite") != suite:
        return False
    if manifest.get("suite_version") != GLM5_PARITY_SUITE_VERSION:
        return False
    configuration = manifest.get("configuration", {})
    if configuration.get("scenario_config_digest") != scenario_config_digest:
        return False
    return (
        fixture_digest is None
        or manifest.get("fixture_digest") == fixture_digest
    )


def _fixture_digest(path: Path) -> str | None:
    manifest = _completed_artifact_manifest(path)
    if manifest is None:
        return None
    value = manifest.get("fixture_digest")
    return value if isinstance(value, str) and value else None


def _state_name(stage: str) -> str:
    return f"{stage}_state.json"


def _run_parity_stage(
    *,
    root: Path,
    environment: dict[str, str],
    log_path: Path,
    stage: str,
    scenario_id: str,
    scenario_config_digest: str,
    expected_output: Path,
) -> None:
    state_name = _state_name(stage)
    attempt = RunAttempt.start(
        log_path.parent,
        kind="parity",
        context={
            "stage": stage,
            "scenario_id": scenario_id,
            "scenario_config_digest": scenario_config_digest,
            "output": str(expected_output),
        },
        state_name=state_name,
    )
    try:
        _run_test(root=root, environment=environment, log_path=log_path)
    except BaseException as error:
        attempt.update("failed", error=repr(error))
        raise
    if not expected_output.exists():
        error = RuntimeError(
            f"parity stage {stage!r} did not create {expected_output}"
        )
        attempt.update("failed", error=repr(error))
        raise error
    attempt.update("completed")


def _assert_parity_states_not_active(run_directory: Path) -> None:
    for stage in ("data", "actual_capture", "expected_capture", "compare", "run"):
        assert_run_not_active(
            run_directory if stage == "run" else run_directory / stage,
            state_name=_state_name(stage),
        )


def _common_environment(
    config: CommonParityConfig,
    scenario_id: str,
) -> dict[str, str]:
    environment = os.environ.copy()
    controlled = {
        key
        for key in environment
        if key.startswith("GLM5_PARITY_")
    }
    for key in controlled:
        environment.pop(key)
    environment.update(
        {
            "GLM5_PARITY_SCENARIO_ID": scenario_id,
            "GLM5_PARITY_MODEL_CONFIG": config.model_config,
            "GLM5_PARITY_SCENARIO_CONFIG_DIGEST": _config_digest(
                config, scenario_id
            ),
            "GLM5_PARITY_SCENARIO_CONFIG_JSON": _config_json(
                config, scenario_id
            ),
            "GLM5_PARITY_DATA_CASE": config.data_case,
            "GLM5_PARITY_DATA_SEED": str(config.data_seed),
            "GLM5_PARITY_MODEL_SEED": str(config.model_seed),
            "GLM5_PARITY_BATCH_SIZE": str(config.batch_size),
            "GLM5_PARITY_SEQUENCE_LENGTH": str(config.sequence_length),
            "GLM5_PARITY_LAYERS": config.layers,
            "GLM5_PARITY_COMPONENTS": config.components,
            "GLM5_PARITY_COMPONENT_EXECUTION": config.component_execution,
            "GLM5_PARITY_HF_ROUTED_EXPERT_COMPUTE": (
                config.hf_routed_expert_compute
            ),
            "GLM5_PARITY_TITAN_ROUTED_EXPERT_COMPUTE": (
                config.titan_routed_expert_compute
            ),
        }
    )
    environment.update(config.model.environment())
    return environment


def _run_test(
    *,
    root: Path,
    environment: dict[str, str],
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    configuration = json.loads(environment["GLM5_PARITY_SCENARIO_CONFIG_JSON"])
    invocation = [sys.executable, *sys.argv]
    configuration["command"] = invocation
    configuration["runtime_log"] = str(log_path)
    (log_path.parent / "experiment.json").write_text(
        json.dumps(configuration, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (log_path.parent / "command.txt").write_text(
        (
            subprocess.list2cmdline(invocation)
            if os.name == "nt"
            else shlex.join(invocation)
        )
        + "\n",
        encoding="utf-8",
    )
    (log_path.parent / "README.md").write_text(
        "\n".join(
            (
                f"# GLM-5 parity run: {configuration['scenario_id']}",
                "",
                "This directory contains the execution evidence for one parity "
                "scenario and configuration.",
                "",
                "- `command.txt`: exact user-facing command.",
                "- `experiment.json`: resolved experiment configuration.",
                f"- `{log_path.name}`: complete pytest stdout and stderr.",
                "",
            )
        ),
        encoding="utf-8",
    )
    command = [sys.executable, "-m", "pytest", TEST_TARGET, "-s"]
    print(f"command: {' '.join(command)}")
    print(f"log: {log_path}")
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def _print_configuration(
    config: CommonParityConfig,
    scenario_id: str,
    paths: dict[str, Path],
) -> None:
    payload: dict[str, Any] = {
        "scenario_id": scenario_id,
        "scenario_config_digest": _config_digest(config, scenario_id),
        "configuration": asdict(config),
        "paths": {name: str(path) for name, path in paths.items()},
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def _capture_options(
    role: str,
    endpoint: OfflineEndpointConfig,
) -> list[str]:
    name = endpoint.name.replace("_", "-").lower()
    invalid = any(
        not (character.isalnum() or character == "-")
        for character in name
    )
    if not name or invalid:
        raise ValueError(f"invalid offline endpoint name: {endpoint.name!r}")
    options = [f"--{role}-capture"]
    named_option = f"--{name}-capture"
    if named_option not in options:
        options.append(named_option)
    return options


def _configure_capture_environment(
    environment: dict[str, str],
    endpoint: OfflineEndpointConfig,
    artifact: Path,
) -> None:
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["ASCEND_RT_VISIBLE_DEVICES"] = ""
    if endpoint.visible_devices_env:
        environment[endpoint.visible_devices_env] = endpoint.visible_device
    environment.update(
        {
            "GLM5_PARITY_DEVICE": endpoint.device_type,
            "GLM5_PARITY_MODE": "capture",
            "GLM5_PARITY_ENDPOINT": endpoint.endpoint,
            "GLM5_PARITY_ARTIFACT": str(artifact),
        }
    )


def run_offline_cli(
    config: OfflineParityConfig,
    script_path: str | os.PathLike[str],
) -> None:
    parser = argparse.ArgumentParser(
        description="Run one configured GLM-5.2 offline parity stage."
    )
    actual_options = _capture_options("actual", config.actual)
    expected_options = _capture_options("expected", config.expected)
    duplicate_options = set(actual_options).intersection(expected_options)
    if duplicate_options:
        raise ValueError(
            "offline endpoint names must produce distinct capture flags: "
            + ", ".join(sorted(duplicate_options))
        )
    stages = parser.add_mutually_exclusive_group(required=True)
    stages.add_argument("--data", action="store_const", dest="stage", const="data")
    stages.add_argument(
        *actual_options,
        action="store_const",
        dest="stage",
        const="actual_capture",
        help=f"capture actual endpoint {config.actual.endpoint}",
    )
    stages.add_argument(
        *expected_options,
        action="store_const",
        dest="stage",
        const="expected_capture",
        help=f"capture expected endpoint {config.expected.endpoint}",
    )
    stages.add_argument(
        "--compare", action="store_const", dest="stage", const="compare"
    )
    stages.add_argument(
        "--print-config",
        action="store_const",
        dest="stage",
        const="print_config",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "start a new generation for the selected stage; forcing data "
            "removes every dependent capture and report"
        ),
    )
    _add_config_arguments(parser, config)
    arguments = parser.parse_args()
    original_config = config
    config = _apply_config_arguments(config, arguments)

    root = _repo_root()
    scenario_id = _scenario_id(script_path)
    if config != original_config:
        scenario_id += "-" + _config_digest(config, scenario_id)[:10]
    fixture = _path(
        root,
        config.fixture_root,
        scenario_id,
        config.fixture_name,
    )
    actual_artifact = _path(
        root,
        config.artifact_root,
        scenario_id,
        config.actual.artifact_name,
    )
    expected_artifact = _path(
        root,
        config.artifact_root,
        scenario_id,
        config.expected.artifact_name,
    )
    report = _path(
        root,
        config.report_root,
        scenario_id,
        config.report_name,
    )
    run_directory = _path(
        root,
        config.run_root,
        scenario_id,
    )
    stage_run_directory = run_directory / arguments.stage
    log = stage_run_directory / "runtime.log"
    paths = {
        "fixture": fixture,
        "actual_artifact": actual_artifact,
        "expected_artifact": expected_artifact,
        "report": report,
        "log": log,
    }
    if arguments.stage == "print_config":
        _print_configuration(config, scenario_id, paths)
        return

    scenario_config_digest = _config_digest(config, scenario_id)
    if arguments.force:
        _assert_parity_states_not_active(run_directory)
        if arguments.stage == "data":
            reset_output_generation(
                (
                    fixture.parent,
                    actual_artifact.parent,
                    report.parent,
                    run_directory,
                ),
                label="parity scenario",
            )
        else:
            selected_outputs: list[Path] = [
                stage_run_directory,
            ]
            if arguments.stage == "actual_capture":
                selected_outputs.extend((actual_artifact, report))
            elif arguments.stage == "expected_capture":
                selected_outputs.extend((expected_artifact, report))
            else:
                selected_outputs.append(report)
            reset_output_generation(
                selected_outputs,
                label=f"parity {arguments.stage}",
            )

    environment = _common_environment(config, scenario_id)
    environment["GLM5_PARITY_FIXTURE"] = str(fixture)
    environment["GLM5_PARITY_RUN_ID"] = f"{scenario_id}-{arguments.stage}"
    if arguments.stage == "data":
        if fixture.exists() and not arguments.force:
            if _artifact_matches_scenario(
                fixture,
                suite=GLM5_PARITY_FIXTURE_SUITE,
                scenario_config_digest=scenario_config_digest,
            ):
                print(f"Skip completed parity fixture: {fixture}")
                return
            fixture_manifest = _completed_artifact_manifest(fixture)
            if (
                fixture_manifest is not None
                and fixture_manifest.get("status") == "success"
            ):
                raise FileExistsError(
                    "parity fixture belongs to a different configuration or "
                    f"suite version; pass --force to replace it: {fixture}"
                )
            dependent_outputs = [
                path
                for path in (actual_artifact, expected_artifact, report)
                if path.exists()
            ]
            if dependent_outputs:
                raise RuntimeError(
                    "incomplete fixture has dependent outputs; pass --force "
                    "to reset the complete parity generation: "
                    + ", ".join(str(path) for path in dependent_outputs)
                )
            archived = archive_previous_output(fixture)
            print(f"Retry incomplete parity fixture; archived: {archived}")
        environment["GLM5_PARITY_MODE"] = "prepare"
    elif arguments.stage == "actual_capture":
        fixture_value = _fixture_digest(fixture)
        if not fixture.is_dir() or fixture_value is None:
            raise FileNotFoundError(f"fixture not found: {fixture}")
        if not _artifact_matches_scenario(
            fixture,
            suite=GLM5_PARITY_FIXTURE_SUITE,
            scenario_config_digest=scenario_config_digest,
        ):
            raise RuntimeError(
                "fixture configuration or suite version does not match this "
                f"scenario: {fixture}"
            )
        if actual_artifact.exists() and not arguments.force:
            if _artifact_matches_scenario(
                actual_artifact,
                suite=GLM5_PARITY_CAPTURE_SUITE,
                scenario_config_digest=scenario_config_digest,
                fixture_digest=fixture_value,
            ):
                print(f"Skip completed parity capture: {actual_artifact}")
                return
            actual_manifest = _completed_artifact_manifest(actual_artifact)
            if (
                actual_manifest is not None
                and actual_manifest.get("status") == "success"
            ):
                raise FileExistsError(
                    "actual capture belongs to a different fixture, "
                    "configuration, or suite version; pass --force to replace "
                    f"it: {actual_artifact}"
                )
            archived = archive_previous_output(actual_artifact)
            print(f"Retry incomplete actual capture; archived: {archived}")
            archive_previous_output(report)
        _configure_capture_environment(
            environment,
            config.actual,
            actual_artifact,
        )
    elif arguments.stage == "expected_capture":
        fixture_value = _fixture_digest(fixture)
        if not fixture.is_dir() or fixture_value is None:
            raise FileNotFoundError(f"fixture not found: {fixture}")
        if not _artifact_matches_scenario(
            fixture,
            suite=GLM5_PARITY_FIXTURE_SUITE,
            scenario_config_digest=scenario_config_digest,
        ):
            raise RuntimeError(
                "fixture configuration or suite version does not match this "
                f"scenario: {fixture}"
            )
        if expected_artifact.exists() and not arguments.force:
            if _artifact_matches_scenario(
                expected_artifact,
                suite=GLM5_PARITY_CAPTURE_SUITE,
                scenario_config_digest=scenario_config_digest,
                fixture_digest=fixture_value,
            ):
                print(f"Skip completed parity capture: {expected_artifact}")
                return
            expected_manifest = _completed_artifact_manifest(expected_artifact)
            if (
                expected_manifest is not None
                and expected_manifest.get("status") == "success"
            ):
                raise FileExistsError(
                    "expected capture belongs to a different fixture, "
                    "configuration, or suite version; pass --force to replace "
                    f"it: {expected_artifact}"
                )
            archived = archive_previous_output(expected_artifact)
            print(f"Retry incomplete expected capture; archived: {archived}")
            archive_previous_output(report)
        _configure_capture_environment(
            environment,
            config.expected,
            expected_artifact,
        )
    else:
        missing = [
            str(path)
            for path in (actual_artifact, expected_artifact)
            if not path.is_dir()
        ]
        if missing:
            raise FileNotFoundError(
                "capture artifacts not found: " + ", ".join(missing)
            )
        report.parent.mkdir(parents=True, exist_ok=True)
        environment.update(
            {
                "GLM5_PARITY_MODE": "compare",
                "GLM5_PARITY_ACTUAL_ARTIFACT": str(actual_artifact),
                "GLM5_PARITY_EXPECTED_ARTIFACT": str(expected_artifact),
                "GLM5_PARITY_REPORT": str(report),
            }
        )
    _print_configuration(config, scenario_id, paths)
    expected_output = {
        "data": fixture,
        "actual_capture": actual_artifact,
        "expected_capture": expected_artifact,
        "compare": report,
    }[arguments.stage]
    _run_parity_stage(
        root=root,
        environment=environment,
        log_path=log,
        stage=arguments.stage,
        scenario_id=scenario_id,
        scenario_config_digest=scenario_config_digest,
        expected_output=expected_output,
    )


def run_paired_cli(
    config: PairedParityConfig,
    script_path: str | os.PathLike[str],
) -> None:
    parser = argparse.ArgumentParser(
        description="Run one configured GLM-5.2 paired parity scenario."
    )
    stages = parser.add_mutually_exclusive_group(required=True)
    stages.add_argument("--run", action="store_const", dest="stage", const="run")
    stages.add_argument(
        "--print-config",
        action="store_const",
        dest="stage",
        const="print_config",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="remove the previous paired report and log before rerunning",
    )
    _add_config_arguments(parser, config)
    arguments = parser.parse_args()
    original_config = config
    config = _apply_config_arguments(config, arguments)

    root = _repo_root()
    scenario_id = _scenario_id(script_path)
    if config != original_config:
        scenario_id += "-" + _config_digest(config, scenario_id)[:10]
    report = _path(
        root,
        config.report_root,
        scenario_id,
        config.report_name,
    )
    run_directory = _path(
        root,
        config.run_root,
        scenario_id,
    )
    log = run_directory / "runtime.log"
    paths = {"report": report, "log": log}
    if arguments.stage == "print_config":
        _print_configuration(config, scenario_id, paths)
        return

    scenario_config_digest = _config_digest(config, scenario_id)
    if arguments.force:
        assert_run_not_active(run_directory, state_name=_state_name("run"))
        reset_output_generation(
            (report, run_directory),
            label="paired parity",
        )

    report.parent.mkdir(parents=True, exist_ok=True)
    environment = _common_environment(config, scenario_id)
    environment.update(
        {
            "GLM5_PARITY_MODE": "paired",
            "GLM5_PARITY_DEVICE": config.device,
            "GLM5_PARITY_ACTUAL": config.actual_endpoint,
            "GLM5_PARITY_EXPECTED": config.expected_endpoint,
            "GLM5_PARITY_REPORT": str(report),
        }
    )
    if config.device == "cuda":
        environment["CUDA_VISIBLE_DEVICES"] = config.visible_device
        environment["ASCEND_RT_VISIBLE_DEVICES"] = ""
    elif config.device == "npu":
        environment["ASCEND_RT_VISIBLE_DEVICES"] = config.visible_device
        environment["CUDA_VISIBLE_DEVICES"] = ""
    _print_configuration(config, scenario_id, paths)
    _run_parity_stage(
        root=root,
        environment=environment,
        log_path=log,
        stage="run",
        scenario_id=scenario_id,
        scenario_config_digest=scenario_config_digest,
        expected_output=report,
    )
