#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Run the official MindStudio graph-accuracy stages under one experiment."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.glm5_2_common.execution import TrainingFeature  # noqa: E402
from tests.glm5_2_common.topology import standard_topologies  # noqa: E402
from tests.glm5_2_graph.config import GraphFeatureConfig  # noqa: E402
from tests.glm5_2_mindstudio.config import (  # noqa: E402
    MindStudioExperimentConfig,
    MsProbeCompileConfig,
    MsProbeDumpConfig,
    MsProbeMonitorConfig,
)
from tests.glm5_2_mindstudio.workflow import run_mindstudio_cli  # noqa: E402
from tests.glm5_2_mindstudio.migration_benchmark import (  # noqa: E402
    CONFIG as EAGER_ACCURACY_CONFIG,
)
from tests.glm5_2_precision.workflow import (  # noqa: E402
    FormalTrainingConfig,
    TrainingEndpoint,
)


TOPOLOGIES = standard_topologies()
ALL_DEVICES = "0,1,2,3,4,5,6,7"
NPU_INDUCTOR_FALLBACKS = "aten.sum,_c10d_functional.all_reduce"

TRAINING = FormalTrainingConfig(
    steps=500,
    local_batch_size=8,
    global_batch_size=64,
    sequence_length=128,
    seed=61,
    deterministic=True,
    training_dtype="float32",
    mixed_precision_param="bfloat16",
    checkpoint_kind="random_seed",
)

def _stage_configs(
    device: str,
    *,
    graph_backend: str = "inductor",
    codegen_backend: str | None = None,
) -> tuple[
    MindStudioExperimentConfig,
    dict[str, MindStudioExperimentConfig],
]:
    if device not in {"gpu", "npu"}:
        raise ValueError(f"unsupported graph accuracy device: {device}")
    if graph_backend not in {"inductor", "npugraphs"}:
        raise ValueError(f"unsupported graph backend: {graph_backend}")
    if device == "gpu" and graph_backend != "inductor":
        raise ValueError("GPU graph accuracy currently supports Inductor only")
    if codegen_backend is None:
        codegen_backend = "ascend-triton" if device == "npu" else "triton"
    allowed_codegen = (
        {"ascend-triton", "dvm"} if device == "npu" else {"triton"}
    )
    if codegen_backend not in allowed_codegen:
        raise ValueError(
            f"codegen backend {codegen_backend!r} is invalid for {device}; "
            f"choose from {sorted(allowed_codegen)}"
        )
    device_type = "cuda" if device == "gpu" else "npu"
    npu_codegen = codegen_backend if device == "npu" else None
    if device == "npu":
        eager_feature = GraphFeatureConfig(
            mode="eager",
            npu_codegen=npu_codegen,
        ).feature(device_type=device_type)
        graph_feature = GraphFeatureConfig(
            mode=graph_backend,
            npu_codegen=npu_codegen,
        ).feature(device_type=device_type)
    else:
        eager_feature = TrainingFeature(
            name="graph:eager",
            metadata={"mode": "eager"},
        )
        graph_feature = TrainingFeature(
            name=f"graph:{graph_backend}",
            arguments=(
                "--compile.enable",
                "--compile.components=model",
                f"--compile.backend={graph_backend}",
            ),
            metadata={
                "mode": graph_backend,
                "components": ["model"],
                "codegen_backend": codegen_backend,
            },
        )
    eager_endpoint = TrainingEndpoint(
        name=f"{device}-eager-reference",
        device_type=device_type,
        visible_devices=ALL_DEVICES,
        topology=TOPOLOGIES["single"],
        repeats=1,
        environment=eager_feature.environment,
        extra_args=eager_feature.arguments,
    )
    graph_environment = dict(graph_feature.environment)
    if device == "npu" and graph_backend == "inductor":
        graph_environment["NPU_INDUCTOR_FALLBACK_LIST"] = (
            NPU_INDUCTOR_FALLBACKS
        )
    graph_endpoint = TrainingEndpoint(
        name=f"{device}-graph-candidate",
        device_type=device_type,
        visible_devices=ALL_DEVICES,
        topology=TOPOLOGIES["single"],
        repeats=1,
        environment=graph_environment,
        extra_args=graph_feature.arguments,
    )
    unscoped = MindStudioExperimentConfig(
        name="glm5-2-official-graph-accuracy",
        workflow="migration",
        reference=eager_endpoint,
        candidate=graph_endpoint,
        training=TRAINING,
        dump=MsProbeDumpConfig(
            task="statistics",
            level="L0",
            steps=(0, 1),
            summary_mode="statistics",
        ),
        execution_branch=(
            f"graph/{device}-{graph_backend}-{codegen_backend}"
        ),
        reuse_eager_role_as_reference=(
            "candidate" if device == "npu" else "reference"
        ),
        owns_fixture=False,
    )
    base = replace(
        unscoped,
        experiment_storage_name=EAGER_ACCURACY_CONFIG.storage_name,
    )
    checker_endpoint = replace(
        eager_endpoint,
        name=f"{device}-compile-checker",
    )
    stages = {
        "config-check": replace(
            base,
            workflow="config-check",
            training=replace(TRAINING, steps=1),
            fixture_training=TRAINING,
            owns_fixture=False,
        ),
        "observation": replace(
            base,
            workflow="observation",
        ),
        "monitor": replace(
            base,
            workflow="monitor",
            training=replace(TRAINING, steps=100),
            fixture_training=TRAINING,
            monitor=MsProbeMonitorConfig(
                ranks=(0,),
                start_step=0,
                step_interval=1,
                step_count_per_record=10,
                collect_times=100,
                weight_grad=True,
            ),
        ),
        "dump": base,
        "compile-checker": replace(
            base,
            workflow="compile",
            reference=checker_endpoint,
            candidate=checker_endpoint,
            training=replace(TRAINING, steps=1),
            fixture_training=TRAINING,
            dump=replace(base.dump, steps=(0,)),
            compile=MsProbeCompileConfig(
                backend=graph_backend,
                dump_graphs=True,
                capture_input=True,
                policy="glm5-block",
            ),
            reuse_eager_role_as_reference=None,
            owns_fixture=False,
        ),
    }
    return base, stages


BASE_CONFIG, STAGE_CONFIGS = _stage_configs("npu")


def _select_options(
    arguments: list[str],
) -> tuple[str, str, str, str, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--stage",
        choices=tuple(STAGE_CONFIGS),
        default="observation",
    )
    parser.add_argument("--device", choices=("gpu", "npu"), required=True)
    parser.add_argument(
        "--graph-backend",
        choices=("inductor", "npugraphs"),
        default="inductor",
    )
    parser.add_argument(
        "--codegen-backend",
        choices=("triton", "ascend-triton", "dvm"),
        required=True,
    )
    parsed, remaining = parser.parse_known_args(arguments)
    return (
        parsed.stage,
        parsed.device,
        parsed.graph_backend,
        parsed.codegen_backend,
        remaining,
    )


def _select_stage(arguments: list[str]) -> tuple[str, list[str]]:
    stage, _device, _graph_backend, _codegen_backend, remaining = (
        _select_options(arguments)
    )
    return stage, remaining


if __name__ == "__main__":
    stage, device, graph_backend, codegen_backend, arguments = (
        _select_options(sys.argv[1:])
    )
    _base_config, stage_configs = _stage_configs(
        device,
        graph_backend=graph_backend,
        codegen_backend=codegen_backend,
    )
    if device == "gpu":
        print(
            "Warning: GPU graph accuracy is wired through native Inductor but "
            "has not completed the server validation matrix.",
            flush=True,
        )
    sys.argv = [sys.argv[0], *arguments]
    run_mindstudio_cli(stage_configs[stage], __file__)
