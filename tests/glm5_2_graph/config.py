# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Graph-mode execution features shared by graph-aware experiments.

Graph mode is modeled as an orthogonal training feature. ``eager`` contributes
no compile arguments; ``inductor`` and ``npugraphs`` contribute explicit
backend/component policy plus optional diagnostics. CUDA supports the upstream
Inductor backend; ``npugraphs`` and NPU codegen selection remain NPU-only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from tests.glm5_2_common.execution import TrainingFeature


GraphMode = Literal["eager", "inductor", "npugraphs"]
NPU_GRAPH_COMPATIBILITY_ENVIRONMENT = {
    # The validated graph profile uses synchronous task submission. Leaving
    # this unset lets Turbo select its ordinary asynchronous default, which
    # can strand PP peers during first-step metadata inference.
    "TASK_QUEUE_ENABLE": "0",
    "TORCHTITAN_TASK_QUEUE_ENABLE": "0",
    # ProcessGroupHCCL's batched object P2P can corrupt PipelineStage's
    # serialized metadata size. Turbo replaces only the one-time metadata
    # exchange with ordinary object P2P when this value is zero.
    "TORCHTITAN_PIPELINE_META_USE_BATCH": "0",
    # Preserve PyTorch's real dynamic metadata exchange, but execute the
    # metadata-only probe under the public compiler force-eager stance so cold
    # compilation is not serialized inside startup P2P.
    "TORCHTITAN_PIPELINE_METADATA_FORCE_EAGER": "1",
    # Reuse the actual inputs observed by PyTorch's dynamic metadata probe to
    # build each local stage's forward/backward cache before schedule P2P.
    "TORCHTITAN_PIPELINE_REAL_INPUT_PRECOMPILE": "1",
    # Align DDP ranks after rank-local cold compilation and immediately before
    # their first real gradient all-reduce. Tensor reduction remains HCCL.
    "TORCHTITAN_FIRST_ALL_REDUCE_HOST_BARRIER": "1",
    # Eight graph-compiling ranks must not each create a pool of device-aware
    # compiler workers; the validated launcher uses one worker per rank.
    "TORCHINDUCTOR_COMPILE_THREADS": "1",
}


def npu_codegen_environment(backend: str | None) -> dict[str, str]:
    """Translate the experiment name to TorchNPU's installed loader names."""
    if backend is None:
        return {}
    if backend not in ("dvm", "ascend-triton"):
        raise ValueError(f"unsupported NPU codegen backend: {backend}")
    return {"TORCHINDUCTOR_NPU_BACKEND": "dvm" if backend == "dvm" else "default"}


def add_npu_codegen_argument(parser) -> None:
    parser.add_argument(
        "--npu-codegen", choices=("dvm", "ascend-triton"), default=None,
        help="NPU Inductor codegen backend; omitted preserves existing policy. "
             "Does not enable whole-model compilation by itself.",
    )


def validate_graph_training_args(
    *,
    device_type: str,
    arguments: Sequence[str],
) -> None:
    """Enforce the current device boundary for raw compile arguments."""

    compile_requested = any(
        argument.startswith("--compile.") for argument in arguments
    )
    if compile_requested and device_type not in {"npu", "cuda"}:
        raise NotImplementedError(
            "graph-mode experiments support only NPU and CUDA endpoints"
        )


@dataclass(frozen=True)
class GraphFeatureConfig:
    """Translate one graph policy into conflict-checkable args/environment."""

    mode: GraphMode = "eager"
    components: tuple[str, ...] = ("model",)
    diagnostics: bool = False
    npu_codegen: str | None = None

    def feature(self, *, device_type: str) -> TrainingFeature:
        codegen_env = npu_codegen_environment(self.npu_codegen)
        if codegen_env and device_type != "npu":
            raise ValueError("NPU compiler controls require an NPU endpoint")
        npu_metadata = {
            **({"npu_codegen": self.npu_codegen} if codegen_env else {}),
        }
        if self.mode == "eager":
            return TrainingFeature(
                name="graph:eager",
                environment=codegen_env,
                metadata={"mode": "eager", **npu_metadata},
            )
        validate_graph_training_args(
            device_type=device_type,
            arguments=("--compile.enable",),
        )
        if self.mode == "npugraphs" and device_type != "npu":
            raise ValueError("npugraphs requires an NPU endpoint")
        if self.mode == "npugraphs" and self.components != ("model",):
            raise ValueError("npugraphs supports model compilation only")
        environment = (
            {
                "TORCH_LOGS": "graph_breaks,recompiles,dynamic",
                "GLM5_GRAPH_CAPTURE_DIAGNOSTICS": "true",
            }
            if self.diagnostics
            else {}
        )
        if device_type == "npu":
            environment = {
                **environment,
                **NPU_GRAPH_COMPATIBILITY_ENVIRONMENT,
            }
        return TrainingFeature(
            name=f"graph:{self.mode}",
            arguments=(
                "--compile.enable",
                f"--compile.components={','.join(self.components)}",
                f"--compile.backend={self.mode}",
            ),
            environment={**environment, **codegen_env},
            metadata={
                "mode": self.mode,
                "components": list(self.components),
                **npu_metadata,
            },
        )


def graph_modes() -> dict[str, GraphFeatureConfig]:
    modes: tuple[GraphMode, ...] = ("eager", "inductor", "npugraphs")
    return {
        mode: GraphFeatureConfig(mode)
        for mode in modes
    }


__all__ = [
    "GraphFeatureConfig",
    "GraphMode",
    "NPU_GRAPH_COMPATIBILITY_ENVIRONMENT",
    "graph_modes",
    "validate_graph_training_args",
]
