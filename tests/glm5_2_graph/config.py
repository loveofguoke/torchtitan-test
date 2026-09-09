# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Graph-mode execution features shared by graph-aware experiments.

Graph mode is modeled as an orthogonal training feature. ``eager`` contributes
no compile arguments; ``inductor`` and ``npugraphs`` contribute explicit
backend/component policy plus optional diagnostics. The CUDA interface is kept
at the type/CLI boundary but raises until a validated GPU policy is defined,
preventing accidental claims from an unimplemented backend.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from tests.glm5_2_common.execution import TrainingFeature


GraphMode = Literal["eager", "inductor", "npugraphs"]


def npu_flexattention_environment(mode: str | None) -> dict[str, str]:
    """Select TorchNPU's mask-in or mask-out lowering path."""
    if mode is None:
        return {}
    if mode not in ("mask-in", "mask-out"):
        raise ValueError(f"unsupported NPU FlexAttention mask mode: {mode}")
    return {
        "TORCHINDUCTOR_FLEXATTENTION_MASKOUT": "1" if mode == "mask-out" else "0"
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


def add_npu_flexattention_argument(parser) -> None:
    parser.add_argument(
        "--npu-flexattention-mask-mode",
        choices=("mask-in", "mask-out"),
        default=None,
        help="TorchNPU FlexAttention lowering path; omitted preserves the "
        "installed default. This also affects internally compiled "
        "FlexAttention when --graph=eager.",
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
    if compile_requested and device_type != "npu":
        raise NotImplementedError(
            "graph-mode experiments currently support only NPU endpoints; "
            "the CUDA interface is reserved until the torch.compile policy "
            "is defined"
        )


@dataclass(frozen=True)
class GraphFeatureConfig:
    """Translate one graph policy into conflict-checkable args/environment."""

    mode: GraphMode = "eager"
    components: tuple[str, ...] = ("model",)
    diagnostics: bool = False
    npu_codegen: str | None = None
    npu_flexattention_mask_mode: str | None = None

    def feature(self, *, device_type: str) -> TrainingFeature:
        codegen_env = npu_codegen_environment(self.npu_codegen)
        flexattention_env = npu_flexattention_environment(
            self.npu_flexattention_mask_mode
        )
        if (codegen_env or flexattention_env) and device_type != "npu":
            raise ValueError("NPU compiler controls require an NPU endpoint")
        npu_environment = {**codegen_env, **flexattention_env}
        npu_metadata = {
            **({"npu_codegen": self.npu_codegen} if codegen_env else {}),
            **(
                {"npu_flexattention_mask_mode": self.npu_flexattention_mask_mode}
                if flexattention_env
                else {}
            ),
        }
        if self.mode == "eager":
            return TrainingFeature(
                name="graph:eager",
                environment=npu_environment,
                metadata={"mode": "eager", **npu_metadata},
            )
        validate_graph_training_args(
            device_type=device_type,
            arguments=("--compile.enable",),
        )
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
        return TrainingFeature(
            name=f"graph:{self.mode}",
            arguments=(
                "--compile.enable",
                f"--compile.components={','.join(self.components)}",
                f"--compile.backend={self.mode}",
            ),
            environment={**environment, **npu_environment},
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
    "add_npu_flexattention_argument",
    "graph_modes",
    "npu_flexattention_environment",
    "validate_graph_training_args",
]
