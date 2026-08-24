# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Graph-mode execution features shared by graph-aware experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from tests.glm5_2_common.execution import TrainingFeature


GraphMode = Literal["eager", "inductor", "npugraphs"]


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
    """Translate one graph policy into TorchTitan training arguments."""

    mode: GraphMode = "eager"
    components: tuple[str, ...] = ("model",)
    diagnostics: bool = False

    def feature(self, *, device_type: str) -> TrainingFeature:
        if self.mode == "eager":
            return TrainingFeature(name="graph:eager", metadata={"mode": "eager"})
        validate_graph_training_args(
            device_type=device_type,
            arguments=("--compile.enable",),
        )
        if self.mode == "npugraphs" and self.components != ("model",):
            raise ValueError("npugraphs supports model compilation only")
        environment = (
            {"TORCH_LOGS": "graph_breaks,recompiles,dynamic"}
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
            environment=environment,
            metadata={"mode": self.mode, "components": list(self.components)},
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
    "graph_modes",
    "validate_graph_training_args",
]
