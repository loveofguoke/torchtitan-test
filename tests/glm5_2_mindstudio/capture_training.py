#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Run TorchTitan with one official msProbe instrumentation mode."""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any

from tests.glm5_2_mindstudio.artifacts import write_json
from tests.glm5_2_mindstudio.msprobe_adapter import validate_compile_report


MODE_ENV = "GLM5_MINDSTUDIO_MODE"
CONFIG_ENV = "GLM5_MINDSTUDIO_CONFIG"
OUTPUT_ENV = "GLM5_MINDSTUDIO_OUTPUT"
SEED_ENV = "GLM5_MINDSTUDIO_SEED"
DETERMINISTIC_ENV = "GLM5_MINDSTUDIO_DETERMINISTIC"
SHELL_PATH_ENV = "GLM5_MINDSTUDIO_SHELL_PATH"
logger = logging.getLogger(__name__)


def _require_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must name an experiment path")
    return Path(value).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_from_official_tool() -> None:
    from msprobe.pytorch import seed_all

    seed_all(
        seed=int(os.environ.get(SEED_ENV, "61")),
        mode=os.environ.get(DETERMINISTIC_ENV, "1") == "1",
        rm_dropout=False,
    )


def _install_dump_capture() -> None:
    from msprobe.pytorch import PrecisionDebugger
    from torchtitan.trainer import Trainer

    config_path = _require_path(CONFIG_ENV)
    debugger = PrecisionDebugger(config_path=str(config_path))
    original_train_step = Trainer.train_step

    def train_step_with_msprobe(self: Trainer, data_iterator):
        debugger.start(model=self.model_parts)
        training_error: BaseException | None = None
        try:
            result = original_train_step(self, data_iterator)
        except BaseException as error:
            training_error = error
            raise
        finally:
            cleanup_error: Exception | None = None
            for operation in (debugger.stop, debugger.step):
                try:
                    operation()
                except Exception as error:
                    if cleanup_error is None:
                        cleanup_error = error
                    logger.exception(
                        "PrecisionDebugger cleanup failed after a training step"
                    )
            if cleanup_error is not None and training_error is None:
                raise cleanup_error
        return result

    Trainer.train_step = train_step_with_msprobe


def _write_compile_rank_report(
    *,
    checker_reports: list[tuple[int, Path]],
    destination: Path,
) -> None:
    """Merge virtual-stage reports while preserving the official columns."""

    header: list[str] | None = None
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        for part_index, source_path in checker_reports:
            with source_path.open(
                "r", encoding="utf-8-sig", errors="replace", newline=""
            ) as source:
                reader = csv.reader(source)
                current_header = next(reader, None)
                if current_header is None:
                    continue
                if header is None:
                    header = current_header
                    writer.writerow(header)
                elif header != current_header:
                    raise RuntimeError(
                        f"PrecisionChecker report headers differ: {source_path}"
                    )
                module_column = (
                    current_header.index("module_name")
                    if "module_name" in current_header
                    else None
                )
                for row in reader:
                    if module_column is not None and row[module_column] != "LOSS":
                        row[module_column] = f"part{part_index}.{row[module_column]}"
                    writer.writerow(row)


def _install_compile_checker() -> None:
    from msprobe.pytorch.compile_accuracy_checker.precision_checker import (
        PrecisionChecker,
    )
    from torchtitan.models.glm5.model import Glm5TransformerBlock
    from torchtitan.trainer import Trainer

    settings = _load_json(_require_path(CONFIG_ENV))
    output_directory = _require_path(OUTPUT_ENV)
    output_directory.mkdir(parents=True, exist_ok=True)
    target_step = int(settings["target_training_step"])
    original_forward_backward_step = Trainer.forward_backward_step
    original_train_step = Trainer.train_step
    state: dict[int, dict[str, Any]] = {}

    def forward_backward_with_loss(self: Trainer, **kwargs):
        loss = original_forward_backward_step(self, **kwargs)
        if self.step == target_step:
            state.setdefault(id(self), {})["loss"] = loss
        return loss

    def train_step_with_checker(self: Trainer, data_iterator):
        if self.step != target_step:
            return original_train_step(self, data_iterator)

        trainer_state = state.setdefault(id(self), {})
        checkers = []
        coverage_parts: list[dict[str, Any]] = []
        for part_index, model_part in enumerate(self.model_parts):
            expected_blocks = [
                name or "<root>"
                for name, module in model_part.named_modules()
                if isinstance(module, Glm5TransformerBlock)
            ]
            if not expected_blocks:
                raise RuntimeError(
                    "PrecisionChecker compile policy found no "
                    f"Glm5TransformerBlock in model part {part_index}"
                )
            checker = PrecisionChecker(
                backend=settings["backend"],
                dump_graphs=bool(settings["dump_graphs"]),
                graph_dir=str(output_directory / "graphs" / f"part{part_index}"),
                capture_input=bool(settings["capture_input"]),
                single_pass=True,
            )
            if settings["policy"] == "glm5-block":
                checker.wrap_by_policy(model_part, (Glm5TransformerBlock,))
            else:
                checker.wrap_all_children(
                    model_part,
                    depth=int(settings["children_depth"]),
                )
            checker.install(model_part)
            checkers.append((part_index, checker))
            coverage_parts.append(
                {
                    "part": part_index,
                    "expected_glm_block_count": len(expected_blocks),
                    "expected_glm_block_names": expected_blocks,
                }
            )
        trainer_state["checkers"] = checkers

        result = original_train_step(self, data_iterator)
        loss = trainer_state.get("loss")
        rank = int(os.environ.get("RANK", "0"))
        part_reports: list[tuple[int, Path]] = []
        for part_index, checker in checkers:
            comparison = checker.collect(loss)
            part_path = output_directory / (
                f"precision_rank{rank}_part{part_index}.csv"
            )
            checker.report(comparison, csv_path=str(part_path))
            coverage_parts[part_index]["report"] = part_path.name
            coverage_parts[part_index]["decisive_rows"] = validate_compile_report(
                part_path
            )
            part_reports.append((part_index, part_path))
        write_json(
            output_directory / f"compile_coverage_rank{rank}.json",
            {
                "schema": "torchtitan.glm5_2.msprobe_compile_coverage",
                "schema_version": 1,
                "rank": rank,
                "policy": {
                    "name": settings["policy"],
                    "children_depth": int(settings["children_depth"]),
                },
                "parts": coverage_parts,
            },
        )
        _write_compile_rank_report(
            checker_reports=part_reports,
            destination=output_directory / f"precision_rank{rank}.csv",
        )
        return result

    Trainer.forward_backward_step = forward_backward_with_loss
    Trainer.train_step = train_step_with_checker


def _install_config_checker() -> None:
    from msprobe.core.config_check import ConfigChecker
    import torch
    from torchtitan.trainer import Trainer

    output_directory = _require_path(OUTPUT_ENV)
    shell_path = _require_path(SHELL_PATH_ENV)
    original_train = Trainer.train

    def train_with_config_check(self: Trainer):
        rank = int(os.environ.get("RANK", "0"))
        output_directory.mkdir(parents=True, exist_ok=True)
        model = (
            self.model_parts[0]
            if len(self.model_parts) == 1
            else torch.nn.ModuleList(self.model_parts)
        )
        ConfigChecker(
            model,
            [str(shell_path)],
            str(output_directory / f"config_check_rank{rank}.zip"),
            "pytorch",
        )
        return original_train(self)

    Trainer.train = train_with_config_check


def _install_training_monitor() -> None:
    """Attach the public Monitor V2 API at TorchTitan train-step boundaries."""

    from msprobe.core.monitor_v2.trainer import TrainerMonitorV2
    from torchtitan.trainer import Trainer

    config_path = _require_path(CONFIG_ENV)
    original_train = Trainer.train
    original_train_step = Trainer.train_step
    active_monitors: dict[int, TrainerMonitorV2] = {}

    def train_step_with_monitor(self: Trainer, data_iterator):
        result = original_train_step(self, data_iterator)
        monitor = active_monitors.get(id(self))
        if monitor is None:
            raise RuntimeError("TrainerMonitorV2 was not started for this Trainer")
        # The generated config fixes patch_optimizer_step=false. Calling step
        # here records exactly one official monitor step per TorchTitan step.
        monitor.step()
        return result

    def train_with_monitor(self: Trainer):
        monitor = TrainerMonitorV2(str(config_path), fr="pytorch")
        model = (
            self.model_parts[0]
            if len(self.model_parts) == 1
            else self.model_parts
        )
        monitor.start(
            model=model,
            optimizer=self.optimizers,
            grad_acc_steps=self.gradient_accumulation_steps,
        )
        active_monitors[id(self)] = monitor
        training_error: BaseException | None = None
        try:
            return original_train(self)
        except BaseException as error:
            training_error = error
            raise
        finally:
            active_monitors.pop(id(self), None)
            try:
                monitor.stop()
            except Exception:
                if training_error is None:
                    raise
                logger.exception(
                    "TrainerMonitorV2.stop failed while preserving the "
                    "original training exception"
                )

    Trainer.train_step = train_step_with_monitor
    Trainer.train = train_with_monitor


def main() -> None:
    mode = os.environ.get(MODE_ENV)
    if mode == "config-check":
        # Official dynamic configuration collection requires these patches to
        # be installed before TorchTitan constructs the model or data loader.
        from msprobe.core.config_check import ConfigChecker

        ConfigChecker.apply_patches("pytorch")

    device_type = os.environ.get("TORCHTITAN_DEVICE", "gpu")
    if device_type == "npu":
        import torchtitanturbo  # noqa: F401
    elif device_type != "gpu":
        raise ValueError(f"unsupported TorchTitan device: {device_type!r}")

    _seed_from_official_tool()

    from tests.glm5_2_precision.fixed_token_dataloader import (
        install_fixed_token_dataloader,
    )

    install_fixed_token_dataloader()
    if mode == "config-check":
        _install_config_checker()
    elif mode == "dump":
        _install_dump_capture()
    elif mode == "compile":
        _install_compile_checker()
    elif mode == "monitor":
        _install_training_monitor()
    else:
        raise ValueError(
            f"{MODE_ENV} must be config-check, dump, compile, or monitor, "
            f"got {mode!r}"
        )

    from torchtitan.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
