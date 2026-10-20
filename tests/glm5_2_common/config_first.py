"""Compatibility helpers for TorchTitan's config-first CLI migration."""

import json
import os
import sys


_LEGACY_CONFIG_STATE: dict[str, object] = {}


def parse_legacy_dotted_override(argument: str) -> tuple[str, object]:
    """Translate one legacy dotted option into a config-first field/value."""
    if not argument.startswith("--") or "." not in argument:
        raise ValueError(f"Expected a dotted legacy option, got {argument!r}.")

    option = argument[2:]
    if "=" in option:
        target, raw_value = option.split("=", 1)
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
    else:
        target, value = option, True
        owner_name, separator, attribute_name = target.rpartition(".")
        if attribute_name.startswith("no-"):
            attribute_name = attribute_name.removeprefix("no-")
            target = f"{owner_name}{separator}{attribute_name}"
            value = False

    return target.replace("-", "_"), value


def translate_config_first_cli(argv: list[str]) -> list[str]:
    """Translate the stable experiment CLI to TorchTitan's config-first CLI."""
    if os.environ.get("TORCHTITAN_CONFIG_FIRST_CLI") != "1":
        return argv

    translated = [argv[0]]
    module_name = config_name = None
    overrides: dict[str, object] = {}
    index = 1
    while index < len(argv):
        argument = argv[index]
        if argument in {"--module", "--config"}:
            value = argv[index + 1]
            if argument == "--module":
                module_name = value
            else:
                config_name = value
            index += 2
            continue
        if argument.startswith("--dump_folder="):
            translated.extend(["--output-dir", argument.split("=", 1)[1]])
            index += 1
            continue
        if not argument.startswith("--") or "." not in argument:
            translated.append(argument)
            index += 1
            continue

        target, value = parse_legacy_dotted_override(argument)
        overrides[target] = value
        index += 1

    if module_name != "glm5":
        raise ValueError(
            "The config-first migration adapter currently supports only the "
            f"GLM experiment contract, got module={module_name!r}."
        )
    factories = {
        "debugmodel": "glm5_debugmodel",
        "glm5_debugmodel": "glm5_debugmodel",
        "glm5_2_debugmodel_mtp": "glm5_2_debugmodel_mtp",
        "glm5_2_scaled_debugmodel": "glm5_2_scaled_debugmodel",
    }
    try:
        factory_name = factories[str(config_name)]
    except KeyError as error:
        raise ValueError(
            f"Unsupported config-first GLM recipe {config_name!r}."
        ) from error
    _LEGACY_CONFIG_STATE.update(factory=factory_name, overrides=overrides)
    sys.modules["torchtitan_test_config_first"] = sys.modules[__name__]
    translated.extend(
        ["--module", "torchtitan_test_config_first", "--config", "legacy_config"]
    )
    return translated


def legacy_config():
    """Build and mutate the official current-tree GLM test recipe."""
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan_recipes.tests.models import glm5 as glm5_recipes

    factory = getattr(glm5_recipes, str(_LEGACY_CONFIG_STATE["factory"]))
    overrides = dict(_LEGACY_CONFIG_STATE["overrides"])

    parallelism = ParallelismConfig()
    for dotted_name, value in overrides.items():
        if not dotted_name.startswith("parallelism."):
            continue
        attribute = dotted_name.removeprefix("parallelism.")
        if attribute == "spmd_backend":
            continue
        if not hasattr(parallelism, attribute):
            raise AttributeError(
                f"Current TorchTitan config has no field {dotted_name!r}."
            )
        setattr(parallelism, attribute, value)

    factory_kwargs = {"parallelism": parallelism}
    if "training.max_context_length" in overrides:
        factory_kwargs["seq_len"] = int(overrides["training.max_context_length"])
    config = factory(**factory_kwargs)
    for dotted_name, value in overrides.items():
        if dotted_name == "parallelism.spmd_backend":
            continue
        owner = config
        parts = dotted_name.split(".")
        for part in parts[:-1]:
            owner = getattr(owner, part)
        attribute = parts[-1]
        if not hasattr(owner, attribute):
            raise AttributeError(
                f"Current TorchTitan config has no field {dotted_name!r}."
            )
        setattr(owner, attribute, value)
    return config
