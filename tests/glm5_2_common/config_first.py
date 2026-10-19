"""Compatibility helpers for TorchTitan's config-first CLI migration."""

import json


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
