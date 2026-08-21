# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Stable, compact names for reproducible experiment outputs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


def slug(value: str) -> str:
    """Return a filesystem-safe lowercase name."""

    result = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not result:
        raise ValueError(f"value does not produce a usable name: {value!r}")
    return result


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_value(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    return value


def config_digest(identity: Any, *, length: int = 8) -> str:
    """Hash capture-defining settings, independent of report presentation."""

    if length < 6:
        raise ValueError("config digest length must be at least 6")
    payload = json.dumps(
        _json_value(identity),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def config_name(
    base: str,
    identity: Any,
    *,
    digest_length: int = 8,
    max_length: int = 112,
) -> str:
    """Append one stable config digest while respecting path component limits."""

    normalized = slug(base)
    digest = config_digest(identity, length=digest_length)
    suffix = f"-{digest}"
    available = max_length - len(suffix)
    if available < 1:
        raise ValueError("max_length is too small for the config digest")
    return f"{normalized[:available].rstrip('-')}{suffix}"

