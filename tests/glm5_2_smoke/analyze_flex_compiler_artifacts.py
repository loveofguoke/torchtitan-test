#!/usr/bin/env python3
"""Inventory and compare compiler evidence from CP capture and replay."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


TEXT_SUFFIXES = {".cpp", ".json", ".log", ".py", ".txt"}
SIGNALS = (
    "flex_attention",
    "flexattention",
    "benchmark elapsed time",
    "autotune",
    "kernel_name",
    "num_warps",
    "grid=",
    "workspace",
    "tangents_1, none",
    "tangents_1, mul_",
)
KERNEL_PATTERN = re.compile(
    r"\b(?:triton|dvm)_(?:poi|tem|per)_fused_flex_attention[A-Za-z0-9_]*"
)
BUFFER_PATTERN = re.compile(r"\b(?:buf\d+|arg\d+_\d+|primals_\d+|tangents_\d+)\b")
BACKWARD_DEFINITION_PATTERN = re.compile(
    r"def\s+(?P<kernel>\w*flex_attention_(?:backward|bwd)\w*)\((?P<args>[^)]*)\)"
)
BACKWARD_LAUNCH_PATTERN = re.compile(
    r"(?P<kernel>\w*flex_attention_(?:backward|bwd)\w*)\.run\((?P<args>.*)\)"
)
LIFETIME_SIGNALS = (
    "empty_strided",
    "reinterpret_tensor",
    "as_strided",
    ".reuse(",
    "del ",
    "delta",
    "flex_attention_backward",
    "flex_attention_bwd",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_top_level_arguments(arguments: str) -> list[str]:
    values = []
    start = 0
    depth = 0
    for index, character in enumerate(arguments):
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        elif character == "," and depth == 0:
            values.append(arguments[start:index].strip())
            start = index + 1
    values.append(arguments[start:].strip())
    return values


def _inspect_tree(root: Path) -> dict[str, object]:
    files = []
    signals: Counter[str] = Counter()
    kernels: Counter[str] = Counter()
    snippets = []
    backward_definitions = []
    backward_launches = []
    buffer_events: dict[str, list[dict[str, object]]] = {}
    if not root.is_dir():
        return {
            "exists": False,
            "files": files,
            "signals": {},
            "kernels": {},
            "backward_definitions": [],
            "backward_launches": [],
            "buffer_lifetime_events": {},
        }
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        record = {
            "path": relative,
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        files.append(record)
        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 64 << 20:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lowered = text.lower()
        for signal in SIGNALS:
            signals[signal] += lowered.count(signal)
        kernels.update(KERNEL_PATTERN.findall(text))
        for line_number, line in enumerate(text.splitlines(), 1):
            lowered_line = line.lower()
            if len(snippets) < 80 and any(
                signal in lowered_line for signal in SIGNALS
            ):
                snippets.append(
                    {"file": relative, "line": line_number, "text": line[:500]}
                )
            definition = BACKWARD_DEFINITION_PATTERN.search(line)
            if definition:
                arguments = _split_top_level_arguments(definition.group("args"))
                backward_definitions.append(
                    {
                        "file": relative,
                        "line": line_number,
                        "kernel": definition.group("kernel"),
                        "delta_argument_indices": [
                            index
                            for index, value in enumerate(arguments)
                            if "DELTA" in value.upper()
                        ],
                        "arguments": arguments,
                    }
                )
            if (
                (
                    "flex_attention_backward" in lowered_line
                    or "flex_attention_bwd" in lowered_line
                )
                and "def " not in lowered_line
                and (".run(" in lowered_line or "(" in lowered_line)
            ):
                launch = BACKWARD_LAUNCH_PATTERN.search(line)
                record = {"file": relative, "line": line_number, "text": line[:1000]}
                if launch:
                    record["kernel"] = launch.group("kernel")
                    record["arguments"] = _split_top_level_arguments(
                        launch.group("args")
                    )
                backward_launches.append(record)
            if any(signal in lowered_line for signal in LIFETIME_SIGNALS):
                for buffer_name in set(BUFFER_PATTERN.findall(line)):
                    events = buffer_events.setdefault(buffer_name, [])
                    if len(events) < 40:
                        events.append(
                            {"file": relative, "line": line_number, "text": line[:1000]}
                        )
    definitions_by_kernel = {
        record["kernel"]: record for record in backward_definitions
    }
    for launch in backward_launches:
        definition = definitions_by_kernel.get(launch.get("kernel"))
        arguments = launch.get("arguments")
        if definition is None or arguments is None:
            continue
        launch["delta_arguments"] = [
            arguments[index]
            for index in definition["delta_argument_indices"]
            if index < len(arguments)
        ]

    return {
        "exists": True,
        "file_count": len(files),
        "total_bytes": sum(item["size"] for item in files),
        "files": files,
        "signals": dict(signals),
        "kernels": dict(kernels),
        "snippets": snippets,
        "backward_definitions": backward_definitions,
        "backward_launches": backward_launches[:200],
        "buffer_lifetime_events": buffer_events,
    }


def analyze(capture_root: Path) -> dict[str, object]:
    compiler_root = capture_root / "compiler"
    capture = {
        directory.name: _inspect_tree(directory)
        for directory in sorted(compiler_root.glob("rank*"))
        if directory.is_dir()
    }
    replay = _inspect_tree(compiler_root / "replay")
    kernel_sets = {
        name: sorted(evidence.get("kernels", {}))
        for name, evidence in {**capture, "replay": replay}.items()
    }
    return {
        "capture_root": str(capture_root.resolve()),
        "capture": capture,
        "replay": replay,
        "kernel_name_sets": kernel_sets,
        "kernel_names_common_to_all": sorted(
            set.intersection(*(set(value) for value in kernel_sets.values()))
            if kernel_sets and all(kernel_sets.values())
            else []
        ),
    }


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    lines = [
        "# FlexAttention compiler evidence comparison",
        "",
        "This report inventories compiler evidence; matching names alone do not prove "
        "matching launch arguments or buffers.",
        "",
        "| Context | Files | Bytes | grad_lse=None | grad_lse=tensor | Autotune | Kernels |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    contexts = {**report["capture"], "replay": report["replay"]}
    for name, evidence in contexts.items():
        signals = evidence.get("signals", {})
        lines.append(
            f"| {name} | {evidence.get('file_count', 0)} | "
            f"{evidence.get('total_bytes', 0)} | "
            f"{signals.get('tangents_1, none', 0)} | "
            f"{signals.get('tangents_1, mul_', 0)} | "
            f"{signals.get('autotune', 0)} | {len(evidence.get('kernels', {}))} |"
        )
    lines.extend(["", "## Kernel names by context", ""])
    for name, kernels in report["kernel_name_sets"].items():
        lines.append(f"- `{name}`: {', '.join(f'`{item}`' for item in kernels) or 'none'}")
    lines.extend(
        [
            "",
            "## Backward DELTA and buffer-lifetime evidence",
            "",
            "The JSON records backward kernel definitions, every `DELTA` formal "
            "argument position, matching launch lines, and allocation/reuse/delete "
            "events grouped by generated buffer name. Map the actual argument at "
            "the recorded DELTA position, then inspect that buffer's event list to "
            "identify the object that reuses its storage.",
            "",
            "The JSON companion contains file hashes and source/log snippets for "
            "launch, workspace, autotune, grid, and kernel inspection.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_root", type=Path)
    args = parser.parse_args()
    report = analyze(args.capture_root.resolve())
    json_path = args.capture_root / "compiler_comparison.json"
    markdown_path = args.capture_root / "compiler_comparison.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_markdown(markdown_path, report)
    print(f"Compiler comparison JSON: {json_path}", flush=True)
    print(f"Compiler comparison report: {markdown_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
