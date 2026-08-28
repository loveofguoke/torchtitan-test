#!/usr/bin/env python3
"""Upload and download experiment directories through GitHub Releases."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote


DEFAULT_REPOSITORY = "loveofguoke/torchtitan-test"
EXPERIMENT_ROOTS = (
    "parity_fixtures",
    "parity_artifacts",
    "parity_reports",
    "precision_fixtures",
    "precision_artifacts",
    "precision_runs",
    "precision_reports",
    "performance_runs",
    "performance_artifacts",
    "performance_reports",
    "performance_dynamic",
    "stability_fixtures",
    "stability_artifacts",
    "stability_runs",
    "stability_reports",
    "checkpoint_fixtures",
    "checkpoint_artifacts",
    "checkpoint_runs",
    "checkpoint_reports",
    "combination_artifacts",
    "combination_runs",
    "combination_reports",
    "combination_reports/precision",
    "graph_debug_runs",
    "smoke_runs",
    "train_runs",
)
EXPERIMENT_BOUNDARY_FILES = {
    "capture_state.json",
    "complete.json",
    "fixture.json",
    "manifest.json",
    "run_state.json",
}

# ``analysis`` archives are intended for local report review and follow-up
# diagnosis.  They retain rendered/parsed results and compact experiment
# metadata while omitting checkpoints, raw trainer state, compiler caches, and
# unparsed CANN collection trees.  ``full`` remains the lossless transfer mode
# used when another machine must resume or re-parse an experiment.
ANALYSIS_RUN_FILES = {
    "analysis.json",
    "artifacts.json",
    "capture_state.json",
    "command_history.jsonl",
    "complete.json",
    "input_contract.json",
    "invocation.json",
    "manifest.json",
    "metrics.jsonl",
    "raw_metrics.jsonl",
    "readme.md",
    "report.md",
    "run_state.json",
    "runtime.log",
    "training_contract.json",
}
ANALYSIS_RUN_DIRECTORIES = {
    "ASCEND_PROFILER_OUTPUT",
    "advisor",
    "cluster",
    "cluster_time_summary",
    "compare",
    "free_analysis",
    "graph_visualization",
    "memory_timeline",
    "mindstudio_flamegraphs",
    "mindstudio_profiler_output",
    "reports",
    "tensorboard",
    "tool_commands",
}


def run_gh(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run GitHub CLI without invoking a shell."""
    if shutil.which("gh") is None:
        raise RuntimeError(
            "GitHub CLI 'gh' was not found. Install it and run 'gh auth login' first."
        )
    command = ["gh", *args]
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, check=check, text=True)


def run_wget(url: str, output_path: Path) -> None:
    """Download a public release asset with certificate checks disabled."""
    if shutil.which("wget") is None:
        raise RuntimeError("'wget' was not found")
    command = [
        "wget",
        "--no-check-certificate",
        "--output-document",
        str(output_path),
        url,
    ]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def validate_experiment_name(experiment: str) -> str:
    """Require one safe directory/tag component."""
    if not experiment or experiment in {".", ".."}:
        raise ValueError("experiment name must not be empty, '.' or '..'")
    if Path(experiment).name != experiment or "/" in experiment or "\\" in experiment:
        raise ValueError("experiment name must be a single directory name")
    if experiment.startswith("-") or any(char.isspace() for char in experiment):
        raise ValueError(
            "experiment name must not start with '-' or contain whitespace"
        )
    return experiment


def validate_repository(repository: str) -> tuple[str, str]:
    """Validate and split a GitHub OWNER/REPO identifier."""
    parts = repository.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("repository must use the OWNER/REPO format")
    invalid_part = any(
        part in {".", ".."} or any(char.isspace() for char in part)
        for part in parts
    )
    if invalid_part:
        raise ValueError("repository owner and name must not contain whitespace")
    return parts[0], parts[1]


def find_experiment_paths(repository_root: Path, experiment: str) -> list[Path]:
    """Find standard artifact directories with the requested experiment name."""
    paths: list[Path] = []
    for root in EXPERIMENT_ROOTS:
        experiment_root = repository_root / root
        if not experiment_root.is_dir():
            continue
        # Performance and graph outputs are scoped by card count, topology,
        # backend, or action before the experiment name. Walk standard roots
        # recursively, but stop below a matched experiment directory so raw
        # profiler traces are never traversed just to rediscover descendants.
        for directory, child_names, file_names in os.walk(experiment_root):
            current = Path(directory)
            if current.name == experiment:
                paths.append(current)
                child_names.clear()
                continue
            for file_name in file_names:
                candidate = current / file_name
                if candidate.stem == experiment:
                    paths.append(candidate)
            # A manifest/state-bearing directory is one experiment output.
            # Experiment outputs never contain another experiment, while raw
            # profiler trees can contain millions of descendants.
            if EXPERIMENT_BOUNDARY_FILES.intersection(file_names):
                child_names.clear()
    paths = list(dict.fromkeys(paths))
    # A matched directory already includes every descendant. Avoid adding a
    # same-named report or file below it to the archive a second time.
    paths = [
        path
        for path in paths
        if not any(
            parent != path and parent.is_dir() and path.is_relative_to(parent)
            for parent in paths
        )
    ]
    if not paths:
        roots = ", ".join(EXPERIMENT_ROOTS)
        raise FileNotFoundError(
            f"no output named {experiment!r} was found under: {roots}"
        )
    return paths


def create_archive(
    repository_root: Path,
    experiment: str,
    archive_path: Path,
    *,
    include: tuple[str, ...] = (),
    content: str = "full",
) -> list[Path]:
    """Archive standard outputs for an experiment and explicit dependencies."""
    if content not in {"full", "analysis"}:
        raise ValueError(f"unsupported archive content mode: {content!r}")
    names = tuple(dict.fromkeys((experiment, *include)))
    paths = list(
        dict.fromkeys(
            path
            for name in names
            for path in find_experiment_paths(repository_root, name)
        )
    )
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in paths:
            relative_path = path.relative_to(repository_root)
            print(f"Adding {relative_path}", flush=True)
            archive.add(
                path,
                arcname=relative_path.as_posix(),
                recursive=True,
                filter=(
                    None
                    if content == "full"
                    else _analysis_archive_filter
                ),
            )
    return paths


def _analysis_archive_filter(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
    """Keep portable analysis evidence and discard heavy source/intermediate data."""

    path = PurePosixPath(member.name)
    if not path.parts:
        return None
    root = path.parts[0]

    # Fixtures contain token plans and often multi-GB checkpoints.  They are
    # required for resuming/capturing, not for reading an existing result.
    if root.endswith("_fixtures"):
        return None

    # Reports and compact artifacts are already curated by their experiment.
    if root.endswith("_reports") or root.endswith("_artifacts"):
        return member

    # Preserve directory entries so tarfile can continue walking towards an
    # allowed descendant.  Unselected files below them are filtered later.
    if member.isdir():
        return member

    if root.endswith("_runs") or root in {
        "graph_debug_runs",
        "performance_dynamic",
    }:
        if path.name in ANALYSIS_RUN_FILES:
            return member
        if any(part in ANALYSIS_RUN_DIRECTORIES for part in path.parts):
            return member
        if any(part.startswith("communication_bottleneck") for part in path.parts):
            return member
        if "flamegraph" in path.name.lower() or path.suffix.lower() in {
            ".svg",
            ".png",
        }:
            return member
        return None

    # A directly named report file can be matched outside a *_reports root.
    return member if path.suffix.lower() in {".html", ".json", ".md"} else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksum(archive_path: Path) -> Path:
    checksum_path = archive_path.with_name(archive_path.name + ".sha256")
    checksum_path.write_text(
        f"{sha256_file(archive_path)}  {archive_path.name}\n",
        encoding="ascii",
    )
    return checksum_path


def upload(args: argparse.Namespace) -> None:
    repository_root = args.repository_root.resolve()
    experiment = validate_experiment_name(args.experiment)
    with tempfile.TemporaryDirectory(prefix="torchtitan-release-") as temp_dir:
        archive_path = Path(temp_dir) / f"{experiment}.tar.gz"
        include = tuple(validate_experiment_name(name) for name in args.include)
        paths = create_archive(
            repository_root,
            experiment,
            archive_path,
            include=include,
            content=args.content,
        )
        checksum_path = write_checksum(archive_path)
        assets = [str(archive_path), str(checksum_path)]

        release_exists = (
            run_gh(
                "release",
                "view",
                experiment,
                "--repo",
                args.repo,
                check=False,
            ).returncode
            == 0
        )
        if release_exists:
            run_gh(
                "release",
                "upload",
                experiment,
                *assets,
                "--clobber",
                "--repo",
                args.repo,
            )
        else:
            included = "\n".join(
                f"- `{path.relative_to(repository_root).as_posix()}`" for path in paths
            )
            notes = (
                f"Experiment artifacts for `{experiment}` "
                f"(content: `{args.content}`).\n\n{included}"
            )
            run_gh(
                "release",
                "create",
                experiment,
                *assets,
                "--title",
                experiment,
                "--notes",
                notes,
                "--latest=false",
                "--repo",
                args.repo,
            )

    print(f"Uploaded release: {experiment}")


def validate_archive_members(
    archive: tarfile.TarFile,
    destination: Path,
    overwrite: bool,
) -> None:
    destination = destination.resolve()
    conflicts: list[Path] = []
    for member in archive.getmembers():
        relative_path = PurePosixPath(member.name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"unsafe archive member: {member.name}")
        target = destination.joinpath(*relative_path.parts)
        if target.resolve().is_relative_to(destination) is False:
            raise RuntimeError(f"archive member escapes destination: {member.name}")
        if not overwrite and member.isfile() and target.exists():
            conflicts.append(target)
    if conflicts:
        examples = "\n".join(f"  {path}" for path in conflicts[:5])
        raise FileExistsError(
            "download would overwrite existing files; rerun with --overwrite. "
            f"Examples:\n{examples}"
        )


def download_assets_with_gh(
    *,
    repository: str,
    experiment: str,
    archive_name: str,
    checksum_name: str,
    download_dir: Path,
) -> None:
    run_gh(
        "release",
        "download",
        experiment,
        "--pattern",
        archive_name,
        "--pattern",
        checksum_name,
        "--dir",
        str(download_dir),
        "--repo",
        repository,
    )


def download_assets_with_wget(
    *,
    repository: str,
    experiment: str,
    archive_name: str,
    checksum_name: str,
    download_dir: Path,
) -> None:
    owner, repository_name = validate_repository(repository)
    release_url = (
        f"https://github.com/{quote(owner, safe='')}/"
        f"{quote(repository_name, safe='')}/releases/download/"
        f"{quote(experiment, safe='')}"
    )
    print(
        "WARNING: wget backend disables TLS certificate verification.",
        file=sys.stderr,
        flush=True,
    )
    for asset_name in (archive_name, checksum_name):
        asset_url = f"{release_url}/{quote(asset_name, safe='')}"
        run_wget(asset_url, download_dir / asset_name)


def download(args: argparse.Namespace) -> None:
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    experiment = validate_experiment_name(args.experiment)
    archive_name = f"{experiment}.tar.gz"
    checksum_name = f"{archive_name}.sha256"

    with tempfile.TemporaryDirectory(prefix="torchtitan-release-") as temp_dir:
        download_dir = Path(temp_dir)
        download_assets = (
            download_assets_with_gh
            if args.backend == "gh"
            else download_assets_with_wget
        )
        download_assets(
            repository=args.repo,
            experiment=experiment,
            archive_name=archive_name,
            checksum_name=checksum_name,
            download_dir=download_dir,
        )
        archive_path = download_dir / archive_name
        checksum_path = download_dir / checksum_name
        expected = checksum_path.read_text(encoding="ascii").split()[0]
        actual = sha256_file(archive_path)
        if actual != expected:
            raise RuntimeError(
                f"SHA-256 mismatch for {archive_name}: "
                f"expected {expected}, got {actual}"
            )

        with tarfile.open(archive_path, "r:gz") as archive:
            validate_archive_members(archive, destination, args.overwrite)
            archive.extractall(destination, filter="data")

    print(f"Downloaded and extracted release {experiment} into {destination}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer generated experiment directories through a GitHub Release "
            "whose tag and title equal the experiment name."
        )
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPOSITORY,
        help=f"GitHub OWNER/REPO (default: {DEFAULT_REPOSITORY})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload_parser = subparsers.add_parser(
        "upload", help="archive and upload an experiment"
    )
    upload_parser.add_argument("experiment", help="experiment directory name")
    upload_parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "also archive outputs with this name; repeat for shared fixtures "
            "or reports whose names differ from the primary experiment"
        ),
    )
    upload_parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="torchtitan-test root (default: current directory)",
    )
    upload_parser.add_argument(
        "--content",
        choices=("full", "analysis"),
        default="full",
        help=(
            "full preserves every matched output; analysis keeps reports, "
            "compact artifacts, parsed profiler/graph visualizations, logs, "
            "and metrics while excluding fixtures and raw intermediates"
        ),
    )
    upload_parser.set_defaults(func=upload)

    download_parser = subparsers.add_parser(
        "download", help="download and extract an experiment"
    )
    download_parser.add_argument("experiment", help="release/experiment name")
    download_parser.add_argument(
        "--backend",
        choices=("gh", "wget"),
        default="gh",
        help=(
            "release download backend (default: gh); wget disables TLS "
            "certificate verification"
        ),
    )
    download_parser.add_argument(
        "--destination",
        type=Path,
        default=Path.cwd(),
        help="extraction root (default: current directory)",
    )
    download_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing files with files from the release archive",
    )
    download_parser.set_defaults(func=download)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
