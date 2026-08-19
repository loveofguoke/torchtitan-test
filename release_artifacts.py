#!/usr/bin/env python3
"""Upload and download experiment directories through GitHub Releases."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


DEFAULT_REPOSITORY = "loveofguoke/torchtitan-test"
EXPERIMENT_ROOTS = (
    "parity_artifacts",
    "parity_fixtures",
    "precision_artifacts",
    "precision_fixtures",
    "precision_runs",
    "stability_runs",
    "checkpoint_runs",
    "checkpoint_fixtures",
)


def run_gh(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run GitHub CLI without invoking a shell."""
    if shutil.which("gh") is None:
        raise RuntimeError(
            "GitHub CLI 'gh' was not found. Install it and run 'gh auth login' first."
        )
    command = ["gh", *args]
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, check=check, text=True)


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


def find_experiment_paths(repository_root: Path, experiment: str) -> list[Path]:
    """Find standard artifact directories with the requested experiment name."""
    paths = [
        repository_root / root / experiment
        for root in EXPERIMENT_ROOTS
        if (repository_root / root / experiment).is_dir()
    ]
    if not paths:
        roots = ", ".join(EXPERIMENT_ROOTS)
        raise FileNotFoundError(
            f"no directory named {experiment!r} was found under: {roots}"
        )
    return paths


def create_archive(
    repository_root: Path,
    experiment: str,
    archive_path: Path,
) -> list[Path]:
    """Archive all standard directories belonging to an experiment."""
    paths = find_experiment_paths(repository_root, experiment)
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in paths:
            relative_path = path.relative_to(repository_root)
            print(f"Adding {relative_path}", flush=True)
            archive.add(path, arcname=relative_path.as_posix(), recursive=True)
    return paths


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
        paths = create_archive(repository_root, experiment, archive_path)
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
            notes = f"Experiment artifacts for `{experiment}`.\n\n{included}"
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


def download(args: argparse.Namespace) -> None:
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    experiment = validate_experiment_name(args.experiment)
    archive_name = f"{experiment}.tar.gz"
    checksum_name = f"{archive_name}.sha256"

    with tempfile.TemporaryDirectory(prefix="torchtitan-release-") as temp_dir:
        run_gh(
            "release",
            "download",
            experiment,
            "--pattern",
            archive_name,
            "--pattern",
            checksum_name,
            "--dir",
            temp_dir,
            "--repo",
            args.repo,
        )
        archive_path = Path(temp_dir) / archive_name
        checksum_path = Path(temp_dir) / checksum_name
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
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="torchtitan-test root (default: current directory)",
    )
    upload_parser.set_defaults(func=upload)

    download_parser = subparsers.add_parser(
        "download", help="download and extract an experiment"
    )
    download_parser.add_argument("experiment", help="release/experiment name")
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
