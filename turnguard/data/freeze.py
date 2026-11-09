"""Freeze the complete test release; checksums never traverse outside its directory."""

import hashlib
import json
from pathlib import Path


def _files(directory: Path) -> list[Path]:
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Release contains a symlink: {path}")
        if path.is_file() and path != directory / "CHECKSUMS.sha256":
            files.append(path)
    return files


def verify_frozen(directory: Path) -> None:
    """Refuse changed, missing, extra or escaping checksum paths."""
    if not (directory / "FROZEN").is_file():
        raise ValueError("Test split is not frozen")
    expected = {}
    for line in (directory / "CHECKSUMS.sha256").read_text("utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        path = Path(name)
        if (
            not separator
            or len(digest) != 64
            or path.is_absolute()
            or ".." in path.parts
            or name in expected
        ):
            raise ValueError("Invalid frozen checksum entry")
        expected[name] = digest
    actual = {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in _files(directory)
    }
    if actual != expected:
        raise ValueError(
            "Frozen split checksum mismatch (changed, missing or extra files)"
        )


def freeze_split(split: str, *, root: Path | None = None) -> Path:
    """Freeze test only: hash every file (FROZEN included) except the checksums."""
    if split != "test":
        raise ValueError("Only split=test can be frozen")
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    directory = root / "data/v1" / split
    directories = [directory]
    companion = directory.with_name("test_cueflip")
    if companion.exists():
        directories.append(companion)
    # Validate every companion before freezing any: a broken cue-flip release
    # must not leave the main release locked while its companion is mutable.
    for candidate in directories:
        _validate_release(candidate)
    for candidate in directories:
        if not (candidate / "FROZEN").exists():
            _freeze_directory(candidate)
    return directory


def _validate_release(directory: Path) -> None:
    if (directory / "FROZEN").exists():
        verify_frozen(directory)
        return
    for name in (
        "episodes.jsonl",
        "rejected.jsonl",
        "manifest.json",
        "dataset_card.md",
        "lexsi_provenance.json",
        "summary.json",
    ):
        if not (directory / name).is_file():
            raise ValueError(
                f"Cannot freeze incomplete test release {directory.name}: "
                f"missing {name}"
            )
    summary = json.loads((directory / "summary.json").read_text("utf-8"))
    if summary.get("status") != "complete" or summary.get("teacher") == "stub":
        raise ValueError("Cannot freeze a stopped or stub build")
    from turnguard.data.loaders import _load_release

    _load_release(directory.name, directory)
    _files(directory)


def _freeze_directory(directory: Path) -> None:
    marker = directory / "FROZEN"
    marker.write_text(
        "TurnGuard test release frozen; verify CHECKSUMS.sha256 before use.\n",
        encoding="utf-8",
    )
    try:
        (directory / "CHECKSUMS.sha256").write_text(
            "".join(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
                f"{path.relative_to(directory)}\n"
                for path in _files(directory)
            ),
            encoding="utf-8",
        )
    except Exception:
        marker.unlink()
        raise
