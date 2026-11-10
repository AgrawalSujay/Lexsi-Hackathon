"""Standard-library-only integrity helpers, usable before editable installation."""

import hashlib
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath

MANIFEST = "KAGGLE_MANIFEST.json"
SPLITS = ("train", "dev", "test", "test_cueflip")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_name(name: str) -> None:
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or str(path) != name
    ):
        raise ValueError(f"Unsafe archive path: {name}")


def verify_release(directory: Path, *, frozen: bool = False) -> None:
    """Mirror frozen-release exact file-set checks without importing dependencies."""
    if frozen and not (directory / "FROZEN").is_file():
        raise ValueError(f"Test split is not frozen: {directory.name}")
    checksum = directory / "CHECKSUMS.sha256"
    if not checksum.exists():
        if frozen or (directory / "FROZEN").exists():
            raise ValueError("Missing frozen checksums")
        return  # Unfrozen train/dev are protected by the outer pack manifest.
    expected = {}
    for line in checksum.read_text().splitlines():
        digest, sep, name = line.partition("  ")
        safe_name(name)
        if not sep or len(digest) != 64 or name in expected:
            raise ValueError("Invalid release checksum entry")
        expected[name] = digest
    actual = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Release contains a symlink")
        if path.is_file() and path != checksum:
            actual[path.relative_to(directory).as_posix()] = sha(path.read_bytes())
    if actual != expected:
        raise ValueError(f"Release checksum mismatch: {directory.name}")


def unpack(bundle: Path, target: Path) -> dict:
    """Verify the entire zip before extraction; refuse changed existing sources."""
    with zipfile.ZipFile(bundle) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate archive entries")
        for info in infos:
            safe_name(info.filename)
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Unexpected directory or symlink entry")
        manifest = json.loads(archive.read(MANIFEST))
        if manifest.get("version") != 1 or set(names) != set(manifest["files"]) | {
            MANIFEST
        }:
            raise ValueError("Bundle file list mismatch")
        for name, digest in manifest["files"].items():
            if sha(archive.read(name)) != digest:
                raise ValueError(f"Bundle checksum mismatch: {name}")
        target = target.resolve()
        # Validate existing paths in a separate pass before writing any files.
        for name in names:
            dest = target / name
            if any(p.is_symlink() for p in (dest, *dest.parents)):
                raise ValueError("Extraction target contains a symlink")
            if dest.exists() and (
                not dest.is_file() or sha(dest.read_bytes()) != sha(archive.read(name))
            ):
                raise ValueError(
                    f"Existing checkout differs: {name}; use a new --work-dir"
                )
        for name in names:
            dest = target / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.write_bytes(archive.read(name))
    for split in SPLITS:
        verify_release(target / "data/v1" / split, frozen=split.startswith("test"))
    return manifest
