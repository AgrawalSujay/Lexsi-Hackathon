"""Pack committed source plus read-only built releases and pairs for Kaggle."""

import argparse
import io
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path

from kaggle_bundle import MANIFEST, SPLITS, safe_name, sha, verify_release


def pack(
    repo: Path,
    data_root: Path,
    output: Path,
    commit: str = "HEAD",
    pairs_root: Path | None = None,
) -> dict:
    repo, data_root, output = repo.resolve(), data_root.resolve(), output.resolve()
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"], cwd=repo, text=True
    ).strip()
    payload = subprocess.check_output(["git", "archive", revision], cwd=repo)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        for member in archive:
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError("Committed archive contains a symlink or special file")
            safe_name(member.name)
            # Built data always comes from the explicitly selected data root.
            if member.name.startswith(("data/v1/", "data/pairs/")):
                continue
            files[member.name] = archive.extractfile(member).read()
    for name in ("scripts/kaggle_run.py", "scripts/kaggle_bundle.py"):
        if name not in files:
            raise ValueError("Commit the Kaggle runner before packing its source")
    for split in SPLITS:
        directory = data_root / "data/v1" / split
        for name in ("episodes.jsonl", "manifest.json"):
            if not (directory / name).is_file():
                raise ValueError(f"Missing built split file: {split}/{name}")
        verify_release(directory, frozen=split.startswith("test"))
    roots = [(data_root, data_root / "data/v1" / split) for split in SPLITS]
    pairs_root = (pairs_root or repo).resolve()
    if (pairs_root / "data/pairs").exists():
        roots.append((pairs_root, pairs_root / "data/pairs"))
    snapshots = {}
    for base, directory in roots:
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Data contains a symlink: {path}")
            if path.is_file():
                if path.resolve() == output:
                    raise ValueError("Output zip cannot be inside the packaged data")
                content = path.read_bytes()
                files[path.relative_to(base).as_posix()] = content
                snapshots[path] = sha(content)
    # Main may be building data while a separate source worktree packs it.
    # Refuse a changing release instead of shipping a mixed snapshot.
    actual_paths = {
        p for _, directory in roots for p in directory.rglob("*") if p.is_file()
    }
    if actual_paths != set(snapshots) or any(
        path.is_symlink() or sha(path.read_bytes()) != digest
        for path, digest in snapshots.items()
    ):
        raise ValueError("Data changed while packing; retry after the build finishes")
    if MANIFEST in files:
        raise ValueError("Reserved pack manifest already exists in source")
    manifest = {
        "version": 1,
        "git_commit": revision,
        "source": "git archive (committed files only)",
        "splits": list(SPLITS),
        "files": {name: sha(data) for name, data in sorted(files.items())},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Fixed zip timestamps and sorted entries make identical inputs reproducible.
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(
            {
                **files,
                MANIFEST: (json.dumps(manifest, indent=2) + "\n").encode(),
            }.items()
        ):
            archive.writestr(
                zipfile.ZipInfo(name, (2020, 1, 1, 0, 0, 0)),
                data,
                compress_type=zipfile.ZIP_DEFLATED,
            )
    return {
        "zip": str(output),
        "git_commit": revision,
        "files": len(files),
        "zip_sha256": sha(output.read_bytes()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--data-root", type=Path, help="Read-only checkout with data/v1 and data/pairs"
    )
    parser.add_argument(
        "--pairs-root", type=Path, help="Checkout with data/pairs (default: --repo)"
    )
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--out", type=Path, default=Path("turnguard_kaggle.zip"))
    args = parser.parse_args(argv)
    print(
        json.dumps(
            pack(
                args.repo,
                args.data_root or args.repo,
                args.out,
                args.commit,
                args.pairs_root,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
