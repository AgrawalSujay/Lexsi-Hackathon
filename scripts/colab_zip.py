"""Build a source archive of HEAD for Colab, with BUILD_INFO.json inside.

Runs from an archive have no .git, so runs.py reads BUILD_INFO.json to record
which commit produced them. The archive contains exactly HEAD: uncommitted
changes are not included, and a warning is printed if any exist.

Usage: python scripts/colab_zip.py [--out ~/Downloads/turnguard_src.zip]
"""

import argparse
import datetime
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", default=str(Path.home() / "Downloads" / "turnguard_src.zip")
    )
    args = parser.parse_args(argv)
    out = Path(args.out).expanduser()
    commit = git("rev-parse", "HEAD")
    uncommitted = bool(git("status", "--porcelain"))
    out.parent.mkdir(parents=True, exist_ok=True)
    git("archive", "--format=zip", "-o", str(out), "HEAD")
    info = {
        "git_commit": commit,
        # The archive is exactly HEAD, so its code is never dirty.
        "git_dirty": False,
        "working_tree_had_uncommitted_changes": uncommitted,
        "built_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }
    with zipfile.ZipFile(out, "a") as archive:
        archive.writestr("BUILD_INFO.json", json.dumps(info, indent=2) + "\n")
    print(f"wrote {out} from {commit[:7]}")
    if uncommitted:
        print(
            "warning: uncommitted changes exist and are NOT in the archive",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
