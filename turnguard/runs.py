"""Create run directories and record environment metadata."""

import importlib
import json
import logging
import platform
import re
import subprocess
from datetime import datetime
from importlib import metadata
from pathlib import Path

logger = logging.getLogger(__name__)

_EXTRA_PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "safetune",
    "auditkit",
    "curatorkit",
    "circuitkit",
    "trl",
    "bitsandbytes",
    "transformer-lens",
)


def _environment(root: Path | None = None) -> dict:
    repo_root = root if root is not None else Path(__file__).resolve().parents[1]
    packages = {}
    for package in _EXTRA_PACKAGES:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue

    gpu_name = None
    if "torch" in packages:
        try:
            torch = importlib.import_module("torch")
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name()
        except (ImportError, OSError, RuntimeError):
            logger.warning("Could not inspect CUDA availability", exc_info=True)

    git_commit = git_dirty = provenance = None
    build_info = repo_root / "BUILD_INFO.json"
    if build_info.exists():
        # An unpacked `make colab-zip` archive: its BUILD_INFO.json is the
        # source of truth (never git, which could belong to an enclosing repo).
        try:
            info = json.loads(build_info.read_text(encoding="utf-8"))
            git_commit = info.get("git_commit")
            git_dirty = info.get("git_dirty")
            provenance = "BUILD_INFO.json"
        except (OSError, ValueError):
            logger.warning("Unreadable BUILD_INFO.json", exc_info=True)
    else:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            git_commit = result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            git_dirty = bool(status.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
        if git_commit is not None or git_dirty is not None:
            provenance = "git"

    return {
        "python_version": platform.python_version(),
        "packages": packages,
        "gpu_name": gpu_name,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "provenance": provenance,
    }


def new_run_dir(
    arm: str, model: str, split: str, *, root: str | Path | None = None
) -> Path:
    """Create a unique run under the source repo (or override root)/runs."""
    for label, value in (("arm", arm), ("split", split)):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
            raise ValueError(f"{label} must be a nonempty directory-name component")
    model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", model).strip(".-")
    if not model_slug:
        raise ValueError("model must contain at least one letter or digit")

    repo_root = (
        Path(root).resolve()
        if root is not None
        else Path(__file__).resolve().parents[1]
    )
    environment = _environment(repo_root)
    runs_root = repo_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M")
    name = f"{arm}__{model_slug}__{split}__{timestamp}"
    suffix = 1
    while True:
        run_dir = runs_root / (name if suffix == 1 else f"{name}-{suffix}")
        try:
            # mkdir is atomic: another process cannot claim the same run directory.
            run_dir.mkdir()
            break
        except FileExistsError:
            suffix += 1
    with (run_dir / "env.json").open("x", encoding="utf-8") as stream:
        json.dump(environment, stream, indent=2)
        stream.write("\n")
    return run_dir
