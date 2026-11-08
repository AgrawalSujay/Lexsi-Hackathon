"""Provenance for runs from a git archive (task 02b step 5)."""

import importlib.util
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from turnguard.runs import _environment

ROOT = Path(__file__).resolve().parents[1]


def test_build_info_fallback_without_git(tmp_path: Path) -> None:
    (tmp_path / "BUILD_INFO.json").write_text(
        json.dumps({"git_commit": "abc123", "git_dirty": False})
    )
    env = _environment(tmp_path)
    assert env["git_commit"] == "abc123"
    assert env["git_dirty"] is False
    assert env["provenance"] == "BUILD_INFO.json"


def test_no_git_and_no_build_info_is_null(tmp_path: Path) -> None:
    env = _environment(tmp_path)
    assert env["git_commit"] is None and env["provenance"] is None


def test_build_info_takes_precedence_inside_another_repo(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    archive = tmp_path / "unpacked"
    archive.mkdir()
    (archive / "BUILD_INFO.json").write_text(json.dumps({"git_commit": "abc123"}))
    env = _environment(archive)
    assert env["provenance"] == "BUILD_INFO.json" and env["git_commit"] == "abc123"


def test_build_info_is_gitignored() -> None:
    assert "/BUILD_INFO.json" in (ROOT / ".gitignore").read_text().splitlines()


@pytest.mark.skipif(not (ROOT / ".git").exists(), reason="needs a git checkout")
def test_colab_zip_contains_head_and_build_info(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "colab_zip", ROOT / "scripts/colab_zip.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "src.zip"
    assert module.main(["--out", str(out)]) == 0
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    with zipfile.ZipFile(out) as archive:
        names = archive.namelist()
        info = json.loads(archive.read("BUILD_INFO.json"))
    assert "turnguard/harness.py" in names and "AGENTS.md" in names
    assert not any(n.startswith((".venv/", "runs/", ".git/")) for n in names)
    assert info["git_commit"] == head and info["git_dirty"] is False
