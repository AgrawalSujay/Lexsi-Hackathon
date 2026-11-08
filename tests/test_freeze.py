"""The test split freezes once, refuses tampering, and is evaluation-only."""

import json
import shutil
from pathlib import Path

import pytest

from turnguard.data import builder
from turnguard.data.freeze import freeze_split, verify_frozen
from turnguard.data.loaders import load_split
from turnguard.schema import make_clean_twin


def release(root, *, teacher="litellm", status="complete"):
    bank, scenarios = builder.load()
    scenario = next(s for s in scenarios if s["id"] == "email-interview-slot")
    episode = builder.injected(
        scenario,
        bank,
        "exfiltrate",
        "imperative",
        "none",
        "first",
        "body",
        split="test",
    )
    episode["id"] = "test-ep-0"
    directory = root / "data/v1/test"
    directory.mkdir(parents=True)
    rows = [episode, make_clean_twin(episode)]
    (directory / "episodes.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8"
    )
    for name in ("rejected.jsonl", "dataset_card.md"):
        (directory / name).write_text("", "utf-8")
    for name in ("manifest.json", "lexsi_provenance.json"):
        (directory / name).write_text("{}", "utf-8")
    (directory / "summary.json").write_text(
        json.dumps({"status": status, "teacher": teacher}), "utf-8"
    )
    return directory


def test_freeze_then_eval_load_works_and_other_purposes_are_refused(tmp_path):
    directory = release(tmp_path)
    freeze_split("test", root=tmp_path)
    assert (directory / "FROZEN").is_file()
    verify_frozen(directory)
    assert len(load_split("test", purpose="eval", root=tmp_path)) == 2
    for purpose in ("train", "steer", "tune", "calibrate"):
        with pytest.raises(ValueError, match="evaluation-only"):
            load_split("test", purpose=purpose, root=tmp_path)


def test_tampering_after_freeze_is_detected(tmp_path):
    directory = release(tmp_path)
    freeze_split("test", root=tmp_path)
    with (directory / "episodes.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_split("test", purpose="eval", root=tmp_path)
    (directory / "episodes.jsonl").write_text("", "utf-8")
    (directory / "extra.txt").write_text("x", "utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_frozen(directory)


def test_stub_or_stopped_builds_cannot_be_frozen(tmp_path):
    release(tmp_path / "a", teacher="stub")
    with pytest.raises(ValueError, match="stopped or stub"):
        freeze_split("test", root=tmp_path / "a")
    release(tmp_path / "b", status="stopped")
    with pytest.raises(ValueError, match="stopped or stub"):
        freeze_split("test", root=tmp_path / "b")
    with pytest.raises(ValueError, match="Only split=test"):
        freeze_split("dev", root=tmp_path / "a")


def test_build_refuses_to_write_into_a_frozen_split(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    shutil.copytree(repo / "configs", tmp_path / "configs")
    shutil.copytree(repo / "turnguard/data", tmp_path / "turnguard/data")
    release(tmp_path)
    freeze_split("test", root=tmp_path)
    # Refused before any teacher call, so no settings or network are needed.
    with pytest.raises(ValueError, match="frozen"):
        builder.build_data("test", teacher="litellm", limit=1, root=tmp_path)
