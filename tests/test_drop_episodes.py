"""Removal is explicit, audited, twin-closed, freeze-valid and rollback-safe."""

import copy
import hashlib
import json

import pytest

from turnguard.data import builder
from turnguard.data.drop_episodes import drop_episodes
from turnguard.data.freeze import freeze_split, verify_frozen
from turnguard.schema import make_clean_twin


def snapshot(directory):
    return {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}


@pytest.fixture
def release(tmp_path):
    bank, scenarios = builder.load()
    scenario = next(s for s in scenarios if s["id"] == "email-interview-slot")
    base = builder.injected(
        scenario,
        bank,
        "exfiltrate",
        "imperative",
        "none",
        "first",
        "body",
        split="test",
    )
    base["id"] = "base-1"
    other = copy.deepcopy(base)
    other["id"] = "base-2"
    auth = builder.authorized(scenario, scenario["authorized"][0], 0, split="test")
    auth["id"] = "auth-1"
    rows = [base, make_clean_twin(base), other, make_clean_twin(other), auth]
    for split in ("test", "test_cueflip"):
        directory = tmp_path / "data/v1" / split
        directory.mkdir(parents=True)
        (directory / "episodes.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        summary = {
            "status": "complete",
            "teacher": "litellm",
            "counts": {"injected": 2, "clean": 2, "authorized": 1},
            "requested_injected": 2,
            "plan_targets": {"injected": 2, "clean": 2, "authorized": 1},
        }
        (directory / "summary.json").write_text(json.dumps(summary))
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "turnguard": {"counts": summary["counts"]},
                    "stage_counts": {"EpisodeExporter": {"exported_count": 5}},
                }
            )
        )
        (directory / "lexsi_provenance.json").write_text("{}")
        (directory / "rejected.jsonl").write_text("")
        (directory / "dataset_card.md").write_text("Dataset card\n")
    return tmp_path


def write_ids(root, episode="base-1", **extra):
    path = root / "ids.jsonl"
    path.write_text(
        json.dumps(
            {
                "episode_id": episode,
                "reason": "confirmed wrong target",
                "item_key": "item-42",
                **extra,
            }
        )
        + "\n"
    )
    return path


def verify_checksums(directory):
    for line in (directory / "checksums.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("split", ["test", "test_cueflip"])
@pytest.mark.parametrize("requested", ["base-1", "base-1-clean"])
def test_drop_closes_twins_audits_and_passes_freeze(release, split, requested):
    directory = release / "data/v1" / split
    other = (
        release / "data/v1" / ("test" if split == "test_cueflip" else "test_cueflip")
    )
    untouched = snapshot(other)
    result = drop_episodes(split, write_ids(release, requested), root=release)
    assert result["removed"] == 2
    rows = [
        json.loads(line)
        for line in (directory / "episodes.jsonl").read_text().splitlines()
    ]
    assert {row["id"] for row in rows} == {"base-2", "base-2-clean", "auth-1"}
    summary = json.loads((directory / "summary.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    trail = [
        json.loads(line)
        for line in (directory / "dropped.jsonl").read_text().splitlines()
    ]
    assert trail == summary["dropped"] == manifest["turnguard"]["dropped"]
    assert all(
        row["reason"] == "confirmed wrong target" and row["item_key"] == "item-42"
        for row in trail
    )
    assert {row["twin_id"] for row in trail} == {"base-1", "base-1-clean"}
    assert summary["counts"] == {"injected": 1, "clean": 1, "authorized": 1}
    assert manifest["turnguard"]["counts"] == summary["counts"]
    assert manifest["stage_counts"]["EpisodeExporter"]["exported_count"] == 3
    assert summary["requested_shortfall"] == 1
    assert (directory / "cue_audit.json").exists()
    verify_checksums(directory)
    assert snapshot(other) == untouched
    # A second confirmed removal appends history, never replaces it.
    before = (directory / "dropped.jsonl").read_bytes()
    drop_episodes(split, write_ids(release, "auth-1", item_key="item-99"), root=release)
    assert (directory / "dropped.jsonl").read_bytes().startswith(before)
    freeze_split("test", root=release)
    verify_frozen(directory)


@pytest.mark.parametrize("split", ["test", "test_cueflip"])
def test_frozen_refusal_leaves_every_file_unchanged(release, split):
    freeze_split("test", root=release)
    directory = release / "data/v1" / split
    before = snapshot(directory)
    with pytest.raises(ValueError, match="frozen"):
        drop_episodes(split, write_ids(release), root=release)
    assert snapshot(directory) == before


def test_unknown_id_or_missing_audit_detail_leaves_no_edits(release):
    directory = release / "data/v1/test"
    before = snapshot(directory)
    with pytest.raises(ValueError, match="Unknown"):
        drop_episodes("test", write_ids(release, "missing"), root=release)
    path = release / "plain.txt"
    path.write_text("base-1\n")
    with pytest.raises(ValueError, match="reason and cross-check item key required"):
        drop_episodes("test", path, root=release)
    assert snapshot(directory) == before


def test_plain_ids_can_resolve_reviewed_disagreement_and_drop_is_idempotent(release):
    out = release / "data/crosscheck/test"
    out.mkdir(parents=True)
    (out / "disagreements.jsonl").write_text(
        json.dumps(
            {
                "key": "item-5",
                "issue": "negation dropped",
                "episode_ids": ["base-1"],
                "test_cueflip_episode_ids": [],
            }
        )
        + "\n"
    )
    path = release / "ids.txt"
    path.write_text("base-1\n")
    assert drop_episodes("test", path, root=release)["removed"] == 2
    assert drop_episodes("test", path, root=release)["removed"] == 0


def test_failed_replace_rolls_back_audit_and_metadata(release, monkeypatch):
    import turnguard.data.drop_episodes as module

    directory = release / "data/v1/test"
    before = snapshot(directory)
    original = module.os.replace
    calls = []

    def fail_second(source, target):
        calls.append(target)
        if len(calls) == 2:
            raise OSError("injected failure")
        return original(source, target)

    monkeypatch.setattr(module.os, "replace", fail_second)
    with pytest.raises(OSError, match="injected failure"):
        drop_episodes("test", write_ids(release), root=release)
    assert snapshot(directory) == before


def test_cueflip_disagreement_rows_and_cli(release, monkeypatch, capsys):
    import turnguard.data.drop_episodes as module
    from turnguard.cli import main

    original = module.drop_episodes
    monkeypatch.setattr(
        module,
        "drop_episodes",
        lambda *args, **kwargs: original(*args, root=release, **kwargs),
    )
    path = release / "confirmed.jsonl"
    path.write_text(
        json.dumps(
            {
                "key": "item-100",
                "issue": "wrong branch",
                "episode_ids": ["base-2"],
                "test_cueflip_episode_ids": ["base-1"],
            }
        )
        + "\n"
    )
    assert (
        main(["drop-episodes", "--split", "test_cueflip", "--ids-file", str(path)]) == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert {row["episode_id"] for row in report["dropped"]} == {
        "base-1",
        "base-1-clean",
    }
    assert all(row["item_key"] == "item-100" for row in report["dropped"])
