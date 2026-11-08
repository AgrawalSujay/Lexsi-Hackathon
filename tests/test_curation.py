"""Real CuratorKIT CPU stages: filler-only dedup and lossless grouped export."""

import copy
import json
import random
import string
from pathlib import Path

import pytest
import yaml

pytest.importorskip("curatorkit")
pytest.importorskip("detect_secrets")

from turnguard.data import builder
from turnguard.data.curation import CuratorPass, EpisodeExporter, FillerBlock
from turnguard.schema import make_clean_twin

REPO = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((REPO / "configs/data.yaml").read_text())["curator"]


def block(name, text, variant=0, language="en"):
    return FillerBlock(name, name, variant, language, {}, {"/body": text}, [])


def test_exact_and_minhash_are_on_filler_only_and_stop_above_five_percent():
    curator = CuratorPass(CONFIG, clean=False)
    text = (
        "The workshop discusses release planning, mentoring, testing and operations. "
        * 20
    )
    blocks = [
        block("first", text),
        block("exact", text),
        block("near", text + " Closing remarks."),
    ]
    report = curator.dedup(blocks)
    assert report["removed"] == 2
    assert {row["reason"] for row in report["removals"]} == {
        "ExactDeduplicator",
        "MinHashDeduplicator",
    }
    assert report["removal_rate"] == pytest.approx(2 / 3)
    assert report["stopped"]
    assert blocks[0].fields["/body"] == text  # No cleaning or rewriting.


def test_removal_below_five_percent_can_continue():
    rng = random.Random(1234)
    texts = [
        " ".join("".join(rng.choices(string.ascii_lowercase, k=6)) for _ in range(15))
        for _ in range(21)
    ]
    blocks = [block(str(i), text) for i, text in enumerate(texts)]
    blocks.append(block("duplicate", texts[0], variant=1))
    report = CuratorPass(CONFIG).dedup(blocks)
    assert report["removed"] == 1
    assert report["removal_rate"] == pytest.approx(1 / 22)
    assert not report["stopped"]


def test_adapter_keeps_multiple_injections_and_twins_and_lossless_export(tmp_path):
    bank, scenarios = builder.load()
    scenario = next(s for s in scenarios if s["id"] == "email-reschedule")
    curator = CuratorPass(CONFIG)
    curator.dedup(
        [
            block(
                "shared",
                "Different useful background information for this email scenario.",
            )
        ]
    )
    episodes = []
    for index, goal in enumerate(("exfiltrate", "destroy")):
        episode = builder.injected(
            scenario, bank, goal, "imperative", "none", "first", "body"
        )
        episode["id"] = f"base-{index}"
        episode["meta"]["filler_block"] = "shared"
        episodes += [episode, make_clean_twin(episode)]
    original = copy.deepcopy(episodes)
    passed, rejected = curator.gate(episodes, {})
    assert passed == original
    assert not rejected
    assert all(
        sample.metadata["episode"] == episode
        for sample, episode in zip(curator.result.passed, original, strict=True)
    )
    EpisodeExporter().export(curator.result.passed, tmp_path)
    assert [
        json.loads(line)
        for line in (tmp_path / "episodes.jsonl").read_text().splitlines()
    ] == original


def test_secret_drops_entire_filler_group_without_redaction():
    bank, scenarios = builder.load()
    scenario = next(s for s in scenarios if s["id"] == "email-reschedule")
    curator = CuratorPass(CONFIG)
    curator.dedup([block("shared", "Routine email scheduling background.")])
    base = builder.injected(
        scenario, bank, "exfiltrate", "imperative", "none", "first", "body"
    )
    base["meta"]["filler_block"] = "shared"
    twin = make_clean_twin(base)
    authorized = builder.authorized(scenario, scenario["authorized"][0], 0)
    authorized["meta"]["filler_block"] = "shared"
    # A documented recognizable private-key marker; no actual credential.
    base["initial_state"]["files"]["key.txt"] = "-----BEGIN RSA PRIVATE KEY-----"
    passed, rejected = curator.gate([base, twin, authorized], {})
    assert not passed
    assert len(rejected) == 3
    assert any("secret_detected" in row["reasons"][0] for row in rejected)
    assert any(row["reasons"] == ["filler_group_rejected"] for row in rejected)
    assert (
        base["initial_state"]["files"]["key.txt"] == "-----BEGIN RSA PRIVATE KEY-----"
    )


def test_missing_secrets_dependency_fails_loudly(monkeypatch):
    import curatorkit.hygiene.secrets as module

    def missing():
        raise ImportError("missing")

    monkeypatch.setattr(module, "_ensure_detect_secrets", missing)
    with pytest.raises(ImportError, match="SecretsGate needs detect-secrets"):
        CuratorPass(CONFIG)
    with pytest.raises(ValueError, match="clean=False"):
        CuratorPass(CONFIG, clean=True)
