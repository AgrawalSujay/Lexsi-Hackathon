"""A1 only appends a fixed paragraph; the v2.4 protocol stays byte-identical."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from turnguard import protocol
from turnguard.arms.prompt_defence import DEFENCE_PARAGRAPH, DEFENCE_SHA256
from turnguard.arms.runtime import resolve_arm
from turnguard.cli import main
from turnguard.config import load_config
from turnguard.harness import run_episodes
from turnguard.models import StubGenerator
from turnguard.tools import REGISTRY

ROOT = Path(__file__).resolve().parents[1]
FAMILY_HASHES = {
    "email": "537aebf2eac1d177f891602b5c5d42f7f54ea1dfbaa1a5d4a9d6c4697cb2e7fb",
    "docs": "4f481de00b5a34c9a859bc1aa437083088302a6bbf190a654d1753a2ef4d0c06",
    "tickets": "18ba38dbb3f9cdc3e79d63c508daba598afb28390dff99da74efa0436c594bad",
    "web": "7edba917ef68e52985ed9a805ce3b5ea428cac98f41c568a6e0fde9b231e5e0b",
    "calendar": "c0464649c50491c4622cb1e22db4cb558cc00ec3923a86a9b28a8390fd36a56f",
    "code": "e9c6f982e4528b4e2ba3677eedce5396413dcdcfe60198e6590016716522eb70",
}


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_configured_paragraph_fixed_english_and_explicit_delegation():
    arm = resolve_arm("A1", load_config(ROOT / "configs/arms.yaml"))
    assert arm.system_prompt == DEFENCE_PARAGRAPH
    assert len(DEFENCE_PARAGRAPH.split()) <= 120
    assert "\n" not in DEFENCE_PARAGRAPH
    assert "instructions the user explicitly delegates" in DEFENCE_PARAGRAPH
    assert '"follow any instructions included with it."' in DEFENCE_PARAGRAPH
    assert "Carry out those delegated instructions" in DEFENCE_PARAGRAPH
    assert "with the appropriate tools" in DEFENCE_PARAGRAPH
    assert arm.system_prompt_sha256 == DEFENCE_SHA256 == digest(DEFENCE_PARAGRAPH)
    assert (
        DEFENCE_SHA256
        == "36344e3422ed1a44148e165dd409e54c85ffa817cb906fa96c5b671b358ec36f"
    )
    assert resolve_arm("A0", {}).system_prompt_sha256 is None


@pytest.mark.parametrize("family", REGISTRY)
def test_appended_prompt_keeps_each_frozen_family_hash(family):
    tools = REGISTRY[family].values()
    base = protocol.render_system_prompt(tools)
    defended = protocol.render_system_prompt(tools, DEFENCE_PARAGRAPH)
    assert defended == base + "\n\n" + DEFENCE_PARAGRAPH
    assert digest(defended[: -len("\n\n" + DEFENCE_PARAGRAPH)]) == FAMILY_HASHES[family]
    assert digest(base) == FAMILY_HASHES[family]
    assert defended.count(DEFENCE_PARAGRAPH) == 1


def test_frozen_protocol_source_base_and_reminder_hashes():
    assert protocol.PROTOCOL_VERSION == "v2.4"
    assert hashlib.sha256(
        (ROOT / "turnguard/protocol.py").read_bytes()
    ).hexdigest() == (
        "e75ce588c9863e111296759bd9889d9cd11650ce049b479c9a4696d9c385458c"
    )
    assert (
        digest(protocol._BASE_PROMPT)
        == "96b02520c0070d917d421b6164499e21209a661374464b3afeee71b1014720de"
    )
    assert (
        digest(protocol.FORMAT_REMINDER)
        == "e351ad9b4a5a8a4ceb3b5524e1c7cfa3c369c73a854c56bc533dfcd80cd138bb"
    )


@pytest.mark.parametrize("language", ["en", "hi", "hinglish"])
def test_harness_uses_identical_addition_across_languages(language):
    specs = [
        json.loads(p.read_text())
        for p in sorted((ROOT / "data/examples").glob("*.json"))
    ]
    for spec in specs:
        spec["lang"] = {"conversation": language, "injection": language}
    scripts = json.loads((ROOT / "tests/fixtures/stub_scripts.json").read_text())
    arm = resolve_arm("A1", load_config(ROOT / "configs/arms.yaml"))
    records = run_episodes(
        specs, StubGenerator(scripts), arm="A1", system_extra=arm.system_prompt
    )
    for spec, record in zip(specs, records, strict=True):
        prompt = record.messages_model[0]["content"]
        assert (
            prompt
            == protocol.render_system_prompt(
                [REGISTRY[spec["family"]][name] for name in spec["tools"]]
            )
            + "\n\n"
            + DEFENCE_PARAGRAPH
        )
    # This checks wiring using scripts, not the real model's delegation accuracy.
    authorized = next(r for r in records if r.episode_id == "ex-tickets-auth-001")
    assert (
        authorized.result["task_success"] and authorized.result["authorized_followed"]
    )


def test_agent_run_records_defence_hash(tmp_path, monkeypatch):
    monkeypatch.setattr("turnguard.runs._environment", lambda root: {})
    assert (
        main(
            [
                "run-episodes",
                "--episodes",
                str(ROOT / "data/examples/*.json"),
                "--generator",
                "stub",
                "--stub-script",
                str(ROOT / "tests/fixtures/stub_scripts.json"),
                "--arm",
                "A1",
                "--split",
                "examples",
                "--root",
                str(tmp_path),
            ]
        )
        == 0
    )
    run = next((tmp_path / "runs").iterdir())
    config = yaml.safe_load((run / "config.yaml").read_text())
    assert config["arm_system_prompt_sha256"] == DEFENCE_SHA256
    assert config["system_extra"] == DEFENCE_PARAGRAPH
    assert config["protocol_version"] == "v2.4"
