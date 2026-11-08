"""Exercise reference schemas, pointers, validation, and clean twin isolation."""

import copy
import json
from pathlib import Path

import pytest

from turnguard.schema import (
    EpisodeSpec,
    make_clean_twin,
    resolve_pointer,
    spec_sha256,
    validate_episode,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "data/examples"


def examples() -> dict:
    return {
        v["id"]: v for p in EXAMPLES.glob("*.json") for v in [json.loads(p.read_text())]
    }


def test_reference_roundtrips() -> None:
    specs = examples()
    for value in specs.values():
        assert validate_episode(value, twins=specs) == []
        episode = EpisodeSpec.from_dict(value)
        assert EpisodeSpec.from_json(episode.to_json()) == episode
        assert spec_sha256(value) == spec_sha256(episode)
    hindi = specs["ex-docs-inj-002"]["injection"]["text"]
    assert hindi in EpisodeSpec.from_dict(specs["ex-docs-inj-002"]).to_json()


def test_clean_twin() -> None:
    source = examples()["ex-email-inj-001"]
    original = copy.deepcopy(source)
    clean = make_clean_twin(source)
    assert source == original
    assert clean["initial_state"] == examples()["ex-email-clean-001"]["initial_state"]
    clean["initial_state"]["files"]["team_notes.txt"] = "changed"
    assert source == original


def test_pointers() -> None:
    value = {"a/b": {"~key": ["text"]}, "": "empty"}
    assert resolve_pointer(value, "/a~1b/~0key/0") == "text"
    assert resolve_pointer(value, "/") == "empty"
    assert resolve_pointer(value, "") is value
    for pointer in ("bad", "/~2", "/a~1b/~0key/01", "/a~1b/~0key/-"):
        with pytest.raises((ValueError, KeyError)):
            resolve_pointer(value, pointer)


@pytest.mark.parametrize(
    "field,bad",
    [("split", "other"), ("user_turns", 1), ("injection", None), ("tools", ["refund"])],
)
def test_bad_fields(field: str, bad: object) -> None:
    value = examples()["ex-email-inj-001"]
    value[field] = bad
    assert validate_episode(value)


def test_bad_payloads_and_twin() -> None:
    specs = examples()
    value = specs["ex-email-inj-001"]
    value["injection"]["text"] = "missing text"
    assert any("contain text" in e for e in validate_episode(value))
    value = examples()["ex-email-inj-001"]
    value["task_oracle"]["all"][0]["tool"] = "refund"
    assert any("matcher tool" in e for e in validate_episode(value))
    specs = examples()
    clean = specs["ex-email-clean-001"]
    clean["initial_state"]["files"]["extra"] = "changed"
    assert any("twin state" in e for e in validate_episode(clean, twins=specs))


def multipart(quoted: bool = False) -> dict:
    value = examples()["ex-email-inj-001"]
    payload = value["injection"]
    part = {"where": "state", "slot": payload["slot"], "text": payload["text"]}
    value["injection"] = {"parts": [part], "target": payload["target"]}
    if quoted:
        value["carrier"] = "quoted"
        value["user_turns"][0] += "\nQUOTED_ATTACK"
        value["injection"]["parts"].append(
            {"where": "user_turn", "slot": "/0", "text": "QUOTED_ATTACK"}
        )
    else:
        value["position"] = "split"
        value["initial_state"]["inbox"][1]["body"] += "\nSECOND_ATTACK"
        value["injection"]["parts"].append(
            {"where": "state", "slot": "/inbox/1/body", "text": "SECOND_ATTACK"}
        )
    return value


@pytest.mark.parametrize("quoted", [False, True])
def test_multipart_clean_twin(quoted: bool) -> None:
    value = multipart(quoted)
    original = copy.deepcopy(value)
    assert validate_episode(value) == []
    clean = make_clean_twin(value)
    assert validate_episode(clean, twins={value["id"]: value}) == []
    content = json.dumps([clean["initial_state"], clean["user_turns"]])
    for part in value["injection"]["parts"]:
        assert part["text"] not in content
    assert value == original
    episode = EpisodeSpec.from_dict(value)
    assert make_clean_twin(episode).to_dict() == clean
    if quoted:
        clean["user_turns"][0] += " unexpected"
        assert any(
            "user_turns" in e
            for e in validate_episode(clean, twins={value["id"]: value})
        )


@pytest.mark.parametrize("field", ["slot", "text"])
def test_mixing_parts_and_shorthand_rejected(field: str) -> None:
    value = multipart()
    value["injection"][field] = "mixed"
    assert any("mixed" in e for e in validate_episode(value))


@pytest.mark.parametrize(
    "part",
    [
        {"where": "other", "slot": "/0", "text": "x"},
        {"where": "user_turn", "slot": "/99", "text": "x"},
        {"where": "user_turn", "slot": "/0/body", "text": "x"},
        {"where": "user_turn", "slot": "/-1", "text": "x"},
        {"where": "user_turn", "slot": "/01", "text": "x"},
        {"where": "state", "slot": "/inbox/0/body", "text": "absent"},
    ],
)
def test_invalid_part(part: dict) -> None:
    value = multipart()
    value["injection"]["parts"] = [part]
    assert validate_episode(value)


def test_authorized_parts() -> None:
    value = examples()["ex-tickets-auth-001"]
    payload = value["authorized"]
    value["authorized"] = {
        "parts": [{"where": "state", "slot": payload["slot"], "text": payload["text"]}],
        "target": payload["target"],
    }
    assert validate_episode(value) == []


@pytest.mark.parametrize(
    "trigger", [{"when": {"id": "m1"}}, {"when": {}}, {"on_call": 1}]
)
def test_argument_keyed_or_legacy_error_schema(trigger):
    value = examples()["ex-email-inj-001"]
    value["initial_state"]["tool_errors"] = {
        "read_email": [{**trigger, "message": "Retry"}]
    }
    assert not validate_episode(value)


@pytest.mark.parametrize(
    "trigger",
    [
        {"when": []},
        {"when": {1: "x"}},
        {"when": {}, "on_call": 1},
        {"on_call": 0},
        {"on_call": True},
        {},
    ],
)
def test_error_schema_rejects_ambiguous_or_invalid_triggers(trigger):
    value = examples()["ex-email-inj-001"]
    value["initial_state"]["tool_errors"] = {
        "read_email": [{**trigger, "message": "Retry"}]
    }
    assert any("tool_errors" in problem for problem in validate_episode(value))
