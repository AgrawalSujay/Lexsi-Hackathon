"""Define episode specs, validate them, and construct matched clean twins."""

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

# Shared vocabulary; the executable registry added in step 2 uses these names.
FAMILY_TOOLS = {
    "email": "list_inbox read_email read_file send_email archive_old".split(),
    "docs": "list_files read_file write_summary share_file delete_file".split(),
    "tickets": "get_ticket list_tickets update_ticket reply_customer refund".split(),
    "web": "search fetch_page save_note send_email".split(),
    "calendar": "list_events create_event cancel_event send_invite".split(),
    "code": "list_repo read_repo_file open_pr add_collaborator delete_branch".split(),
}


@dataclass
class EpisodeSpec:
    id: str
    split: str
    episode_type: str
    twin_of: str | None
    family: str
    carrier: str | None
    goal: str | None
    position: str | None
    style: str | None
    lang: dict
    user_turns: list[str]
    tools: list[str]
    initial_state: dict
    injection: dict | None
    authorized: dict | None
    task_oracle: dict
    meta: dict

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: dict) -> "EpisodeSpec":
        problems = validate_episode(value)
        if problems:
            raise ValueError("; ".join(problems))
        return cls(**copy.deepcopy(value))

    @classmethod
    def from_json(cls, value: str) -> "EpisodeSpec":
        return cls.from_dict(json.loads(value))


def canonical_json(value: dict) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def spec_sha256(spec: EpisodeSpec | dict) -> str:
    value = spec.to_dict() if isinstance(spec, EpisodeSpec) else spec
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _tokens(pointer: str) -> list[str]:
    if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
        raise ValueError("invalid JSON Pointer")
    if re.search(r"~(?![01])", pointer):
        raise ValueError("invalid JSON Pointer escape")
    return (
        [t.replace("~1", "/").replace("~0", "~") for t in pointer[1:].split("/")]
        if pointer
        else []
    )


def _child(value: Any, token: str) -> Any:
    if isinstance(value, list):
        if not re.fullmatch(r"0|[1-9][0-9]*", token):
            raise ValueError("invalid array index")
        return value[int(token)]
    if isinstance(value, dict):
        return value[token]
    raise ValueError("JSON Pointer traverses a scalar")


def resolve_pointer(value: Any, pointer: str) -> Any:
    for token in _tokens(pointer):
        value = _child(value, token)
    return value


def validate_episode(
    spec: EpisodeSpec | dict, *, twins: dict | None = None
) -> list[str]:
    value = spec.to_dict() if isinstance(spec, EpisodeSpec) else spec
    errors = []
    if not isinstance(value, dict):
        return ["episode must be an object"]
    types = {
        "id": str,
        "split": str,
        "episode_type": str,
        "family": str,
        "lang": dict,
        "user_turns": list,
        "tools": list,
        "initial_state": dict,
        "task_oracle": dict,
        "meta": dict,
    }
    types.update(
        dict.fromkeys(
            ("twin_of", "carrier", "goal", "position", "style"), (str, type(None))
        )
    )
    types.update(dict.fromkeys(("injection", "authorized"), (dict, type(None))))
    for key, expected in types.items():
        if key not in value or not isinstance(value[key], expected):
            errors.append(f"{key}: missing or wrong type")
    if errors:
        return errors
    for key, choices in (
        ("split", ("examples", "train", "dev", "test")),
        ("episode_type", ("injected", "clean", "authorized")),
        ("family", FAMILY_TOOLS),
    ):
        if value[key] not in choices:
            errors.append(f"{key}: invalid value")
    if not value["id"]:
        errors.append("id must not be empty")
    for key in ("user_turns", "tools"):
        if not value[key] or not all(isinstance(x, str) for x in value[key]):
            errors.append(f"{key}: expected nonempty list of strings")
    if any(
        not isinstance(value["lang"].get(k), str) for k in ("conversation", "injection")
    ):
        errors.append("lang: expected conversation and injection strings")
    tools = value["tools"]
    for tool in tools:
        if tool not in FAMILY_TOOLS.get(value["family"], []):
            errors.append(f"tools: {tool!r} does not exist for family")

    def matchers(items: Any, label: str) -> None:
        if not isinstance(items, list) or not items:
            errors.append(f"{label}: expected nonempty matcher list")
            return
        for item in items:
            if not isinstance(item, dict):
                errors.append(f"{label}: matcher must be object")
            elif item.get("kind") == "call":
                if item.get("tool") not in tools:
                    errors.append(f"{label}: matcher tool missing from tools")
                for field in ("args_match", "args_contains"):
                    if field in item and not isinstance(item[field], dict):
                        errors.append(f"{label}: {field} must be object")
                if isinstance(item.get("args_contains"), dict):
                    for terms in item["args_contains"].values():
                        if not isinstance(terms, list) or not all(
                            isinstance(t, str) for t in terms
                        ):
                            errors.append(
                                f"{label}: args_contains values must be string lists"
                            )
            elif item.get("kind") == "answer":
                terms = item.get("contains")
                if not isinstance(terms, list) or not all(
                    isinstance(t, str) for t in terms
                ):
                    errors.append(f"{label}: contains must be string list")
                if "turn" in item and (
                    type(item["turn"]) is not int
                    or not 1 <= item["turn"] <= len(value["user_turns"])
                ):
                    errors.append(f"{label}: turn out of range")
            else:
                errors.append(f"{label}: unknown matcher kind")

    matchers(value["task_oracle"].get("all"), "task_oracle.all")
    for key in ("injection", "authorized"):
        payload = value[key]
        required = (
            value["episode_type"]
            == {"injection": "injected", "authorized": "authorized"}[key]
        )
        if required != (payload is not None):
            errors.append(f"{key}: inconsistent with episode_type")
        if payload is None:
            continue
        try:
            parts = payload_parts(payload)
        except ValueError as error:
            errors.append(f"{key}: {error}")
            parts = []
        for part in parts:
            try:
                container = value[
                    "initial_state" if part["where"] == "state" else "user_turns"
                ]
                target = resolve_pointer(container, part["slot"])
                if not isinstance(target, str) or part["text"] not in target:
                    errors.append(f"{key}: slot must contain text")
            except (KeyError, IndexError, TypeError, ValueError):
                errors.append(f"{key}: invalid slot")
        matchers(payload.get("target"), f"{key}.target")
    tool_errors = value["initial_state"].get("tool_errors", {})
    if not isinstance(tool_errors, dict):
        errors.append("tool_errors must be object")
    else:
        for tool, entries in tool_errors.items():
            if tool not in tools or not isinstance(entries, list):
                errors.append("tool_errors: invalid tool or entries")
                continue
            for entry in entries:
                if (
                    not isinstance(entry, dict)
                    or type(entry.get("on_call")) is not int
                    or entry["on_call"] < 1
                    or not isinstance(entry.get("message"), str)
                ):
                    errors.append("tool_errors: invalid on_call or message")
    if twins and value["episode_type"] == "clean" and value["twin_of"] in twins:
        try:
            expected = make_clean_twin(twins[value["twin_of"]])
            expected = (
                expected.to_dict() if isinstance(expected, EpisodeSpec) else expected
            )
            if value["initial_state"] != expected["initial_state"]:
                errors.append("clean twin state differs from stripped injected state")
            if value["user_turns"] != expected["user_turns"]:
                errors.append(
                    "clean twin user_turns differ from stripped injected turns"
                )
        except ValueError:
            errors.append("invalid injected twin")
    return errors


def make_clean_twin(spec: EpisodeSpec | dict) -> EpisodeSpec | dict:
    value = spec.to_dict() if isinstance(spec, EpisodeSpec) else copy.deepcopy(spec)
    problems = validate_episode(value)
    if problems or value["episode_type"] != "injected":
        raise ValueError("Expected valid injected episode: " + "; ".join(problems))
    for part in payload_parts(value["injection"]):
        tokens = _tokens(part["slot"])
        parent = value["initial_state" if part["where"] == "state" else "user_turns"]
        for token in tokens[:-1]:
            parent = _child(parent, token)
        key = int(tokens[-1]) if isinstance(parent, list) else tokens[-1]
        parent[key] = parent[key].replace(part["text"], "").strip()
    value.update(
        id=value["id"] + "-clean",
        twin_of=value["id"],
        episode_type="clean",
        injection=None,
    )
    return EpisodeSpec.from_dict(value) if isinstance(spec, EpisodeSpec) else value


def payload_parts(payload: dict) -> list[dict]:
    """Normalize shorthand without modifying the payload; validate part structure."""
    if "parts" in payload:
        if "slot" in payload or "text" in payload:
            raise ValueError("shorthand and parts must not be mixed")
        parts = payload["parts"]
    else:
        parts = [
            {"where": "state", "slot": payload.get("slot"), "text": payload.get("text")}
        ]
    if not isinstance(parts, list) or not parts:
        raise ValueError("parts must be a nonempty list")
    for part in parts:
        if not isinstance(part, dict) or part.get("where") not in (
            "state",
            "user_turn",
        ):
            raise ValueError("part where must be state or user_turn")
        if not isinstance(part.get("text"), str) or not part["text"]:
            raise ValueError("part text must be a nonempty string")
        tokens = _tokens(part.get("slot"))
        if not tokens:
            raise ValueError("part slot must address a string")
        if part["where"] == "user_turn" and not re.fullmatch(
            r"/(0|[1-9][0-9]*)", part["slot"]
        ):
            raise ValueError("user_turn slot must be /<index>")
    return copy.deepcopy(parts)


@dataclass
class EpisodeRecord:
    """A deterministic episode trace and its independently computed outcome."""

    episode_id: str
    spec_sha256: str
    arm: str
    model: str
    messages_model: list[dict]
    messages_openai: list[dict]
    calls: list[dict]
    answers: list[dict]
    final_state: dict
    steps: int
    malformed_turns: int
    result: dict
    # Protocol v2: every raw model output, and ACTION turns that also wrote FINAL.
    raw_outputs: list[dict] = field(default_factory=list)
    action_with_trailing_final: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, value: str) -> "EpisodeRecord":
        return cls(**json.loads(value))
