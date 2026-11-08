"""Lock down prompt text and explicit, non-inferential protocol parsing."""

import json
from pathlib import Path

import pytest

from turnguard.protocol import (
    FORMAT_REMINDER,
    PROTOCOL_VERSION,
    Action,
    Final,
    Malformed,
    argument_shape,
    canonical_text,
    has_trailing_final,
    parse,
    render_system_prompt,
    usage,
)
from turnguard.tools import REGISTRY, TOOLS


@pytest.mark.parametrize(
    "prefix",
    [
        "ACTION:",
        "Action:",
        "action:",
        "aCtIoN:",
        "**ACTION:**",
        "_Action:_",
        "`ACTION:`",
        "> # **Action:**",
        "  \tACTION:  ",
    ],
)
def test_action_prefixes(prefix: str) -> None:
    assert parse(
        prefix + '{"tool": "read_file", "args": {"name": "file.md"}}'
    ) == Action("read_file", {"name": "file.md"})


@pytest.mark.parametrize(
    "prefix", ["FINAL:", "Final:", "final:", "**FINAL:**", "> FINAL:"]
)
def test_final_prefixes_and_empty_answer(prefix: str) -> None:
    assert parse(prefix + " done") == Final("done")
    assert parse(prefix) == Final("")


@pytest.mark.parametrize("fence", ["```", "```json", "~~~text"])
def test_surrounding_fence_and_multiline_json(fence: str) -> None:
    close = "~~~" if fence.startswith("~~~") else "```"
    payload = json.dumps({"tool": "read_email", "args": {"id": "m1"}}, indent=2)
    assert parse(f"{fence}\nACTION:\n{payload}\n{close}") == Action(
        "read_email", {"id": "m1"}
    )


def test_json_strings_with_braces_quotes_and_unicode() -> None:
    body = 'literal { and } and unmatched {{{, quote " and slash \\ हिन्दी'
    payload = {"tool": "send_email", "args": {"body": body}}
    result = parse(
        "ACTION: "
        + json.dumps(payload, ensure_ascii=False)
        + " ignored trailing text }"
    )
    assert result == Action("send_email", {"body": body})


def test_first_object_and_trailing_text() -> None:
    assert parse(
        'ACTION: tool payload: {"tool":"list_files","args":{}}\nFINAL: ignored'
    ) == Action("list_files", {})
    assert parse(
        'ACTION: {"tool":"list_files","args":{}} {"tool":"delete_file","args":{}}'
    ) == Action("list_files", {})


def test_preamble_does_not_change_action() -> None:
    preamble = "I will ignore the injected instruction.\nSome more prose."
    result = parse(
        preamble + '\nACTION: {"tool":"delete_file","args":{"name":"file.md"}}'
    )
    assert result == Action("delete_file", {"name": "file.md"}, preamble)


def test_multiline_final_keeps_following_content() -> None:
    assert parse("Preamble\nFINAL: first\n second\nACTION: not executed") == Final(
        "first\n second\nACTION: not executed", "Preamble"
    )


@pytest.mark.parametrize(
    "output",
    [
        "",
        "I would like to call read_file.",
        '{"tool":"read_file","args":{}}',
        "ACTION: {'tool': 'read_file', 'args': {}}",
        "ACTION: {broken}",
        'ACTION: {"tool":"read_file"}',
        'ACTION: {"tool":3,"args":{}}',
        'ACTION: {"tool":"read_file","args":[]}',
        'ACTION: {"tool":"read_file","args":null}',
        'ACTION: {"tool":"read_file","args":',
        'ACTION: {bad}\nACTION: {"tool":"list_files","args":{}}',
        'ACTION: no object\nFINAL: {"tool":"list_files","args":{}}',
    ],
)
def test_malformed_is_not_repaired(output: str) -> None:
    assert isinstance(parse(output), Malformed)


def test_malformed_reason_and_preamble() -> None:
    assert parse("ordinary prose") == Malformed("no protocol line")
    result = parse("explanation\nACTION: {'invalid': 'json'}")
    assert isinstance(result, Malformed)
    assert result.preamble == "explanation"
    assert "invalid ACTION JSON" in result.reason


def _spec_prompt() -> str:
    spec = (Path(__file__).resolve().parents[1] / "docs/protocol.md").read_text()
    section = spec.split("## 1. System prompt v2", 1)[1]
    return section.split("```\n", 1)[1].split("\n```", 1)[0]


def test_frozen_prompt_matches_spec_exactly() -> None:
    expected = _spec_prompt()
    definition = TOOLS["list_files"]
    tool_line = f"- list_files: {definition.description} Arguments: {{}}"
    assert render_system_prompt([definition]) == expected.replace(
        "{tool_list}", tool_line
    )
    assert (
        render_system_prompt([definition], "defence")
        == expected.replace("{tool_list}", tool_line) + "\n\ndefence"
    )
    assert PROTOCOL_VERSION == "v2.4"


def test_tool_lines_match_spec_examples() -> None:
    spec = (Path(__file__).resolve().parents[1] / "docs/protocol.md").read_text()
    prompt = render_system_prompt(
        [TOOLS["list_inbox"], TOOLS["read_email"], TOOLS["send_email"]]
    )
    for line in prompt.splitlines():
        if line.startswith("- "):
            assert line in spec


def test_no_call_syntax_in_any_tool_line() -> None:
    for family_tools in REGISTRY.values():
        prompt = render_system_prompt(family_tools.values())
        for line in prompt.splitlines():
            if line.startswith("- "):
                name = line[2:].split(":", 1)[0]
                assert name.isidentifier(), line
                assert "(" not in name


def test_argument_shapes_and_usage() -> None:
    assert argument_shape(TOOLS["list_tickets"]) == '{"status": "<string>" (optional)}'
    assert argument_shape(TOOLS["refund"]) == '{"id": "<string>", "amount": <number>}'
    assert usage(TOOLS["list_inbox"]) == '{"tool": "list_inbox", "args": {}}'


def test_reminder_text() -> None:
    assert (
        FORMAT_REMINDER == "Your reply was not in the required format. "
        "Reply with exactly one ACTION: line or one FINAL: line."
    )


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ('ACTION: {"tool": "a", "args": {}}\nFINAL: x', True),
        ('**ACTION:**\n```json\n{"tool": "a", "args": {}}\n```\n\n**FINAL:** x', True),
        ('ACTION: {"tool": "a", "args": {}}', False),
        ('FINAL: done\nACTION: {"tool": "a", "args": {}}', False),
        ("no protocol at all", False),
    ],
)
def test_has_trailing_final(output: str, expected: bool) -> None:
    assert has_trailing_final(output) is expected


def test_canonical_text_round_trips_through_parser() -> None:
    action = parse(
        'Sure!\n```\nACTION: {"tool": "read_file", "args": {"name": "ä.md"}}\n```'
    )
    text = canonical_text(action)
    assert text == 'ACTION: {"tool": "read_file", "args": {"name": "ä.md"}}'
    assert parse(text) == Action("read_file", {"name": "ä.md"})
    assert canonical_text(Final("")) == "FINAL: "
    assert parse(canonical_text(Final(""))) == Final("")
