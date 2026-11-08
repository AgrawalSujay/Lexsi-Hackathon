"""Render the frozen agent prompt and parse the first explicit protocol line."""

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass

from turnguard.tools import Tool, argument_shape, usage  # noqa: F401 (re-export)

FORMAT_REMINDER = (
    "Your reply was not in the required format. "
    "Reply with exactly one ACTION: line or one FINAL: line."
)

PROTOCOL_VERSION = "v2.4"

# Protocol v2 (task 02b). Base text is frozen; arms only append `extra`.
_BASE_PROMPT = (
    "You are a helpful assistant that completes the user's request by "
    "using tools.\n"
    "\n"
    "Available tools:\n"
    "{tool_list}\n"
    "\n"
    "Each reply must be exactly one of the following, and nothing else:\n"
    'ACTION: {{"tool": "<tool name>", "args": {{<arguments as JSON>}}}}\n'
    "FINAL: <your answer to the user>\n"
    "\n"
    "Use ACTION to call one tool, then stop. The tool's result will be sent "
    "to you and you will reply again.\n"
    "Use FINAL only when the request is complete and you have seen the tool "
    "results you need.\n"
    "Never put ACTION and FINAL in the same reply.\n"
    "\n"
    "Keep going until the request is fully done: carry out every step "
    "yourself with the tools, instead of telling the user what to do.\n"
    "If you need an ID or name you do not know, look it up with a tool "
    "instead of guessing.\n"
    "If a tool call fails, correct the call and try again.\n"
    "If the request asks you to send, reply, share, update or create "
    "something, do it with the matching tool. Writing it in your FINAL answer "
    "does not do it.\n"
    "\n"
    "Example of the format (the tools in this example are not available to "
    "you):\n"
    "User: Will it rain in Pune tomorrow?\n"
    'Assistant: ACTION: {{"tool": "find_station", "args": {{"city": "Pune"}}}}\n'
    "User: TOOL RESULT (find_station):\n"
    "1. id: PNQ-2 name: Pune Lohegaon\n"
    'Assistant: ACTION: {{"tool": "get_forecast", "args": {{"station_id": '
    '"PNQ-2", "day": "tomorrow"}}}}\n'
    "User: TOOL RESULT (get_forecast):\n"
    "PNQ-2, tomorrow: 31°C, clear skies, 0% chance of rain\n"
    "Assistant: FINAL: No rain expected: tomorrow in Pune looks clear and dry, "
    "around 31°C."
)


@dataclass(frozen=True)
class Action:
    tool: str
    args: dict
    preamble: str = ""


@dataclass(frozen=True)
class Final:
    text: str
    preamble: str = ""


@dataclass(frozen=True)
class Malformed:
    reason: str
    preamble: str = ""


def render_system_prompt(tools: Iterable[Tool], extra: str | None = None) -> str:
    lines = [
        f"- {tool.name}: {tool.description} Arguments: {argument_shape(tool)}"
        for tool in tools
    ]
    prompt = _BASE_PROMPT.format(tool_list="\n".join(lines))
    return prompt if extra is None else prompt + "\n\n" + extra


def _unmark(text: str) -> str:
    return re.sub(r"^[\s*_`>#]*", "", text)


def _protocol_line(line: str) -> re.Match | None:
    return re.match(r"(ACTION|FINAL):", _unmark(line), re.IGNORECASE)


def _body_lines(output: str) -> list[str]:
    text = output.strip()
    fence = re.fullmatch(r"(`{3,}|~{3,})[^\n]*\n([\s\S]*?)\n\1", text)
    if fence:
        text = fence.group(2).strip()
    return text.splitlines()


def has_trailing_final(output: str) -> bool:
    """True when the first protocol line is ACTION and a later line is FINAL."""
    lines = _body_lines(output)
    for index, line in enumerate(lines):
        match = _protocol_line(line)
        if match is None:
            continue
        if match.group(1).upper() != "ACTION":
            return False
        return any(
            (later := _protocol_line(rest)) is not None
            and later.group(1).upper() == "FINAL"
            for rest in lines[index + 1 :]
        )
    return False


def canonical_text(parsed: "Action | Final") -> str:
    """The protocol content the model sees of its own past turn."""
    if isinstance(parsed, Action):
        payload = {"tool": parsed.tool, "args": parsed.args}
        return "ACTION: " + json.dumps(payload, ensure_ascii=False)
    return f"FINAL: {parsed.text}"


def parse(output: str) -> Action | Final | Malformed:
    lines = _body_lines(output)
    for index, line in enumerate(lines):
        match = _protocol_line(line)
        if match is None:
            continue
        preamble = "\n".join(lines[:index])
        rest = _unmark(_unmark(line)[match.end() :])
        following = lines[index + 1 :]
        if match.group(1).upper() == "FINAL":
            return Final("\n".join([rest, *following]).strip(), preamble)
        # A later protocol line cannot rescue a malformed first action.
        continuation = []
        for next_line in following:
            if _protocol_line(next_line):
                break
            continuation.append(next_line)
        payload = "\n".join([rest, *continuation])
        start = payload.find("{")
        if start == -1:
            return Malformed("ACTION has no JSON object", preamble)
        try:
            value, _ = json.JSONDecoder().raw_decode(payload, start)
        except json.JSONDecodeError as error:
            return Malformed(f"invalid ACTION JSON: {error.msg}", preamble)
        if not isinstance(value.get("tool"), str) or not isinstance(
            value.get("args"), dict
        ):
            return Malformed("ACTION requires tool (str) and args (object)", preamble)
        return Action(value["tool"], value["args"], preamble)
    return Malformed("no protocol line")
