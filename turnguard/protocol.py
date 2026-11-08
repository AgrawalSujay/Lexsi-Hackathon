"""Render the frozen agent prompt and parse the first explicit protocol line."""

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass

from turnguard.tools import _MISSING, Tool

FORMAT_REMINDER = (
    "Your reply was not in the required format. "
    "Reply with exactly one ACTION: line or one FINAL: line."
)

PROTOCOL_VERSION = "v2"

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
    "Example of the format (the tools in this example are not available to "
    "you):\n"
    "User: Where is order 991?\n"
    'Assistant: ACTION: {{"tool": "lookup_order", "args": {{"order_id": '
    '"991"}}}}\n'
    "User: TOOL RESULT (lookup_order):\n"
    "status: shipped, arriving Friday\n"
    "Assistant: FINAL: Order 991 has shipped and should arrive on Friday."
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


_PLACEHOLDERS = {
    "str": '"<string>"',
    "int": "<integer>",
    "float": "<number>",
    "list[str]": '["<string>", ...]',
}


def argument_shape(tool: Tool) -> str:
    """Render a tool's arguments as the JSON shape the model should copy."""
    parts = []
    for parameter in tool.parameters:
        part = f'"{parameter.name}": {_PLACEHOLDERS[parameter.type]}'
        if parameter.default is not _MISSING:
            part += " (optional)"
        parts.append(part)
    return "{" + ", ".join(parts) + "}"


def usage(tool: Tool) -> str:
    """One-line usage hint in the same JSON shape as the tool list."""
    return f'{{"tool": "{tool.name}", "args": {argument_shape(tool)}}}'


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


def parse(output: str) -> Action | Final | Malformed:
    text = output.strip()
    fence = re.fullmatch(r"(`{3,}|~{3,})[^\n]*\n([\s\S]*?)\n\1", text)
    if fence:
        text = fence.group(2).strip()
    lines = text.splitlines()
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
