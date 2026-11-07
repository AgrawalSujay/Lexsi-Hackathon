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

_BASE_PROMPT = (
    "You are a helpful assistant that completes the user's request by "
    "using tools.\n"
    "\n"
    "Available tools:\n"
    "{tool_list}\n"
    "\n"
    "On every turn, reply in exactly one of these two forms:\n"
    'ACTION: {{"tool": "<tool name>", "args": {{<arguments as JSON>}}}'
    "}\n"
    "FINAL: <your answer to the user>\n"
    "\n"
    "Use ACTION to call one tool. You will then receive the tool's res"
    "ult and take your next turn.\n"
    "Use FINAL when the request is complete."
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
    lines = []
    for tool in tools:
        parameters = []
        for parameter in tool.parameters:
            text = f"{parameter.name}: {parameter.type}"
            if parameter.default is not _MISSING:
                text += " = " + json.dumps(parameter.default, ensure_ascii=False)
            parameters.append(text)
        lines.append(f"- {tool.name}({', '.join(parameters)}) — {tool.description}")
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
