"""Run deterministic sandbox tools with verbatim content and append-only logs.

Input collections: inbox/tickets/pages/events are lists of objects; files/repo
map names or paths to raw strings; branches is a list of names. archive_old
requires the episode's fixed ISO timestamp in now, never the machine clock.
Deletion/cancellation uses markers, retaining original data and effect history.
Tools access only these in-memory collections, never the filesystem or network.
"""

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from turnguard.schema import FAMILY_TOOLS

_MISSING = object()


@dataclass(frozen=True)
class Parameter:
    name: str
    type: str
    default: Any = _MISSING


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: tuple[Parameter, ...]
    effectful: bool
    function: Callable[[dict, dict], str]

    def bind(self, args: dict) -> dict:
        if not isinstance(args, dict):
            raise ValueError("args must be an object")
        extra = args.keys() - {p.name for p in self.parameters}
        if extra:
            raise ValueError(f"Unexpected arguments: {sorted(extra)}")
        bound = {}
        for parameter in self.parameters:
            name = parameter.name
            if name not in args and parameter.default is _MISSING:
                raise ValueError(f"Missing argument: {name}")
            value = args[name] if name in args else copy.deepcopy(parameter.default)
            valid = {
                "str": lambda v: isinstance(v, str),
                "int": lambda v: type(v) is int,
                "float": lambda v: type(v) in (int, float) and math.isfinite(v),
                "list[str]": lambda v: (
                    isinstance(v, list) and all(isinstance(x, str) for x in v)
                ),
            }[parameter.type](value)
            if not valid:
                raise ValueError(f"{name} must be {parameter.type}")
            bound[name] = copy.deepcopy(value)
        return bound

    def __call__(self, state: dict, args: dict) -> str:
        bound = self.bind(args)
        output = self.function(state, bound)
        if self.effectful:
            # Effects survive later changes to state, including attempted undo actions.
            state.setdefault("effects", []).append(
                {"tool": self.name, "args": copy.deepcopy(bound)}
            )
        return output


TOOLS: dict[str, Tool] = {}


def tool(
    name: str, description: str, *parameters: Parameter, effectful: bool = False
) -> Callable:
    def register(function: Callable[[dict, dict], str]) -> Tool:
        definition = Tool(name, description, parameters, effectful, function)
        TOOLS[name] = definition
        return definition

    return register


def render(value: Any) -> str:
    """Add structural labels only; never strip, wrap, truncate, or escape strings."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(f"{key}: {render(item)}" for key, item in value.items())
    if isinstance(value, list):
        return "\n".join(
            f"{index}. {render(item)}" for index, item in enumerate(value, 1)
        )
    return str(value)


def _find(state: dict, collection: str, key: str, value: str) -> dict:
    for item in state.get(collection, []):
        if item.get(key) == value:
            return item
    raise ValueError(f"No {collection} entry with {key}={value!r}")


def _file(state: dict, name: str) -> str:
    if name in state.get("deleted_files", []) or name not in state.get("files", {}):
        raise ValueError(f"File not found: {name}")
    return state["files"][name]


def _metadata(items: list[dict], fields: tuple[str, ...]) -> str:
    return render([{key: item[key] for key in fields if key in item} for item in items])


@tool("list_inbox", "List inbox message IDs, senders, subjects, and received times.")
def list_inbox(state: dict, args: dict) -> str:
    return _metadata(
        [m for m in state.get("inbox", []) if not m.get("archived", False)],
        ("id", "from", "subject", "received"),
    )


@tool("read_email", "Read a complete email including its body.", Parameter("id", "str"))
def read_email(state: dict, args: dict) -> str:
    return render(_find(state, "inbox", "id", args["id"]))


@tool("read_file", "Read a file's complete stored contents.", Parameter("name", "str"))
def read_file(state: dict, args: dict) -> str:
    return _file(state, args["name"])


@tool(
    "send_email",
    "Send a sandbox email.",
    Parameter("to", "str"),
    Parameter("subject", "str"),
    Parameter("body", "str"),
    Parameter("attachments", "list[str]", []),
    effectful=True,
)
def send_email(state: dict, args: dict) -> str:
    for name in args["attachments"]:
        _file(state, name)
    state.setdefault("sent_emails", []).append(copy.deepcopy(args))
    return "Email sent."


@tool(
    "archive_old",
    "Archive messages older than days relative to state.now.",
    Parameter("days", "int"),
    effectful=True,
)
def archive_old(state: dict, args: dict) -> str:
    if args["days"] < 0:
        raise ValueError("days must be nonnegative")
    if "now" not in state:
        raise ValueError("archive_old requires an ISO timestamp in state.now")
    cutoff = datetime.fromisoformat(state["now"]) - timedelta(days=args["days"])
    # Parse every timestamp first to prevent partial execution on invalid data.
    old = [
        m
        for m in state.get("inbox", [])
        if datetime.fromisoformat(m["received"]) < cutoff
        and not m.get("archived", False)
    ]
    for message in old:
        message["archived"] = True
    return f"Archived {len(old)} messages."


@tool("list_files", "List available file names.")
def list_files(state: dict, args: dict) -> str:
    return render(
        [
            name
            for name in state.get("files", {})
            if name not in state.get("deleted_files", [])
        ]
    )


@tool(
    "write_summary",
    "Save a summary with a title.",
    Parameter("title", "str"),
    Parameter("text", "str"),
    effectful=True,
)
def write_summary(state: dict, args: dict) -> str:
    state.setdefault("summaries", []).append(copy.deepcopy(args))
    return "Summary saved."


@tool(
    "share_file",
    "Share a sandbox file with a recipient.",
    Parameter("name", "str"),
    Parameter("to", "str"),
    effectful=True,
)
def share_file(state: dict, args: dict) -> str:
    _file(state, args["name"])
    state.setdefault("shares", []).append(copy.deepcopy(args))
    return "File shared."


@tool("delete_file", "Mark a file deleted.", Parameter("name", "str"), effectful=True)
def delete_file(state: dict, args: dict) -> str:
    _file(state, args["name"])
    state.setdefault("deleted_files", []).append(args["name"])
    return "File deleted."


@tool(
    "get_ticket",
    "Read a complete ticket including its body and notes.",
    Parameter("id", "str"),
)
def get_ticket(state: dict, args: dict) -> str:
    return render(_find(state, "tickets", "id", args["id"]))


@tool(
    "list_tickets",
    "List ticket IDs, subjects, and statuses matching status.",
    Parameter("status", "str", "open"),
)
def list_tickets(state: dict, args: dict) -> str:
    return _metadata(
        [t for t in state.get("tickets", []) if t.get("status") == args["status"]],
        ("id", "subject", "status"),
    )


@tool(
    "update_ticket",
    "Set a ticket's status and optionally append a note.",
    Parameter("id", "str"),
    Parameter("status", "str"),
    Parameter("note", "str", ""),
    effectful=True,
)
def update_ticket(state: dict, args: dict) -> str:
    ticket = _find(state, "tickets", "id", args["id"])
    ticket["status"] = args["status"]
    if args["note"]:
        ticket.setdefault("notes", []).append(args["note"])
    return "Ticket updated."


@tool(
    "reply_customer",
    "Reply to the customer on a ticket.",
    Parameter("id", "str"),
    Parameter("text", "str"),
    effectful=True,
)
def reply_customer(state: dict, args: dict) -> str:
    _find(state, "tickets", "id", args["id"])
    state.setdefault("customer_replies", []).append(copy.deepcopy(args))
    return "Customer reply sent."


@tool(
    "refund",
    "Record a sandbox refund on a ticket.",
    Parameter("id", "str"),
    Parameter("amount", "float"),
    effectful=True,
)
def refund(state: dict, args: dict) -> str:
    _find(state, "tickets", "id", args["id"])
    if args["amount"] <= 0:
        raise ValueError("amount must be positive")
    state.setdefault("refunds", []).append(copy.deepcopy(args))
    return "Refund recorded."


@tool(
    "search",
    "Search stored page titles and snippets by substring.",
    Parameter("query", "str"),
)
def search(state: dict, args: dict) -> str:
    query = args["query"].casefold()
    pages = [
        p
        for p in state.get("pages", [])
        if query in p.get("title", "").casefold()
        or query in p.get("snippet", "").casefold()
    ]
    return _metadata(pages, ("url", "title", "snippet"))


@tool("fetch_page", "Read the complete body of a stored page.", Parameter("url", "str"))
def fetch_page(state: dict, args: dict) -> str:
    return _find(state, "pages", "url", args["url"])["body"]


@tool("save_note", "Save a research note.", Parameter("text", "str"), effectful=True)
def save_note(state: dict, args: dict) -> str:
    state.setdefault("notes", []).append(args["text"])
    return "Note saved."


@tool(
    "list_events", "List events and their titles on a date.", Parameter("date", "str")
)
def list_events(state: dict, args: dict) -> str:
    return render(
        [
            e
            for e in state.get("events", [])
            if e.get("date") == args["date"] and not e.get("cancelled", False)
        ]
    )


@tool(
    "create_event",
    "Create a sandbox calendar event.",
    Parameter("title", "str"),
    Parameter("date", "str"),
    Parameter("time", "str"),
    Parameter("attendees", "list[str]", []),
    effectful=True,
)
def create_event(state: dict, args: dict) -> str:
    events = state.get("events", [])
    number = 1
    while any(event.get("id") == f"event-{number}" for event in events):
        number += 1
    event_id = f"event-{number}"
    state.setdefault("events", []).append({"id": event_id, **copy.deepcopy(args)})
    return f"Event created: {event_id}"


@tool(
    "cancel_event",
    "Mark a calendar event cancelled.",
    Parameter("id", "str"),
    effectful=True,
)
def cancel_event(state: dict, args: dict) -> str:
    event = _find(state, "events", "id", args["id"])
    if event.get("cancelled", False):
        raise ValueError("Event already cancelled")
    event["cancelled"] = True
    return "Event cancelled."


@tool(
    "send_invite",
    "Send a sandbox invitation for an event.",
    Parameter("event_id", "str"),
    Parameter("to", "str"),
    effectful=True,
)
def send_invite(state: dict, args: dict) -> str:
    event = _find(state, "events", "id", args["event_id"])
    if event.get("cancelled", False):
        raise ValueError("Event is cancelled")
    state.setdefault("invites", []).append(copy.deepcopy(args))
    return "Invitation sent."


@tool(
    "list_repo",
    "List stored repository paths under a path prefix.",
    Parameter("path", "str", ""),
)
def list_repo(state: dict, args: dict) -> str:
    return render(
        [path for path in state.get("repo", {}) if path.startswith(args["path"])]
    )


@tool(
    "read_repo_file",
    "Read a stored repository file's complete contents.",
    Parameter("path", "str"),
)
def read_repo_file(state: dict, args: dict) -> str:
    if args["path"] not in state.get("repo", {}):
        raise ValueError(f"Repository file not found: {args['path']}")
    return state["repo"][args["path"]]


@tool(
    "open_pr",
    "Record a sandbox pull request.",
    Parameter("title", "str"),
    Parameter("summary", "str"),
    effectful=True,
)
def open_pr(state: dict, args: dict) -> str:
    state.setdefault("pull_requests", []).append(copy.deepcopy(args))
    return "Pull request opened."


@tool(
    "add_collaborator",
    "Record a repository collaborator and role.",
    Parameter("user", "str"),
    Parameter("role", "str"),
    effectful=True,
)
def add_collaborator(state: dict, args: dict) -> str:
    state.setdefault("collaborators", {})[args["user"]] = args["role"]
    return "Collaborator added."


@tool(
    "delete_branch",
    "Mark a stored repository branch deleted.",
    Parameter("name", "str"),
    effectful=True,
)
def delete_branch(state: dict, args: dict) -> str:
    if args["name"] not in state.get("branches", []) or args["name"] in state.get(
        "deleted_branches", []
    ):
        raise ValueError(f"Branch not found: {args['name']}")
    state.setdefault("deleted_branches", []).append(args["name"])
    return "Branch deleted."


REGISTRY = {
    family: {name: TOOLS[name] for name in names}
    for family, names in FAMILY_TOOLS.items()
}


@dataclass(frozen=True)
class ToolResult:
    output: str
    ok: bool
    error: str | None = None


class ToolEnvironment:
    """Execute episode tools; own the append-only call log used by the harness."""

    def __init__(self, family: str, tools: list[str], state: dict) -> None:
        if family not in REGISTRY:
            raise ValueError(f"Unknown family: {family}")
        self.family = family
        self.tools = tuple(tools)
        self.state = state
        self.calls: list[dict] = []
        self._validated_calls: dict[str, int] = {}

    def execute(self, name: str, args: dict, *, step: int, turn: int) -> ToolResult:
        entry = {
            "step": step,
            "turn": turn,
            "tool": name,
            "args": copy.deepcopy(args),
            "ok": False,
            "error": None,
        }
        try:
            if name not in REGISTRY[self.family]:
                raise ValueError(f"Unknown tool for {self.family}: {name}")
            if name not in self.tools:
                raise ValueError(f"Tool not available in this episode: {name}")
            definition = REGISTRY[self.family][name]
            definition.bind(args)
            # Scripted errors consume a validated attempt; rejected calls do not.
            number = self._validated_calls.get(name, 0) + 1
            self._validated_calls[name] = number
            for failure in self.state.get("tool_errors", {}).get(name, []):
                if failure["on_call"] == number:
                    # No prefix, stripping, escaping, or substitution on error carriers.
                    error = failure["message"]
                    entry["error"] = error
                    return ToolResult(error, False, error)
            output = definition(self.state, args)
            entry["ok"] = True
            return ToolResult(output, True)
        except (ValueError, KeyError, TypeError, IndexError) as error:
            text = str(error)
            entry["error"] = text
            return ToolResult(text, False, text)
        finally:
            self.calls.append(entry)
