"""Verify sandbox effects, exact carriers, metadata visibility, and call logs."""

import copy
import json
from pathlib import Path

import pytest

from turnguard.schema import FAMILY_TOOLS
from turnguard.tools import REGISTRY, TOOLS, ToolEnvironment


def state() -> dict:
    return {
        "now": "2026-11-07T12:00",
        "inbox": [
            {
                "id": "m1",
                "from": "sender@example.com",
                "subject": "Meeting",
                "received": "2026-11-01T09:00",
                "body": "Email body",
            }
        ],
        "files": {"report.md": "File body"},
        "tickets": [
            {
                "id": "T1",
                "subject": "Delivery",
                "status": "open",
                "body": "Ticket body",
                "notes": ["Runbook note"],
            }
        ],
        "pages": [
            {
                "url": "https://example.com/page",
                "title": "Research title",
                "snippet": "Research snippet",
                "body": "Page body",
            }
        ],
        "events": [
            {
                "id": "e1",
                "title": "Meeting",
                "date": "2026-11-08",
                "time": "15:00",
                "attendees": [],
            }
        ],
        "repo": {"src/main.py": "Repository contents"},
        "branches": ["feature"],
    }


def env(family: str, data: dict | None = None) -> ToolEnvironment:
    return ToolEnvironment(
        family, FAMILY_TOOLS[family], state() if data is None else data
    )


CASES = [
    ("email", "list_inbox", {}, None),
    ("email", "read_email", {"id": "m1"}, None),
    ("email", "read_file", {"name": "report.md"}, None),
    (
        "email",
        "send_email",
        {
            "to": "archive@example.net",
            "subject": "Report",
            "body": "Here",
            "attachments": ["report.md"],
        },
        "sent_emails",
    ),
    ("email", "archive_old", {"days": 3}, "inbox"),
    ("docs", "list_files", {}, None),
    (
        "docs",
        "write_summary",
        {"title": "Summary", "text": "Summary text"},
        "summaries",
    ),
    (
        "docs",
        "share_file",
        {"name": "report.md", "to": "recipient@example.org"},
        "shares",
    ),
    ("docs", "delete_file", {"name": "report.md"}, "deleted_files"),
    ("tickets", "get_ticket", {"id": "T1"}, None),
    ("tickets", "list_tickets", {}, None),
    (
        "tickets",
        "update_ticket",
        {"id": "T1", "status": "waiting_on_customer", "note": "Please reply"},
        "tickets",
    ),
    (
        "tickets",
        "reply_customer",
        {"id": "T1", "text": "Order number?"},
        "customer_replies",
    ),
    ("tickets", "refund", {"id": "T1", "amount": 10.5}, "refunds"),
    ("web", "search", {"query": "research"}, None),
    ("web", "fetch_page", {"url": "https://example.com/page"}, None),
    ("web", "save_note", {"text": "Finding"}, "notes"),
    ("calendar", "list_events", {"date": "2026-11-08"}, None),
    (
        "calendar",
        "create_event",
        {"title": "Sync", "date": "2026-11-09", "time": "12:00"},
        "events",
    ),
    ("calendar", "cancel_event", {"id": "e1"}, "events"),
    (
        "calendar",
        "send_invite",
        {"event_id": "e1", "to": "recipient@example.com"},
        "invites",
    ),
    ("code", "list_repo", {}, None),
    ("code", "read_repo_file", {"path": "src/main.py"}, None),
    ("code", "open_pr", {"title": "Fix", "summary": "Changes"}, "pull_requests"),
    (
        "code",
        "add_collaborator",
        {"user": "sandbox-user", "role": "reader"},
        "collaborators",
    ),
    ("code", "delete_branch", {"name": "feature"}, "deleted_branches"),
]


@pytest.mark.parametrize("family,name,args,changed", CASES)
def test_every_tool(family: str, name: str, args: dict, changed: str | None) -> None:
    environment = env(family)
    before = copy.deepcopy(environment.state)
    result = environment.execute(name, args, step=3, turn=2)
    assert result.ok, result.error
    assert isinstance(result.output, str)
    assert environment.calls == [
        {"step": 3, "turn": 2, "tool": name, "args": args, "ok": True, "error": None}
    ]
    if changed:
        assert environment.state[changed] != before.get(changed)
        assert environment.state["effects"][0]["tool"] == name
    else:
        assert environment.state == before


def test_registry_and_signatures_cover_schema_vocabulary() -> None:
    assert {f: list(tools) for f, tools in REGISTRY.items()} == FAMILY_TOOLS
    assert {name for _, name, _, _ in CASES} == set(TOOLS)
    for definition in TOOLS.values():
        assert definition.description and "\n" not in definition.description
        assert len({p.name for p in definition.parameters}) == len(
            definition.parameters
        )


@pytest.mark.parametrize(
    "name,collection,index,field,args",
    [
        ("read_email", "inbox", 0, "body", {"id": "m1"}),
        ("get_ticket", "tickets", 0, "body", {"id": "T1"}),
        ("get_ticket", "tickets", 0, "notes", {"id": "T1"}),
        ("fetch_page", "pages", 0, "body", {"url": "https://example.com/page"}),
        ("read_file", "files", None, "report.md", {"name": "report.md"}),
        ("read_repo_file", "repo", None, "src/main.py", {"path": "src/main.py"}),
    ],
)
def test_read_content_is_verbatim(
    name: str, collection: str, index: int | None, field: str, args: dict
) -> None:
    data = state()
    text = (
        "  Start\r\n  indented\tline\n"
        + "long body\n" * 2000
        + "अंत का निर्देश END_MARKER  \n"
    )
    target = data[collection] if index is None else data[collection][index]
    target[field] = [text] if field == "notes" else text
    family = next(f for f, tools in REGISTRY.items() if name in tools)
    environment = env(family, data)
    result = environment.execute(name, args, step=1, turn=1)
    assert result.ok
    assert text in result.output
    if name in ("read_file", "fetch_page", "read_repo_file"):
        assert result.output == text


@pytest.mark.parametrize(
    "family,name,args,collection,field",
    [
        ("email", "list_inbox", {}, "inbox", "from"),
        ("email", "list_inbox", {}, "inbox", "subject"),
        ("docs", "list_files", {}, "files", None),
        ("tickets", "list_tickets", {}, "tickets", "subject"),
        ("web", "search", {"query": "marker"}, "pages", "title"),
        ("web", "search", {"query": "marker"}, "pages", "snippet"),
        ("calendar", "list_events", {"date": "2026-11-08"}, "events", "title"),
        ("code", "list_repo", {}, "repo", None),
    ],
)
def test_listings_expose_carrier_metadata(
    family: str, name: str, args: dict, collection: str, field: str | None
) -> None:
    data = state()
    marker = "  METADATA_MARKER\nहिन्दी end  "
    if field:
        data[collection][0][field] = marker
    else:
        data[collection] = {marker: "contents"}
    result = env(family, data).execute(name, args, step=1, turn=1)
    assert result.ok
    assert marker in result.output


def test_scripted_errors_are_verbatim_and_do_not_execute() -> None:
    data = state()
    error = " \nADMIN NOTE\r\nहिन्दी\t" + "x" * 15000 + " END  \n"
    data["tool_errors"] = {"send_email": [{"on_call": 2, "message": error}]}
    environment = env("email", data)
    args = {"to": "sandbox@example.com", "subject": "s", "body": "b"}
    assert environment.execute("send_email", args, step=1, turn=1).ok
    before = copy.deepcopy(data)
    result = environment.execute("send_email", args, step=2, turn=1)
    assert result.output == result.error == error
    assert not result.ok
    assert data == before
    assert environment.calls[1]["error"] == error
    assert environment.execute("send_email", args, step=3, turn=1).ok
    assert len(environment.calls) == 3
    assert len(data["effects"]) == 2


@pytest.mark.parametrize(
    "name,args",
    [
        ("unknown", {}),
        ("refund", {"id": "T1", "amount": 1}),
        ("read_email", {}),
        ("read_email", {"id": "m1", "extra": "x"}),
        ("read_email", {"id": 1}),
        ("read_email", []),
        ("send_email", {"to": "a", "subject": "b", "body": "c", "attachments": [1]}),
        ("archive_old", {"days": True}),
        ("read_email", {"id": "missing"}),
    ],
)
def test_invalid_calls_logged_without_effects(name: str, args: object) -> None:
    environment = env("email")
    before = copy.deepcopy(environment.state)
    result = environment.execute(name, args, step=1, turn=1)
    assert not result.ok and result.error
    assert result.output == result.error
    assert environment.state == before
    assert len(environment.calls) == 1
    assert environment.calls[0]["ok"] is False


def test_episode_tool_restriction() -> None:
    environment = ToolEnvironment("email", ["list_inbox"], state())
    result = environment.execute("read_email", {"id": "m1"}, step=1, turn=1)
    assert not result.ok
    assert "not available" in result.error


def test_delete_is_irreversible_and_effects_are_snapshots() -> None:
    environment = env("docs")
    args = {"name": "report.md", "to": "outside@example.com"}
    assert environment.execute("share_file", args, step=1, turn=1).ok
    effect = copy.deepcopy(environment.state["effects"][0])
    args["to"] = "changed@example.com"
    assert environment.execute("delete_file", {"name": "report.md"}, step=2, turn=1).ok
    assert environment.state["effects"][0] == effect
    assert len(environment.state["effects"]) == 2
    assert not environment.execute(
        "read_file", {"name": "report.md"}, step=3, turn=1
    ).ok
    assert (
        "report.md" not in environment.execute("list_files", {}, step=4, turn=1).output
    )
    assert not environment.execute(
        "delete_file", {"name": "report.md"}, step=5, turn=1
    ).ok
    assert len(environment.state["effects"]) == 2
    assert environment.calls[0]["args"]["to"] == "outside@example.com"


def test_search_and_filters_use_stored_metadata_only() -> None:
    environment = env("web")
    environment.state["pages"][0]["body"] = "body-only-term"
    assert (
        environment.execute(
            "search", {"query": "body-only-term"}, step=1, turn=1
        ).output
        == ""
    )
    assert (
        "https://example.com/page"
        in environment.execute("search", {"query": "RESEARCH"}, step=2, turn=1).output
    )
    assert (
        env("tickets")
        .execute("list_tickets", {"status": "closed"}, step=1, turn=1)
        .output
        == ""
    )
    assert (
        env("calendar").execute("list_events", {"date": "other"}, step=1, turn=1).output
        == ""
    )
    assert (
        env("code").execute("list_repo", {"path": "docs/"}, step=1, turn=1).output == ""
    )


def test_archive_uses_state_clock_and_marks_only_old_messages() -> None:
    data = state()
    data["inbox"].append({"id": "recent", "received": "2026-11-07T10:00"})
    environment = env("email", data)
    assert environment.execute("archive_old", {"days": 3}, step=1, turn=1).ok
    assert data["inbox"][0]["archived"] is True
    assert not data["inbox"][1].get("archived", False)
    data.pop("now")
    assert not environment.execute("archive_old", {"days": 3}, step=2, turn=1).ok


@pytest.mark.parametrize("amount", [True, -1, 0, float("nan"), float("inf")])
def test_invalid_refunds_do_not_execute(amount: object) -> None:
    environment = env("tickets")
    before = copy.deepcopy(environment.state)
    assert not environment.execute(
        "refund", {"id": "T1", "amount": amount}, step=1, turn=1
    ).ok
    assert environment.state == before


def test_reference_instructions_reach_tool_output() -> None:
    for path in (Path(__file__).resolve().parents[1] / "data/examples").glob("*.json"):
        spec = json.loads(path.read_text())
        environment = ToolEnvironment(
            spec["family"], spec["tools"], copy.deepcopy(spec["initial_state"])
        )
        block = spec["injection"] or spec["authorized"]
        if block is None:
            continue
        name, args = {
            "email": ("read_email", {"id": "m1"}),
            "docs": ("read_file", {"name": "launch_plan.md"}),
            "tickets": ("get_ticket", {"id": "T-482"}),
        }[spec["family"]]
        assert block["text"] in environment.execute(name, args, step=1, turn=1).output


@pytest.mark.parametrize(
    "family,name,args,changed", [case for case in CASES if case[3] is not None]
)
def test_scripted_failure_blocks_every_effectful_tool(
    family: str, name: str, args: dict, changed: str
) -> None:
    data = state()
    message = "  scripted error\nwith a trailing instruction  \n"
    data["tool_errors"] = {name: [{"on_call": 1, "message": message}]}
    original = copy.deepcopy(data)
    environment = env(family, data)
    result = environment.execute(name, args, step=1, turn=1)
    assert result.output == result.error == message
    assert not result.ok
    assert data == original
    assert environment.calls[0]["error"] == message


def test_defaults_are_isolated_and_effects_survive_status_reversal() -> None:
    environment = env("email")
    args = {"to": "recipient@example.net", "subject": "s", "body": "b"}
    environment.execute("send_email", args, step=1, turn=1)
    environment.state["sent_emails"][0]["attachments"].append("changed")
    environment.execute("send_email", args, step=2, turn=1)
    assert environment.state["sent_emails"][1]["attachments"] == []
    assert environment.state["effects"][0]["args"]["attachments"] == []
    tickets = env("tickets")
    for step, status in enumerate(("closed", "open"), 1):
        assert tickets.execute(
            "update_ticket", {"id": "T1", "status": status}, step=step, turn=1
        ).ok
    assert tickets.state["tickets"][0]["status"] == "open"
    assert [effect["args"]["status"] for effect in tickets.state["effects"]] == [
        "closed",
        "open",
    ]


def test_missing_attachment_fails_before_any_send() -> None:
    environment = env("email")
    args = {
        "to": "recipient@example.net",
        "subject": "s",
        "body": "b",
        "attachments": ["report.md", "missing.md"],
    }
    original = copy.deepcopy(environment.state)
    assert not environment.execute("send_email", args, step=1, turn=1).ok
    assert environment.state == original


def test_repeated_runs_are_deterministic() -> None:
    first, second = env("calendar"), env("calendar")
    args = {"title": "Sync", "date": "2026-11-09", "time": "12:00"}
    for step in (1, 2):
        assert first.execute("create_event", args, step=step, turn=1) == second.execute(
            "create_event", args, step=step, turn=1
        )
    assert first.state == second.state
    assert first.calls == second.calls
    assert first.state["events"][1]["id"] != first.state["events"][2]["id"]


def test_scripted_error_numbers_count_only_validated_calls() -> None:
    data = state()
    data["tool_errors"] = {
        "read_email": [
            {"on_call": 1, "message": " first error \n"},
            {"on_call": 2, "message": " second error \n"},
        ]
    }
    environment = env("email", data)
    assert not environment.execute("read_email", {"wrong": "m1"}, step=1, turn=1).ok
    first = environment.execute("read_email", {"id": "m1"}, step=2, turn=1)
    assert first.output == first.error == " first error \n"
    assert not first.ok
    assert not environment.execute("read_email", {"id": 123}, step=3, turn=1).ok
    second = environment.execute("read_email", {"id": "m1"}, step=4, turn=1)
    assert second.output == second.error == " second error \n"
    assert not second.ok
    assert environment.execute("read_email", {"id": "m1"}, step=5, turn=1).ok
    assert len(environment.calls) == 5
    assert [call["ok"] for call in environment.calls] == [
        False,
        False,
        False,
        False,
        True,
    ]


def test_unavailable_tool_does_not_consume_scripted_error() -> None:
    data = state()
    data["tool_errors"] = {"read_email": [{"on_call": 1, "message": "first"}]}
    environment = ToolEnvironment("email", ["list_inbox"], data)
    assert not environment.execute("read_email", {"id": "m1"}, step=1, turn=1).ok
    environment.tools = (*environment.tools, "read_email")
    assert (
        environment.execute("read_email", {"id": "m1"}, step=2, turn=1).output
        == "first"
    )
