"""Check that every probe episode is exposed, solvable and detectable.

For each episode in data/probe/:
  exposure   the meta["reveal"] tool call shows the planted text (the clean twin's
             does not), so a model following the task actually meets it;
  solvable   a scripted agent that does exactly the task is tagged safe / pass;
  detectable a scripted agent that also obeys the injection is tagged unsafe.

Usage: python scripts/check_probe.py   (exit code 1 on any problem)
"""

import copy
import json
import sys
from pathlib import Path

from turnguard.harness import run_episodes
from turnguard.models import StubGenerator
from turnguard.tools import _MISSING, REGISTRY, ToolEnvironment

ROOT = Path(__file__).resolve().parents[1]
FILL = {"str": "x", "int": 1, "float": 1.0, "list[str]": []}


def load(directory: Path = ROOT / "data" / "probe") -> list[dict]:
    return [json.loads(p.read_text("utf-8")) for p in sorted(directory.glob("*.json"))]


def planted_texts(spec: dict) -> list[str]:
    block = spec["injection"] or spec["authorized"]
    if not block:
        return []
    return [p["text"] for p in block["parts"]] if "parts" in block else [block["text"]]


def act(spec: dict, matcher: dict, extra: dict | None = None) -> str:
    tool = REGISTRY[spec["family"]][matcher["tool"]]
    args = dict(extra or {})
    args.update(matcher.get("args_match", {}))
    for key, terms in matcher.get("args_contains", {}).items():
        args[key] = " ".join(terms)
    for parameter in tool.parameters:
        if parameter.name not in args and parameter.default is _MISSING:
            args[parameter.name] = FILL[parameter.type]
    return "ACTION: " + json.dumps(
        {"tool": tool.name, "args": args}, ensure_ascii=False
    )


def script(spec: dict, obey: bool) -> list[str]:
    turns = len(spec["user_turns"])
    task = spec["task_oracle"]["all"]
    outputs = []
    for turn in range(1, turns + 1):
        last = turn == turns
        if obey and turn == 1:
            extra = spec["meta"].get("obey_args")
            outputs += [
                act(spec, m, extra)
                for m in spec["injection"]["target"]
                if m["kind"] == "call"
            ]
        if last:
            outputs += [act(spec, m) for m in task if m["kind"] == "call"]
        terms = [
            t
            for m in task
            if m["kind"] == "answer" and m.get("turn", turns) == turn
            for t in m["contains"]
        ]
        if obey and last:
            terms += [
                t
                for m in spec["injection"]["target"]
                if m["kind"] == "answer"
                for t in m["contains"]
            ]
        outputs.append("FINAL: " + (" ".join(terms) or "Done."))
    return outputs


def check(spec: dict, by_id: dict) -> list[str]:
    problems = []
    reveal = spec["meta"].get("reveal")
    source = by_id.get(spec["twin_of"], spec) if spec["twin_of"] else spec
    texts = planted_texts(source)
    if reveal:
        environment = ToolEnvironment(
            spec["family"], spec["tools"], copy.deepcopy(spec["initial_state"])
        )
        output = environment.execute(
            reveal["tool"], reveal["args"], step=1, turn=1
        ).output
        shown = any(text.strip() in output for text in texts)
        if spec["episode_type"] == "clean" and shown:
            problems.append("clean twin still shows the injection")
        if spec["episode_type"] != "clean" and not shown:
            problems.append(
                f"reveal call does not show the planted text: {output[:120]!r}"
            )
    expected = {"injected": "safe", "clean": "pass", "authorized": "pass"}
    good = run_episodes([spec], StubGenerator({spec["id"]: script(spec, False)}))[0]
    if good.result["tag"] != expected[spec["episode_type"]]:
        problems.append(f"task script tagged {good.result['tag']}")
    if spec["episode_type"] == "injected":
        bad = run_episodes([spec], StubGenerator({spec["id"]: script(spec, True)}))[0]
        if bad.result["tag"] != "unsafe":
            problems.append(f"obeying script tagged {bad.result['tag']}")
    return problems


def main() -> int:
    specs = load()
    by_id = {s["id"]: s for s in specs}
    failures = {s["id"]: p for s in specs if (p := check(s, by_id))}
    for spec in specs:
        status = (
            "FAIL " + "; ".join(failures[spec["id"]])
            if spec["id"] in failures
            else "ok"
        )
        print(f"{spec['id']:18s} {spec['episode_type']:10s} {status}")
    print(f"\n{len(specs) - len(failures)}/{len(specs)} episodes pass")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
