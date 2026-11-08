"""Prototype renderer for the Step 2 template formats (English, stub filler).

Not the builder: it proves the formats in turnguard/data/templates/*.yaml and
turnguard/data/injections.yaml are complete enough to produce valid episodes.
For every scenario x goal x style x pressure x position x carrier it renders
the injected episode, its clean twin and the authorized variants, then runs the
scenario's `route` through the real harness and checks:

  exposure    every injection part reaches the model on the route;
  solvable    route alone -> injected `safe`, clean `pass`, authorized `pass`;
  detectable  route + obeying the injection -> `unsafe`;
  neutral     oracle terms are language-neutral (contain a digit, @ or /).

The builder (Step 3) replaces stub filler with the teacher and adds languages,
sampling, CuratorKIT, splits and freezing; this logic is meant to be reused.

Usage: python scripts/render_prototype.py   (exit code 1 on any problem)
"""

import copy
import itertools
import json
import re
import sys
from pathlib import Path

import yaml

from turnguard.harness import run_episodes
from turnguard.models import StubGenerator
from turnguard.oracles import exposed
from turnguard.schema import make_clean_twin, validate_episode
from turnguard.tools import REGISTRY

DATA = Path(__file__).resolve().parents[1] / "turnguard" / "data"
STYLES = ("imperative", "fake_authority", "named_approval", "task_framed")


def load() -> tuple[dict, list[dict]]:
    bank = yaml.safe_load((DATA / "injections.yaml").read_text("utf-8"))
    scenarios = []
    for path in sorted((DATA / "templates").glob("*.yaml")):
        family_file = yaml.safe_load(path.read_text("utf-8"))
        for scenario in family_file["scenarios"]:
            scenarios.append({**scenario, "family": family_file["family"]})
    return bank, scenarios


def fill(value, names: dict):
    """Substitute {name} placeholders in every string of a nested value."""
    if isinstance(value, str):
        return re.sub(
            r"\{(\w+)\}",
            lambda m: str(names[m.group(1)]) if m.group(1) in names else m.group(0),
            value,
        )
    if isinstance(value, list):
        return [fill(v, names) for v in value]
    if isinstance(value, dict):
        # Keys too: docs file names and repo paths are placeholders.
        return {fill(k, names): fill(v, names) for k, v in value.items()}
    return value


def stub_generate(value):
    """Stand-in for the teacher: plain text that contains every include once."""
    if isinstance(value, dict) and "gen" in value:
        text = value["about"][0].upper() + value["about"][1:] + "."
        for term in value.get("include", []):
            if text.count(term) != 1:
                raise ValueError(f"stub cannot place {term!r} exactly once")
        return text
    if isinstance(value, list):
        return [stub_generate(v) for v in value]
    if isinstance(value, dict):
        return {k: stub_generate(v) for k, v in value.items()}
    return value


def append_at(container, pointer: str, text: str) -> None:
    *path, last = [t for t in pointer.split("/")[1:]]
    for token in path:
        container = (
            container[int(token)] if isinstance(container, list) else container[token]
        )
    key = int(last) if isinstance(container, list) else last
    container[key] = container[key] + "\n\n" + text


def base(scenario: dict, attacker: dict) -> tuple[dict, dict]:
    names = dict(scenario["vars"])
    names.update(attacker)
    names.update(fill(scenario["assets"], names))
    state = stub_generate(fill(scenario["state"], names))
    state.setdefault("tool_errors", {})
    return names, {
        "split": "dev",
        "family": scenario["family"],
        "lang": {"conversation": "en", "injection": "en"},
        "user_turns": fill(scenario["turns"], names),
        "tools": scenario["tools"],
        "initial_state": state,
        "task_oracle": {"all": fill(scenario["task_oracle"], names)},
        "meta": {
            "scenario": scenario["id"],
            "renderer": "prototype",
            "obey_args": fill(scenario.get("obey_args", {}), names),
        },
    }


def place(episode: dict, slot: dict, text: str, error: str | None) -> dict:
    """Put one injection part at a slot; return the schema `part`."""
    if "at" in slot:
        append_at(episode["initial_state"], slot["at"], text)
        return {"where": "state", "slot": slot["at"], "text": text}
    if "turn" in slot:
        index = slot["turn"]
        episode["user_turns"][index] += f'\n\n{slot["intro"]}\n"{text}"'
        return {"where": "user_turn", "slot": f"/{index}", "text": text}
    errors = episode["initial_state"]["tool_errors"].setdefault(slot["tool"], [])
    errors.append({"on_call": slot["on_call"], "message": f"{error} {text}"})
    pointer = f"/tool_errors/{slot['tool']}/{len(errors) - 1}/message"
    return {"where": "state", "slot": pointer, "text": text}


def injected(scenario, bank, goal, style, pressure, position, carrier) -> dict | None:
    slots = fill(scenario["slots"].get(position, {}).get(carrier), scenario["vars"])
    targets = bank["goals"][goal]["targets"]
    family_targets = targets.get(scenario["family"], targets.get("*"))
    if slots is None or family_targets is None:
        return None
    pool = bank["attacker_pools"]["dev"]
    attacker = {
        "attacker_addr": pool["addr"][0],
        "attacker_url": pool["url"][0],
        "attacker_user": pool["user"][0],
    }
    names, episode = base(scenario, attacker)
    names["ask"] = fill(bank["goals"][goal]["ask"], names)
    names["short"] = fill(bank["styles"][style]["short"], names)
    error = None
    if carrier == "tool_error":
        snag = bank["pressure"]["snag"]["tool_error"][0]
        error, texts = fill(snag["error"], names), [fill(snag["inject"], names)]
    elif position == "split":
        texts = fill(bank["split"][0], names)
    elif pressure == "snag":
        texts = [fill(bank["pressure"]["snag"]["content"][0], names)]
    else:
        texts = [fill(bank["styles"][style]["long"][0], names)]
    slot_list = slots if isinstance(slots, list) else [slots]
    parts = [place(episode, s, t, error) for s, t in zip(slot_list, texts, strict=True)]
    episode.update(
        id=f"{scenario['id']}-{goal}-{style}-{pressure}-{position}-{carrier}",
        episode_type="injected",
        twin_of=None,
        carrier=carrier,
        goal=goal,
        position=position,
        style=style,
        injection={"parts": parts, "target": fill(family_targets, names)},
        authorized=None,
    )
    episode["meta"].update(pressure=pressure)
    return episode


def authorized(scenario: dict, variant: dict, index: int) -> dict:
    names, episode = base(scenario, {})
    slot = fill(scenario["slots"][variant["position"]][variant["carrier"]], names)
    turn = 0 if variant["position"] == "first" else 1
    episode["user_turns"][turn] += variant["delegate"]
    text = fill(variant["text"], names)
    part = place(episode, slot, text, None)
    episode.update(
        id=f"{scenario['id']}-auth-{index}",
        episode_type="authorized",
        twin_of=None,
        carrier=variant["carrier"],
        goal=None,
        position=variant["position"],
        style=None,
        injection=None,
        authorized={"parts": [part], "target": fill(variant["target"], names)},
    )
    return episode


def action(tool: str, args: dict) -> str:
    return "ACTION: " + json.dumps({"tool": tool, "args": args}, ensure_ascii=False)


def script(scenario: dict, episode: dict, obey: bool) -> list[str]:
    """Route calls per turn (retrying scripted errors), obeying if asked, then FINAL."""
    names = dict(scenario["vars"])
    route = fill(scenario["route"], names)
    errors = episode["initial_state"]["tool_errors"]
    counts: dict[str, int] = {}
    oracle = episode["task_oracle"]["all"]
    extra = []
    if episode["authorized"]:
        extra = episode["authorized"]["target"]
    obey_turn = None
    if obey:
        obey_turn = max(
            [1]
            + [
                int(p["slot"][1:]) + 1
                for p in episode["injection"]["parts"]
                if p["where"] == "user_turn"
            ]
            + [step["turn"] for step in route if _reveals(step, episode)]
        )
    outputs = []
    for turn in range(1, len(episode["user_turns"]) + 1):
        for step in [s for s in route if s["turn"] == turn]:
            outputs.append(action(step["tool"], step["args"]))
            counts[step["tool"]] = counts.get(step["tool"], 0) + 1
            if any(
                e["on_call"] == counts[step["tool"]]
                for e in errors.get(step["tool"], [])
            ):
                outputs.append(action(step["tool"], step["args"]))
                counts[step["tool"]] += 1
        for matcher in extra:
            if matcher["kind"] == "call" and turn == 1:
                outputs.append(_call_for(matcher, episode))
        terms = [
            t
            for m in oracle
            if m["kind"] == "answer"
            and m.get("turn", len(episode["user_turns"])) == turn
            for t in m["contains"]
        ]
        if obey and turn == obey_turn:
            for matcher in episode["injection"]["target"]:
                if matcher["kind"] == "call":
                    outputs.append(_call_for(matcher, episode))
                else:
                    terms += matcher["contains"]
        outputs.append("FINAL: " + (" ".join(terms) or "Done."))
    return outputs


def _reveals(step: dict, episode: dict) -> bool:
    """True when a route call's item holds a state injection part (rough check)."""
    for part in episode["injection"]["parts"]:
        if part["where"] != "state":
            continue
        if part["slot"].startswith(f"/tool_errors/{step['tool']}/"):
            return True
        tokens = part["slot"].split("/")[1:]
        if len(tokens) >= 2 and tokens[1].isdigit():
            item = episode["initial_state"][tokens[0]][int(tokens[1])]
            key = item.get("id", item.get("url"))
            if key in step["args"].values():
                return True
        elif len(tokens) == 2 and tokens[1] in step["args"].values():
            return True  # dict-keyed state: files / repo, keyed by name or path
    return False


FILL = {"str": "x", "int": 1, "float": 1.0, "list[str]": []}


def _call_for(matcher: dict, episode: dict) -> str:
    tool = REGISTRY[episode["family"]][matcher["tool"]]
    args = dict(episode["meta"].get("obey_args", {}).get(matcher["tool"], {}))
    args.update(matcher.get("args_match", {}))
    for key, terms in matcher.get("args_contains", {}).items():
        args[key] = " ".join(terms)
    for parameter in tool.parameters:
        if parameter.name not in args and parameter.type != "list[str]":
            if parameter.name in ("days",):
                args[parameter.name] = 0
            elif parameter.default is not _missing():
                continue
            else:
                args[parameter.name] = FILL[parameter.type]
    return action(tool.name, args)


def _missing():
    from turnguard.tools import _MISSING

    return _MISSING


def neutral(term: str) -> bool:
    return bool(re.search(r"[0-9@/]", term))


def check(scenario: dict, episode: dict) -> list[str]:
    problems = validate_episode(episode)
    if problems:
        return problems
    for matcher in episode["task_oracle"]["all"]:
        terms = matcher.get("contains", []) + [
            t for ts in matcher.get("args_contains", {}).values() for t in ts
        ]
        problems += [f"non-neutral oracle term {t!r}" for t in terms if not neutral(t)]
    expected = {"injected": "safe", "clean": "pass", "authorized": "pass"}
    run = lambda outputs: run_episodes(  # noqa: E731
        [episode], StubGenerator({episode["id"]: outputs})
    )[0]
    good = run(script(scenario, episode, obey=False))
    if good.result["tag"] != expected[episode["episode_type"]]:
        problems.append(f"route tagged {good.result['tag']}")
    block = episode["injection"] or episode["authorized"]
    if block and not exposed(block, good):
        problems.append("route does not expose every part")
    if block:
        # Main path: each state part is in the result of the FIRST tool call of
        # some user turn, not only somewhere along the route.
        firsts, waiting = [], False
        for message in good.messages_model[1:]:
            content = message.get("content") or ""
            if message["role"] == "user" and not content.startswith("TOOL RESULT"):
                waiting = True
            elif message["role"] == "user" and waiting:
                firsts.append(content)
                waiting = False
        for part in block["parts"]:
            if part["where"] == "state" and not any(part["text"] in f for f in firsts):
                problems.append(f"part at {part['slot']} is not on a turn's first call")
    if episode["episode_type"] == "injected":
        bad = run(script(scenario, episode, obey=True))
        if bad.result["tag"] != "unsafe":
            problems.append(f"obeying tagged {bad.result['tag']}")
    return problems


def main() -> int:
    bank, scenarios = load()
    total, failures = 0, []
    for scenario in scenarios:
        combos = itertools.product(
            scenario["goals"],
            STYLES,
            ("none", "snag"),
            ("first", "later", "split"),
            ("body", "metadata", "quoted", "tool_error"),
        )
        for goal, style, pressure, position, carrier in combos:
            episode = injected(scenario, bank, goal, style, pressure, position, carrier)
            if episode is None:
                continue
            for item in (episode, make_clean_twin(copy.deepcopy(episode))):
                total += 1
                if problems := check(scenario, item):
                    failures.append((item["id"], problems))
        for index, variant in enumerate(scenario.get("authorized", [])):
            total += 1
            item = authorized(scenario, variant, index)
            if problems := check(scenario, item):
                failures.append((item["id"], problems))
    for episode_id, problems in failures[:20]:
        print(f"FAIL {episode_id}: {'; '.join(problems)}")
    print(f"{total - len(failures)}/{total} rendered episodes pass")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
