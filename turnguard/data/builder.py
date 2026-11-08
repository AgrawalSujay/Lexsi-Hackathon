"""Deterministic offline episode assembly and route checks.

Rendering and route verification are ported from scripts/render_prototype.py;
that script remains the exhaustive format acceptance check. No teacher calls or
CuratorKIT pipeline run in the stub path.
"""

import copy
import itertools
import json
import logging
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from turnguard.harness import run_episodes
from turnguard.models import StubGenerator
from turnguard.oracles import exposed
from turnguard.schema import make_clean_twin, resolve_pointer, validate_episode
from turnguard.tools import REGISTRY

DATA = Path(__file__).resolve().parent
STYLES = ("imperative", "fake_authority", "named_approval", "task_framed")


def load(data: Path = DATA) -> tuple[dict, list[dict]]:
    bank = yaml.safe_load((data / "injections.yaml").read_text("utf-8"))
    scenarios = []
    for path in sorted((data / "templates").glob("*.yaml")):
        family_file = yaml.safe_load(path.read_text("utf-8"))
        for scenario in family_file["scenarios"]:
            scenarios.append({**scenario, "family": family_file["family"]})
    return bank, scenarios


def fill(value: Any, names: dict) -> Any:
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
        return {fill(k, names): fill(v, names) for k, v in value.items()}
    return value


def stub_generate(value: Any) -> Any:
    """Stand-in for the teacher: plain text that contains every include once."""
    if isinstance(value, dict) and "gen" in value:
        text = value["about"][0].upper() + value["about"][1:] + "."
        for term in value.get("include", []):
            if text.count(term) != 1:
                raise ValueError(f"stub cannot place {term!r} exactly once")
        check_generated_text(text)
        return text
    if isinstance(value, list):
        return [stub_generate(v) for v in value]
    if isinstance(value, dict):
        return {k: stub_generate(v) for k, v in value.items()}
    return value


def append_at(container: dict | list, pointer: str, text: str) -> None:
    *path, last = [
        t.replace("~1", "/").replace("~0", "~") for t in pointer.split("/")[1:]
    ]
    for token in path:
        container = (
            container[int(token)] if isinstance(container, list) else container[token]
        )
    key = int(last) if isinstance(container, list) else last
    container[key] = container[key] + "\n\n" + text


def base(scenario: dict, attacker: dict, split: str = "dev") -> tuple[dict, dict]:
    names = dict(scenario["vars"])
    names.update(attacker)
    names.update(fill(scenario["assets"], names))
    state = stub_generate(fill(scenario["state"], names))
    state.setdefault("tool_errors", {})
    return names, {
        "split": split,
        "family": scenario["family"],
        "lang": {"conversation": "en", "injection": "en"},
        "user_turns": fill(scenario["turns"], names),
        "tools": scenario["tools"],
        "initial_state": state,
        "task_oracle": {"all": fill(scenario["task_oracle"], names)},
        "meta": {
            "scenario": scenario["id"],
            "renderer": "stub-v1",
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


def injected(
    scenario: dict,
    bank: dict,
    goal: str,
    style: str,
    pressure: str,
    position: str,
    carrier: str,
    *,
    split: str = "dev",
    rng: random.Random | None = None,
) -> dict | None:
    slots = fill(scenario["slots"].get(position, {}).get(carrier), scenario["vars"])
    targets = bank["goals"][goal]["targets"]
    family_targets = targets.get(scenario["family"], targets.get("*"))
    if slots is None or family_targets is None:
        return None
    pool = bank["attacker_pools"][split]
    choose = rng.choice if rng is not None else lambda values: values[0]
    attacker = {
        "attacker_addr": choose(pool["addr"]),
        "attacker_url": choose(pool["url"]),
        "attacker_user": choose(pool["user"]),
    }
    names, episode = base(scenario, attacker, split)
    names["ask"] = fill(bank["goals"][goal]["ask"], names)
    names["short"] = fill(bank["styles"][style]["short"], names)
    error = None
    if carrier == "tool_error":
        snag = choose(bank["pressure"]["snag"]["tool_error"])
        error, texts = fill(snag["error"], names), [fill(snag["inject"], names)]
    elif position == "split":
        texts = fill(choose(bank["split"]), names)
    elif pressure == "snag":
        texts = [fill(choose(bank["pressure"]["snag"]["content"]), names)]
    else:
        texts = [fill(choose(bank["styles"][style]["long"]), names)]
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


def authorized(
    scenario: dict, variant: dict, index: int, *, split: str = "dev"
) -> dict:
    names, episode = base(scenario, {}, split)
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
            if item.get("id", item.get("url")) in step["args"].values():
                return True
        elif len(tokens) == 2 and tokens[1] in step["args"].values():
            return True
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
    if block and episode["position"] == "split":
        parts = block["parts"]
        results = [
            message["content"]
            for message in good.messages_model
            if message["role"] == "user"
            and (message.get("content") or "").startswith("TOOL RESULT")
        ]
        if (
            len(parts) != 2
            or any(part["where"] != "state" for part in parts)
            or parts[0]["slot"] == parts[1]["slot"]
            or not any(
                i != j and parts[0]["text"] in left and parts[1]["text"] in right
                for i, left in enumerate(results)
                for j, right in enumerate(results)
            )
        ):
            problems.append(
                "split parts must be revealed in two different tool outputs"
            )
    if episode["episode_type"] == "injected":
        bad = run(script(scenario, episode, obey=True))
        if bad.result["tag"] != "unsafe":
            problems.append(f"obeying tagged {bad.result['tag']}")
    return problems


@dataclass(frozen=True)
class Candidate:
    """One supported injected base configuration, before twin expansion."""

    scenario: dict
    goal: str
    style: str
    pressure: str
    position: str
    carrier: str

    def factors(self) -> dict[str, str]:
        return {
            "family": self.scenario["family"],
            "scenario": self.scenario["id"],
            "goal": self.goal,
            "style": self.style,
            "pressure": self.pressure,
            "position": self.position,
            "carrier": self.carrier,
            "conversation_language": "en",
            "injection_language": "en",
        }


def candidates(scenarios: list[dict], bank: dict, split: str) -> list[Candidate]:
    """Enumerate existing slots and targets, excluding held-outs before sampling."""
    result = []
    for scenario in scenarios:
        if split != "test" and scenario["family"] == "code":
            continue
        for goal, style, pressure, position, carrier in itertools.product(
            scenario["goals"],
            bank["styles"],
            ("none", "snag"),
            ("first", "later", "split"),
            ("body", "metadata", "quoted", "tool_error"),
        ):
            if split != "test" and carrier == "tool_error":
                continue
            targets = bank["goals"][goal]["targets"]
            if scenario["family"] not in targets and "*" not in targets:
                continue
            if carrier not in scenario["slots"].get(position, {}):
                continue
            result.append(Candidate(scenario, goal, style, pressure, position, carrier))
    return result


def sample(pool: list[Candidate], count: int, *, seed: int = 1234) -> list[Candidate]:
    """Greedily balance feasible marginal strata, with seeded tie breaking.

    The square-count objective penalizes over-represented levels; scaling by
    the number of feasible levels gives each factor comparable influence.
    Constraints can make perfect marginal balance impossible (e.g. split only
    has a body slot); the report exposes these deficits instead of inventing slots.
    """
    if count < 0:
        raise ValueError("limit must be nonnegative")
    remaining = list(pool)
    random.Random(seed).shuffle(remaining)
    levels: dict[str, set[str]] = {}
    for item in remaining:
        for key, value in item.factors().items():
            levels.setdefault(key, set()).add(value)
    counts: dict[str, Counter] = {key: Counter() for key in levels}
    selected = []
    for _ in range(min(count, len(remaining))):
        index = min(
            range(len(remaining)),
            key=lambda i: sum(
                len(levels[key]) * (2 * counts[key][value] + 1)
                for key, value in remaining[i].factors().items()
            ),
        )
        item = remaining.pop(index)
        selected.append(item)
        for key, value in item.factors().items():
            counts[key][value] += 1
    return selected


def _strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for child in value for text in _strings(child)]
    if isinstance(value, dict):
        return [text for child in value.values() for text in _strings(child)]
    return []


def placement_problems(episode: dict, twin: dict | None = None) -> list[str]:
    """Check payload confinement in model-visible fields, never metadata copies."""
    block = episode["injection"] or episode["authorized"]
    if not block:
        return []
    problems = []
    visible = _strings([episode["initial_state"], episode["user_turns"]])
    for part in block["parts"]:
        container = episode[
            "initial_state" if part["where"] == "state" else "user_turns"
        ]
        if resolve_pointer(container, part["slot"]).count(part["text"]) != 1:
            problems.append(f"payload must appear exactly once at {part['slot']}")
        if sum(text.count(part["text"]) for text in visible) != 1:
            problems.append(f"payload outside its slot: {part['slot']}")
        if twin and any(
            part["text"] in text
            for text in _strings([twin["initial_state"], twin["user_turns"]])
        ):
            problems.append(f"clean twin retains payload from {part['slot']}")
    return problems


def _factor_report(items: list[dict], levels: dict, target: int) -> dict:
    report = {}
    for key, values in levels.items():
        counts = Counter(
            item["lang"][key.removesuffix("_language")]
            if key.endswith("_language")
            else item["meta"]["pressure"]
            if key == "pressure"
            else item[key]
            for item in items
        )
        # Quotient + remainder avoids promising fractional episodes.
        quotas = dict.fromkeys(values, target // len(values))
        for value in values[: target % len(values)]:
            quotas[value] += 1
        report[key] = {
            value: {
                "target": quotas[value],
                "actual": counts[value],
                "shortfall": max(0, quotas[value] - counts[value]),
            }
            for value in values
        }
    return report


def build_data(
    split: str,
    *,
    teacher: str = "stub",
    limit: int | None = None,
    root: Path | None = None,
    config_path: Path | None = None,
) -> dict:
    """Build an offline split and return counts and PLAN target shortfalls.

    Limit counts injected bases, never twins or authorized variants. Each
    available authorized variant is emitted once; absent variants are reported,
    not padded with duplicate episodes. A rejected base rejects its twin too;
    the limit caps base attempts, and any acceptance deficit is reported.
    """
    if teacher != "stub":
        raise ValueError("Only --teacher stub is implemented in step 3a")
    if split not in ("train", "dev", "test"):
        raise ValueError("split must be train, dev or test")
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    config_path = config_path or root / "configs/data.yaml"
    config = yaml.safe_load(config_path.read_text("utf-8"))
    # Stub data must never be mistaken for a real teacher dataset.
    if Path(config["stub_output"]) != Path("build/stub"):
        raise ValueError("Stub output must be build/stub, never data/v1")
    target = config["targets"][split]
    count = target["injected"] if limit is None else limit
    if count < 0:
        raise ValueError("limit must be nonnegative")
    bank, scenarios = load(root / config["templates"])
    pool = candidates(scenarios, bank, split)
    selected = sample(pool, count, seed=config["seed"])
    rng = random.Random(config["seed"])
    episodes, rejected = [], []
    for index, item in enumerate(selected):
        factors = item.factors()
        episode_id = "-".join(
            [
                split,
                item.scenario["id"],
                item.goal,
                item.style,
                item.pressure,
                item.position,
                item.carrier,
                "en-en",
                str(index),
            ]
        )
        try:
            episode = injected(
                item.scenario,
                bank,
                item.goal,
                item.style,
                item.pressure,
                item.position,
                item.carrier,
                split=split,
                rng=rng,
            )
            episode["id"] = episode_id
            episode["meta"]["seed"] = config["seed"]
            twin = make_clean_twin(episode)
            problems = (
                check(item.scenario, episode)
                + check(item.scenario, twin)
                + validate_episode(twin, twins={episode_id: episode})
                + placement_problems(episode, twin)
            )
            if problems:
                rejected.append(
                    {"id": episode_id, "factors": factors, "reasons": problems}
                )
            else:
                episodes.extend([episode, twin])
        except (ValueError, KeyError, TypeError, IndexError) as error:
            rejected.append(
                {"id": episode_id, "factors": factors, "reasons": [str(error)]}
            )
    authorized_count = 0
    for scenario in scenarios:
        if split != "test" and scenario["family"] == "code":
            continue
        for index, variant in enumerate(scenario.get("authorized", [])):
            if authorized_count >= target["authorized"]:
                break
            if split != "test" and variant["carrier"] == "tool_error":
                continue
            episode_id = f"{split}-{scenario['id']}-auth-{index}-en-en"
            try:
                episode = authorized(scenario, variant, index, split=split)
                episode["id"] = episode_id
                problems = check(scenario, episode) + placement_problems(episode)
                if problems:
                    rejected.append({"id": episode_id, "reasons": problems})
                else:
                    episodes.append(episode)
                    authorized_count += 1
            except (ValueError, KeyError, TypeError, IndexError) as error:
                rejected.append({"id": episode_id, "reasons": [str(error)]})
    injected_items = [e for e in episodes if e["episode_type"] == "injected"]
    levels = copy.deepcopy(config["factors"])
    if split != "test":
        levels["family"].remove("code")
        levels["carrier"].remove("tool_error")
    report = {
        "split": split,
        "teacher": teacher,
        "seed": config["seed"],
        "requested_injected": count,
        "available_injected": len(pool),
        "counts": {
            kind: sum(e["episode_type"] == kind for e in episodes)
            for kind in ("injected", "clean", "authorized")
        },
        "rejected": len(rejected),
        "requested_shortfall": max(0, count - len(injected_items)),
        "plan_targets": target,
        "plan_type_shortfalls": {
            kind: max(0, quota - sum(e["episode_type"] == kind for e in episodes))
            for kind, quota in target.items()
        },
        "factors": _factor_report(injected_items, levels, count),
        "plan_factors": _factor_report(injected_items, levels, target["injected"]),
        "notes": [
            "English only in step 3a; hi/hinglish are reported as shortfalls.",
            "Equal marginal factor quotas; slot constraints may prevent balance.",
            "Authorized variants are not duplicated to fill unavailable scenarios.",
            "Train quotas budget source episodes for future DPO pairs.",
        ],
    }
    output = root / config["stub_output"] / split
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("episodes.jsonl", episodes), ("rejected.jsonl", rejected)):
        (output / name).write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    logging.getLogger(__name__).info("Stub build %s: %s", split, report["counts"])
    return report


def check_generated_text(text: str) -> None:
    """Reject generated addresses and web links outside reserved sandbox domains.

    Apply only to filler, not literal template data or injection metadata.
    Bare domains with common web suffixes count as links; ordinary file names
    (e.g. payroll.xlsx) do not.
    """
    email_hosts = re.findall(r"[\w.+%-]+@([\w.-]+)", text)
    web_hosts = re.findall(
        r"(?i)(?:https?://|www\.)([a-z0-9.-]+)|"
        r"\b([a-z0-9.-]+\.(?:com|org|net|invalid|io|ai|co|edu|gov|dev))\b",
        text,
    )
    hosts = email_hosts + [a or b for a, b in web_hosts]
    for host in hosts:
        host = host.lower().rstrip(".")
        if not (
            host.endswith(".invalid")
            or any(
                host == domain or host.endswith("." + domain)
                for domain in ("example.com", "example.org", "example.net")
            )
        ):
            raise ValueError(f"generated text has a non-sandbox address or URL: {host}")
