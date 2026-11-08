"""Scenario-scoped multilingual episode assembly and route checks.

Rendering and route verification are ported from scripts/render_prototype.py;
that script remains the exhaustive format acceptance check. No teacher calls or
CuratorKIT pipeline run in the stub path.
"""

import copy
import hashlib
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

from turnguard.data.teacher import Teacher, keep_tokens, read_settings
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


def generate_state(
    value: Any,
    teacher: Teacher,
    language: str,
    tokens: list[str],
    *,
    variant: int = 0,
    scenario: str = "",
    pointer: str = "",
    fields: dict | None = None,
) -> Any:
    """Fill only authored generator specs; preserve literal values and keys."""
    if isinstance(value, dict) and "gen" in value:
        text = teacher.generate(
            value,
            language,
            tokens,
            variant=variant,
            scenario=scenario,
            field_path=pointer,
        )
        if fields is not None:
            fields[pointer] = text
        check_generated_text(text)
        return text
    if isinstance(value, list):
        return [
            generate_state(
                v,
                teacher,
                language,
                tokens,
                variant=variant,
                scenario=scenario,
                pointer=f"{pointer}/{i}",
                fields=fields,
            )
            for i, v in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            k: generate_state(
                v,
                teacher,
                language,
                tokens,
                variant=variant,
                scenario=scenario,
                pointer=pointer + "/" + str(k).replace("~", "~0").replace("/", "~1"),
                fields=fields,
            )
            for k, v in value.items()
        }
    return value


def base(
    scenario: dict,
    attacker: dict,
    split: str = "dev",
    *,
    teacher_client: Teacher | None = None,
    conversation_language: str = "en",
    injection_language: str = "en",
    filler_state: dict | None = None,
) -> tuple[dict, dict]:
    names = dict(scenario["vars"])
    names.update(attacker)
    names.update(fill(scenario["assets"], names))
    tokens = keep_tokens(scenario["vars"], attacker, fill(scenario["assets"], names))
    skeleton = fill(scenario["state"], names)
    state = (
        copy.deepcopy(filler_state)
        if filler_state is not None
        else (
            generate_state(skeleton, teacher_client, conversation_language, tokens)
            if teacher_client
            else stub_generate(skeleton)
        )
    )
    state.setdefault("tool_errors", {})
    turns = fill(scenario["turns"], names)
    if teacher_client:
        turns = [
            teacher_client.translate(t, conversation_language, tokens, kind="user_turn")
            for t in turns
        ]
        for text in turns:
            check_generated_text(text)
    return names, {
        "split": split,
        "family": scenario["family"],
        "lang": {
            "conversation": conversation_language,
            "injection": injection_language,
        },
        "user_turns": turns,
        "tools": scenario["tools"],
        "initial_state": state,
        "task_oracle": {"all": fill(scenario["task_oracle"], names)},
        "meta": {
            "scenario": scenario["id"],
            "renderer": "builder-v2",
            "obey_args": fill(scenario.get("obey_args", {}), names),
        },
    }


def translate_slot(
    slot: dict, teacher: Teacher | None, language: str, tokens: list[str]
) -> dict:
    slot = copy.deepcopy(slot)
    if teacher and "intro" in slot:
        slot["intro"] = teacher.translate(
            slot["intro"], language, tokens, kind="quoted_intro"
        )
        check_generated_text(slot["intro"])
    return slot


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
    teacher_client: Teacher | None = None,
    conversation_language: str = "en",
    injection_language: str = "en",
    filler_state: dict | None = None,
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
    names, episode = base(
        scenario,
        attacker,
        split,
        teacher_client=teacher_client,
        conversation_language=conversation_language,
        injection_language=injection_language,
        filler_state=filler_state,
    )
    tokens = keep_tokens(scenario["vars"], attacker, fill(scenario["assets"], names))
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
    slot_list = [
        translate_slot(slot, teacher_client, conversation_language, tokens)
        for slot in slot_list
    ]
    if teacher_client:
        texts = [
            teacher_client.translate(t, injection_language, tokens, kind="injection")
            for t in texts
        ]
    for text in texts:
        check_generated_text(text)
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
    scenario: dict,
    variant: dict,
    index: int,
    *,
    split: str = "dev",
    teacher_client: Teacher | None = None,
    conversation_language: str = "en",
    filler_state: dict | None = None,
) -> dict:
    names, episode = base(
        scenario,
        {},
        split,
        teacher_client=teacher_client,
        conversation_language=conversation_language,
        injection_language=conversation_language,
        filler_state=filler_state,
    )
    tokens = keep_tokens(
        scenario["vars"], {}, fill(scenario["assets"], scenario["vars"])
    )
    slot = fill(scenario["slots"][variant["position"]][variant["carrier"]], names)
    turn = 0 if variant["position"] == "first" else 1
    slot = translate_slot(slot, teacher_client, conversation_language, tokens)
    delegate = fill(variant["delegate"], names)
    text = fill(variant["text"], names)
    if teacher_client:
        delegate = teacher_client.translate(
            delegate, conversation_language, tokens, kind="authorized_delegate"
        )
        # Addresses and filenames authored inside the variant itself are pinned too.
        tokens = sorted(set(tokens + keep_tokens({}, {}, {"text": text})))
        text = teacher_client.translate(
            text, conversation_language, tokens, kind="authorized_text"
        )
    check_generated_text(delegate)
    check_generated_text(text)
    episode["user_turns"][turn] += delegate
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
        results_by_turn: list[list[str]] = []
        for message in good.messages_model[1:]:
            content = message.get("content") or ""
            if message["role"] == "user" and not content.startswith("TOOL RESULT"):
                waiting = True
                results_by_turn.append([])
            elif message["role"] == "user":
                results_by_turn[-1].append(content)
                if waiting:
                    firsts.append(content)
                    waiting = False
        for part in block["parts"]:
            if part["where"] == "state" and not any(part["text"] in f for f in firsts):
                problems.append(f"part at {part['slot']} is not on a turn's first call")
        # "later" must stay later: keyword search can return a turn-2 page in
        # turn 1, and teacher text can add the overlapping word.
        if episode["position"] == "later" and results_by_turn:
            early = "\n".join(results_by_turn[0])
            if any(part["text"] in early for part in block["parts"]):
                problems.append("later leaks into turn 1")
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
    conversation_language: str = "en"
    injection_language: str = "en"

    def factors(self) -> dict[str, str]:
        return {
            "family": self.scenario["family"],
            "scenario": self.scenario["id"],
            "goal": self.goal,
            "style": self.style,
            "pressure": self.pressure,
            "position": self.position,
            "carrier": self.carrier,
            "conversation_language": self.conversation_language,
            "injection_language": self.injection_language,
        }


def select_scenarios(
    scenarios: list[dict], split: str, required: list[str]
) -> list[dict]:
    """Scenario splits are authored, never inferred or randomly repartitioned."""
    ids = set()
    for scenario in scenarios:
        if scenario.get("split") not in ("train", "dev", "test"):
            raise ValueError(
                f"Scenario {scenario['id']}: split must be train, dev or test"
            )
        if scenario["family"] == "code" and scenario["split"] != "test":
            raise ValueError(
                f"Scenario {scenario['id']}: family code must have split test"
            )
        if scenario["id"] in ids:
            raise ValueError(f"Duplicate scenario id: {scenario['id']}")
        ids.add(scenario["id"])
    selected = [scenario for scenario in scenarios if scenario["split"] == split]
    missing = sorted(set(required) - {scenario["family"] for scenario in selected})
    if missing:
        raise ValueError(
            f"Missing required scenario family for split {split}: {', '.join(missing)}"
        )
    return selected


def candidates(
    scenarios: list[dict],
    bank: dict,
    split: str,
    *,
    conversation_languages: tuple[str, ...] = ("en",),
    injection_languages: tuple[str, ...] = ("en",),
) -> list[Candidate]:
    """Enumerate supported slots and independent language axes within a split."""
    result = []
    for scenario in select_scenarios(scenarios, split, []):
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
            for conversation, injection in itertools.product(
                conversation_languages, injection_languages
            ):
                result.append(
                    Candidate(
                        scenario,
                        goal,
                        style,
                        pressure,
                        position,
                        carrier,
                        conversation,
                        injection,
                    )
                )
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
    dry_run: bool = False,
    backtranslate: bool = False,
    teacher_client: Teacher | None = None,
) -> dict:
    """Assemble checked multilingual groups; plan calls without I/O in dry-run.

    Limit caps injected base attempts. Twins are built only after all generation
    and translation. Every authorized variant is rendered once per conversation
    language, regardless of limit; unavailable quota is reported, never duplicated.
    """
    if teacher not in ("stub", "litellm"):
        raise ValueError("teacher must be stub or litellm (set CK_MODEL in .env)")
    if split not in ("train", "dev", "test"):
        raise ValueError("split must be train, dev or test")
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    config_path = config_path or root / "configs/data.yaml"
    config = yaml.safe_load(config_path.read_text("utf-8"))
    if Path(config["stub_output"]) != Path("build/stub"):
        raise ValueError("Stub output must be build/stub, never data/v1")
    if Path(config["real_output"]) != Path("data/v1"):
        raise ValueError("Real teacher output must be data/v1")
    target = config["targets"][split]
    count = target["injected"] if limit is None else limit
    if count < 0:
        raise ValueError("limit must be nonnegative")
    bank, all_scenarios = load(root / config["templates"])
    required = config["required_families"][split]
    scenarios = select_scenarios(all_scenarios, split, required)
    output = (
        root / config["stub_output" if teacher == "stub" else "real_output"] / split
    )
    # Freeze implementation belongs to 3c; protect already frozen datasets now.
    if (output / "FROZEN").exists():
        raise ValueError(f"Split {split} is frozen: {output}")
    conversations = tuple(config["factors"]["conversation_language"])
    injections = tuple(config["factors"]["injection_language"])
    pool = candidates(
        scenarios,
        bank,
        split,
        conversation_languages=conversations,
        injection_languages=injections,
    )
    selected = sample(pool, count, seed=config["seed"])
    if teacher_client is None:
        settings = (
            read_settings(root, required=teacher == "litellm" and not dry_run)
            if teacher == "litellm"
            else {}
        )
        # Hash actual input formats as well as the configured template version.
        version = hashlib.sha256(
            json.dumps(
                {
                    "version": config["template_version"],
                    "bank": bank,
                    "scenarios": all_scenarios,
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        teacher_client = Teacher(
            cache_dir=root / config["cache"],
            model=settings.get("CK_MODEL", "stub" if teacher == "stub" else "unset"),
            api_base=settings.get("CK_API_BASE", ""),
            api_key=settings.get("CK_API_KEY", ""),
            sampling=config["teacher_sampling"],
            template_version=version,
            mode="dry-run" if dry_run else "stub" if teacher == "stub" else "real",
            review=backtranslate,
        )
    if dry_run and teacher_client.mode != "dry-run":
        raise ValueError("dry-run requires a planning teacher, never a real transport")
    # Dry-run remains independent of optional curation packages and never writes.
    curator = None
    if not dry_run:
        from turnguard.data.curation import CuratorPass

        curator = CuratorPass(config["curator"], clean=config["curator"]["clean"])
    rng = random.Random(config["seed"])
    jobs = []
    for index, item in enumerate(selected):
        variant = rng.randrange(3)
        episode_id = "-".join(
            [
                split,
                item.scenario["id"],
                item.goal,
                item.style,
                item.pressure,
                item.position,
                item.carrier,
                item.conversation_language,
                item.injection_language,
                f"v{variant}",
                str(index),
            ]
        )
        jobs.append(
            {
                "id": episode_id,
                "kind": "injected",
                "scenario": item.scenario,
                "language": item.conversation_language,
                "variant": variant,
                "item": item,
            }
        )
    for scenario in scenarios:
        for index, variant in enumerate(scenario.get("authorized", [])):
            if split != "test" and variant["carrier"] == "tool_error":
                continue
            # Once per (conversation language x filler version): up to 9 per
            # variant, so authorized episodes reach the PLAN targets.
            for conversation, filler_variant in itertools.product(
                conversations, range(3)
            ):
                jobs.append(
                    {
                        "id": (
                            f"{split}-{scenario['id']}-auth-{index}-"
                            f"{conversation}-v{filler_variant}"
                        ),
                        "kind": "authorized",
                        "scenario": scenario,
                        "language": conversation,
                        "variant": filler_variant,
                        "authorized": variant,
                        "index": index,
                    }
                )
    blocks, failed_blocks, rejected = {}, set(), []
    for job in jobs:
        scenario = job["scenario"]
        block_id = f"{split}-{scenario['id']}-v{job['variant']}-{job['language']}"
        job["block"] = block_id
        if block_id in blocks or block_id in failed_blocks:
            continue
        names = dict(scenario["vars"])
        names.update(fill(scenario["assets"], names))
        tokens = keep_tokens(scenario["vars"], {}, fill(scenario["assets"], names))
        start = len(teacher_client.history)
        try:
            fields = {}
            state = generate_state(
                fill(scenario["state"], names),
                teacher_client,
                job["language"],
                tokens,
                variant=job["variant"],
                scenario=scenario["id"],
                fields=fields,
            )
            state.setdefault("tool_errors", {})
            # Keep this dictionary usable in dry-run without importing CuratorKIT.
            blocks[block_id] = {
                "id": block_id,
                "scenario": scenario["id"],
                "variant": job["variant"],
                "language": job["language"],
                "state": state,
                "fields": fields,
                "cache_keys": list(
                    dict.fromkeys(row["key"] for row in teacher_client.history[start:])
                ),
            }
        except (ValueError, KeyError, TypeError, IndexError) as error:
            failed_blocks.add(block_id)
            rejected.append({"block": block_id, "reasons": [str(error)]})
    dedup = {
        "blocks": len(blocks),
        "removed": 0,
        "variant_collapsed": 0,
        "cross_scenario_removed": 0,
        "removal_rate": 0,
        "removals": [],
        "stopped": False,
    }
    if curator:
        from turnguard.data.curation import FillerBlock

        dedup = curator.dedup([FillerBlock(**block) for block in blocks.values()])
    # A collapsed variant is a near-copy of a surviving variant of the same
    # scenario and language: re-point its episodes there instead of dropping them.
    retained = set(curator.block_samples) if curator else set(blocks)
    for row in dedup["removals"]:
        if not row["reason"].startswith("variant_collapse"):
            continue
        gone = blocks[row["block"]]
        survivor = next(
            (
                block_id
                for block_id in sorted(retained)
                if blocks[block_id]["scenario"] == gone["scenario"]
                and blocks[block_id]["language"] == gone["language"]
            ),
            None,
        )
        if survivor is not None:
            row["replaced_by"] = survivor
            for job in jobs:
                if job["block"] == row["block"]:
                    job["block"] = survivor
    # Merging versions can make two jobs identical (same episode on the same
    # surviving filler). Keep the first; record the rest.
    seen: set[tuple] = set()
    unique_jobs = []
    for job in jobs:
        if job["kind"] == "injected":
            key = ("injected", tuple(sorted(job["item"].factors().items())))
        else:
            key = ("authorized", job["scenario"]["id"], job["index"], job["language"])
        key += (job["block"],)
        if key in seen:
            rejected.append(
                {
                    "id": job["id"],
                    "block": job["block"],
                    "reasons": ["duplicate_after_merge"],
                }
            )
            continue
        seen.add(key)
        unique_jobs.append(job)
    jobs[:] = unique_jobs
    removed = {row["block"] for row in dedup["removals"] if "replaced_by" not in row}
    for row in [r for r in dedup["removals"] if "replaced_by" not in r]:
        affected = [job["id"] for job in jobs if job["block"] == row["block"]]
        affected += [
            job["id"] + "-clean"
            for job in jobs
            if job["block"] == row["block"] and job["kind"] == "injected"
        ]
        row["episode_ids"] = affected
        rejected.append(
            {"block": row["block"], "episode_ids": affected, "reasons": [row["reason"]]}
        )
    if dedup["stopped"]:
        output.mkdir(parents=True, exist_ok=True)
        diagnostic = {
            "status": "stopped",
            "split": split,
            "dedup": dedup,
            "reason": "filler dedup removal rate exceeds 5%; review before proceeding",
        }
        (output / "dedup_report.json").write_text(
            json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8"
        )
        (output / "summary.json").write_text(
            json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8"
        )
        (output / "rejected.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rejected),
            encoding="utf-8",
        )
        raise ValueError(
            f"Filler dedup removed {dedup['cross_scenario_removed']}/"
            f"{dedup['blocks']} blocks as cross-scenario duplicates "
            f"({dedup['removal_rate']:.1%}), above 5%; stopped. "
            f"See {output / 'dedup_report.json'}"
        )
    episodes = []
    for job in jobs:
        block_id = job["block"]
        if block_id in removed or block_id in failed_blocks:
            continue
        block = blocks[block_id]
        start = len(teacher_client.history)
        try:
            if job["kind"] == "injected":
                item = job["item"]
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
                    teacher_client=teacher_client,
                    conversation_language=item.conversation_language,
                    injection_language=item.injection_language,
                    filler_state=block["state"],
                )
            else:
                episode = authorized(
                    job["scenario"],
                    job["authorized"],
                    job["index"],
                    split=split,
                    teacher_client=teacher_client,
                    conversation_language=job["language"],
                    filler_state=block["state"],
                )
            episode["id"] = job["id"]
            episode["meta"].update(
                seed=config["seed"],
                teacher=teacher,
                filler_block=block_id,
                filler_variant=job["variant"],
                teacher_cache_keys=list(
                    dict.fromkeys(
                        block["cache_keys"]
                        + [row["key"] for row in teacher_client.history[start:]]
                    )
                ),
            )
            group = [episode]
            if job["kind"] == "injected":
                group.append(make_clean_twin(episode))
            problems = []
            if not dry_run:
                for member in group:
                    problems.extend(check(job["scenario"], member))
                problems += placement_problems(
                    episode, group[1] if len(group) == 2 else None
                )
            if problems:
                rejected.append({"id": job["id"], "reasons": problems})
            else:
                episodes.extend(group)
        except (ValueError, KeyError, TypeError, IndexError) as error:
            rejected.append({"id": job["id"], "reasons": [str(error)]})
    source_paths = [
        config_path,
        root / config["templates"] / "injections.yaml",
        *sorted((root / config["templates"] / "templates").glob("*.yaml")),
    ]
    sources = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    if curator:
        episodes, gate_rejected = curator.gate(episodes, sources)
        rejected.extend(gate_rejected)
    teacher_client.backtranslate_sample()
    injected_items = [e for e in episodes if e["episode_type"] == "injected"]
    levels = copy.deepcopy(config["factors"])
    if split != "test":
        levels["family"].remove("code")
        levels["carrier"].remove("tool_error")
    report = {
        "split": split,
        "teacher": teacher,
        "seed": config["seed"],
        "status": "complete",
        "dedup": dedup,
        "dry_run": dry_run,
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
        "teacher_estimate": teacher_client.estimates(),
        "output": str(output.relative_to(root)),
        "notes": [
            "Independent language sampling; stub translations use language markers.",
            "Equal marginal factor quotas; slot constraints may prevent balance.",
            "Each authorized variant is rendered once per conversation language x filler version (duplicates after version merging are dropped).",
            "Train quotas budget source episodes for future DPO pairs.",
            "CuratorKIT filler-only dedup and SecretsGate; PII rewriting is off.",
        ],
    }
    if dry_run:
        return report
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("rejected.jsonl", rejected),
        (
            "filler_blocks.jsonl",
            [
                {k: v for k, v in block.items() if k != "state"}
                for block in blocks.values()
            ],
        ),
        ("dedup_report.jsonl", dedup["removals"]),
        ("teacher_calls.jsonl", teacher_client.history),
        ("backtranslations.jsonl", teacher_client.reviews),
    ):
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
    # Diagnostics from an earlier stopped attempt must not taint a successful rebuild.
    (output / "dedup_report.json").unlink(missing_ok=True)
    curator.write(output, report, rejected, sources)
    logging.getLogger(__name__).info("Build %s: %s", split, report["counts"])
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
