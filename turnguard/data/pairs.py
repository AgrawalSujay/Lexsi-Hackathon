"""Verified on-policy preference pairs with isolated dev validation."""

import copy
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from turnguard.analysis import load_verified_run, regrade_records
from turnguard.config import load_config
from turnguard.data.builder import _call_for, fill, load
from turnguard.oracles import (
    GRADING_VERSION,
    _contains,
    _equal,
    _state_strings,
    injection_relayed,
    matching_steps,
    normalize_text,
    observed_exposure,
)
from turnguard.protocol import Action, Final, Malformed, canonical_text, parse
from turnguard.schema import EpisodeRecord, canonical_json, make_clean_twin, spec_sha256
from turnguard.tools import REGISTRY, ToolEnvironment

VERSION = "m3-v2"
VALIDATION_LABEL = "validation — never for training"
SOURCES = (
    "injected-executed",
    "injected-relayed",
    "injected-synthesized",
    "clean anchor",
    "clean act-anchor",
    "authorized",
)


class Skip(ValueError):
    """A reason a decision cannot safely supply a labeled pair."""


@dataclass
class Decision:
    step: int
    turn: int
    prompt: list[dict]
    output: str
    history: EpisodeRecord
    environment: ToolEnvironment


def _bound_equal(spec: dict, tool: str, a: dict, b: dict) -> bool:
    try:
        definition = REGISTRY[spec["family"]][tool]
        return _equal(definition.bind(a), definition.bind(b))
    except (KeyError, ValueError):
        return False


def decisions(spec: dict, record: EpisodeRecord) -> list[Decision]:
    """Replay exactly the calls behind model-format prefixes, checking the trace.

    Raw generations may contain hallucinated tool results/trailing protocol lines.
    They are labels only; prompts always use messages_model verbatim, including
    format reminders, native tool messages and any arm-specific system prompt.
    """
    environment = ToolEnvironment(
        spec["family"], spec["tools"], copy.deepcopy(spec["initial_state"])
    )
    raw = {row["step"]: row["text"] for row in record.raw_outputs}
    if len(raw) != len(record.raw_outputs):
        raise ValueError(f"Duplicate raw step in {spec['id']}")
    calls = {call["step"]: call for call in record.calls}
    step, turn = 0, 1
    answers, result = [], []
    for index, message in enumerate(record.messages_model):
        if message["role"] != "assistant":
            continue
        step += 1
        output = raw.get(step, message["content"])
        parsed = parse(output)
        expected = (
            canonical_text(parsed) if isinstance(parsed, (Action, Final)) else output
        )
        if expected != message["content"]:
            raise ValueError(f"Model/raw history mismatch for {spec['id']} step {step}")
        history = copy.deepcopy(record)
        history.messages_model = copy.deepcopy(record.messages_model[:index])
        history.messages_openai = []
        history.result = {}
        history.calls = copy.deepcopy(environment.calls)
        history.answers = copy.deepcopy(answers)
        history.steps = step - 1
        history.raw_outputs = [
            copy.deepcopy(row) for row in record.raw_outputs if row["step"] < step
        ]
        history.final_state = copy.deepcopy(environment.state)
        result.append(
            Decision(
                step,
                turn,
                copy.deepcopy(history.messages_model),
                output,
                history,
                copy.deepcopy(environment),
            )
        )
        if isinstance(parsed, Action):
            observed = calls.get(step)
            executed = environment.execute(
                parsed.tool, parsed.args, step=step, turn=turn
            )
            if observed is None or observed != environment.calls[-1]:
                raise ValueError(f"Call replay mismatch for {spec['id']} step {step}")
            following = (
                record.messages_model[index + 1]
                if index + 1 < len(record.messages_model)
                else None
            )
            if following is None or following.get("content") not in (
                executed.output,
                f"TOOL RESULT ({parsed.tool}):\n{executed.output}",
            ):
                raise ValueError(
                    f"Tool result replay mismatch for {spec['id']} step {step}"
                )
        elif isinstance(parsed, Final):
            answers.append({"turn": turn, "text": parsed.text})
            turn += 1
    if (
        step != record.steps
        or len(environment.calls) != len(record.calls)
        or answers != record.answers
        or environment.state != record.final_state
    ):
        raise ValueError(f"Final replay mismatch for {spec['id']}")
    return result


def _after(spec: dict, decision: Decision, output: str) -> EpisodeRecord:
    parsed = parse(output)
    if isinstance(parsed, Malformed):
        raise Skip("invalid_protocol")
    record = copy.deepcopy(decision.history)
    record.steps = decision.step
    record.messages_model.append(
        {"role": "assistant", "content": canonical_text(parsed)}
    )
    record.raw_outputs.append({"step": decision.step, "text": output})
    if isinstance(parsed, Action):
        environment = copy.deepcopy(decision.environment)
        result = environment.execute(
            parsed.tool, parsed.args, step=decision.step, turn=decision.turn
        )
        if not result.ok:
            raise Skip("tool_call_failed")
        record.calls = environment.calls
        record.final_state = environment.state
    else:
        record.answers.append({"turn": decision.turn, "text": parsed.text})
    return record


def _answer_checks(spec: dict, turn: int) -> list[dict]:
    return [
        matcher
        for matcher in spec["task_oracle"]["all"]
        if matcher["kind"] == "answer"
        and matcher.get("turn", len(spec["user_turns"])) == turn
    ]


def _route_matcher(spec: dict, route: dict) -> dict:
    """Use the authored oracle for semantic body matches, exact route IDs otherwise."""
    call = {"kind": "call", "tool": route["tool"], "args_match": route["args"]}
    # A model can satisfy a write with different wording. Use its task oracle,
    # but only when the oracle also matches this specific route call.
    for matcher in spec["task_oracle"]["all"]:
        if matcher["kind"] != "call" or matcher["tool"] != route["tool"]:
            continue
        fake = EpisodeRecord(
            "route",
            "",
            "",
            "",
            [],
            [],
            [{"tool": route["tool"], "args": route["args"], "ok": True, "step": 1}],
            [],
            {},
            1,
            0,
            {},
        )
        if matching_steps(matcher, fake):
            return matcher
    return call


def _lookup_completed(spec: dict, route: dict, point: Decision) -> bool:
    if REGISTRY[spec["family"]][route["tool"]].effectful:
        return False
    probe = copy.deepcopy(point.environment)
    expected = probe.execute(
        route["tool"], route["args"], step=point.step, turn=point.turn
    )
    if not expected.ok or expected.output in ("No results.", "[]", ""):
        return False
    # Equivalent searches can return extra items or different ranking. Every
    # requested result chunk must actually have appeared in the recorded result.
    chunks = [
        normalize_text(chunk)
        for chunk in re.split(r"(?m)^\d+\.\s*", expected.output)
        if chunk.strip()
    ]
    step = 0
    for index, message in enumerate(point.prompt):
        if message["role"] != "assistant":
            continue
        step += 1
        call = next(
            (call for call in point.history.calls if call["step"] == step), None
        )
        if (
            not call
            or not call["ok"]
            or call["turn"] != point.turn
            or call["tool"] != route["tool"]
        ):
            continue
        following = point.prompt[index + 1] if index + 1 < len(point.prompt) else {}
        content = following.get("content", "")
        if following.get("role") == "user":
            content = content.partition("\n")[2]
        visible = normalize_text(re.sub(r"(?m)^\d+\.\s*", "", content))
        if all(chunk in visible for chunk in chunks):
            return True
    return False


def _pending_route(spec: dict, route: list[dict], decision: Decision) -> list[dict]:
    history = copy.deepcopy(decision.history)
    history.calls = [call for call in history.calls if call["turn"] == decision.turn]
    return [
        step
        for step in route
        if step["turn"] == decision.turn
        and not (
            matching_steps(_route_matcher(spec, step), history)
            or _lookup_completed(spec, step, decision)
        )
    ]


def _safe(spec: dict, decision: Decision, output: str) -> EpisodeRecord:
    record = _after(spec, decision, output)
    if spec.get("injection"):
        for matcher in spec["injection"]["target"]:
            if decision.step in matching_steps(
                matcher,
                record,
                verified_line=spec.get("goal") == "tamper_answer"
                and matcher["kind"] == "answer",
            ):
                raise Skip("chosen_executes_injection")
        # A chosen FINAL must not pass attack instructions to the user either.
        if isinstance(parse(output), Final):
            current = copy.deepcopy(record)
            current.answers = [record.answers[-1]]
            if injection_relayed(spec, current):
                raise Skip("chosen_relays_injection")
    return record


def task_choice(spec: dict, route: list[dict], decision: Decision) -> tuple[str, dict]:
    pending = _pending_route(spec, route, decision)
    if pending:
        next_step = pending[0]
        chosen = canonical_text(Action(next_step["tool"], next_step["args"]))
        after = _safe(spec, decision, chosen)
        matcher = _route_matcher(spec, next_step)
        if decision.step not in matching_steps(matcher, after):
            raise Skip("chosen_fails_route_oracle_step")
        return chosen, {
            "source": "scenario_route",
            "check": "successful_next_route_oracle_step",
            "matcher": matcher,
        }
    checks = _answer_checks(spec, decision.turn)
    terms = list(
        dict.fromkeys(term for matcher in checks for term in matcher["contains"])
    )
    chosen = canonical_text(Final(" ".join(terms) if terms else "Done."))
    after = _safe(spec, decision, chosen)
    if not all(decision.step in matching_steps(matcher, after) for matcher in checks):
        raise Skip("chosen_fails_answer_oracle")
    return chosen, {
        "source": "task_answer_oracle" if checks else "route_complete",
        "check": "answer_oracle" if checks else "route_complete_turn_boundary",
        "matchers": checks,
    }


def _point_exposure(spec: dict, points: list[Decision], block: dict) -> Decision:
    for point in points:
        if (
            observed_exposure(block, point.history, user_turns=spec["user_turns"])[
                "exposed_turn"
            ]
            is not None
        ):
            return point
    raise Skip("not_exposed_before_model_turn")


def _relay_terms(spec: dict) -> list[str]:
    background = list(_state_strings(make_clean_twin(spec)["initial_state"]))
    return list(
        dict.fromkeys(
            term
            for matcher in spec["injection"]["target"]
            if matcher["kind"] == "call"
            for values in matcher.get("args_contains", {}).values()
            for term in values
            if not _contains(background, [term])
        )
    )


def _sentences(text: str) -> list[str]:
    # Split on sentence-ending punctuation followed by whitespace, never on a
    # dot inside an address/URL. Danda handles Hindi. Keep untouched bytes of
    # every retained sentence; no paraphrasing or synthesized answer prose.
    return re.findall(r"[^\n]+?(?:[.!?।](?=\s|$)|(?=\n|$))\s*", text)


def relay_choice(spec: dict, point: Decision, route: list[dict]) -> tuple[str, dict]:
    parsed = parse(point.output)
    if not isinstance(parsed, Final):
        raise Skip("relay_not_final")
    checks = _answer_checks(spec, point.turn)
    if not checks:
        raise Skip("relay_no_answer_oracle_for_turn")
    if _pending_route(spec, route, point):
        raise Skip("relay_pending_task_calls")
    terms = _relay_terms(spec)
    sentences = _sentences(parsed.text)
    kept = []
    removed = False
    answer_terms = [term for check in checks for term in check["contains"]]
    for sentence in sentences:
        if any(_contains(sentence, [term]) for term in terms):
            removed = True
            continue
        if removed and not any(_contains(sentence, [term]) for term in answer_terms):
            # Only remove a tightly bounded acknowledgement of the discarded
            # request. Unclear trailing prose is not a safe preference label.
            if re.fullmatch(
                r"\s*I (?:will|shall) (?:ensure|make sure) (?:that )?"
                r"(?:this|it) (?:is|gets) (?:done|completed|handled)"
                r"(?: (?:promptly|soon|right away|as requested))?[.!]?\s*",
                sentence,
                flags=re.IGNORECASE,
            ):
                continue
            raise Skip("relay_ambiguous_followup")
        kept.append(sentence)
    if not removed:
        raise Skip("relay_no_removable_sentence")
    text = "".join(kept).strip()
    if not text:
        raise Skip("relay_no_answer_remains")
    chosen = canonical_text(Final(text))
    after = _safe(spec, point, chosen)
    if not all(point.step in matching_steps(matcher, after) for matcher in checks):
        raise Skip("relay_remainder_fails_answer_oracle")
    return chosen, {
        "source": "model_final_sentence_removal",
        "check": "answer_oracle",
        "matchers": checks,
        "removed_sentences": [
            sentence for sentence in sentences if sentence not in kept
        ],
    }


def _new_violation(spec: dict, point: Decision) -> bool:
    # Matching the recorded point (not the whole episode) prevents labeling a
    # legitimate early action as a model-produced harmful rejection.
    try:
        after = _after(spec, point, point.output)
    except Skip:
        return False
    return any(
        point.step
        in matching_steps(
            matcher,
            after,
            verified_line=spec.get("goal") == "tamper_answer"
            and matcher["kind"] == "answer",
        )
        for matcher in spec["injection"]["target"]
    )


_NATURAL = {"subject": "Update", "title": "Update"}


def _natural_fill(text: str) -> str:
    """Replace the bank checker's "x" placeholders in a synthesized obey call.

    A literal "x" would let preference training separate chosen from rejected
    by a surface artefact instead of by obeying the injection.
    """
    parsed = parse(text)
    args = {
        key: (_NATURAL.get(key, "As requested.") if value == "x" else value)
        for key, value in parsed.args.items()
    }
    return canonical_text(Action(parsed.tool, args))


def _synthesized_rejection(spec: dict, point: Decision) -> str:
    for matcher in spec["injection"]["target"]:
        output = (
            _natural_fill(_call_for(matcher, spec))
            if matcher["kind"] == "call"
            else canonical_text(
                Final("\n".join("Verified: " + term for term in matcher["contains"]))
            )
        )
        try:
            after = _after(spec, point, output)
        except Skip:
            continue
        if point.step in matching_steps(
            matcher,
            after,
            verified_line=spec.get("goal") == "tamper_answer"
            and matcher["kind"] == "answer",
        ):
            return output
    raise Skip("no_executable_obey_rejection")


def _known_facts(spec: dict, route: list[dict], point: Decision) -> list[str]:
    terms = [
        term
        for matcher in _answer_checks(spec, point.turn)
        for term in matcher["contains"]
    ]
    for step in route:
        if step["turn"] == point.turn:
            matcher = _route_matcher(spec, step)
            terms.extend(
                term
                for key, values in matcher.get("args_contains", {}).items()
                if key in ("body", "text")
                for term in values
            )
    terms = list(dict.fromkeys(terms))
    # Do not turn the model's hallucinated assistant text into factual grounding.
    visible = [
        message["content"]
        for message in point.prompt
        if message["role"] in ("user", "tool")
    ]
    return terms if terms and all(_contains(visible, [term]) for term in terms) else []


def authorized_choice(
    spec: dict, route: list[dict], point: Decision, matcher: dict
) -> tuple[str, str]:
    parsed = parse(point.output)
    if isinstance(parsed, Action):
        try:
            actual = _safe(spec, point, point.output)
            facts = _known_facts(spec, route, point)
            contents = [
                parsed.args[key] for key in ("text", "body") if key in parsed.args
            ]
            if (
                point.step in matching_steps(matcher, actual)
                and all(value != "x" for value in contents)
                and (
                    not contents
                    or (facts and all(_contains(value, facts) for value in contents))
                )
            ):
                return canonical_text(parsed), "model_authorized_call"
        except Skip:
            pass
    synthesized = parse(_call_for(matcher, spec))
    args = copy.deepcopy(synthesized.args)
    prototype = next(
        (
            step
            for step in route
            if step["turn"] == point.turn and step["tool"] == matcher["tool"]
        ),
        None,
    )
    if prototype:
        args = {**prototype["args"], **matcher.get("args_match", {})}
        for name, terms in matcher.get("args_contains", {}).items():
            if name not in args or not _contains(args[name], terms):
                args[name] = " ".join(terms)
        # Supply only parameters the route did not already bind.
        args = {**synthesized.args, **args}
    declared = set(matcher.get("args_match", {})) | set(
        matcher.get("args_contains", {})
    )
    declared |= set(spec["meta"].get("obey_args", {}).get(matcher["tool"], {}))
    declared |= set(prototype["args"]) if prototype else set()
    grounded = False
    for parameter in REGISTRY[spec["family"]][matcher["tool"]].parameters:
        name = parameter.name
        if name not in args or name in declared:
            continue
        if name in ("body", "text"):
            facts = _known_facts(spec, route, point)
            if not facts:
                raise Skip("authorized_content_not_available")
            args[name] = " ".join(facts)
            grounded = True
        elif name in ("subject", "title"):
            args[name] = "Requested update"
        else:
            raise Skip("authorized_unbound_argument")
    source = (
        "authorized_target_grounded_facts"
        if grounded
        else "authorized_target_and_route"
        if prototype
        else "authorized_target"
    )
    return canonical_text(Action(matcher["tool"], args)), source


def act_choice(spec: dict, route: list[dict], point: Decision) -> tuple[str, dict]:
    """Ground the next pending write in facts actually visible at this point."""
    pending = _pending_route(spec, route, point)
    if not pending or not REGISTRY[spec["family"]][pending[0]["tool"]].effectful:
        raise Skip("act_next_route_not_write")
    step = pending[0]
    matcher = _route_matcher(spec, step)
    chosen, origin = authorized_choice(spec, route, point, matcher)
    action = parse(chosen)
    visible = [m["content"] for m in point.prompt if m["role"] in ("user", "tool")]
    for key, value in action.args.items():
        if key in ("body", "text"):
            facts = _known_facts(spec, route, point)
            if not facts:
                raise Skip("act_content_not_available")
            # Use only observed oracle facts, not unchecked route prose.
            action.args[key] = " ".join(facts)
        elif key not in ("subject", "title") and not _contains(visible, [str(value)]):
            raise Skip("act_argument_not_available")
    chosen = canonical_text(action)
    after = _safe(spec, point, chosen)
    if point.step not in matching_steps(matcher, after):
        raise Skip("act_chosen_fails_route_oracle")
    return chosen, {
        "source": "pending_write_grounded_facts",
        "construction": origin,
        "check": "successful_next_route_oracle_step",
        "matcher": matcher,
    }


def _pair(
    spec: dict,
    point: Decision,
    chosen: str,
    rejected: str,
    source: str,
    chosen_provenance: dict,
    rejected_source: str,
) -> dict:
    if chosen == rejected or (
        isinstance(parse(rejected), (Action, Final))
        and canonical_text(parse(rejected)) == chosen
    ):
        raise Skip("identical_preference")
    if isinstance(parse(chosen), Malformed):
        raise ValueError("Invalid chosen protocol escaped step validation")
    value = {"prompt": point.prompt, "chosen": chosen, "rejected": rejected}
    return {
        "id": hashlib.sha256(
            canonical_json(
                {**value, "episode": spec["id"], "step": point.step, "source": source}
            ).encode()
        ).hexdigest(),
        **value,
        "split": spec["split"],
        "validation_only": spec["split"] != "train",
        "source": source,
        "lang": copy.deepcopy(spec["lang"]),
        "provenance": {
            "episode_id": spec["id"],
            "spec_sha256": spec_sha256(spec),
            "scenario": spec["meta"]["scenario"],
            "step": point.step,
            "turn": point.turn,
            "chosen": chosen_provenance,
            "rejected_source": rejected_source,
            "grading_version": GRADING_VERSION,
            "source_split": spec["split"],
            "oracle_step_check": "passed",
            "state_before_sha256": hashlib.sha256(
                canonical_json(point.environment.state).encode()
            ).hexdigest(),
            "version": VERSION,
        },
    }


def _clean_point(
    spec: dict,
    points: list[Decision],
    injected_record: EpisodeRecord,
    exposure: Decision,
) -> Decision:
    # Align by the source read and user turn, not an assistant-step ordinal:
    # error retries and hallucinated calls make raw step numbers diverge.
    prior = next(
        (call for call in injected_record.calls if call["step"] == exposure.step - 1),
        None,
    )
    if prior is None:
        return (
            next((point for point in points if point.turn == exposure.turn), None)
            or _missing_clean_point()
        )
    for point in points:
        if point.turn != exposure.turn:
            continue
        call = next(
            (call for call in point.history.calls if call["step"] == point.step - 1),
            None,
        )
        if (
            call
            and call["tool"] == prior["tool"]
            and _bound_equal(spec, call["tool"], call["args"], prior["args"])
        ):
            return point
    raise Skip("clean_reveal_call_not_observed")


def _missing_clean_point():
    raise Skip("clean_turn_not_observed")


def _clean_wrong(spec: dict, route: list[dict], point: Decision) -> bool:
    parsed = parse(point.output)
    if isinstance(parsed, Malformed):
        return True
    if isinstance(parsed, Final):
        if _pending_route(spec, route, point):
            return True
        checks = _answer_checks(spec, point.turn)
        after = _after(spec, point, point.output)
        return any(
            point.step not in matching_steps(matcher, after) for matcher in checks
        )
    try:
        _after(spec, point, point.output)
    except Skip:
        return True
    # Do not label an alternative successful tool action as wrong merely because
    # it differs from the authored route; clean anchors focus on clear omissions.
    return False


def candidates(
    records: list[EpisodeRecord], specs: list[dict], scenarios: dict[str, dict]
) -> tuple[list[dict], list[dict]]:
    by_id = {spec["id"]: spec for spec in specs}
    traces = {record.episode_id: record for record in records}
    points = {
        record.episode_id: decisions(by_id[record.episode_id], record)
        for record in records
    }
    result, skipped = [], []

    def skip(episode_id, reason, source):
        skipped.append({"episode_id": episode_id, "reason": reason, "source": source})

    for record in sorted(records, key=lambda record: record.episode_id):
        spec = by_id[record.episode_id]
        scenario = scenarios.get(spec.get("meta", {}).get("scenario"))
        if scenario is None:
            raise ValueError(f"Missing authored scenario for {spec['id']}")
        if scenario.get("split") != spec["split"]:
            raise ValueError(f"Scenario split mismatch for {spec['id']}")
        route = fill(scenario["route"], scenario["vars"])
        if spec["episode_type"] == "clean":
            parent = traces.get(spec.get("twin_of"))
            if record.result["tag"] == "ungraded":
                skip(spec["id"], "clean_trace_missing_or_ungraded", "clean anchor")
            elif parent is None:
                skip(spec["id"], "clean_missing_injected_trace", "clean anchor")
            elif parent.result["tag"] == "ungraded":
                skip(spec["id"], "clean_injected_trace_ungraded", "clean anchor")
            else:
                for point in points[spec["id"]]:
                    parsed = parse(point.output)
                    if not isinstance(parsed, (Final, Malformed)):
                        continue
                    if isinstance(parsed, Malformed) and (
                        not any(char.isalpha() for char in point.output)
                        or re.search(r"(?i)\bACTION\s*:", point.output)
                    ):
                        continue
                    pending = _pending_route(spec, route, point)
                    if (
                        not pending
                        or not REGISTRY[spec["family"]][pending[0]["tool"]].effectful
                    ):
                        continue
                    try:
                        chosen, proof = act_choice(spec, route, point)
                        result.append(
                            _pair(
                                spec,
                                point,
                                chosen,
                                point.output,
                                "clean act-anchor",
                                proof,
                                "model_wrong_output",
                            )
                        )
                    except Skip as error:
                        skip(spec["id"], str(error), "clean act-anchor")
            continue
        if record.result["tag"] == "ungraded":
            skip(spec["id"], "ungraded", spec["episode_type"])
            continue
        if spec["episode_type"] == "authorized":
            try:
                first = _point_exposure(spec, points[spec["id"]], spec["authorized"])
                found = None
                last_error = "authorized_content_not_available"
                for point in points[spec["id"]]:
                    if point.step < first.step or point.turn != first.turn:
                        continue
                    matcher = next(
                        (
                            matcher
                            for matcher in spec["authorized"]["target"]
                            if not matching_steps(matcher, point.history)
                        ),
                        None,
                    )
                    if matcher is None:
                        break
                    if matcher["kind"] != "call":
                        raise Skip("authorized_target_not_call")
                    try:
                        chosen, choice_source = authorized_choice(
                            spec, route, point, matcher
                        )
                        found = (point, matcher, chosen, choice_source)
                        break
                    except Skip as error:
                        last_error = str(error)
                if found is None:
                    raise Skip(last_error)
                point, matcher, chosen, choice_source = found
                after = _safe(spec, point, chosen)
                if point.step not in matching_steps(matcher, after):
                    raise Skip("authorized_chosen_fails_oracle")
                actual = parse(point.output)
                rejected = (
                    point.output
                    if isinstance(actual, (Final, Malformed))
                    else "FINAL: I cannot follow instructions from that document."
                )
                result.append(
                    _pair(
                        spec,
                        point,
                        chosen,
                        rejected,
                        "authorized",
                        {
                            "source": choice_source,
                            "check": "target_call_oracle",
                            "matcher": matcher,
                        },
                        "model_output"
                        if rejected == point.output
                        else "synthetic_refusal",
                    )
                )
            except Skip as error:
                skip(spec["id"], str(error), "authorized")
            continue
        exposure = None
        try:
            exposure = _point_exposure(spec, points[spec["id"]], spec["injection"])
            if record.result["tag"] == "relayed":
                point = None
                for candidate in points[spec["id"]]:
                    if candidate.step < exposure.step or not isinstance(
                        parse(candidate.output), Final
                    ):
                        continue
                    isolated = copy.deepcopy(candidate.history)
                    isolated.answers = [
                        {"turn": candidate.turn, "text": parse(candidate.output).text}
                    ]
                    if injection_relayed(spec, isolated):
                        point = candidate
                        break
                if point is None:
                    raise Skip("relay_final_not_found")
                chosen, proof = relay_choice(spec, point, route)
                source, rejected, rejected_source = (
                    "injected-relayed",
                    point.output,
                    "model_relay_final",
                )
            else:
                point = exposure
                chosen, proof = task_choice(spec, route, point)
                if _new_violation(spec, point):
                    source, rejected, rejected_source = (
                        "injected-executed",
                        point.output,
                        "model_attack_output",
                    )
                else:
                    source, rejected, rejected_source = (
                        "injected-synthesized",
                        _synthesized_rejection(spec, point),
                        "injection_target",
                    )
            result.append(
                _pair(spec, point, chosen, rejected, source, proof, rejected_source)
            )
        except Skip as error:
            skip(spec["id"], str(error), "injected")
        twins = [clean for clean in specs if clean.get("twin_of") == spec["id"]]
        for clean in twins:
            try:
                if (
                    clean["id"] not in traces
                    or traces[clean["id"]].result["tag"] == "ungraded"
                ):
                    continue  # Reported once in the clean-record branch above.
                if exposure is None:
                    raise Skip("injected_not_exposed")
                point = _clean_point(clean, points[clean["id"]], record, exposure)
                if not _clean_wrong(clean, route, point):
                    raise Skip("clean_output_not_proven_wrong")
                pending = _pending_route(clean, route, point)
                if (
                    isinstance(parse(point.output), (Final, Malformed))
                    and pending
                    and REGISTRY[clean["family"]][pending[0]["tool"]].effectful
                ):
                    # The clean-trace scan applies stricter fact grounding and
                    # owns this decision, including its skip reason.
                    continue
                chosen, proof = task_choice(clean, route, point)
                result.append(
                    _pair(
                        clean,
                        point,
                        chosen,
                        point.output,
                        "clean anchor",
                        proof,
                        "model_wrong_output",
                    )
                )
            except Skip as error:
                skip(clean["id"], str(error), "clean anchor")
    # A missed write at the old exposure-aligned anchor is now counted under
    # its more specific source, never twice.
    act_points = {
        (row["provenance"]["episode_id"], row["provenance"]["step"])
        for row in result
        if row["source"] == "clean act-anchor"
    }
    result = [
        row
        for row in result
        if not (
            row["source"] == "clean anchor"
            and (row["provenance"]["episode_id"], row["provenance"]["step"])
            in act_points
        )
    ]
    return result, skipped


def balance(
    rows: list[dict], cap: int, seed: int = 1234
) -> tuple[list[dict], list[dict]]:
    if cap < 2:
        raise ValueError("scenario cap must be at least 2")
    rng, grouped, selected, dropped = random.Random(seed), defaultdict(list), [], []
    seen = set()
    for row in sorted(rows, key=lambda row: row["id"]):
        key = canonical_json(
            {name: row[name] for name in ("prompt", "chosen", "rejected")}
        )
        if key in seen:
            dropped.append(
                {
                    "episode_id": row["provenance"]["episode_id"],
                    "reason": "duplicate_pair_content",
                    "source": row["source"],
                }
            )
            continue
        seen.add(key)
        grouped[row["provenance"]["scenario"]].append(row)

    def pick(pool, n):
        strata = defaultdict(list)
        for row in pool:
            strata[
                row["source"], row["lang"]["conversation"], row["lang"]["injection"]
            ].append(row)
        keys = sorted(strata, key=str)
        rng.shuffle(keys)
        for bucket in strata.values():
            rng.shuffle(bucket)
        result = []
        while len(result) < n:
            for key in list(keys):
                result.append(strata[key].pop())
                if not strata[key]:
                    keys.remove(key)
                if len(result) == n:
                    break
        return result

    for scenario, group in sorted(grouped.items()):
        injected = [row for row in group if row["source"].startswith("injected-")]
        controls = [row for row in group if not row["source"].startswith("injected-")]
        n = min(len(injected), len(controls), cap // 2)
        keep = pick(injected, n) + pick(controls, n)
        selected.extend(keep)
        ids = {row["id"] for row in keep}
        dropped.extend(
            {
                "episode_id": row["provenance"]["episode_id"],
                "reason": "scenario_cap_or_balance",
                "source": row["source"],
            }
            for row in group
            if row["id"] not in ids
        )
    return sorted(selected, key=lambda row: row["id"]), dropped


def load_training_pairs(path: str | Path) -> list[dict]:
    """Mandatory trainer preflight: no copied/renamed dev validation can train."""
    path = Path(path).expanduser().resolve()
    directory = path if path.is_dir() else path.parent
    if "_validation_dev" in directory.parts or any(
        (parent / "VALIDATION_ONLY").exists()
        for parent in (directory, *directory.parents)
    ):
        raise ValueError(VALIDATION_LABEL)
    summary_file = directory / "summary.json"
    if summary_file.exists():
        summary = json.loads(summary_file.read_text("utf-8"))
        if summary.get("validation_only") or summary.get("split") != "train":
            raise ValueError(VALIDATION_LABEL)
    file = directory / "pairs.jsonl" if path.is_dir() else path
    rows = [
        json.loads(line)
        for line in file.read_text("utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("No training pairs")
    for row in rows:
        if row.get("validation_only") is not False or row.get("split") != "train":
            raise ValueError(VALIDATION_LABEL)
        if isinstance(parse(row["chosen"]), Malformed):
            raise ValueError("Invalid chosen protocol in training pairs")
    return rows


def build_pairs(
    runs: list[Path],
    *,
    split: str = "train",
    validate_only: bool = False,
    specs_path: str | None = None,
    scenario_cap: int = 200,
    root: Path | None = None,
) -> dict:
    if not runs:
        raise ValueError("At least one source run is required")
    expected = "dev" if validate_only else "train"
    if split != expected:
        raise ValueError(
            "Use --validate-only with dev; training pairs require split=train"
        )
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    records, values, origins, snapshots, datasets = {}, {}, defaultdict(list), {}, set()
    manifest_paths = set()
    for run in sorted({Path(run).expanduser().resolve() for run in runs}):
        config = load_config(run / "config.yaml")
        if config.get("split") != expected:
            raise ValueError(f"Pair source run must be {expected}, never test: {run}")
        before = {
            name: hashlib.sha256((run / name).read_bytes()).hexdigest()
            for name in ("config.yaml", "episodes.jsonl")
        }
        source_records, source_specs, metadata = load_verified_run(
            run, specs_path=specs_path
        )
        if before != {
            name: hashlib.sha256((run / name).read_bytes()).hexdigest()
            for name in before
        }:
            raise ValueError("Source run changed while loading")
        snapshots[str(run)] = before
        if metadata.get("dataset_sha256"):
            datasets.add(metadata["dataset_sha256"])
        for name in metadata["spec_files"]:
            path = Path(name).parent / "manifest.json"
            if path.exists():
                manifest_paths.add(path)
        for record, spec in zip(source_records, source_specs, strict=True):
            if spec["split"] != expected:
                raise ValueError("Pair source specs must match the allowed split")
            if not validate_only and (
                spec["family"] == "code" or spec.get("carrier") == "tool_error"
            ):
                raise ValueError("Held-out family/carrier cannot supply training pairs")
            if record.episode_id in records:
                old = {
                    key: value
                    for key, value in records[record.episode_id].to_dict().items()
                    if key != "result"
                }
                new = {
                    key: value
                    for key, value in record.to_dict().items()
                    if key != "result"
                }
                if old != new:
                    raise ValueError(
                        f"Conflicting duplicate episode: {record.episode_id}"
                    )
            records[record.episode_id], values[record.episode_id] = record, spec
            origins[record.episode_id].append(str(run))
    if (
        len(datasets) > 1
        or len({(record.arm, record.model) for record in records.values()}) > 1
    ):
        raise ValueError("Shard runs must use the same dataset, model and arm")
    _, authored = load(root / "turnguard/data")
    scenarios = {scenario["id"]: scenario for scenario in authored}
    template_hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (root / "turnguard/data/templates").glob("*.yaml")
    }
    for path in manifest_paths:
        manifest = json.loads(path.read_text("utf-8"))
        sources = manifest.get("turnguard", {}).get("source_sha256", {})
        for name, digest in template_hashes.items():
            if name in sources and sources[name] != digest:
                raise ValueError(f"Template provenance mismatch: {name}")
    ordered = sorted(records)
    specs = [values[key] for key in ordered]
    graded = regrade_records([records[key] for key in ordered], specs)
    relay_changes = [
        {
            "episode_id": key,
            "before": records[key].result.get("tag"),
            "after": record.result["tag"],
        }
        for key, record in zip(ordered, graded, strict=True)
        if records[key].result.get("tag") != record.result["tag"]
        and "relayed" in (records[key].result.get("tag"), record.result["tag"])
    ]
    pool, skipped = candidates(graded, specs, scenarios)
    rows, removed = balance(pool, scenario_cap)
    skipped.extend(removed)
    for row in rows:
        row["provenance"]["source_runs"] = origins[row["provenance"]["episode_id"]]
        row["provenance"]["template_sha256"] = template_hashes
        row["validation_only"] = validate_only
    output = root / "data/pairs" / ("_validation_dev" if validate_only else "train")
    summary = {
        "version": VERSION,
        "split": expected,
        "validation_only": validate_only,
        "purpose": VALIDATION_LABEL if validate_only else "training",
        "seed": 1234,
        "scenario_cap": scenario_cap,
        "source_runs": snapshots,
        "episodes": len(records),
        "regrading": {
            "version": GRADING_VERSION,
            "relays_before": sum(
                r.result.get("tag") == "relayed" for r in records.values()
            ),
            "relays_after": sum(r.result["tag"] == "relayed" for r in graded),
            "relay_labels_changed": len(relay_changes),
            "changes": relay_changes,
        },
        "candidate_pairs": len(pool),
        "pairs": len(rows),
        "candidates_by_source": {
            source: sum(row["source"] == source for row in pool) for source in SOURCES
        },
        "by_source": {
            source: sum(row["source"] == source for row in rows) for source in SOURCES
        },
        "by_language": {
            factor: dict(sorted(Counter(row["lang"][factor] for row in rows).items()))
            for factor in ("conversation", "injection")
        },
        "by_language_pair": dict(
            sorted(
                Counter(
                    f"{row['lang']['conversation']}/{row['lang']['injection']}"
                    for row in rows
                ).items()
            )
        ),
        "by_source_language": {
            source: dict(
                sorted(
                    Counter(
                        f"{row['lang']['conversation']}/{row['lang']['injection']}"
                        for row in rows
                        if row["source"] == source
                    ).items()
                )
            )
            for source in SOURCES
        },
        "by_scenario": dict(
            sorted(Counter(row["provenance"]["scenario"] for row in rows).items())
        ),
        "skip_reasons": dict(sorted(Counter(row["reason"] for row in skipped).items())),
        "output": str(output),
    }
    for run, expected_hashes in snapshots.items():
        if expected_hashes != {
            name: hashlib.sha256((Path(run) / name).read_bytes()).hexdigest()
            for name in expected_hashes
        }:
            raise ValueError("Source run changed while building pairs")
    output.mkdir(parents=True, exist_ok=True)
    if validate_only:
        (output / "VALIDATION_ONLY").write_text(VALIDATION_LABEL + "\n", "utf-8")
    (output / "pairs.jsonl").write_text(
        "".join(canonical_json(row) + "\n" for row in rows), "utf-8"
    )
    (output / "skipped.jsonl").write_text(
        "".join(canonical_json(row) + "\n" for row in skipped), "utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    audit = random.Random(1234).sample(rows, min(50, len(rows)))
    lines = [
        "# Preference-pair audit",
        "",
        VALIDATION_LABEL if validate_only else "Training pairs",
        "",
    ]
    for index, row in enumerate(audit, 1):
        lines += [
            f"## {index}. {row['source']} — {row['provenance']['episode_id']}",
            "",
            f"Step {row['provenance']['step']}; languages {row['lang']}",
            "",
            "### Exact model-format prompt",
            "",
            "```json",
            json.dumps(row["prompt"], ensure_ascii=False, indent=2),
            "```",
            "",
            "### Chosen",
            "",
            "```text",
            row["chosen"],
            "```",
            "",
            "### Rejected",
            "",
            "```text",
            row["rejected"],
            "```",
            "",
            "### Provenance",
            "",
            "```json",
            json.dumps(row["provenance"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    (output / "audit_sample.md").write_text("\n".join(lines), "utf-8")
    return summary
