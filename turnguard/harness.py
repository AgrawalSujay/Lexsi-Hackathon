"""Run isolated multi-turn episodes in lockstep and retain both message formats."""

import copy
import logging
from dataclasses import dataclass

from turnguard.models import Generator, GenRequest
from turnguard.oracles import evaluate_episode
from turnguard.protocol import (
    FORMAT_REMINDER,
    Action,
    Final,
    parse,
    render_system_prompt,
)
from turnguard.schema import (
    EpisodeRecord,
    EpisodeSpec,
    canonical_json,
    spec_sha256,
    validate_episode,
)
from turnguard.tools import REGISTRY, ToolEnvironment

logger = logging.getLogger(__name__)


@dataclass
class _Active:
    spec: EpisodeSpec
    record: EpisodeRecord
    environment: ToolEnvironment
    turn: int = 1
    turn_steps: int = 0
    finished: bool = False


def _start(
    spec: EpisodeSpec, generator: Generator, arm: str, extra: str | None
) -> _Active:
    system = render_system_prompt(
        [REGISTRY[spec.family][name] for name in spec.tools], extra
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": spec.user_turns[0]},
    ]
    environment = ToolEnvironment(
        spec.family, spec.tools, copy.deepcopy(spec.initial_state)
    )
    record = EpisodeRecord(
        episode_id=spec.id,
        spec_sha256=spec_sha256(spec),
        arm=arm,
        model=generator.name,
        messages_model=copy.deepcopy(messages),
        messages_openai=copy.deepcopy(messages),
        calls=environment.calls,
        answers=[],
        final_state=environment.state,
        steps=0,
        malformed_turns=0,
        result={},
    )
    return _Active(spec, record, environment)


def _advance(active: _Active, output: str, max_steps: int, role: str) -> None:
    record = active.record
    record.steps += 1
    active.turn_steps += 1
    parsed = parse(output)
    model_message = {"role": "assistant", "content": output}
    record.messages_model.append(model_message)
    if isinstance(parsed, Action):
        call_id = f"call_{record.steps}"
        tool_calls = [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": parsed.tool,
                    "arguments": canonical_json(parsed.args),
                },
            }
        ]
        record.messages_openai.append(
            {
                "role": "assistant",
                "content": parsed.preamble or None,
                "tool_calls": tool_calls,
            }
        )
        result = active.environment.execute(
            parsed.tool, parsed.args, step=record.steps, turn=active.turn
        )
        tool_message = {
            "role": "tool",
            "tool_call_id": call_id,
            "name": parsed.tool,
            "content": result.output,
        }
        record.messages_openai.append(copy.deepcopy(tool_message))
        if role == "tool":
            # Native tool results need a matching assistant tool-call ID.
            model_message["tool_calls"] = copy.deepcopy(tool_calls)
            record.messages_model.append(copy.deepcopy(tool_message))
        else:
            record.messages_model.append(
                {
                    "role": "user",
                    "content": f"TOOL RESULT ({parsed.tool}):\n{result.output}",
                }
            )
    elif isinstance(parsed, Final):
        record.answers.append({"turn": active.turn, "text": parsed.text})
        record.messages_openai.append({"role": "assistant", "content": parsed.text})
        if active.turn == len(active.spec.user_turns):
            active.finished = True
        else:
            active.turn += 1
            active.turn_steps = 0
            user = {"role": "user", "content": active.spec.user_turns[active.turn - 1]}
            record.messages_model.append(copy.deepcopy(user))
            record.messages_openai.append(copy.deepcopy(user))
    else:
        record.malformed_turns += 1
        record.messages_openai.append({"role": "assistant", "content": output})
        reminder = {"role": "user", "content": FORMAT_REMINDER}
        record.messages_model.append(copy.deepcopy(reminder))
        record.messages_openai.append(copy.deepcopy(reminder))
    if active.turn_steps >= max_steps:
        active.finished = True
    if active.finished:
        record.result = evaluate_episode(active.spec, record)


def run_episodes(
    specs: list[EpisodeSpec | dict],
    generator: Generator,
    *,
    max_steps_per_turn: int = 6,
    tool_result_role: str = "user",
    system_extra: str | None = None,
    arm: str = "A0",
) -> list[EpisodeRecord]:
    if type(max_steps_per_turn) is not int or max_steps_per_turn < 1:
        raise ValueError("max_steps_per_turn must be a positive integer")
    if tool_result_role not in ("user", "tool"):
        raise ValueError("tool_result_role must be user or tool")
    episodes = [
        EpisodeSpec.from_dict(s.to_dict() if isinstance(s, EpisodeSpec) else s)
        for s in specs
    ]
    twins = {spec.id: spec for spec in episodes}
    if len(twins) != len(episodes):
        raise ValueError("Episode IDs must be unique")
    for spec in episodes:
        problems = validate_episode(spec, twins=twins)
        if problems:
            raise ValueError(f"{spec.id}: {'; '.join(problems)}")
    states = [_start(spec, generator, arm, system_extra) for spec in episodes]
    while active := [state for state in states if not state.finished]:
        requests = [
            GenRequest(
                s.spec.id, s.record.steps + 1, copy.deepcopy(s.record.messages_model)
            )
            for s in active
        ]
        # Failed batches produce no assistant turns. Preserve observed violations;
        # without a proven violation, a partial run is ungraded, never safe.
        for attempt in range(2):
            try:
                outputs = generator.generate(copy.deepcopy(requests))
                if (
                    not isinstance(outputs, list)
                    or len(outputs) != len(active)
                    or not all(isinstance(o, str) for o in outputs)
                ):
                    raise ValueError("Generator must return one string per request")
                break
            except Exception as error:
                logger.warning(
                    "Generation batch failed (attempt %s): %s", attempt + 1, error
                )
                if attempt == 1:
                    for state in active:
                        state.finished = True
                        partial = evaluate_episode(state.spec, state.record)
                        if partial["injection_executed"]:
                            state.record.result = {**partial, "error": str(error)}
                        else:
                            state.record.result = {
                                "injection_executed": None,
                                "first_violation_step": None,
                                "task_success": None,
                                "authorized_followed": None,
                                "tag": "ungraded",
                                "error": str(error),
                            }
        else:
            continue
        for state, output in zip(active, outputs, strict=True):
            _advance(state, output, max_steps_per_turn, tool_result_role)
    return [state.record for state in states]
