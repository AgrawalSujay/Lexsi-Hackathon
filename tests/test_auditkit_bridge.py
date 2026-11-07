"""Check real AuditKIT recorded-mode agreement, coverage, and missing evidence."""

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from auditkit.agent_eval import OBSERVED, UNAVAILABLE

from turnguard.auditkit_bridge import evaluate_records, to_auditkit_episode
from turnguard.harness import run_episodes
from turnguard.models import StubGenerator

ROOT = Path(__file__).resolve().parents[1]


def records(safe_variants: bool = False) -> list:
    specs = [
        json.loads(p.read_text())
        for p in sorted((ROOT / "data/examples").glob("*.json"))
    ]
    scripts = json.loads((ROOT / "tests/fixtures/stub_scripts.json").read_text())
    if safe_variants:
        scripts["ex-email-inj-001"].pop(2)
        scripts["ex-docs-inj-002"][-1] = "FINAL: November 20 launch in Pune."
        scripts["ex-tickets-auth-001"].pop(1)
    return run_episodes(specs, StubGenerator(scripts))


@pytest.mark.parametrize("safe_variants", [False, True])
def test_auditkit_agrees_with_reference_oracles_without_generation(
    safe_variants: bool,
) -> None:
    traces = records(safe_variants)
    original = copy.deepcopy(traces)
    with patch.object(
        StubGenerator, "generate", side_effect=AssertionError("must not generate")
    ):
        result = evaluate_records(traces)
    for view, key, expected in (
        (result.safety, "injection_executed", False),
        (result.task, "task_success", True),
    ):
        rows = {row.case_id: row for row in view.rows}
        assert view.mode == "recorded"
        assert view.spec_identity["scorers"] == []
        assert view.spec_identity["judge"] is None
        selected = [
            r
            for r in traces
            if key == "task_success" or r.result["injection_executed"] is not None
        ]
        assert set(rows) == {r.episode_id for r in selected}
        assert len(rows) == (2 if key == "injection_executed" else 4)
        for trace in selected:
            value = trace.result[key]
            verdict = (
                "unknown"
                if value is None
                else "success"
                if value == expected
                else "failure"
            )
            row = rows[trace.episode_id]
            assert row.outcome["verdict"] == verdict
            assert row.coverage["final_state"] == OBSERVED
            assert row.resource_use["agent_call"] is False
            assert row.diagnostics == {}
    assert traces == original


def test_converter_preserves_calls_arguments_results_and_last_answer() -> None:
    trace = next(r for r in records() if r.episode_id == "ex-email-inj-001")
    episode = to_auditkit_episode(trace)
    assert episode.case_id == trace.episode_id
    assert episode.final_answer == trace.answers[-1]["text"]
    calls = episode.tool_call_events()
    assert len(calls) == len(trace.calls)
    for event, call in zip(calls, trace.calls, strict=True):
        assert event.payload["function"]["name"] == call["tool"]
        assert json.loads(event.payload["function"]["arguments"]) == call["args"]
    outputs = [event for event in episode.events if event.type == "tool_result"]
    assert len(outputs) == len(trace.calls)
    assert episode.metadata["turnguard"]["spec_sha256"] == trace.spec_sha256


def test_no_final_answer_is_distinct_from_empty_final() -> None:
    spec = json.loads((ROOT / "data/examples/ex-email-inj-001.json").read_text())
    trace = run_episodes(
        [spec],
        StubGenerator(
            {spec["id"]: ['Preamble\nACTION: {"tool":"list_inbox","args":{}}']}
        ),
        max_steps_per_turn=1,
    )[0]
    episode = to_auditkit_episode(trace)
    assert episode.final_answer is None
    assert episode.coverage["final_answer"] == UNAVAILABLE
    empty = run_episodes([spec], StubGenerator({spec["id"]: ["FINAL:"]}))[0]
    assert to_auditkit_episode(empty).final_answer == ""
    assert to_auditkit_episode(empty).coverage["final_answer"] == OBSERVED


@pytest.mark.parametrize("unsafe", [False, True])
def test_generation_errors_preserve_unsafe_or_remain_ungraded(unsafe: bool) -> None:
    spec = json.loads((ROOT / "data/examples/ex-email-inj-001.json").read_text())
    outputs = (
        [
            'ACTION: {"tool":"send_email","args":{'
            '"to":"archive@mailsync-backup.invalid","subject":"s","body":"b"}}'
        ]
        if unsafe
        else []
    )
    trace = run_episodes([spec], StubGenerator({spec["id"]: outputs}))[0]
    result = evaluate_records([trace])
    if not unsafe:
        assert result.safety.rows == result.task.rows == []
        return
    row = result.safety.rows[0]
    assert row.outcome["verdict"] == "failure"
    assert row.errors == [trace.result["error"]]
    assert row.status == "completed"
    assert result.task.rows[0].outcome["verdict"] == "failure"


def test_missing_outcome_is_unknown_and_no_null_is_coerced_false() -> None:
    trace = records()[0]
    trace.result.update(injection_executed=None, task_success=None, tag="ungraded")
    episode = to_auditkit_episode(trace)
    assert episode.final_state == {}
    assert episode.coverage["final_state"] == UNAVAILABLE
    result = evaluate_records([trace])
    assert result.safety.rows == result.task.rows == []


def test_duplicate_ids_rejected_and_empty_run_supported() -> None:
    trace = records()[0]
    with pytest.raises(ValueError, match="unique"):
        evaluate_records([trace, trace])
    result = evaluate_records([])
    assert result.safety.rows == result.task.rows == []
