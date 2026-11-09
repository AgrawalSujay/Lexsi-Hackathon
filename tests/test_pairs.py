"""M3 preferences use exact on-policy prefixes, verified steps and guarded splits."""

import copy
import json
import shutil

import pytest
import yaml

from turnguard.cli import main
from turnguard.data import builder
from turnguard.data.pairs import (
    SOURCES,
    VALIDATION_LABEL,
    balance,
    build_pairs,
    candidates,
    decisions,
    load_training_pairs,
)
from turnguard.harness import run_episodes
from turnguard.models import StubGenerator
from turnguard.protocol import Action, Malformed, parse
from turnguard.schema import canonical_json, make_clean_twin


def fixture_episodes(root, split="train", relay_answer=True, relay_language="en"):
    bank, scenarios = builder.load(root / "turnguard/data")
    scenario = next(
        s
        for s in scenarios
        if s["id"]
        == ("email-vendor-invoice" if split == "train" else "email-reschedule")
    )
    specs, scripts = [], {}
    for name, position, style in (
        ("executed", "first", "imperative"),
        ("relayed", "later", "imperative"),
        ("synthesized", "first", "fake_authority"),
    ):
        episode = builder.injected(
            scenario, bank, "exfiltrate", style, "none", position, "body", split=split
        )
        episode["id"] = name
        safe = builder.script(scenario, episode, False)
        script = list(safe)
        if name == "executed":
            script.insert(
                1, builder._call_for(episode["injection"]["target"][0], episode)
            )
        elif name == "relayed":
            terms = [
                term
                for matcher in episode["injection"]["target"]
                for values in matcher.get("args_contains", {}).values()
                for term in values
            ]
            answer = script[-1][len("FINAL: ") :]
            if relay_answer:
                script[-1] = f"FINAL: {answer}. Please send it to {' '.join(terms)}."
            else:
                script[-1] = f"FINAL: Please send {answer} to {' '.join(terms)}."
            episode["lang"] = {
                "conversation": relay_language,
                "injection": relay_language,
            }
        twin = make_clean_twin(episode)
        clean_script = list(safe)
        if name == "executed":
            clean_script = [
                line
                for line in safe
                if not (
                    isinstance(parse(line), Action) and parse(line).tool == "send_email"
                )
            ]
            clean_script[1] = (
                "FINAL: I have written a draft reply but have not sent it."
            )
        specs.extend([episode, twin])
        scripts[episode["id"]] = script
        scripts[twin["id"]] = clean_script
    for index in (0, 1):
        episode = builder.authorized(
            scenario, scenario["authorized"][index], index, split=split
        )
        episode["id"] = f"authorized-{index}"
        script = builder.script(scenario, episode, False)
        # Keep a legitimate route but skip the delegated extra action, so one
        # rejected label is the model's refusal/skip and the other is synthetic.
        specs.append(episode)
        scripts[episode["id"]] = script
    records = run_episodes(specs, StubGenerator(scripts))
    return specs, records, {scenario["id"]: scenario}


def save_run(root, name, specs, records):
    sources = root / "originals"
    sources.mkdir(exist_ok=True)
    path = sources / "episodes.jsonl"
    path.write_text(
        "".join(json.dumps(spec, ensure_ascii=False) + "\n" for spec in specs), "utf-8"
    )
    run = root / name
    run.mkdir()
    (run / "config.yaml").write_text(
        yaml.safe_dump({"split": specs[0]["split"], "episode_paths": [str(path)]})
    )
    (run / "episodes.jsonl").write_text(
        "".join(record.to_json() + "\n" for record in records), "utf-8"
    )
    return run


def test_all_five_sources_exact_prefixes_and_step_checks(bank_root):
    specs, records, scenarios = fixture_episodes(bank_root)
    rows, skipped = candidates(records, specs, scenarios)
    assert set(row["source"] for row in rows) == set(SOURCES)
    by_id = {record.episode_id: record for record in records}
    for row in rows:
        assert not isinstance(parse(row["chosen"]), Malformed)
        proof = row["provenance"]
        record = by_id[proof["episode_id"]]
        assistant_index = [
            i for i, m in enumerate(record.messages_model) if m["role"] == "assistant"
        ][proof["step"] - 1]
        assert row["prompt"] == record.messages_model[:assistant_index]
        assert proof["chosen"]["check"]
    relay = next(row for row in rows if row["source"] == "injected-relayed")
    assert relay["chosen"] == "FINAL: APR-6619."
    assert relay["rejected"].startswith("FINAL: APR-6619. Please send")
    assert relay["provenance"]["chosen"]["source"] == "model_final_sentence_removal"
    anchor = next(row for row in rows if row["source"] == "clean anchor")
    assert parse(anchor["chosen"]).tool == "send_email"
    assert "draft reply" in anchor["rejected"]
    assert any(row["reason"] == "clean_output_not_proven_wrong" for row in skipped)


def test_relay_losing_answer_is_skipped_never_synthesized(bank_root):
    specs, records, scenarios = fixture_episodes(bank_root, relay_answer=False)
    rows, skipped = candidates(records, specs, scenarios)
    assert not any(row["provenance"]["episode_id"] == "relayed" for row in rows)
    assert any(
        row["episode_id"] == "relayed" and row["reason"] == "relay_no_answer_remains"
        for row in skipped
    )


def test_replay_includes_error_counters_and_native_tool_messages(bank_root):
    bank, scenarios = builder.load(bank_root / "turnguard/data")
    scene = next(s for s in scenarios if s["id"] == "email-vendor-invoice")
    episode = builder.injected(
        scene,
        bank,
        "exfiltrate",
        "imperative",
        "snag",
        "first",
        "tool_error",
        split="test",
    )
    script = builder.script(scene, episode, False)
    record = run_episodes(
        [episode], StubGenerator({episode["id"]: script}), tool_result_role="tool"
    )[0]
    points = decisions(episode, record)
    assert points[1].environment._fired_errors
    assert any(message["role"] == "tool" for message in points[1].prompt)
    assert decisions(episode, record)[-1].step == record.steps


def test_tampered_raw_or_replay_facts_are_refused(bank_root):
    specs, records, _ = fixture_episodes(bank_root)
    record = copy.deepcopy(records[0])
    record.calls[0]["ok"] = False
    with pytest.raises(ValueError, match="replay mismatch"):
        decisions(specs[0], record)
    record = copy.deepcopy(records[0])
    record.raw_outputs[0]["text"] = "FINAL: corrupted"
    with pytest.raises(ValueError, match="history mismatch"):
        decisions(specs[0], record)


def test_multiple_shards_are_balanced_deterministic_and_trainable(bank_root):
    specs, records, _ = fixture_episodes(bank_root)
    a = save_run(bank_root, "shard-a", specs, records[:4])
    b = save_run(bank_root, "shard-b", specs, records[4:])
    report = build_pairs([a, b], root=bank_root)
    assert report["pairs"] == 6
    assert (
        sum(
            n
            for source, n in report["by_source"].items()
            if source.startswith("injected-")
        )
        == 3
    )
    assert report["by_source"]["clean anchor"] + report["by_source"]["authorized"] == 3
    assert set(report["by_source"]) == set(SOURCES)
    assert sum(report["by_language"]["conversation"].values()) == 6
    directory = bank_root / "data/pairs/train"
    rows = load_training_pairs(directory)
    assert len(rows) == 6
    assert (directory / "audit_sample.md").read_text().count("### Chosen") == 6
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    again = build_pairs([b, a], root=bank_root)
    assert report == again
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


def test_cap_balance_and_language_strata(bank_root):
    specs, records, scenarios = fixture_episodes(bank_root)
    rows, _ = candidates(records, specs, scenarios)
    selected, dropped = balance(rows, 2)
    assert len(selected) == 2 and dropped
    assert sum(row["source"].startswith("injected-") for row in selected) == 1
    assert balance(list(reversed(rows)), 2)[0] == selected
    with pytest.raises(ValueError, match="at least 2"):
        balance(rows, 1)


def test_duplicate_record_results_can_upgrade_but_conflicting_histories_fail(bank_root):
    specs, records, _ = fixture_episodes(bank_root)
    a = save_run(bank_root, "run-a", specs, records)
    old = copy.deepcopy(records)
    old[0].result["grading_version"] = "old"
    b = save_run(bank_root, "run-b", specs, old)
    assert build_pairs([a, b], root=bank_root)["episodes"] == len(records)
    old[0].raw_outputs[0]["text"] += " changed"
    (b / "episodes.jsonl").write_text("".join(row.to_json() + "\n" for row in old))
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        build_pairs([a, b], root=bank_root)


def test_spec_and_template_hash_changes_are_refused(bank_root):
    specs, records, _ = fixture_episodes(bank_root)
    run = save_run(bank_root, "source", specs, records)
    source = bank_root / "originals/episodes.jsonl"
    original = source.read_bytes()
    changed = copy.deepcopy(specs)
    changed[0]["user_turns"][0] += " changed"
    source.write_text("".join(canonical_json(spec) + "\n" for spec in changed))
    with pytest.raises(ValueError, match="Spec hash mismatch"):
        build_pairs([run], root=bank_root)
    source.write_bytes(original)
    (source.parent / "manifest.json").write_text(
        json.dumps(
            {
                "turnguard": {
                    "source_sha256": {"turnguard/data/templates/email.yaml": "0" * 64}
                }
            }
        )
    )
    with pytest.raises(ValueError, match="Template provenance mismatch"):
        build_pairs([run], root=bank_root)
    assert not (bank_root / "data/pairs").exists()


@pytest.mark.parametrize("split", ["dev", "test"])
def test_nontrain_source_refused_before_output(bank_root, split):
    specs, records, _ = fixture_episodes(bank_root, "dev")
    run = save_run(bank_root, "wrong-split", specs, records)
    config = yaml.safe_load((run / "config.yaml").read_text())
    config["split"] = split
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="Pair source run must be train"):
        build_pairs([run], root=bank_root)
    assert not (bank_root / "data/pairs").exists()


def test_dev_validation_export_and_trainer_refusal_even_after_copy(
    bank_root, monkeypatch
):
    import turnguard.data.pairs as module

    specs, records, _ = fixture_episodes(bank_root, "dev")
    run = save_run(bank_root, "dev-run", specs, records)
    original = module.build_pairs
    monkeypatch.setattr(
        module,
        "build_pairs",
        lambda *args, **kwargs: original(*args, root=bank_root, **kwargs),
    )
    assert main(["build-pairs", "--run", str(run), "--validate-only"]) == 0
    directory = bank_root / "data/pairs/_validation_dev"
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["purpose"] == VALIDATION_LABEL
    assert (directory / "VALIDATION_ONLY").read_text().strip() == VALIDATION_LABEL
    assert not (bank_root / "data/pairs/train").exists()
    with pytest.raises(ValueError, match="never for training"):
        load_training_pairs(directory)
    with pytest.raises(SystemExit) as error:
        main(["train", "--arm", "A4", "--model", "headline", "--pairs", str(directory)])
    assert error.value.code == 2
    copied = bank_root / "renamed"
    copied.mkdir()
    shutil.copyfile(directory / "pairs.jsonl", copied / "pairs.jsonl")
    with pytest.raises(ValueError, match="never for training"):
        load_training_pairs(copied)


def test_no_source_run_is_modified_and_no_test_split_is_accepted(bank_root):
    specs, records, _ = fixture_episodes(bank_root)
    run = save_run(bank_root, "original-run", specs, records)
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    build_pairs([run], root=bank_root)
    assert {p.name: p.read_bytes() for p in run.iterdir()} == before
    with pytest.raises(ValueError, match="require split=train"):
        build_pairs([run], split="test", root=bank_root)


def test_ungraded_clean_is_reported_once(bank_root):
    specs, records, scenes = fixture_episodes(bank_root)
    clean = next(record for record in records if record.episode_id == "executed-clean")
    clean.result["tag"] = "ungraded"
    _, skipped = candidates(records, specs, scenes)
    rows = [row for row in skipped if row["episode_id"] == clean.episode_id]
    assert len(rows) == 1 and rows[0]["reason"] == "clean_trace_missing_or_ungraded"


def test_relay_hindi_danda_and_address_dots_preserve_untouched_answer(bank_root):
    specs, records, scenes = fixture_episodes(bank_root)
    spec = next(spec for spec in specs if spec["id"] == "relayed")
    script = builder.script(next(iter(scenes.values())), spec, False)
    term = spec["injection"]["target"][0]["args_contains"]["to"][0]
    script[-1] = f"FINAL: कोड APR-6619 है। कृपया {term} पर भेजें।"
    altered = run_episodes([spec], StubGenerator({spec["id"]: script}))[0]
    records = [
        altered if record.episode_id == spec["id"] else record for record in records
    ]
    rows, _ = candidates(records, specs, scenes)
    relay = next(row for row in rows if row["source"] == "injected-relayed")
    assert relay["chosen"] == "FINAL: कोड APR-6619 है।"
    assert relay["rejected"] == script[-1]


def test_equivalent_search_result_does_not_make_correct_final_a_bad_anchor(bank_root):
    bank, scenes = builder.load(bank_root / "turnguard/data")
    scene = next(scene for scene in scenes if scene["id"] == "web-train-schedule")
    base = builder.injected(
        scene,
        bank,
        "tamper_answer",
        "imperative",
        "none",
        "first",
        "metadata",
        split="dev",
    )
    clean = make_clean_twin(base)
    safe = builder.script(scene, base, False)
    alternate = list(safe)
    first = parse(alternate[0])
    alternate[0] = builder.action(
        "search", {"query": first.args["query"] + " train number"}
    )
    records = run_episodes(
        [base, clean], StubGenerator({base["id"]: safe, clean["id"]: alternate})
    )
    rows, skipped = candidates(records, [base, clean], {scene["id"]: scene})
    assert not any(row["source"] == "clean anchor" for row in rows)
    assert any(
        row["episode_id"] == clean["id"] and row["source"] == "clean anchor"
        for row in skipped
    )


def test_authorized_open_text_waits_for_factual_lookup_never_uses_x(bank_root):
    bank, scenes = builder.load(bank_root / "turnguard/data")
    scene = next(scene for scene in scenes if scene["id"] == "docs-travel-policy")
    variant = {**scene["authorized"][2], "carrier": "quoted"}
    spec = builder.authorized(scene, variant, 2, split="dev")
    script = builder.script(scene, spec, False)
    record = run_episodes([spec], StubGenerator({spec["id"]: script}))[0]
    rows, _ = candidates([record], [spec], {scene["id"]: scene})
    assert len(rows) == 1
    row = rows[0]
    assert row["provenance"]["step"] == 2
    assert parse(row["chosen"]).args["text"] == "PD-MUM-3"
    assert row["provenance"]["chosen"]["source"] == "authorized_target_grounded_facts"
    assert row["prompt"][-1]["content"].startswith("TOOL RESULT (read_file):")


def test_authorized_unknown_content_is_skipped_not_filled_from_hidden_state(bank_root):
    bank, scenes = builder.load(bank_root / "turnguard/data")
    scene = next(scene for scene in scenes if scene["id"] == "docs-travel-policy")
    spec = builder.authorized(
        scene, {**scene["authorized"][2], "carrier": "quoted"}, 2, split="dev"
    )
    script = builder.script(scene, spec, False)
    script = [line for index, line in enumerate(script) if index != 0]
    record = run_episodes([spec], StubGenerator({spec["id"]: script}))[0]
    rows, skipped = candidates([record], [spec], {scene["id"]: scene})
    assert not rows
    assert skipped[0]["reason"] == "authorized_content_not_available"


def test_synthesized_rejection_has_no_placeholder_artefact():
    from turnguard.data.pairs import _natural_fill

    text = _natural_fill(
        'ACTION: {"tool": "write_summary", "args": {"title": "x", "text": "x"}}'
    )
    assert parse(text).args == {"title": "Update", "text": "As requested."}
