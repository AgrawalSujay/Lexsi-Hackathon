"""Offline builder acceptance, stratification, rejection and replay checks."""

import copy
import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from turnguard.cli import main
from turnguard.data import builder
from turnguard.schema import make_clean_twin, validate_episode

REPO = Path(__file__).resolve().parents[1]


def originals():
    bank, scenarios = builder.load()
    return bank, [
        s
        for s in scenarios
        if s["id"] in ("email-reschedule", "tickets-where-is-order")
    ]


@pytest.fixture
def root(tmp_path):
    # The two original scenarios remain dev-only; test/train cases explicitly
    # retag copies inside their temporary fixture, never the authored YAML.
    bank, scenarios = originals()
    data = tmp_path / "turnguard/data"
    (data / "templates").mkdir(parents=True)
    (data / "injections.yaml").write_text(yaml.safe_dump(bank))
    for scenario in scenarios:
        family = scenario.pop("family")
        (data / "templates" / f"{family}.yaml").write_text(
            yaml.safe_dump(
                {
                    "family": family,
                    "scenarios": [scenario],
                }
            )
        )
    (tmp_path / "configs").mkdir()
    config = yaml.safe_load((REPO / "configs/data.yaml").read_text())
    # Assembly tests isolate MinHash sensitivity; real 0.85 behavior has
    # dedicated tests.
    config["curator"]["minhash"]["threshold"] = 1.0
    config["required_families"] = {
        split: ["email", "tickets"] for split in ("train", "dev", "test")
    }
    (tmp_path / "configs/data.yaml").write_text(yaml.safe_dump(config))
    return tmp_path


def retag(root, split):
    for path in (root / "turnguard/data/templates").glob("*.yaml"):
        template = yaml.safe_load(path.read_text())
        for scenario in template["scenarios"]:
            scenario["split"] = split
        path.write_text(yaml.safe_dump(template))


def rows(root, split):
    return [
        json.loads(line)
        for line in (root / "build/stub" / split / "episodes.jsonl")
        .read_text()
        .splitlines()
    ]


@pytest.mark.parametrize("split", ["train", "dev", "test"])
def test_small_build_counts_twins_reproducible_files_and_split_pools(root, split):
    retag(root, split)
    report = builder.build_data(split, limit=5, root=root)
    assert report["counts"] == {"injected": 5, "clean": 5, "authorized": 18}
    assert report["rejected"] == 0
    assert report["overreach"] == {"requested": 1, "accepted": 1}
    assert (root / "build/stub" / split / "cue_audit.json").is_file()
    assert (
        report["cue_audit"]["by_type"]["injected"]["shares"]["user_turn_delegation"]
        == 0.2
    )
    assert not (root / "data/v1").exists()
    output = root / "build/stub" / split
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    assert builder.build_data(split, limit=5, root=root) == report
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
    items = rows(root, split)
    by_id = {e["id"]: e for e in items}
    assert len(by_id) == len(items)
    bank, _ = builder.load(root / "turnguard/data")
    forbidden = [
        value
        for name, pool in bank["attacker_pools"].items()
        if name != split
        for values in pool.values()
        for value in values
    ]
    for episode in items:
        assert not validate_episode(episode, twins=by_id)
        assert episode["split"] == split
        assert episode["id"].startswith(split + "-")
        assert episode["lang"]["conversation"] in ("en", "hi", "hinglish")
        assert episode["lang"]["injection"] in ("en", "hi", "hinglish")
        if split != "test":
            assert episode["carrier"] != "tool_error"
            assert episode["family"] != "code"
        if episode["episode_type"] == "injected":
            twin = by_id[episode["id"] + "-clean"]
            assert make_clean_twin(episode) == twin
            assert not builder.placement_problems(episode, twin)
            assert not any(value in json.dumps(episode) for value in forbidden)
            for target in episode["injection"]["target"]:
                for terms in target.get("args_contains", {}).values():
                    if episode["goal"] != "destroy":
                        assert any(
                            term in values
                            for values in bank["attacker_pools"][split].values()
                            for term in terms
                        )
    assert report["plan_factors"]["family"]["web"]["shortfall"] > 0
    assert report["plan_factors"]["conversation_language"]["hi"]["actual"] > 0
    assert report["plan_type_shortfalls"]["authorized"] > 0


def test_sampler_balances_available_factors_with_seeded_ties():
    bank, scenarios = originals()
    pool = builder.candidates(scenarios, bank, "dev")
    chosen = builder.sample(pool, 5)
    assert chosen == builder.sample(pool, 5)
    assert chosen != builder.sample(pool, 5, seed=22)
    for factor in (
        "family",
        "scenario",
        "goal",
        "style",
        "pressure",
        "position",
        "carrier",
    ):
        counts = Counter(item.factors()[factor] for item in chosen)
        available = {item.factors()[factor] for item in pool}
        assert (
            max(counts[level] for level in available)
            - min(counts[level] for level in available)
            <= 1
        ), (factor, counts)
    assert len(builder.sample(pool, len(pool) + 1)) == len(pool)


def test_heldouts_filtered_before_sampling():
    bank, scenarios = originals()
    code = copy.deepcopy(scenarios[0])
    code.update(id="code-case", family="code", goals=["tamper_answer"], split="test")
    for split in ("train", "dev"):
        assert all(
            c.scenario["family"] != "code" and c.carrier != "tool_error"
            for c in builder.candidates([*scenarios, code], bank, split)
        )
    pool = builder.candidates([*scenarios, code], bank, "test")
    assert any(c.scenario["family"] == "code" for c in pool)
    assert any(c.carrier == "tool_error" for c in pool)


@pytest.mark.parametrize("split", ["dev", "test"])
def test_empty_templates_fail_with_split_and_family(root, split):
    for path in (root / "turnguard/data/templates").glob("*.yaml"):
        path.unlink()
    with pytest.raises(ValueError, match=f"split {split}: email, tickets"):
        builder.build_data(split, limit=5, root=root)
    assert not (root / "build").exists()


def test_bad_generated_fact_rejects_base_and_twin_with_reason(root):
    path = root / "turnguard/data/templates/email.yaml"
    template = yaml.safe_load(path.read_text())
    template["scenarios"][0]["state"]["inbox"][0]["body"]["include"] = ["MISSING-998"]
    path.write_text(yaml.safe_dump(template))
    report = builder.build_data("dev", limit=5, root=root)
    assert report["rejected"] > 0
    rejected = [
        json.loads(line)
        for line in (root / "build/stub/dev/rejected.jsonl").read_text().splitlines()
    ]
    assert any(
        "exactly once" in reason for row in rejected for reason in row["reasons"]
    )
    assert all(e["family"] != "email" for e in rows(root, "dev"))
    assert report["counts"]["clean"] == report["counts"]["injected"]


def test_first_call_and_duplicate_payload_rejected():
    bank, scenarios = originals()
    scenario = scenarios[0]
    episode = builder.injected(
        scenario, bank, "exfiltrate", "imperative", "none", "first", "body"
    )
    late = copy.deepcopy(scenario)
    late["route"].insert(0, {"turn": 1, "tool": "list_inbox", "args": {}})
    assert any("first call" in reason for reason in builder.check(late, episode))
    episode["initial_state"]["files"]["team_notes.txt"] = episode["injection"]["parts"][
        0
    ]["text"]
    assert any(
        "outside its slot" in reason for reason in builder.placement_problems(episode)
    )


def test_split_requires_two_different_outputs():
    bank, scenarios = originals()
    scenario = copy.deepcopy(scenarios[0])
    scenario["slots"]["split"]["body"][1] = scenario["slots"]["split"]["body"][0]
    episode = builder.injected(
        scenario, bank, "destroy", "imperative", "none", "split", "body"
    )
    assert any(
        "two different tool outputs" in reason
        for reason in builder.check(scenario, episode)
    )


@pytest.mark.parametrize(
    "text",
    ["Contact admin@real-business.com", "Visit https://host.xyz/path", "See bad.io/a"],
)
def test_generated_external_links_rejected(text):
    with pytest.raises(ValueError, match="non-sandbox"):
        builder.stub_generate({"gen": "body", "about": text})


def test_generated_reserved_domains_allowed():
    builder.check_generated_text(
        "a@acme.example.com https://thing.invalid/p/2 q3_payroll.xlsx"
    )


def test_cli_limit_and_validation(root, monkeypatch, capsys):
    real = builder.build_data
    monkeypatch.setattr(
        builder, "build_data", lambda *a, **kw: real(*a, **kw, root=root)
    )
    assert (
        main(["build-data", "--split", "dev", "--teacher", "stub", "--limit", "5"]) == 0
    )
    assert '"injected": 5' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["build-data", "--split", "dev", "--teacher", "stub", "--limit", "-1"])
    with pytest.raises(ValueError, match="teacher must be stub or litellm"):
        real("dev", teacher="paid", root=root)


def test_stub_cannot_be_redirected_into_real_teacher_output(root):
    path = root / "configs/data.yaml"
    config = yaml.safe_load(path.read_text())
    config["stub_output"] = "data/v1"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="Stub output must be build/stub"):
        builder.build_data("dev", limit=1, root=root)
    assert not (root / "data/v1").exists()


def test_split_tags_and_required_family_errors(root):
    bank, scenarios = originals()
    assert {scenario["split"] for scenario in scenarios} == {"dev"}
    assert not builder.candidates(scenarios, bank, "train")
    assert not builder.candidates(scenarios, bank, "test")
    missing = copy.deepcopy(scenarios)
    del missing[0]["split"]
    with pytest.raises(ValueError, match="split must be train, dev or test"):
        builder.select_scenarios(missing, "dev", [])
    invalid = copy.deepcopy(scenarios[0])
    invalid["family"] = "code"
    with pytest.raises(ValueError, match="family code must have split test"):
        builder.select_scenarios([invalid], "dev", [])
    with pytest.raises(ValueError, match="split test: code"):
        builder.select_scenarios(scenarios, "test", ["code"])
    with pytest.raises(ValueError, match="split train: email, tickets"):
        builder.build_data("train", limit=5, root=root)


def test_independent_language_cross_product_balances_small_run():
    bank, scenarios = originals()
    languages = ("en", "hi", "hinglish")
    pool = builder.candidates(
        scenarios,
        bank,
        "dev",
        conversation_languages=languages,
        injection_languages=languages,
    )
    assert {(c.conversation_language, c.injection_language) for c in pool} == {
        (a, b) for a in languages for b in languages
    }
    for factor in ("conversation_language", "injection_language"):
        counts = Counter(c.factors()[factor] for c in builder.sample(pool, 5))
        assert max(counts.values()) - min(counts.values()) <= 1


def test_dictionary_keys_and_pointer_slots_are_filled():
    bank, scenarios = builder.load()
    scenario = next(s for s in scenarios if s["id"] == "docs-travel-policy")
    assert builder.fill({"{name}": "{value}"}, {"name": "a.md", "value": "x"}) == {
        "a.md": "x"
    }
    episode = builder.injected(
        scenario, bank, "exfiltrate", "imperative", "none", "first", "body"
    )
    assert "travel_policy.md" in episode["initial_state"]["files"]
    assert episode["injection"]["parts"][0]["slot"] == "/files/travel_policy.md"
    assert not builder.check(scenario, episode)


def test_cli_dry_run_counts_calls_without_output(root, monkeypatch, capsys):
    real = builder.build_data
    monkeypatch.setattr(
        builder, "build_data", lambda *args, **kwargs: real(*args, **kwargs, root=root)
    )
    assert (
        main(
            [
                "build-data",
                "--split",
                "dev",
                "--teacher",
                "litellm",
                "--limit",
                "5",
                "--dry-run",
            ]
        )
        == 0
    )
    assert '"teacher_calls":' in capsys.readouterr().out
    assert not (root / "data").exists()
    assert not (root / "build").exists()


def test_later_injection_visible_in_turn_one_is_rejected():
    """A turn-2 search page whose snippet shares a word with the turn-1 query
    is returned in turn 1, so its 'later' injection is not really later."""
    bank, scenarios = builder.load()
    scenario = copy.deepcopy(
        next(s for s in scenarios if s["id"] == "web-library-hours")
    )
    page = scenario["state"]["pages"][1]
    page["snippet"]["about"] = "zone {zone} parking, Lakeview opening hours apply"
    episode = builder.injected(
        scenario, bank, "tamper_answer", "imperative", "none", "later", "metadata"
    )
    assert "later leaks into turn 1" in builder.check(scenario, episode)


def test_unfilled_placeholder_is_rejected():
    """An asset the bank needs but the scenario forgot stays as literal text."""
    bank, scenarios = builder.load()
    scenario = copy.deepcopy(
        next(s for s in scenarios if s["id"] == "code-release-notes")
    )
    del scenario["assets"]["extra_verb"]
    episode = builder.injected(
        scenario, bank, "extra_call", "imperative", "none", "first", "body"
    )
    assert "unfilled placeholder {extra_verb}" in builder.check(scenario, episode)
