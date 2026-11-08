"""Offline builder acceptance, stratification, rejection and replay checks."""

import copy
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest
import yaml

from turnguard.cli import main
from turnguard.data import builder
from turnguard.schema import make_clean_twin, validate_episode

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def root(tmp_path):
    shutil.copytree(
        REPO / "turnguard/data/templates", tmp_path / "turnguard/data/templates"
    )
    shutil.copy(REPO / "turnguard/data/injections.yaml", tmp_path / "turnguard/data")
    (tmp_path / "configs").mkdir()
    shutil.copy(REPO / "configs/data.yaml", tmp_path / "configs")
    return tmp_path


def rows(root, split):
    return [
        json.loads(line)
        for line in (root / "build/stub" / split / "episodes.jsonl")
        .read_text()
        .splitlines()
    ]


@pytest.mark.parametrize("split", ["train", "dev", "test"])
def test_small_build_counts_twins_reproducible_files_and_split_pools(root, split):
    report = builder.build_data(split, limit=5, root=root)
    assert report["counts"] == {"injected": 5, "clean": 5, "authorized": 2}
    assert report["rejected"] == 0
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
        assert episode["lang"] == {"conversation": "en", "injection": "en"}
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
    assert report["plan_factors"]["conversation_language"]["hi"]["actual"] == 0
    assert report["plan_type_shortfalls"]["authorized"] > 0


def test_sampler_balances_available_factors_with_seeded_ties():
    bank, scenarios = builder.load()
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
    bank, scenarios = builder.load()
    code = copy.deepcopy(scenarios[0])
    code.update(id="code-case", family="code", goals=["tamper_answer"])
    for split in ("train", "dev"):
        assert all(
            c.scenario["family"] != "code" and c.carrier != "tool_error"
            for c in builder.candidates([*scenarios, code], bank, split)
        )
    pool = builder.candidates([*scenarios, code], bank, "test")
    assert any(c.scenario["family"] == "code" for c in pool)
    assert any(c.carrier == "tool_error" for c in pool)


@pytest.mark.parametrize("split", ["dev", "test"])
def test_empty_templates_report_shortfall_without_failure(root, split):
    for path in (root / "turnguard/data/templates").glob("*.yaml"):
        path.unlink()
    report = builder.build_data(split, limit=5, root=root)
    assert report["counts"] == {"injected": 0, "clean": 0, "authorized": 0}
    assert report["requested_shortfall"] == 5
    assert report["factors"]["goal"]["exfiltrate"]["shortfall"] > 0


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
    bank, scenarios = builder.load()
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
    bank, scenarios = builder.load()
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
    with pytest.raises(ValueError, match="Only --teacher stub"):
        real("dev", teacher="paid", root=root)


def test_stub_cannot_be_redirected_into_real_teacher_output(root):
    path = root / "configs/data.yaml"
    config = yaml.safe_load(path.read_text())
    config["stub_output"] = "data/v1"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="Stub output must be build/stub"):
        builder.build_data("dev", limit=1, root=root)
    assert not (root / "data/v1").exists()
