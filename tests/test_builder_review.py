"""C2 configuration, mechanical slot conversion and authorized coverage checks."""

import copy
import json

import pytest
import yaml
from test_builder import root as _root_fixture

from turnguard.data import builder
from turnguard.schema import make_clean_twin

builder_root = _root_fixture


def test_every_error_slot_is_keyed_to_its_turns_first_route_call():
    _, scenarios = builder.load()
    seen = 0
    for scenario in scenarios:
        for position, carriers in scenario["slots"].items():
            if "tool_error" not in carriers:
                continue
            slot = carriers["tool_error"]
            route = next(
                step
                for step in scenario["route"]
                if step["turn"] == (1 if position == "first" else 2)
            )
            assert slot["tool"] == route["tool"]
            assert slot["when"] == route["args"]
            assert "on_call" not in slot
            seen += 1
    assert seen == 2 * len(scenarios)


def test_later_error_survives_repeated_reads_of_first_item():
    bank, scenarios = builder.load()
    scenario = next(s for s in scenarios if s["id"] == "email-interview-slot")
    episode = builder.injected(
        scenario,
        bank,
        "exfiltrate",
        "imperative",
        "snag",
        "later",
        "tool_error",
        split="test",
    )
    rule = episode["initial_state"]["tool_errors"]["read_email"][0]
    assert rule["when"] == {"id": scenario["vars"]["later_msg"]}
    assert not builder.check(scenario, episode)
    assert not builder.check(scenario, make_clean_twin(episode))
    from turnguard.tools import ToolEnvironment

    environment = ToolEnvironment(
        "email", episode["tools"], copy.deepcopy(episode["initial_state"])
    )
    first = next(
        step
        for step in builder.fill(scenario["route"], scenario["vars"])
        if step["turn"] == 1
    )
    for step in range(1, 4):
        assert environment.execute(first["tool"], first["args"], step=step, turn=1).ok
    delivered = environment.execute("read_email", rule["when"], step=4, turn=2)
    assert delivered.error == rule["message"]
    assert environment.execute("read_email", rule["when"], step=5, turn=2).ok


def test_authorized_carrier_expansion_is_complete_and_deduplicated():
    _, scenarios = builder.load()
    for scenario in scenarios:
        variants = builder.authorized_variants(scenario)
        expected = {
            (index, carrier)
            for index, variant in enumerate(scenario["authorized"])
            for carrier in scenario["slots"][variant["position"]]
            if carrier in ("body", "metadata", "quoted")
        }
        assert {(index, variant["carrier"]) for index, variant in variants} == expected
        assert all(variant["carrier"] != "tool_error" for _, variant in variants)
    scenario = copy.deepcopy(scenarios[0])
    before = builder.authorized_variants(scenario)
    scenario["authorized"].append(copy.deepcopy(scenario["authorized"][0]))
    assert builder.authorized_variants(scenario) == before


def test_train_authorized_budget_reaches_five_hundred_with_all_axes():
    report = builder.build_data("train", limit=5, dry_run=True)
    assert report["counts"]["authorized"] >= 500
    coverage = report["authorized_coverage"]
    assert set(coverage["by_carrier"]) == {"body", "metadata", "quoted"}
    assert set(coverage["by_language"]) == {"en", "hi", "hinglish"}
    assert set(coverage["by_filler_variant"]) == {"0", "1", "2"}
    assert sum(coverage["by_carrier"].values()) == report["counts"]["authorized"]
    assert report["teacher_estimate"]["calls_by_task"]["equivalence"] > 0
    assert report["teacher_estimate"]["calls_by_task"]["backtranslation"] > 0


@pytest.mark.parametrize("share", [0, 0.25])
def test_overreach_config_changes_quota_and_preserves_twins(builder_root, share):
    path = builder_root / "configs/data.yaml"
    config = yaml.safe_load(path.read_text())
    config["overreach_share"] = share
    path.write_text(yaml.safe_dump(config))
    report = builder.build_data("dev", limit=5, root=builder_root)
    requested = round(5 * share)
    assert report["overreach"] == {"requested": requested, "accepted": requested}
    rows = [
        json.loads(line)
        for line in (builder_root / "build/stub/dev/episodes.jsonl")
        .read_text()
        .splitlines()
    ]
    index = {e["id"]: e for e in rows}
    for episode in rows:
        if episode["episode_type"] == "injected":
            assert index[episode["id"] + "-clean"] == make_clean_twin(episode)


@pytest.mark.parametrize("share", [-0.1, 1.1, True, "0.4", float("nan")])
def test_overreach_config_rejects_invalid_values(builder_root, share):
    path = builder_root / "configs/data.yaml"
    config = yaml.safe_load(path.read_text())
    config["overreach_share"] = share
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="overreach_share"):
        builder.build_data("dev", limit=5, root=builder_root, dry_run=True)
