"""C4 source integrity, address validation and per-type limit regressions."""

import copy
import hashlib
import json

import pytest

from turnguard.data import builder


def test_all_authorized_targets_are_unchanged_from_a2d9921():
    _, scenarios = builder.load()
    targets = {s["id"]: [v["target"] for v in s["authorized"]] for s in scenarios}
    assert (
        hashlib.sha256(json.dumps(targets, sort_keys=True).encode()).hexdigest()
        == "c9a196ca18e46a019394592be965dbeb843bac7fc965484831721f7062b1c7d8"
    )


def test_every_template_authorized_instruction_passes_source_and_rendered_guards():
    _, scenarios = builder.load()
    assert len(scenarios) == 35
    for scenario in scenarios:
        for index, variant in builder.authorized_variants(scenario):
            episode = builder.authorized(
                scenario, variant, index, split=scenario["split"]
            )
            assert not builder.check(scenario, episode), (scenario["id"], index)


def test_check_rejects_a_missing_literal_authorized_target_term():
    _, scenarios = builder.load()
    scenario = next(s for s in scenarios if s["id"] == "email-reschedule")
    episode = builder.authorized(scenario, scenario["authorized"][0], 0)
    episode["authorized"]["target"][0]["args_contains"]["to"] = [
        "missing-desk@acme.example.org"
    ]
    assert builder.check(scenario, episode) == [
        "authorized target term 'missing-desk@acme.example.org' not in instruction text"
    ]


def test_english_source_is_checked_before_any_teacher_call():
    _, scenarios = builder.load()
    scenario = copy.deepcopy(
        next(s for s in scenarios if s["id"] == "email-reschedule")
    )
    variant = scenario["authorized"][0]
    variant["text"] = "Please acknowledge this."

    class NoTeacherCalls:
        def __getattr__(self, name):
            raise AssertionError(f"teacher must not be called: {name}")

    with pytest.raises(ValueError, match="authorized target term"):
        builder.authorized(
            scenario,
            variant,
            0,
            teacher_client=NoTeacherCalls(),
            injection_language="hi",
        )


@pytest.mark.parametrize(
    "address",
    [
        "user@team..invalid",
        "user@team.invalid.example.org",
        "user@team.example.com.invalid",
        "user@team.invalid.invalid",
        "https://team.example.org..example.net/2",
        "team.invalid.example.org/3",
    ],
)
@pytest.mark.parametrize("where", ("bank", "template"))
def test_load_rejects_malformed_addresses_anywhere(bank_root, address, where):
    data = bank_root / "turnguard/data"
    path = (
        data / "injections.yaml"
        if where == "bank"
        else next((data / "templates").glob("*.yaml"))
    )
    with path.open("a") as stream:
        stream.write(f"\n# Contact {address}\n")
    with pytest.raises(ValueError, match="malformed address"):
        builder.load(data)


def test_generated_and_translated_addresses_cannot_append_a_second_suffix():
    with pytest.raises(ValueError, match="malformed address"):
        builder.check_generated_text("Reply to desk@school.invalid..example.org")
    builder.check_generated_text("Reply to desk@school.invalid.")


@pytest.mark.parametrize("split", ("train", "dev", "test"))
@pytest.mark.parametrize("limit", (0, 1, 5))
def test_limit_caps_each_type_and_teacher_work_is_bounded(bank_root, split, limit):
    report = builder.build_data(split, root=bank_root, limit=limit)
    assert report["counts"] == dict.fromkeys(("injected", "clean", "authorized"), limit)
    assert report["authorized_cap"] == limit
    if limit == 0:
        assert report["teacher_estimate"]["requests"] == 0


def test_lookup_codes_are_not_required_in_authorized_text():
    assert (
        builder.authorized_target_problems(
            "Please email the zone code to permits@ward9.example.org",
            [
                {
                    "args_contains": {
                        "to": ["permits@ward9.example.org"],
                        "body": ["PZ-14"],
                    }
                }
            ],
        )
        == []
    )


def test_limited_build_cue_audit_is_indicative_and_silent(bank_root):
    limited = builder.build_data("dev", root=bank_root, limit=5, dry_run=True)
    assert limited["cue_audit"]["status"] == "indicative only (limited build)"
    assert limited["cue_audit"]["warnings"] == []
    full = builder.build_data("dev", root=bank_root, dry_run=True)
    assert full["cue_audit"]["status"] == "full"
