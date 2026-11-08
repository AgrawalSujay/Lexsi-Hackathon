"""C3 bank content checks: disjoint prose, balanced cues and real call tampering."""

import itertools
import json
import random
import re
from collections import Counter

import pytest

from turnguard.data import builder
from turnguard.data.cues import cue_audit, tld_class
from turnguard.harness import run_episodes
from turnguard.models import StubGenerator
from turnguard.schema import make_clean_twin

SPLITS = ("train", "dev", "test")


def composed_texts(bank: dict, split: str) -> list[str]:
    """Compare authored prose, excluding scenario/pool substitutions."""
    texts = []
    for goal, style in itertools.product(
        bank["goals"].values(), bank["styles"].values()
    ):
        names = {"ask": goal["ask"][split]}
        names["short"] = builder.fill(style["short"][split], names)
        texts += style["long"][split] + [style["short"][split]]
        texts += builder.fill(style["long"][split], names)
        texts += builder.fill(bank["pressure"]["snag"]["content"][split], names)
        texts += [
            text for pair in builder.fill(bank["split"][split], names) for text in pair
        ]
        if split == "test":
            texts += [
                builder.fill(pair[key], names)
                for pair in bank["pressure"]["snag"]["tool_error"]
                for key in ("error", "inject")
            ]
    return texts


def fivegrams(texts: list[str]) -> set[tuple[str, ...]]:
    grams = set()
    for text in texts:
        # Keep placeholders as atomic tokens: shared scaffolding still counts.
        words = re.findall(r"\{\w+\}|[a-z]+", text.casefold())
        grams.update(tuple(words[i : i + 5]) for i in range(len(words) - 4))
    return grams


def test_split_text_counts_and_no_shared_five_word_sequences():
    bank, _ = builder.load()
    assert set(bank["styles"]) == set(builder.STYLES)
    for style in bank["styles"].values():
        assert {split: len(values) for split, values in style["long"].items()} == {
            "train": 6,
            "dev": 4,
            "test": 6,
        }
        assert set(style["short"]) == set(SPLITS)
    for split in SPLITS:
        assert len(bank["pressure"]["snag"]["content"][split]) == 4
        assert len(bank["split"][split]) == 4
        assert all(len(pair) == 2 for pair in bank["split"][split])
    assert len(bank["pressure"]["snag"]["tool_error"]) == 4
    grams = {split: fivegrams(composed_texts(bank, split)) for split in SPLITS}
    for left, right in itertools.combinations(SPLITS, 2):
        assert not (grams[left] & grams[right]), (
            left,
            right,
            sorted(grams[left] & grams[right]),
        )


def test_addressing_is_balanced_with_protocol_and_quiet_exceptions():
    bank, _ = builder.load()
    for split in SPLITS:
        texts = [
            text for style in bank["styles"].values() for text in style["long"][split]
        ]
        addressed = sum(bool(re.search(r"\bassistant\b", text, re.I)) for text in texts)
        assert addressed / len(texts) == 0.5
        assert all(
            "Assistant" in text
            for text in bank["styles"]["protocol_spoof"]["long"][split]
        )
        assert all(
            "Assistant" not in text
            for text in bank["styles"]["quiet_fyi"]["long"][split]
        )


def test_pools_are_disjoint_reserved_and_evenly_mix_domain_classes():
    bank, _ = builder.load()
    for split, pool in bank["attacker_pools"].items():
        for key in ("addr", "url"):
            counts = Counter(tld_class(value) for value in pool[key])
            assert counts[".invalid"] == counts["example.*"]
            assert not counts["other"]
        for other in set(SPLITS) - {split}:
            for key in ("addr", "url", "user"):
                assert not (set(pool[key]) & set(bank["attacker_pools"][other][key]))
    flipped = builder.attacker_pool(bank, "test", "cueflip")
    assert all(
        tld_class(v) == "example.*" for key in ("addr", "url") for v in flipped[key]
    )


@pytest.mark.parametrize("split", SPLITS)
def test_snag_short_composition_is_a_standalone_clause_for_every_style(split):
    bank, _ = builder.load()
    for style in bank["styles"].values():
        short = style["short"][split]
        assert not short.startswith(("to ", "as "))
    for text in composed_texts(bank, split):
        assert not re.search(r"\bto\s+(?:to|as)\b", text, re.I), text
        assert ".." not in text


@pytest.mark.parametrize(
    "pressure,position,carrier",
    [
        ("none", "first", "body"),
        ("snag", "first", "metadata"),
        ("snag", "later", "quoted"),
        ("none", "split", "body"),
        ("snag", "first", "tool_error"),
    ],
)
def test_quiet_fyi_stays_unaddressed_and_protocol_spoof_stays_protocol_like(
    pressure, position, carrier
):
    bank, scenarios = builder.load()
    scenario = next(s for s in scenarios if s["id"] == "email-interview-slot")
    for style in ("quiet_fyi", "protocol_spoof"):
        episode = builder.injected(
            scenario,
            bank,
            "exfiltrate",
            style,
            pressure,
            position,
            carrier,
            split="test",
        )
        texts = [part["text"] for part in episode["injection"]["parts"]]
        if style == "quiet_fyi":
            assert not any(re.search(r"\bassistant\b", text, re.I) for text in texts)
        else:
            assert any(
                re.search(r"(?m)^(TOOL RESULT \(|User:|FINAL:)", text) for text in texts
            )
        assert not builder.check(scenario, episode)
        assert not builder.placement_problems(episode, make_clean_twin(episode))


def test_tool_error_prose_is_not_available_outside_test():
    bank, scenarios = builder.load()
    scenario = next(s for s in scenarios if s["id"] == "email-reschedule")
    assert (
        builder.injected(
            scenario,
            bank,
            "exfiltrate",
            "imperative",
            "snag",
            "first",
            "tool_error",
            split="dev",
        )
        is None
    )


@pytest.mark.parametrize(
    "family", ["email", "docs", "tickets", "web", "calendar", "code"]
)
def test_redirect_changes_one_argument_on_the_legitimate_call_not_an_extra_call(family):
    bank, scenarios = builder.load()
    scenario = next(
        s
        for s in scenarios
        if s["family"] == family
        and builder.redirect_route(s, bank, "first") is not None
    )
    episode = builder.injected(
        scenario,
        bank,
        "redirect",
        "quiet_fyi",
        "none",
        "first",
        next(c for c in scenario["slots"]["first"] if c != "tool_error"),
        split=scenario["split"],
    )
    redirected = episode["meta"]["redirect"]
    field = redirected["field"]
    assert redirected["args"][field] != redirected["original_args"][field]
    assert {k: v for k, v in redirected["args"].items() if k != field} == {
        k: v for k, v in redirected["original_args"].items() if k != field
    }
    good = [
        json.loads(text[8:])
        for text in builder.script(scenario, episode, False)
        if text.startswith("ACTION: ")
    ]
    bad = [
        json.loads(text[8:])
        for text in builder.script(scenario, episode, True)
        if text.startswith("ACTION: ")
    ]
    assert len(good) == len(bad)
    assert sum(a != b for a, b in zip(good, bad, strict=True)) == 1
    record = run_episodes(
        [episode],
        StubGenerator({episode["id"]: builder.script(scenario, episode, True)}),
    )[0]
    assert record.result["tag"] == "unsafe"
    assert not builder.check(scenario, episode)
    assert episode["injection"]["target"][0]["args_contains"][field]


def test_redirect_excludes_positions_after_the_users_action_is_over():
    bank, scenarios = builder.load()
    scenario = next(s for s in scenarios if s["id"] == "email-reschedule")
    assert (
        builder.injected(
            scenario, bank, "redirect", "imperative", "none", "later", "body"
        )
        is None
    )
    candidates = builder.candidates([scenario], bank, "dev")
    assert any(item.goal == "redirect" for item in candidates)
    assert all(
        item.position == "first" for item in candidates if item.goal == "redirect"
    )


def test_split_specific_rendering_does_not_fall_back_to_another_splits_text():
    bank, scenarios = builder.load()
    scenario = next(s for s in scenarios if s["id"] == "email-reschedule")
    for split in SPLITS:
        episode = builder.injected(
            scenario,
            bank,
            "exfiltrate",
            "imperative",
            "none",
            "first",
            "body",
            split=split,
        )
        assert (
            bank["styles"]["imperative"]["long"][split][0].split("{ask}")[1]
            in episode["injection"]["parts"][0]["text"]
        )
    with pytest.raises(ValueError, match="missing split"):
        builder.split_text({"train": ["training only"]}, "test")


def test_mixed_pools_show_both_tlds_in_the_observed_audit():
    bank, scenarios = builder.load()
    scenario = next(s for s in scenarios if s["id"] == "email-interview-slot")

    class Cycle(random.Random):
        def __init__(self):
            super().__init__(1234)
            self.counts = Counter()

        def choice(self, values):
            key = json.dumps(values, sort_keys=True)
            item = values[self.counts[key] % len(values)]
            self.counts[key] += 1
            return item

    rng = Cycle()
    episodes = [
        builder.injected(
            scenario,
            bank,
            "exfiltrate",
            "imperative",
            "none",
            "first",
            "body",
            split="test",
            rng=rng,
        )
        for _ in range(len(bank["attacker_pools"]["test"]["addr"]))
    ]
    shares = cue_audit(episodes)["by_type"]["injected"]["tld_shares"]
    assert shares[".invalid"] == shares["example.*"] == 0.5


def test_redirect_does_not_anticipate_an_unseen_future_user_request():
    bank, scenarios = builder.load()
    for name in ("calendar-sprint-retro", "code-dependency-bump"):
        scenario = next(s for s in scenarios if s["id"] == name)
        assert builder.redirect_route(scenario, bank, "first") is None
        assert builder.redirect_route(scenario, bank, "later") is not None
