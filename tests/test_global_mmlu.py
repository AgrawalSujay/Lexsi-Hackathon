"""CPU-only Global-MMLU tests; tiny fixtures, no datasets or model downloads."""

import copy
import hashlib
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from turnguard.cli import main
from turnguard.evals import global_mmlu as mmlu
from turnguard.metrics import rate
from turnguard.models import StubGenerator

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs/evals.yaml").read_text())["global_mmlu"]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No network in Global-MMLU tests")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(
        "turnguard.runs._environment",
        lambda root: {"packages": {}, "git_commit": "fixture"},
    )


@pytest.fixture
def rows():
    return [
        {
            "sample_id": "math/test/1",
            "subject": "math",
            "question": "What is two plus two?",
            "option_a": "Three",
            "option_b": "Four",
            "option_c": "Five",
            "option_d": "Six",
            "answer": "B",
        },
        {
            "sample_id": "math/test/2",
            "subject": "math",
            "question": "What is one plus one?",
            "option_a": "One",
            "option_b": "Three",
            "option_c": "Four",
            "option_d": "Two",
            "answer": "D",
        },
    ]


@pytest.fixture
def items(rows):
    en = mmlu.parse_items(rows, "en")
    hi_rows = copy.deepcopy(rows)
    hi_rows[0].update(
        question="दो और दो का योग क्या है?",
        option_a="तीन",
        option_b="चार",
        option_c="पाँच",
        option_d="छह",
    )
    hi_rows[1].update(
        question="एक और एक का योग क्या है?",
        option_a="एक",
        option_b="तीन",
        option_c="चार",
        option_d="दो",
    )
    return en + mmlu.parse_items(hi_rows, "hi")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("B", "B"),
        (" b ", "B"),
        ("(C).", "C"),
        ("**D**", "D"),
        ("The answer is **B**.", "B"),
        ("Answer: c", "C"),
        ("उत्तर: B", "B"),
        ("सही विकल्प (C) है।", "C"),
        ("मेरा जवाब D है।", "D"),
        ("सही उत्तर है: (बी)।", "B"),
        ("डी", "D"),
        ("यह विकल्प A है।", "A"),
        ("B सही है।", "B"),
        ("Ａ", "A"),
        ("उत्तर: Ｂ", "B"),
        ("A and B are wrong. Final answer: C", "C"),
        ("A या B? अंतिम उत्तर: डी।", "D"),
        ("<think>A or C</think>\nB", "B"),
        ("", None),
        ("I do not know.", None),
        ("Because I am uncertain.", None),
        ("A good explanation requires more information.", None),
        ("A/B", None),
        ("उत्तर: B या C", None),
        ("Answer: B or D", None),
        ("उत्तर: ए या बी", None),
        ("AB", None),
        ("E", None),
        ("उत्तर: बीमार", None),
    ],
)
def test_extract_letter_hindi_markers_and_ambiguity(text, expected):
    assert mmlu.extract_answer(text) == expected


def test_unparseable_counts_wrong_per_language_and_preserves_raw(items):
    outputs = ["Answer: B", "A or D", "सही उत्तर: बी।", "उत्तर नहीं पता"]
    generator = StubGenerator(
        {item.id: [output] for item, output in zip(items, outputs, strict=True)}
    )
    records, metrics = mmlu.evaluate_mmlu(
        items, generator, instructions=CONFIG["instructions"], batch_size=2
    )
    assert [row["response"] for row in records] == outputs
    for language in ("en", "hi"):
        group = metrics["languages"][language]
        assert group["total"] == group["graded"] == 2
        assert group["accuracy"] == rate(1, 2)
        assert group["unparseable"] == 1
        assert group["generation_errors"] == 0
    assert records[1]["correct"] is False
    assert records[1]["predicted"] is None
    assert records[0]["target"] == "B"
    assert len({item.id for item in items}) == 4
    assert (
        items[0].source_id == items[2].source_id
    )  # Language namespaces cannot collide.


def test_localized_plain_chat_and_shared_system_prompt(items):
    generator = StubGenerator({item.id: ["B"] for item in items})
    records, _ = mmlu.evaluate_mmlu(
        items,
        generator,
        instructions=CONFIG["instructions"],
        system_prompt="Exact shared defence",
    )
    for item, record in zip(items, records, strict=True):
        assert record["messages"][0] == {
            "role": "system",
            "content": "Exact shared defence",
        }
        content = record["messages"][1]["content"]
        assert content.endswith(CONFIG["instructions"][item.language])
        assert all(
            f"{letter}. {choice}" in content
            for letter, choice in zip("ABCD", item.choices, strict=True)
        )
        assert "ACTION:" not in content and "FINAL:" not in content


def test_failed_generation_ungraded_but_empty_completion_is_wrong(items):
    class Broken(StubGenerator):
        def generate(self, requests):
            raise RuntimeError("do not echo model errors")

    _, metrics = mmlu.evaluate_mmlu(
        items, Broken({}), instructions=CONFIG["instructions"]
    )
    for group in metrics["languages"].values():
        assert group["generation_errors"] == 2
        assert group["accuracy"] == rate(0, 0)
        assert group["unparseable"] == 0
    _, empty = mmlu.evaluate_mmlu(
        items,
        StubGenerator({item.id: [""] for item in items}),
        instructions=CONFIG["instructions"],
    )
    for group in empty["languages"].values():
        assert group["generation_errors"] == 0
        assert group["accuracy"] == rate(0, 2)
        assert group["unparseable"] == 2


def test_stratified_total_limit_and_full_default(items):
    chosen = mmlu.select_items(items, 3)
    assert chosen == mmlu.select_items(items, 3)
    assert [item.language for item in chosen] == ["en", "hi", "en"]
    assert mmlu.select_items(items, None) == items
    assert len(mmlu.select_items(items, 100)) == 4
    with pytest.raises(ValueError, match="positive"):
        mmlu.select_items(items, 0)


def test_loader_pins_revision_and_reads_all_items_in_both_languages(
    rows, tmp_path, monkeypatch
):
    calls = []

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return rows

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load))
    config = copy.deepcopy(CONFIG)
    config["expected_counts"] = {"en": 2, "hi": 2}
    items = mmlu.load_items(config, cache_dir=tmp_path)
    assert len(items) == 4
    assert [call[1]["name"] for call in calls] == ["en", "hi"]
    assert all(
        call[0] == "CohereLabs/Global-MMLU-Lite"
        and call[1]["split"] == "test"
        and call[1]["revision"] == "36c2fd756f19ccf13a9a96c8e53ccecc02192b8b"
        for call in calls
    )
    config["expected_counts"]["en"] = 400
    with pytest.raises(ValueError, match="en: expected 400 items, got 2"):
        mmlu.load_items(config, cache_dir=tmp_path)
    config["revision"] = "main"
    with pytest.raises(ValueError, match="immutable"):
        mmlu.load_items(config, cache_dir=tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("answer", None),
        ("answer", "AB"),
        ("option_a", ""),
        ("sample_id", ""),
        ("question", None),
    ],
)
def test_invalid_dataset_row_fails_loudly(rows, field, value):
    rows[0][field] = value
    with pytest.raises(ValueError):
        mmlu.parse_items(rows, "en")


def test_auditkit_accuracy_agrees_including_unparseable(items, tmp_path):
    outputs = ["B", "unparseable", "उत्तर बी", "उत्तर A"]
    records, metrics = mmlu.evaluate_mmlu(
        items,
        StubGenerator(
            {item.id: [text] for item, text in zip(items, outputs, strict=True)}
        ),
        instructions=CONFIG["instructions"],
    )
    audit = mmlu.auditkit_scores(records, cache_dir=tmp_path)
    for language in ("en", "hi"):
        assert (
            audit[language]["headline"]["accuracy"]
            == metrics["languages"][language]["accuracy"]["rate"]
        )
        assert len(audit[language]["predictions"]) == 2
        assert (
            audit[language]["predictions"][0]["metadata"]["scores"][0]["direction"]
            == "maximize"
        )
    assert audit["en"]["predictions"][1]["raw_output"] == "unparseable"


def test_cli_hf_generator_short_limit_config_artifacts_and_a1(
    items, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(mmlu, "load_items", lambda *a, **kw: items)
    calls, requests = [], []

    class RecordingStub(StubGenerator):
        def generate(self, batch):
            requests.extend(copy.deepcopy(batch))
            return super().generate(batch)

    def factory(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return RecordingStub({item.id: [item.answer] for item in items})

    monkeypatch.setattr("turnguard.cli.HFGenerator", factory)
    args = [
        "eval-mmlu",
        "--model",
        "dev",
        "--arm",
        "A0",
        "--limit",
        "4",
        "--root",
        str(tmp_path),
    ]
    assert main(args) == 0
    assert calls == [
        (
            "CohereLabs/tiny-aya-global",
            {"max_new_tokens": 16, "batch_size": 8, "seed": 1234},
        )
    ]
    assert all(len(request.messages) == 1 for request in requests)
    assert "Run directory:" in capsys.readouterr().out
    run = next((tmp_path / "runs").iterdir())
    assert run.name.startswith("A0__dev__mmlu__")
    assert {
        "config.yaml",
        "episodes.jsonl",
        "metrics.json",
        "env.json",
        "auditkit.json",
    } <= {path.name for path in run.iterdir()}
    config = yaml.safe_load((run / "config.yaml").read_text())
    assert config["decoding"] == {
        "do_sample": False,
        "max_new_tokens": 16,
        "chat_template": "tokenizer",
        "padding_side": "left",
    }
    assert config["global_mmlu"]["revision"] == CONFIG["revision"]
    metrics = json.loads((run / "metrics.json").read_text())
    assert all(
        group["accuracy"]["rate"] == 1 for group in metrics["languages"].values()
    )
    arms = tmp_path / "arms.yaml"
    arms.write_text('arms:\n  A1: {system_prompt: "Exact defence text"}\n')
    args[args.index("A0")] = "A1"
    args += ["--arms-config", str(arms)]
    assert main(args) == 0
    assert all(
        request.messages[0]["content"] == "Exact defence text"
        for request in requests[-4:]
    )

    a1_run = next((tmp_path / "runs").glob("A1__*"))
    a1_config = yaml.safe_load((a1_run / "config.yaml").read_text())
    assert (
        a1_config["arm_system_prompt_sha256"]
        == hashlib.sha256(b"Exact defence text").hexdigest()
    )


def test_cli_teacher_judging_rejected_before_model_load(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Should not load a model or dataset")

    monkeypatch.setattr(mmlu, "load_items", forbidden)
    monkeypatch.setattr("turnguard.cli.HFGenerator", forbidden)
    with pytest.raises(SystemExit):
        main(
            [
                "eval-mmlu",
                "--model",
                "dev",
                "--judge",
                "teacher",
                "--root",
                str(tmp_path),
            ]
        )
    assert not (tmp_path / "runs").exists()
