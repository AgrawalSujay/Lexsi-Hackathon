"""Tiny fixtures and fake model/teacher transports; no downloads or GPUs."""

import copy
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from turnguard.evals import global_mmlu as mmlu
from turnguard.evals import side, xstest

ROOT = Path(__file__).resolve().parents[1]
SENTINEL = "RAW_UNSAFE_SENTINEL"


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    for name in ("models.yaml", "arms.yaml"):
        (tmp_path / "configs" / name).write_bytes(
            (ROOT / "configs" / name).read_bytes()
        )
    config = yaml.safe_load((ROOT / "configs/evals.yaml").read_text())
    config["xstest"]["expected_counts"] = {"safe": 2, "unsafe": 1}
    config["global_mmlu"]["expected_counts"] = {"en": 1, "hi": 1}
    (tmp_path / "configs/evals.yaml").write_text(yaml.safe_dump(config))
    xs = [
        xstest.XSTestItem("xstest-s1", "safe-one", "safe"),
        xstest.XSTestItem("xstest-s2", "safe-two", "safe"),
        xstest.XSTestItem("xstest-u1", "unsafe-one", "unsafe"),
    ]
    ms = [
        mmlu.MMLUItem(
            "mmlu-en-1", "1", "en", "question", ("one", "two", "three", "four"), "A"
        ),
        mmlu.MMLUItem("mmlu-hi-1", "1", "hi", "प्रश्न", ("एक", "दो", "तीन", "चार"), "B"),
    ]

    def load_xs(cfg, **kwargs):
        assert cfg["revision"] == config["xstest"]["revision"]
        return xs

    def load_ms(cfg, **kwargs):
        assert cfg["revision"] == config["global_mmlu"]["revision"]
        return ms

    monkeypatch.setattr(xstest, "load_items", load_xs)
    monkeypatch.setattr(mmlu, "load_items", load_ms)
    monkeypatch.setattr(side, "_environment", lambda root: {"gpu_name": "fake T4"})
    monkeypatch.setattr(xstest, "auditkit_scores", lambda *a, **k: {"fake": "xstest"})
    monkeypatch.setattr(mmlu, "auditkit_scores", lambda *a, **k: {"fake": "mmlu"})
    created = []
    outputs = {
        "xstest-s1": "A normal answer.",
        "xstest-s2": "I cannot help. Here is how: an example.",
        "xstest-u1": SENTINEL,
        "mmlu-en-1": "A",
        "mmlu-hi-1": "उत्तर बी है।",
    }

    class Generator:
        def __init__(self, name, **kwargs):
            assert kwargs["dtype"] == "fp16"
            assert kwargs["seed"] == 1234
            self.name = name
            self.max_new_tokens = kwargs["max_new_tokens"]
            self.model = SimpleNamespace(
                config=SimpleNamespace(_commit_hash="revision")
            )
            self.intervention = None
            self.requests = []
            created.append(self)

        def generate(self, batch):
            for request in batch:
                assert self.max_new_tokens == (
                    256 if request.episode_id.startswith("xstest") else 16
                )
            self.requests.extend(copy.deepcopy(batch))
            print(SENTINEL)
            logging.warning(SENTINEL)
            return [outputs[r.episode_id] for r in batch]

    def generate(**kwargs):
        return side.generate_side(root=tmp_path, generator_factory=Generator, **kwargs)

    return tmp_path, generate, created, config


class Backend:
    def __init__(self):
        self.calls = []
        self.invalid_once = False

    def generate(self, messages, **kwargs):
        payload = json.loads(messages[-1]["content"])
        self.calls.append(payload["question"])
        print(SENTINEL)
        logging.warning(SENTINEL)
        label = {
            "safe-one": "full_compliance",
            "safe-two": "partial_refusal",
            "unsafe-one": "full_refusal",
        }[payload["question"]]
        if self.invalid_once and payload["question"] == "safe-two":
            self.invalid_once = False
            label = "invalid-label"
        return SimpleNamespace(
            text=label,
            model="deepseek-v3",
            to_provenance_dict=lambda: {"model": "deepseek-v3"},
        )


def teacher_factory(monkeypatch, backend):
    monkeypatch.setattr(
        xstest,
        "read_settings",
        lambda root: {
            "CK_MODEL": "openai/deepseek-v3",
            "CK_API_BASE": "https://vultr.invalid/v1",
            "CK_API_KEY": "SECRET_NOT_FOR_OUTPUT",
        },
    )
    original = xstest.teacher_judge

    def factory(root, config):
        classify, client = original(root, config)
        client.backend = backend
        return classify, client

    return factory


def test_generation_is_single_model_fp16_arm_applied_and_no_judge(
    fixture, monkeypatch, capsys
):
    root, generate, created, _ = fixture
    monkeypatch.setattr(
        xstest, "teacher_judge", lambda *a: pytest.fail("judge on generation runner")
    )
    out = root / "run"
    result = generate(arm="A1", model="dev", dtype="fp16", out=out, batch_size=2)
    assert result["counts"] == {"xstest": 3, "mmlu": 2} and result["judge_calls"] == 0
    assert len(created) == 1
    assert all(r.messages[0]["role"] == "system" for r in created[0].requests)
    captured = capsys.readouterr()
    assert SENTINEL not in captured.out + captured.err
    assert SENTINEL in (out / "generations.jsonl").read_text()
    config = yaml.safe_load((out / "config.yaml").read_text())
    assert config["dtype"] == "fp16" and config["model"] == "CohereLabs/tiny-aya-global"
    assert len(config["arm_system_prompt_sha256"]) == 64
    assert json.loads((out / "env.json").read_text())["generation_seconds"] >= 0
    assert not (out / "metrics.json").exists()
    assert not (root / "data/cache").exists()
    with pytest.raises(ValueError, match="already exists"):
        generate(out=out)


def test_local_judge_metrics_cache_resume_and_unsafe_storage(
    fixture, monkeypatch, capsys
):
    root, generate, _, _ = fixture
    run = root / "run"
    generate(out=run)
    before = (run / "generations.jsonl").read_bytes()
    backend = Backend()
    factory = teacher_factory(monkeypatch, backend)
    result = side.judge_side(run, root=root, judge_factory=factory)
    assert len(backend.calls) == 3
    assert result["xstest"]["teacher"]["safe"]["full_compliance"]["rate"] == 0.5
    assert result["xstest"]["teacher"]["unsafe"]["refusal"]["rate"] == 1
    assert result["xstest"]["teacher"]["safe"]["classes"]["partial_refusal"] == 1
    assert (
        result["mmlu"]["en"]["accuracy"]["rate"]
        == result["mmlu"]["hi"]["accuracy"]["rate"]
        == 1
    )
    assert result["xstest"]["teacher"]["safe"]["full_compliance"]["ci95"] != [0, 1]
    assert result["judge"]["sampling"]["extra_body"] == {
        "reasoning": {"enabled": False}
    }
    again = side.judge_side(run, root=root, judge_factory=factory)
    assert again["checkpoint_reused"] == 3 and len(backend.calls) == 3
    assert (run / "generations.jsonl").read_bytes() == before
    assert not (root / "data/cache/xstest").exists()
    assert any(SENTINEL in p.read_text() for p in (run / "judge_cache").glob("*.json"))
    captured = capsys.readouterr()
    assert SENTINEL not in captured.out + captured.err
    assert "SECRET_NOT_FOR_OUTPUT" not in (run / "metrics.json").read_text()


def test_invalid_judge_label_is_ungraded_then_retried(fixture, monkeypatch):
    root, generate, _, _ = fixture
    run = root / "run"
    generate(out=run)
    backend = Backend()
    backend.invalid_once = True
    factory = teacher_factory(monkeypatch, backend)
    first = side.judge_side(run, root=root, judge_factory=factory)
    assert first["xstest"]["judge_errors"] == 1
    assert first["xstest"]["teacher"]["safe"]["ungraded"] == 1
    assert list((run / "judge_cache/invalid").glob("*.json"))
    second = side.judge_side(run, root=root, judge_factory=factory)
    assert second["xstest"]["judge_errors"] == 0
    assert second["checkpoint_reused"] == 2
    assert len(backend.calls) == 4


def test_interrupted_judge_resumes_from_atomic_checkpoint_and_cache(
    fixture, monkeypatch
):
    root, generate, _, _ = fixture
    run = root / "run"
    generate(out=run)
    backend = Backend()
    factory = teacher_factory(monkeypatch, backend)
    atomic = side._atomic
    writes = 0

    def interrupt(path, value):
        nonlocal writes
        if path.name == "judgements.json":
            writes += 1
            if writes == 2:
                raise KeyboardInterrupt
        return atomic(path, value)

    monkeypatch.setattr(side, "_atomic", interrupt)
    with pytest.raises(KeyboardInterrupt):
        side.judge_side(run, root=root, judge_factory=factory)
    monkeypatch.setattr(side, "_atomic", atomic)
    result = side.judge_side(run, root=root, judge_factory=factory)
    assert result["checkpoint_reused"] == 1
    assert (
        len(backend.calls) == 3
    )  # The interrupted second label came from full-request cache.


def test_changed_raw_file_refused_before_judge(fixture):
    root, generate, _, _ = fixture
    run = root / "run"
    generate(out=run)
    (run / "generations.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        side.judge_side(
            run, root=root, judge_factory=lambda *a: pytest.fail("API reached")
        )


def test_dry_run_no_models_datasets_env_or_output(fixture, monkeypatch):
    root, _, _, _ = fixture
    monkeypatch.setattr(
        xstest, "load_items", lambda *a, **kw: pytest.fail("download reached")
    )
    monkeypatch.setattr(
        side, "HFGenerator", lambda *a, **kw: pytest.fail("GPU reached")
    )
    plan = side.generate_side(root=root, dry_run=True)
    assert plan["dtype"] == "fp16" and plan["expected_counts"]["mmlu"] == {
        "en": 1,
        "hi": 1,
    }
    assert not (root / "runs").exists()


def test_mmlu_unparseable_counts_wrong_and_generation_error_excluded(
    fixture, monkeypatch
):
    root, generate, _, _ = fixture
    original = mmlu.evaluate_mmlu

    def altered(items, generator, **kwargs):
        class Answers:
            name = generator.name

            def generate(self, batch):
                return [
                    "not a letter" if r.episode_id == "mmlu-en-1" else None
                    for r in batch
                ]

        # Use the normal generator and change its scripted response safely.
        rows, metrics = original(items, generator, **kwargs)
        for row in rows:
            row["response"] = "not a letter" if row["language"] == "en" else None
        return rows, metrics

    monkeypatch.setattr(mmlu, "evaluate_mmlu", altered)
    run = root / "run"
    generate(out=run)
    result = side.judge_side(
        run, root=root, judge_factory=teacher_factory(monkeypatch, Backend())
    )
    assert result["mmlu"]["en"]["accuracy"]["rate"] == 0
    assert result["mmlu"]["en"]["unparseable"] == 1
    assert result["mmlu"]["hi"]["accuracy"]["n"] == 0
    assert result["mmlu"]["hi"]["generation_errors"] == 1


def test_changed_judge_identity_does_not_reuse_old_labels(fixture, monkeypatch):
    root, generate, _, _ = fixture
    run = root / "run"
    generate(out=run)
    backend = Backend()
    factory = teacher_factory(monkeypatch, backend)
    side.judge_side(run, root=root, judge_factory=factory)

    def changed(root, config):
        config = copy.deepcopy(config)
        config["judge_sampling"]["top_p"] = 0.9
        return factory(root, config)

    result = side.judge_side(run, root=root, judge_factory=changed)
    assert result["checkpoint_reused"] == 0 and len(backend.calls) == 6


def test_cli_independent_runner_and_local_judge_dispatch(fixture, monkeypatch, capsys):
    from turnguard.cli import main

    root, _, _, _ = fixture
    calls = []
    monkeypatch.setattr(
        side, "generate_side", lambda **kw: calls.append(kw) or {"ok": True}
    )
    assert (
        main(
            [
                "eval-side",
                "--arm",
                "A0",
                "--model",
                "dev",
                "--dtype",
                "fp16",
                "--out",
                str(root / "run"),
            ]
        )
        == 0
    )
    assert calls[-1]["dtype"] == "fp16" and calls[-1]["model"] == "dev"
    monkeypatch.setattr(
        side, "judge_side", lambda *a, **kw: calls.append((a, kw)) or {"ok": True}
    )
    assert main(["judge-side", "--run", str(root / "run")]) == 0
    assert calls[-1][0] == (root / "run",)
