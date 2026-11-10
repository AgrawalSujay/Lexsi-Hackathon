"""Recorded-baseline tuning with deterministic type balance and no real models."""

import copy
import json
from contextlib import nullcontext
from pathlib import Path

import pytest
import yaml

from turnguard.arms import steer_tune as tuning
from turnguard.config import load_config
from turnguard.data.loaders import _rank, load_split
from turnguard.harness import run_episodes
from turnguard.models import StubGenerator
from turnguard.schema import EpisodeSpec

ROOT = Path(__file__).resolve().parents[1]
MODEL = "CohereLabs/aya-expanse-8b"


@pytest.fixture
def tuning_run(tmp_path):
    specs = [
        json.loads((ROOT / "data/examples" / name).read_text())
        for name in (
            "ex-email-inj-001.json",
            "ex-email-clean-001.json",
            "ex-tickets-auth-001.json",
        )
    ]
    for spec in specs:
        spec["split"] = "dev"
    dev = tmp_path / "data/v1/dev"
    dev.mkdir(parents=True)
    (dev / "episodes.jsonl").write_text("".join(json.dumps(s) + "\n" for s in specs))
    (dev / "manifest.json").write_text("{}")
    (dev / "summary.json").write_text('{"status":"complete"}')
    scripts = json.loads((ROOT / "tests/fixtures/stub_scripts.json").read_text())
    generator = StubGenerator(scripts)
    generator.name = MODEL
    records = run_episodes(specs, generator, arm="A0")
    run = tmp_path / "a0"
    run.mkdir()
    config = load_config(ROOT / "configs/models.yaml")
    config.update(
        arm="A0",
        model=MODEL,
        generator="hf",
        protocol_version="v2.4",
        system_extra=None,
        split="dev",
        episode_paths=[str(dev / "episodes.jsonl")],
    )
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    (run / "episodes.jsonl").write_text("".join(r.to_json() + "\n" for r in records))
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/models.yaml").write_bytes(
        (ROOT / "configs/models.yaml").read_bytes()
    )
    snapshot = tuning.prepare_baseline(dev, run, MODEL, n=1)
    snapshot_path = tmp_path / "baseline.json"
    snapshot_path.write_text(json.dumps(snapshot))
    return tmp_path, dev, run, snapshot_path, specs, scripts


def test_type_composition_hash_order_and_twins(tuning_run):
    _, _, _, _, values, _ = tuning_run
    specs = []
    for i in range(70):
        attack, clean, auth = copy.deepcopy(values)
        attack["id"] = f"base-{i}"
        clean.update(id=f"base-{i}-clean", twin_of=attack["id"])
        auth["id"] = f"auth-{i}"
        specs.extend(EpisodeSpec(**s) for s in (attack, clean, auth))
    chosen = tuning.tuning_subset(specs)
    assert len(chosen) == 150
    assert [s.id for s in chosen] == [
        s.id for s in tuning.tuning_subset(list(reversed(specs)))
    ]
    injected = [s for s in chosen if s.episode_type == "injected"]
    assert {s.twin_of for s in chosen if s.episode_type == "clean"} == {
        s.id for s in injected
    }
    assert len([s for s in chosen if s.episode_type == "authorized"]) == 50
    assert [s.id for s in injected] == sorted(
        (f"base-{i}" for i in range(70)), key=lambda k: (_rank(k), k)
    )[:50]
    with pytest.raises(ValueError, match="at least 71"):
        tuning.tuning_subset(specs, n=71)
    no_twins = [s for s in specs if s.episode_type != "clean"]
    with pytest.raises(ValueError, match="one clean twin"):
        tuning.tuning_subset(no_twins)


def test_baseline_regraded_in_memory_inputs_unchanged(tuning_run):
    _, dev, run, _, _, _ = tuning_run
    rows = [
        json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()
    ]
    rows[0]["result"]["tag"] = "relayed"  # old false label; recorded effect is executed
    rows[0]["result"]["grading_version"] = "old"
    (run / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    snapshot = tuning.prepare_baseline(dev, run, MODEL, n=1)
    assert snapshot["grading_version"] == "c8-v1"
    assert snapshot["relays_changed"] == 1
    assert snapshot["records"][0]["result"]["tag"] == "unsafe"
    assert {p.name: p.read_bytes() for p in run.iterdir()} == before
    assert len(snapshot["selected_ids"]) == 3


def test_changed_specs_wrong_model_and_missing_baseline_ids_refused(tuning_run):
    _, dev, run, _, _, _ = tuning_run
    with pytest.raises(ValueError, match="must match"):
        tuning.prepare_baseline(dev, run, "different-model", n=1)
    path = dev / "episodes.jsonl"
    values = [json.loads(line) for line in path.read_text().splitlines()]
    values[0]["user_turns"][0] += " changed"
    path.write_text("".join(json.dumps(v) + "\n" for v in values))
    with pytest.raises(ValueError, match="Spec hash mismatch"):
        tuning.prepare_baseline(dev, run, MODEL, n=1)


def point(layer, alpha, asr, clean, auth, ungraded=0):
    return dict(
        layer=layer,
        alpha=alpha,
        asr_incl_relay=asr,
        clean_task_success=clean,
        authorized_followed=auth,
        ungraded=ungraded,
    )


def test_predeclared_guardrails_ties_fallback_and_ungraded():
    baseline = point(0, 0, 0.5, 0.8, 0.6)
    valid = point(18, 4, 0.1, 0.76, 0.55)
    too_low_clean = point(14, 4, 0, 0.72, 0.6)
    assert tuning.pick_point([too_low_clean, valid], baseline)["layer"] == 18
    tie = point(14, 4, 0.1, 0.76, 0.55)
    assert tuning.pick_point([valid, tie], baseline)["layer"] == 14
    assert tuning.pick_point([valid, tie], baseline)["qualified"]
    fallback = tuning.pick_point(
        [point(22, 8, 0, 0.6, 0.5), point(14, 4, 0.2, 0.73, 0.6)], baseline
    )
    assert fallback["layer"] == 14 and not fallback["qualified"]
    assert (
        tuning.pick_point([point(14, 4, 0, 0.9, 0.9, 1), valid], baseline)["layer"]
        == 18
    )
    with pytest.raises(ValueError, match="fully graded"):
        tuning.pick_point([point(14, 4, 0, 0.9, 0.9, 1)], baseline)


def test_full_grid_generates_only_six_times_same_ids_never_a0(tuning_run, monkeypatch):
    root, dev, run, snapshot, values, scripts = tuning_run
    requested = []

    class ReusableStub:
        name = MODEL
        intervention = None

        def generate(self, batch):
            assert self.intervention is not None
            with self.intervention():
                requested.extend(r.episode_id for r in batch)
                return [scripts[r.episode_id][r.step - 1] for r in batch]

    pairs = root / "train-pairs"
    pairs.mkdir()
    vector = root / "steering/A2/headline.pt"

    def fake_vectors(*args, **kwargs):
        vector.parent.mkdir(parents=True, exist_ok=True)
        vector.write_bytes(b"fake")
        return {"vector": str(vector), "vector_sha256": "f" * 64}

    monkeypatch.setattr(tuning, "build_vectors", fake_vectors)
    hooks = []

    def factory(**kwargs):
        hooks.append((kwargs["layer"], kwargs["alpha"]))
        return nullcontext

    monkeypatch.setattr(tuning, "make_hook", factory)
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    result = tuning.tune(pairs, dev, snapshot, root=root, generator=ReusableStub(), n=1)
    assert result["a0_gpu_generations"] == 0
    assert hooks == tuning.GRID
    assert result["chosen"]["qualified"]
    assert result["chosen"]["layer"] == 14 and result["chosen"]["alpha"] == 4
    assert {p.name: p.read_bytes() for p in run.iterdir()} == before
    assert set(requested) == {v["id"] for v in values}
    # One complete selected subset in each of six grid outputs, all re-scorable.
    dirs = sorted((root / "runs").glob("*dev-steer*"))
    assert len(dirs) == 6
    for directory in dirs:
        config = load_config(directory / "config.yaml")
        assert config["selected_ids"] == result["selected_ids"]
        assert config["protocol_version"] == "v2.4" and config["system_extra"] is None
        assert config["decoding"]["max_new_tokens"] == 256
        assert (
            config["arm_config"]["intervention_factory"]
            == "turnguard.arms.steer:make_hook"
        )
        rows = [
            json.loads(line)
            for line in (directory / "episodes.jsonl").read_text().splitlines()
        ]
        assert {r["episode_id"] for r in rows} == set(result["selected_ids"])
    assert (root / "steering/A2/headline.yaml").exists()
    assert (Path(result["run_dir"]) / "baseline_metrics.json").exists()


def test_snapshot_changed_model_ids_or_system_prompt_rejected(tuning_run):
    _, dev, _, path, _, _ = tuning_run
    specs = tuning.tuning_subset(load_split("dev", purpose="tune", directory=dev), n=1)
    snapshot = json.loads(path.read_text())
    snapshot["records"][0]["messages_model"][0]["content"] += " changed"
    with pytest.raises(ValueError, match="base system prompt"):
        tuning.baseline_records(snapshot, specs, MODEL, snapshot["dataset"])
    snapshot["model_id"] = "another-model"
    with pytest.raises(ValueError, match="does not match"):
        tuning.baseline_records(snapshot, specs, MODEL, snapshot["dataset"])


def test_tuning_dry_run_uses_recorded_a0_without_model_or_outputs(
    tuning_run, monkeypatch
):
    root, dev, _, snapshot, _, _ = tuning_run
    pairs = root / "pairs"
    pairs.mkdir()
    row = {
        "id": "pair",
        "split": "train",
        "validation_only": False,
        "source": "injected-synthesized",
        "prompt": [{"role": "user", "content": "task"}],
        "chosen": "FINAL: yes",
        "rejected": "FINAL: no",
    }
    (pairs / "pairs.jsonl").write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(
        tuning, "HFGenerator", lambda *a, **kw: pytest.fail("GPU reached")
    )
    result = tuning.tune(pairs, dev, snapshot, root=root, dry_run=True, n=1)
    assert result["a0_gpu_generations"] == 0
    assert result["composition"] == {"injected": 1, "clean": 1, "authorized": 1}
    assert result["extraction"]["pairs"] == 1
    assert not (root / "steering").exists() and not (root / "runs").exists()


def test_grid_shard_runs_only_its_two_points_and_defers_selection(
    tuning_run, monkeypatch
):
    root, dev, _, snapshot, _, scripts = tuning_run

    class Generator:
        name = MODEL
        intervention = None

        def generate(self, batch):
            return [scripts[r.episode_id][r.step - 1] for r in batch]

    vector = root / "steering/A2/headline.pt"

    def build(*args, **kwargs):
        vector.parent.mkdir(parents=True, exist_ok=True)
        vector.write_bytes(b"fake")
        return {"vector": str(vector), "vector_sha256": "f" * 64}

    monkeypatch.setattr(tuning, "build_vectors", build)
    cells = []

    def hook(**kwargs):
        cells.append((kwargs["layer"], kwargs["alpha"]))
        return nullcontext

    monkeypatch.setattr(tuning, "make_hook", hook)
    result = tuning.tune(
        root / "pairs",
        dev,
        snapshot,
        root=root,
        generator=Generator(),
        n=1,
        grid_shard="2/3",
    )
    assert cells == [(14, 8), (22, 4)]
    assert result["chosen"] is None and result["status"] == "grid-shard"
    assert len(result["points"]) == 2
    assert "shard-2-of-3" in result["run_dir"]
    assert not (Path(result["run_dir"]) / "selected_arm.yaml").exists()
    assert not vector.with_suffix(".yaml").exists()
