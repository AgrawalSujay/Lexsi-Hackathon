"""Paired comparison, clustered bootstrap and verified rescoring (task C1)."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from turnguard.metrics import mcnemar_exact

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "a_only,b_only,p",
    [(0, 0, 1.0), (0, 5, 0.0625), (5, 0, 0.0625), (2, 8, 0.109375), (4, 4, 1.0)],
)
def test_mcnemar_exact_matches_binomial_tail(a_only, b_only, p):
    assert mcnemar_exact(a_only, b_only) == pytest.approx(p)


def cli(*args, cwd=ROOT):
    result = subprocess.run(
        [sys.executable, "-m", "turnguard", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    assert result.returncode == 0, result.stderr[-800:]
    return result.stdout


def stub_run(tmp_path, name, scripts, episodes):
    script = tmp_path / f"{name}.json"
    script.write_text(json.dumps(scripts), "utf-8")
    cli(
        "run-episodes",
        "--episodes",
        str(episodes / "*.json"),
        "--generator",
        "stub",
        "--stub-script",
        str(script),
        "--split",
        "examples",
        "--arm",
        "A0",
        "--root",
        str(tmp_path / name),
    )
    (run,) = (tmp_path / name / "runs").iterdir()
    return run


@pytest.fixture
def two_runs(tmp_path):
    episodes = tmp_path / "specs"
    shutil.copytree(ROOT / "data/examples", episodes)
    scripts = json.loads((ROOT / "tests/fixtures/stub_scripts.json").read_text())
    # Run A obeys the injection in ex-email-inj-001 (fixture); run B resists it.
    resisting = dict(scripts)
    resisting["ex-email-inj-001"] = [
        line for line in scripts["ex-email-inj-001"] if "mailsync-backup" not in line
    ]
    return (
        stub_run(tmp_path, "A0", scripts, episodes),
        stub_run(tmp_path, "B", resisting, episodes),
        episodes,
    )


def test_compare_counts_the_one_improved_episode(two_runs):
    run_a, run_b, _ = two_runs
    report = json.loads(cli("compare", str(run_a), str(run_b)))
    asr = report["metrics"]["asr"]["shared_graded"]
    assert asr["a_only_positive"] == 1 and asr["b_only_positive"] == 0
    assert asr["improved_in_b"] == 1 and asr["regressed_in_b"] == 0
    assert asr["mcnemar_exact_p"] == pytest.approx(1.0)


def test_metrics_include_clustered_bootstrap_and_are_deterministic(two_runs):
    run_a, _, _ = two_runs
    metrics = json.loads((run_a / "metrics.json").read_text())
    boot = metrics["scenario_bootstrap"]
    assert boot["resamples"] == 1000 and boot["seed"] == 1234
    first = json.loads(cli("rescore", str(run_a)))
    second = json.loads(cli("rescore", str(run_a)))
    assert first == second


def test_rescore_refuses_a_changed_spec(two_runs):
    run_a, _, episodes = two_runs
    path = episodes / "ex-email-inj-001.json"
    spec = json.loads(path.read_text())
    spec["user_turns"][0] += " (edited)"
    path.write_text(json.dumps(spec), "utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "turnguard", "rescore", str(run_a)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode != 0
    assert "has changed since the run" in result.stderr
    assert (run_a / "episodes.jsonl").read_text()  # untouched
