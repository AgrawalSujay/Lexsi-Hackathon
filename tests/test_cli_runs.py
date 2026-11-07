"""Check complete stub runs, input failures, provenance, and ungraded reporting."""

import json
from pathlib import Path

import pytest
import yaml

from turnguard.cli import main
from turnguard.metrics import compute_metrics
from turnguard.schema import EpisodeRecord, EpisodeSpec

ROOT = Path(__file__).resolve().parents[1]


def arguments(root: Path) -> list[str]:
    return [
        "run-episodes",
        "--episodes",
        str(ROOT / "data/examples/*.json"),
        "--generator",
        "stub",
        "--stub-script",
        str(ROOT / "tests/fixtures/stub_scripts.json"),
        "--arm",
        "A0",
        "--split",
        "examples",
        "--root",
        str(root),
    ]


@pytest.fixture(autouse=True)
def cpu_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "turnguard.runs._environment",
        lambda root: {
            "python_version": "3.12",
            "packages": {},
            "gpu_name": None,
            "git_commit": "abc",
            "git_dirty": True,
        },
    )


def test_cli_writes_complete_run(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(arguments(tmp_path)) == 0
    run = next((tmp_path / "runs").iterdir())
    assert {p.name for p in run.iterdir()} == {
        "config.yaml",
        "episodes.jsonl",
        "metrics.json",
        "env.json",
    }
    records = [
        EpisodeRecord.from_json(line)
        for line in (run / "episodes.jsonl").read_text().splitlines()
    ]
    config = yaml.safe_load((run / "config.yaml").read_text())
    specs = [
        EpisodeSpec.from_json(Path(path).read_text())
        for path in config["episode_paths"]
    ]
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics == compute_metrics(records, specs)
    assert metrics["asr"]["rate"] == 1.0
    assert metrics["clean_task_success"]["rate"] == 1.0
    assert metrics["authorized_follow_rate"]["rate"] == 1.0
    assert metrics["first_violation_step"] == {"3": 1, "4": 1}
    assert config["seed"] == 1234
    assert config["decoding"]["max_steps_per_user_turn"] == 6
    assert len(config["stub_script_sha256"]) == 64
    assert all(Path(p).is_absolute() for p in config["episode_paths"])
    assert json.loads((run / "env.json").read_text())["git_dirty"] is True
    output = capsys.readouterr().out
    for record in records:
        assert record.episode_id in output and record.result["tag"] in output
    assert "95% CI" in output
    assert main(arguments(tmp_path)) == 0
    assert len(list((tmp_path / "runs").iterdir())) == 2


@pytest.mark.parametrize(
    "option,value",
    [
        ("--episodes", "missing-*.json"),
        ("--split", "dev"),
        ("--stub-script", "missing.json"),
    ],
)
def test_bad_input_creates_no_run(tmp_path: Path, option: str, value: str) -> None:
    args = arguments(tmp_path)
    args[args.index(option) + 1] = value
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert not (tmp_path / "runs").exists()


def test_exhausted_script_is_reported_and_not_counted_safe(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    path = tmp_path / "scripts.json"
    scripts = json.loads((ROOT / "tests/fixtures/stub_scripts.json").read_text())
    scripts["ex-email-inj-001"] = []
    path.write_text(json.dumps(scripts))
    args = arguments(tmp_path)
    args[args.index("--stub-script") + 1] = str(path)
    assert main(args) == 1
    run = next((tmp_path / "runs").iterdir())
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["counts"]["ungraded"] == 4
    assert metrics["asr"]["n"] == 0
    assert "Ungraded: 4" in capsys.readouterr().out
