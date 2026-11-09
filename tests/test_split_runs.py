"""Built releases, stable shards and transport checks without GPU or Modal calls."""

import copy
import importlib.util
import io
import json
import sys
import tarfile
import types
from pathlib import Path

import pytest
import yaml

from turnguard.analysis import compare_runs, rescore_run
from turnguard.cli import main
from turnguard.data.freeze import _freeze_directory
from turnguard.data.loaders import load_split, parse_shard, select_episodes

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def release(tmp_path):
    directory = tmp_path / "data/v1/dev"
    directory.mkdir(parents=True)
    rows = []
    for path in sorted((ROOT / "data/examples").glob("*.json")):
        row = json.loads(path.read_text())
        row["split"] = "dev"
        rows.append(row)
    (directory / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (directory / "manifest.json").write_text('{"version": 1}')
    (directory / "summary.json").write_text('{"status": "complete"}')
    return directory


def run_split(directory, output, *extra):
    assert (
        main(
            [
                "run-episodes",
                "--episodes",
                str(directory),
                "--split",
                directory.name,
                "--generator",
                "stub",
                "--stub-script",
                str(ROOT / "tests/fixtures/stub_scripts.json"),
                "--root",
                str(output),
                *extra,
            ]
        )
        == 0
    )
    return next((output / "runs").iterdir())


@pytest.mark.parametrize("filename", [False, True])
def test_load_split_file_and_directory_and_timing(release, tmp_path, filename):
    source = release / "episodes.jsonl" if filename else release
    # run_split uses the release name for --split, so invoke with a file directly.
    out = tmp_path / "out"
    assert (
        main(
            [
                "run-episodes",
                "--episodes",
                str(source),
                "--split",
                "dev",
                "--generator",
                "stub",
                "--stub-script",
                str(ROOT / "tests/fixtures/stub_scripts.json"),
                "--root",
                str(out),
            ]
        )
        == 0
    )
    run = next((out / "runs").iterdir())
    config = yaml.safe_load((run / "config.yaml").read_text())
    assert len(config["dataset"]["sha256"]) == 64
    assert set(config["dataset"]["files"]) == {"episodes.jsonl", "manifest.json"}
    env = json.loads((run / "env.json").read_text())
    assert env["elapsed_seconds"] >= env["generation_seconds"] >= 0
    assert env["finished_at"] >= env["started_at"]
    assert rescore_run(run)["counts"]["episodes"] == 4
    metrics = json.loads((run / "metrics.json").read_text())
    for factor in ("conversation", "injection", "conversation_x_injection"):
        cells = metrics["by_language"][factor]
        assert sum(cell["counts"]["episodes"] for cell in cells.values()) == 4
        for name in ("asr_incl_relay", "clean_task_success", "authorized_follow_rate"):
            assert sum(cell[name]["n"] for cell in cells.values()) == metrics[name]["n"]


def test_shards_disjoint_complete_stable_and_limit_before_partition(release):
    base = load_split("dev", purpose="eval", directory=release)[0]
    specs = []
    for i in range(100):
        spec = copy.deepcopy(base)
        spec.id = f"episode-{i}"
        specs.append(spec)
    shards = [select_episodes(specs, shard=f"{i}/3") for i in range(1, 4)]
    ids = [{spec.id for spec in shard} for shard in shards]
    assert set.union(*ids) == {spec.id for spec in specs}
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    for i, group in enumerate(shards, 1):
        assert [s.id for s in group] == [
            s.id for s in select_episodes(list(reversed(specs)), shard=f"{i}/3")
        ]
    limited = [select_episodes(specs, limit=5, shard=f"{i}/3") for i in range(1, 4)]
    assert set.union(*[{s.id for s in group} for group in limited]) == {
        s.id for s in select_episodes(specs, limit=5)
    }


@pytest.mark.parametrize("value", ["0/2", "3/2", "1/0", "1", "x/2", "-1/2"])
def test_invalid_shards(value):
    with pytest.raises(ValueError, match="shard"):
        parse_shard(value)


def test_limit_run_and_singletons_still_work(release, tmp_path):
    run = run_split(release, tmp_path / "limited", "--limit", "1")
    assert len((run / "episodes.jsonl").read_text().splitlines()) == 1
    # Hash checking uses the whole dataset, even when only one spec was run.
    (release / "episodes.jsonl").write_text(
        (release / "episodes.jsonl").read_text() + "\n"
    )
    with pytest.raises(ValueError, match="Dataset hash mismatch"):
        rescore_run(run)


def test_changed_manifest_refused_by_rescore_and_compare(release, tmp_path):
    run_a = run_split(release, tmp_path / "a")
    run_b = run_split(release, tmp_path / "b")
    before = (run_a / "episodes.jsonl").read_bytes()
    (release / "manifest.json").write_text('{"version": 2}')
    for operation in (lambda: rescore_run(run_a), lambda: compare_runs(run_a, run_b)):
        with pytest.raises(ValueError, match="Dataset hash mismatch"):
            operation()
    assert (run_a / "episodes.jsonl").read_bytes() == before


@pytest.mark.parametrize("split", ["test", "test_cueflip"])
def test_test_release_requires_freeze(release, tmp_path, split):
    target = release.with_name(split)
    release.rename(target)
    rows = [
        json.loads(line)
        for line in (target / "episodes.jsonl").read_text().splitlines()
    ]
    for row in rows:
        row["split"] = "test"
    (target / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    with pytest.raises(ValueError, match="not frozen"):
        load_split(split, purpose="eval", directory=target)
    with pytest.raises(SystemExit):
        run_split(target, tmp_path / "unfrozen")
    assert not (tmp_path / "unfrozen/runs").exists()
    _freeze_directory(target)
    run = run_split(target, tmp_path / "frozen")
    assert json.loads((run / "metrics.json").read_text())["counts"]["episodes"] == 4


@pytest.fixture
def modal_runner(monkeypatch):
    # Import pure packaging helpers with a minimal decorator/image double.
    # No Modal SDK, credentials, remote function, or model is used by these tests.
    class Noop:
        def __getattr__(self, name):
            return lambda *args, **kwargs: self

    class App(Noop):
        def function(self, **kwargs):
            return lambda function: function

        def local_entrypoint(self):
            return lambda function: function

    module = types.ModuleType("modal")
    module.is_local = lambda: False
    module.Image = types.SimpleNamespace(debian_slim=lambda **kwargs: Noop())
    module.Volume = types.SimpleNamespace(from_name=lambda *args, **kwargs: Noop())
    module.Secret = module.Volume
    module.App = lambda *args, **kwargs: App()
    monkeypatch.setitem(sys.modules, "modal", module)
    spec = importlib.util.spec_from_file_location(
        "modal_runner", ROOT / "scripts/modal_run.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def test_split_upload_contains_all_files_and_verifies_bytes(
    release, tmp_path, modal_runner
):
    modal_runner.ROOT = release.parents[2]
    payload, info = modal_runner._split_upload("dev")
    assert set(info["files"]) == {"episodes.jsonl", "manifest.json", "summary.json"}
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        assert set(archive.getnames()) == set(info["files"])
    directory = modal_runner._unpack_split(payload, info, tmp_path / "remote")
    assert (directory / "episodes.jsonl").read_bytes() == (
        release / "episodes.jsonl"
    ).read_bytes()
    bad = copy.deepcopy(info)
    bad["files"]["episodes.jsonl"] = "0" * 64
    with pytest.raises(ValueError, match="checksum mismatch"):
        modal_runner._unpack_split(payload, bad, tmp_path / "bad")


def test_frozen_upload_includes_checksums_and_marker(release, tmp_path, modal_runner):
    target = release.with_name("test")
    release.rename(target)
    path = target / "episodes.jsonl"
    path.write_text(path.read_text().replace('"split": "dev"', '"split": "test"'))
    _freeze_directory(target)
    modal_runner.ROOT = target.parents[2]
    payload, info = modal_runner._split_upload("test")
    assert {"manifest.json", "episodes.jsonl", "CHECKSUMS.sha256", "FROZEN"} <= info[
        "files"
    ].keys()
    modal_runner._unpack_split(payload, info, tmp_path / "remote")


def test_modal_split_command_uses_configured_batch_and_shard(modal_runner):
    [(name, command)] = modal_runner._steps(
        "split", "headline", split="test_cueflip", shard="2/3", batch_size=7
    )
    assert name == "split"
    assert command[command.index("--batch-size") + 1] == "7"
    assert command[command.index("--shard") + 1] == "2/3"
    assert command[command.index("--arm") + 1] == "A0"
    assert (
        command[command.index("--episodes") + 1]
        == "data/v1/test_cueflip/episodes.jsonl"
    )


def test_language_summary_prints_counts_without_outputs(
    release, tmp_path, modal_runner, capsys
):
    run = run_split(release, tmp_path / "out")
    capsys.readouterr()
    modal_runner._print_split_metrics(run.parent)
    output = capsys.readouterr().out
    for text in (
        "conversation=en",
        "injection=en",
        "ungraded=0",
        "CI95",
        "asr_incl_relay",
    ):
        assert text in output
    assert "FINAL:" not in output


def test_limit_is_mixed_and_shards_keep_twins_together():
    from types import SimpleNamespace

    from turnguard.data.loaders import select_episodes

    specs = []
    for i in range(40):
        specs.append(SimpleNamespace(id=f"dev-a-auth-{i}", twin_of=None))
        specs.append(SimpleNamespace(id=f"dev-z-inj-{i}", twin_of=None))
        specs.append(
            SimpleNamespace(id=f"dev-z-inj-{i}-clean", twin_of=f"dev-z-inj-{i}")
        )
    picked = select_episodes(specs, limit=30)
    assert len(picked) <= 30
    assert any("auth" in s.id for s in picked) and any("inj" in s.id for s in picked)
    for part in [select_episodes(specs, shard=f"{i}/3") for i in (1, 2, 3)] + [picked]:
        ids = {s.id for s in part}
        for s in part:
            if s.twin_of:
                assert s.twin_of in ids
            elif f"{s.id}-clean" in {x.id for x in specs}:
                assert f"{s.id}-clean" in ids
    union = [s.id for i in (1, 2, 3) for s in select_episodes(specs, shard=f"{i}/3")]
    assert sorted(union) == sorted(s.id for s in specs)
