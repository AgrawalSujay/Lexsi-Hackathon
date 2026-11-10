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


def training_pairs(tmp_path):
    directory = tmp_path / "pairs"
    directory.mkdir()
    rows = [
        dict(
            id=str(i),
            split="train",
            validation_only=False,
            prompt=[{"role": "user", "content": "Save code-1."}],
            chosen='ACTION: {"tool":"save_note","args":{"text":"code-1"}}',
            rejected="FINAL: Done.",
        )
        for i in range(2)
    ]
    (directory / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (directory / "summary.json").write_text(
        json.dumps({"split": "train", "validation_only": False})
    )
    return directory


def test_modal_train_upload_verified_before_training(tmp_path, modal_runner):
    directory = training_pairs(tmp_path)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    payload, info = modal_runner._pairs_upload(directory)
    assert set(info["files"]) == {"pairs.jsonl", "summary.json"}
    unpacked = modal_runner._unpack_pairs(payload, info, tmp_path / "remote")
    assert (unpacked / "pairs.jsonl").read_bytes() == before["pairs.jsonl"]
    bad = copy.deepcopy(info)
    bad["files"]["pairs.jsonl"] = "0" * 64
    with pytest.raises(ValueError, match="checksum mismatch"):
        modal_runner._unpack_pairs(payload, bad, tmp_path / "bad")
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    (directory / "VALIDATION_ONLY").write_text("validation — never for training")
    with pytest.raises(ValueError, match="never for training"):
        modal_runner._pairs_upload(directory)


def test_modal_train_background_and_smoke_command(tmp_path, modal_runner, monkeypatch):
    directory = training_pairs(tmp_path)
    [(name, command)] = modal_runner._steps("train-a4", "dev", smoke=True)
    assert name == "train-a4" and "--smoke" in command
    assert command[command.index("--pairs") + 1] == "data/pairs/train"
    assert command[command.index("--arm") + 1] == "A4"
    captured = []

    class Remote:
        def with_options(self, **kwargs):
            assert kwargs == {"gpu": "H100", "timeout": 6 * 60 * 60}
            return self

        def spawn(self, *args):
            captured.append(args)
            return types.SimpleNamespace(object_id="fake-job")

        def remote(self, *args):
            pytest.fail("background must spawn")

    monkeypatch.setattr(modal_runner, "run_task", Remote())
    monkeypatch.setattr(
        modal_runner,
        "_source_archive",
        lambda: (
            b"source",
            {"git_commit": "1234567", "working_tree_had_uncommitted_changes": False},
        ),
    )
    modal_runner.main(
        task="train-a4",
        model="dev",
        gpu="H100",
        background=True,
        pairs=str(directory),
        smoke=True,
    )
    args = captured[0]
    assert args[0] == "train-a4" and args[8] is True and args[-3:] == ("A0", None, None)
    assert args[5]["files"]["pairs.jsonl"]
    with pytest.raises(SystemExit, match="requires --gpu"):
        modal_runner.main(task="train-a4", gpu="L40S")


def test_modal_train_persists_adapter_logs_and_upload_checksums(
    tmp_path, modal_runner, monkeypatch
):
    directory = training_pairs(tmp_path)
    monkeypatch.setattr(modal_runner, "CHECKPOINT_ROOT", tmp_path / "checkpoints")
    payload, info = modal_runner._pairs_upload(directory)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w"):
        pass

    def run(command, cwd):
        cwd = Path(cwd)
        job = cwd / "runs/A4__headline__train-a4__fake"
        job.mkdir(parents=True)
        (job / "training_log.json").write_text("[]")
        adapter = cwd / "adapters/A4/headline"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"fake")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(modal_runner.subprocess, "run", run)
    original = modal_runner.shutil.copytree

    def copytree(src, dst, *args, **kwargs):
        if str(dst) == "/cache/runs":
            dst = tmp_path / "volume/runs"
        return original(src, dst, *args, **kwargs)

    monkeypatch.setattr(modal_runner.shutil, "copytree", copytree)
    monkeypatch.setattr(
        modal_runner, "cache", types.SimpleNamespace(commit=lambda: None)
    )
    result = modal_runner.run_task(
        "train-a4", "headline", buf.getvalue(), {}, payload, info
    )
    assert result["steps"] == {"train-a4": 0}
    job = tmp_path / "volume/runs/A4__headline__train-a4__fake"
    assert (job / "training_log.json").exists()
    assert (job / "adapters/A4/headline/adapter_model.safetensors").exists()
    assert json.loads((job / "PAIRS_UPLOAD.json").read_text()) == info
    with tarfile.open(fileobj=io.BytesIO(result["runs_tar"])) as archive:
        assert "adapters/A4/headline/adapter_model.safetensors" in archive.getnames()


def test_train_a4_step_passes_persistent_checkpoint_dir(modal_runner):
    ((name, command),) = modal_runner._steps(
        "train-a4", "headline", checkpoint_dir="/cache/checkpoints/A4-x"
    )
    assert command[-2:] == ["--checkpoint-dir", "/cache/checkpoints/A4-x"]


def test_split_step_runs_the_requested_arm(modal_runner):
    ((name, command),) = modal_runner._steps("split", "headline", split="dev", arm="A4")
    assert command[command.index("--arm") + 1] == "A4"


def test_tuning_upload_contains_verified_recorded_a0_and_checksums(
    release, tmp_path, modal_runner, monkeypatch
):
    from turnguard.arms import steer_tune
    from turnguard.data.loaders import split_fingerprint

    directory = training_pairs(tmp_path)
    snapshot = {
        "dataset": split_fingerprint(release),
        "selected_ids": ["fake"],
        "records": [],
        "model_id": "CohereLabs/aya-expanse-8b",
    }
    monkeypatch.setattr(steer_tune, "prepare_baseline", lambda *a, **kw: snapshot)
    payload, info = modal_runner._tune_upload(
        str(directory), str(release), "read-only-a0", "headline"
    )
    assert set(info["files"]) == {"pairs.tar.gz", "dev.tar.gz", "baseline.json"}
    work = tmp_path / "remote-tune"
    work.mkdir()
    modal_runner._unpack_tune(payload, info, work)
    assert json.loads((work / "baseline.json").read_text()) == snapshot
    assert (work / "data/pairs/train/pairs.jsonl").exists()
    assert (work / "data/v1/dev/episodes.jsonl").exists()
    bad = copy.deepcopy(info)
    bad["files"]["baseline.json"] = "0" * 64
    with pytest.raises(ValueError, match="tuning checksum mismatch"):
        modal_runner._unpack_tune(payload, bad, tmp_path / "bad-tune")


def test_tuning_background_uses_dedicated_job_and_never_a0_gpu(
    modal_runner, monkeypatch
):
    calls = []

    class Remote:
        def with_options(self, **kwargs):
            assert kwargs == {"gpu": "L40S", "timeout": 6 * 60 * 60}
            return self

        def spawn(self, *args):
            calls.append(args)
            return types.SimpleNamespace(object_id="fake-tune")

    def upload(*args):
        calls.append(args)
        return b"packed-inputs", {"files": {"baseline.json": "verified"}}

    monkeypatch.setattr(modal_runner, "run_steer_tune", Remote())
    monkeypatch.setattr(
        modal_runner,
        "run_task",
        types.SimpleNamespace(
            with_options=lambda **kw: pytest.fail("A4/ordinary GPU job used")
        ),
    )
    monkeypatch.setattr(modal_runner, "_tune_upload", upload)
    monkeypatch.setattr(
        modal_runner,
        "_source_archive",
        lambda: (
            b"source",
            {"git_commit": "1234567", "working_tree_had_uncommitted_changes": False},
        ),
    )
    modal_runner.main(
        task="steer-tune",
        model="headline",
        background=True,
        pairs="pairs",
        dev_dir="dev",
        baseline_run="a0",
    )
    assert calls[0] == ("pairs", "dev", "a0", "headline")
    assert calls[1][0] == "headline" and calls[1][3] == b"packed-inputs"


@pytest.mark.parametrize("exit_code", [0, 1])
def test_tuning_remote_persists_vectors_selection_and_grid(
    release, tmp_path, modal_runner, monkeypatch, exit_code
):
    from turnguard.arms import steer_tune
    from turnguard.data.loaders import split_fingerprint

    directory = training_pairs(tmp_path)
    monkeypatch.setattr(
        steer_tune,
        "prepare_baseline",
        lambda *a, **kw: {"dataset": split_fingerprint(release)},
    )
    payload, info = modal_runner._tune_upload(
        str(directory), str(release), "a0", "headline"
    )
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w"):
        pass

    def run(command, cwd):
        assert command[command.index("--baseline-snapshot") + 1] == "baseline.json"
        assert command[3] == "steer-tune" and "run-episodes" not in command
        work = Path(cwd)
        folder = work / "runs/A2__headline__steer-tune__fake"
        folder.mkdir(parents=True)
        (folder / "summary.json").write_text("{}")
        vector = work / "steering/A2/headline.pt"
        vector.parent.mkdir(parents=True)
        vector.write_bytes(b"fake-vector")
        vector.with_suffix(".yaml").write_text("arms: {}")
        return types.SimpleNamespace(returncode=exit_code)

    monkeypatch.setattr(modal_runner.subprocess, "run", run)
    original = modal_runner.shutil.copytree

    def copytree(src, dst, *args, **kwargs):
        if str(dst).startswith("/cache/"):
            dst = tmp_path / "volume" / Path(dst).name
        return original(src, dst, *args, **kwargs)

    monkeypatch.setattr(modal_runner.shutil, "copytree", copytree)
    monkeypatch.setattr(
        modal_runner, "cache", types.SimpleNamespace(commit=lambda: None)
    )
    result = modal_runner.run_steer_tune("headline", buf.getvalue(), {}, payload, info)
    assert result["steps"] == {"steer-tune": exit_code}
    assert (tmp_path / "volume/steering/A2/headline.pt").exists() == (exit_code == 0)
    run = tmp_path / "volume/runs/A2__headline__steer-tune__fake"
    assert (run / "summary.json").exists() and (
        run / "steering/A2/headline.yaml"
    ).exists()
    assert json.loads((run / "STEER_UPLOAD.json").read_text()) == info


@pytest.fixture
def frozen_suite(release):
    import shutil

    for name in ("test", "test_cueflip"):
        directory = release.with_name(name)
        shutil.copytree(release, directory)
        values = [
            json.loads(line)
            for line in (directory / "episodes.jsonl").read_text().splitlines()
        ]
        for value in values:
            value["split"] = "test"
        (directory / "episodes.jsonl").write_text(
            "".join(json.dumps(v) + "\n" for v in values)
        )
        _freeze_directory(directory)
    return release.parent


def setup_work(work):
    (work / "configs").mkdir(parents=True)
    for name in ("arms.yaml", "models.yaml"):
        (work / "configs" / name).write_bytes((ROOT / "configs" / name).read_bytes())


def test_eval_suite_transport_checks_both_frozen_releases(
    frozen_suite, tmp_path, modal_runner
):
    payload, info = modal_runner._suite_upload(str(frozen_suite))
    assert set(info["splits"]) == {"test", "test_cueflip"}
    work = tmp_path / "remote-suite"
    modal_runner._unpack_suite(payload, info, work)
    for name in ("test", "test_cueflip"):
        assert (work / "data/v1" / name / "FROZEN").exists()
    bad = copy.deepcopy(info)
    bad["files"]["test.tar.gz"] = "0" * 64
    with pytest.raises(ValueError, match="checksum mismatch"):
        modal_runner._unpack_suite(payload, bad, tmp_path / "bad-suite")
    (frozen_suite / "test_cueflip/FROZEN").unlink()
    with pytest.raises(ValueError):
        modal_runner._suite_upload(str(frozen_suite))


@pytest.mark.parametrize("arm", ["A0", "A1", "A2", "A4"])
def test_eval_steps_reuse_split_arm_and_shard(modal_runner, arm):
    suite = modal_runner._steps(
        "eval-suite", "headline", arm=arm, shard="2/2", batch_size=5
    )
    assert [name for name, _ in suite] == ["test", "test_cueflip"]
    for name, command in suite:
        assert command[command.index("--arm") + 1] == arm
        assert command[command.index("--split") + 1] == name
        assert command[command.index("--shard") + 1] == "2/2"
        assert command[command.index("--batch-size") + 1] == "5"
    single = modal_runner._steps("eval-arm", "headline", arm=arm, split="dev")
    assert (
        single[0][1]
        == modal_runner._steps("split", "headline", arm=arm, split="dev")[0][1]
    )


def test_a4_volume_adapter_uses_model_specific_path(
    tmp_path, modal_runner, monkeypatch
):
    work = tmp_path / "work"
    setup_work(work)
    volume = tmp_path / "volume/runs"
    adapter = volume / "train-job/adapters/A4/dev"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "CohereLabs/tiny-aya-global"})
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"fake-lora")
    monkeypatch.setattr(modal_runner, "VOLUME_RUNS", volume)
    modal_runner._stage_arm("A4", "dev", "train-job", None, work)
    config = yaml.safe_load((work / "configs/arms.yaml").read_text())
    actual = Path(config["arms"]["A4"]["adapter"])
    assert actual == work / "adapters/A4/dev"
    assert (actual / "adapter_model.safetensors").read_bytes() == b"fake-lora"
    assert (
        json.loads((work / "ARM_ARTIFACT.json").read_text())["source_run"]
        == "train-job"
    )
    bad_work = tmp_path / "bad-work"
    setup_work(bad_work)
    (adapter / "adapter_config.json").write_text(
        '{"base_model_name_or_path":"wrong-model"}'
    )
    with pytest.raises(ValueError, match="base model"):
        modal_runner._stage_arm("A4", "dev", "train-job", None, bad_work)
    with pytest.raises(ValueError, match="single run name"):
        modal_runner._volume_run("../escape")


def test_a2_volume_selection_uses_exact_winning_shard_vector(
    tmp_path, modal_runner, monkeypatch
):
    from test_steer_merge import make_shards

    from turnguard.arms.steer_merge import merge_grid_runs

    paths = make_shards(tmp_path / "sources")
    # The fake grid uses model ID "model"; evaluation's config must agree.
    volume = tmp_path / "volume/runs"
    merged = volume / "selected-job"
    merge_grid_runs(paths, output=merged)
    work = tmp_path / "work"
    setup_work(work)
    models = yaml.safe_load((work / "configs/models.yaml").read_text())
    models["models"]["headline"] = "model"
    (work / "configs/models.yaml").write_text(yaml.safe_dump(models))
    monkeypatch.setattr(modal_runner, "VOLUME_RUNS", volume)
    modal_runner._stage_arm("A2", "headline", None, "selected-job", work)
    config = yaml.safe_load((work / "configs/arms.yaml").read_text())
    kwargs = config["arms"]["A2"]["intervention_kwargs"]
    assert kwargs["layer"] == 18 and kwargs["alpha"] == 8
    assert (
        Path(kwargs["vector"]).read_bytes()
        == (paths[0] / "steering/A2/headline.pt").read_bytes()
    )
    (merged / "steering/A2/headline.pt").write_bytes(b"changed")
    bad = tmp_path / "bad"
    setup_work(bad)
    (bad / "configs/models.yaml").write_text(yaml.safe_dump(models))
    with pytest.raises(ValueError, match="checksum mismatch"):
        modal_runner._stage_arm("A2", "headline", None, "selected-job", bad)


def test_h100_grid_shard_background_dispatch(modal_runner, monkeypatch):
    calls = []

    class Remote:
        def with_options(self, **kw):
            assert kw == {"gpu": "H100", "timeout": 6 * 60 * 60}
            return self

        def spawn(self, *args):
            calls.append(args)
            return types.SimpleNamespace(object_id="grid-shard")

    monkeypatch.setattr(modal_runner, "run_steer_tune", Remote())
    monkeypatch.setattr(modal_runner, "_tune_upload", lambda *args: (b"inputs", {}))
    monkeypatch.setattr(
        modal_runner,
        "_source_archive",
        lambda: (
            b"source",
            {"git_commit": "abc1234", "working_tree_had_uncommitted_changes": False},
        ),
    )
    modal_runner.main(task="steer-tune", gpu="H100", background=True, grid_shard="2/3")
    assert calls[0][-1] == "2/3"


def test_suite_background_and_cpu_volume_merge_dispatch(modal_runner, monkeypatch):
    calls = []

    class Remote:
        def with_options(self, **kw):
            assert kw == {"gpu": "H100", "timeout": 6 * 60 * 60}
            return self

        def spawn(self, *args):
            calls.append(args)
            return types.SimpleNamespace(object_id="suite")

    monkeypatch.setattr(modal_runner, "run_task", Remote())
    monkeypatch.setattr(
        modal_runner, "_suite_upload", lambda root: (b"suite", {"splits": {}})
    )
    monkeypatch.setattr(
        modal_runner,
        "_source_archive",
        lambda: (
            b"source",
            {"git_commit": "abc1234", "working_tree_had_uncommitted_changes": False},
        ),
    )
    modal_runner.main(
        task="eval-suite",
        arm="A2",
        steer_run="selected-job",
        gpu="H100",
        background=True,
        shard="1/2",
    )
    assert calls[0][0] == "eval-suite" and calls[0][6] == "1/2"
    assert calls[0][-3:] == ("A2", None, "selected-job")

    class Merge:
        # No with_options method: the CPU merge must not receive GPU options.
        def remote(self, *args):
            calls.append(args)
            return {"volume_run": "merged-job"}

    monkeypatch.setattr(modal_runner, "run_steer_merge", Merge())
    modal_runner.main(task="steer-merge", grid_runs="one,two,three")
    assert calls[-1][-1] == ["one", "two", "three"]


def test_suite_job_runs_both_even_after_one_split_failure(
    frozen_suite, tmp_path, modal_runner, monkeypatch
):
    payload, info = modal_runner._suite_upload(str(frozen_suite))
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w"):
        pass
    seen = []

    def run(command, cwd):
        name = command[command.index("--split") + 1]
        seen.append(name)
        (Path(cwd) / "runs" / name).mkdir(parents=True)
        return types.SimpleNamespace(returncode=1 if name == "test" else 0)

    monkeypatch.setattr(modal_runner.subprocess, "run", run)
    monkeypatch.setattr(modal_runner, "_print_split_metrics", lambda runs: None)
    original = modal_runner.shutil.copytree

    def copytree(src, dst, *args, **kw):
        if str(dst) == "/cache/runs":
            dst = tmp_path / "volume/runs"
        return original(src, dst, *args, **kw)

    monkeypatch.setattr(modal_runner.shutil, "copytree", copytree)
    monkeypatch.setattr(
        modal_runner, "cache", types.SimpleNamespace(commit=lambda: None)
    )
    result = modal_runner.run_task(
        "eval-suite", "headline", buffer.getvalue(), {}, payload, info, shard="2/2"
    )
    assert seen == ["test", "test_cueflip"]
    assert result["steps"] == {"test": 1, "test_cueflip": 0}


def test_cpu_merge_reads_volume_results_and_persists_selection(
    tmp_path, modal_runner, monkeypatch
):
    from test_steer_merge import make_shards

    volume = tmp_path / "volume/runs"
    paths = make_shards(volume)
    monkeypatch.setattr(modal_runner, "VOLUME_RUNS", volume)
    monkeypatch.setattr(
        modal_runner,
        "cache",
        types.SimpleNamespace(reload=lambda: None, commit=lambda: None),
    )
    original = modal_runner.shutil.copytree

    def copytree(src, dst, *args, **kwargs):
        if str(dst) == "/cache/runs":
            dst = volume
        return original(src, dst, *args, **kwargs)

    monkeypatch.setattr(modal_runner.shutil, "copytree", copytree)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w"):
        pass
    result = modal_runner.run_steer_merge(
        buffer.getvalue(), {}, [p.name for p in paths]
    )
    selected = volume / result["volume_run"]
    assert (selected / "selected_arm.yaml").exists()
    assert (selected / "steering/A2/headline.pt").read_bytes() == (
        paths[0] / "steering/A2/headline.pt"
    ).read_bytes()


def test_side_task_is_generation_only_and_default_l40s(modal_runner, monkeypatch):
    [(name, command)] = modal_runner._steps("eval-side", "dev", arm="A1", batch_size=4)
    assert name == "eval-side" and command[3] == "eval-side"
    assert command[command.index("--dtype") + 1] == "fp16"
    assert "--judge" not in command and "judge-side" not in command
    calls = []

    class Remote:
        def with_options(self, **kwargs):
            assert kwargs == {"gpu": "L40S", "timeout": 6 * 60 * 60}
            return self

        def spawn(self, *args):
            calls.append(args)
            return types.SimpleNamespace(object_id="side-job")

    monkeypatch.setattr(modal_runner, "run_task", Remote())
    monkeypatch.setattr(
        modal_runner,
        "_source_archive",
        lambda: (
            b"source",
            {"git_commit": "abc1234", "working_tree_had_uncommitted_changes": False},
        ),
    )
    modal_runner.main(task="eval-side", model="dev", arm="A1", background=True)
    assert calls[0][0] == "eval-side" and calls[0][-3:] == ("A1", None, None)


def test_split_dtype_recorded(release, tmp_path):
    run = run_split(release, tmp_path / "out", "--dtype", "fp16")
    assert yaml.safe_load((run / "config.yaml").read_text())["dtype"] == "fp16"
