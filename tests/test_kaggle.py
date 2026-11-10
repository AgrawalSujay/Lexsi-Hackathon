"""Kaggle archive and subprocess contracts: tiny fixtures, no GPU/network/install."""

import argparse
import importlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from turnguard.data.freeze import _freeze_directory

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return tuple(
        importlib.import_module(f"kaggle_{n}") for n in ("pack", "run", "bundle")
    )


@pytest.fixture
def packaged(tmp_path, scripts):
    pack, _, _ = scripts
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "scripts").mkdir()
    for path in (ROOT / "scripts").glob("kaggle*.py"):
        (repo / "scripts" / path.name).write_bytes(path.read_bytes())
    (repo / "configs").mkdir()
    for name in ("models.yaml", "arms.yaml"):
        (repo / "configs" / name).write_bytes((ROOT / "configs" / name).read_bytes())
    (repo / "tracked.txt").write_text("committed")
    for command in (
        ["git", "init", "-q"],
        ["git", "add", "."],
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.org",
            "commit",
            "-qm",
            "Fixture",
        ],
    ):
        subprocess.run(command, cwd=repo, check=True, capture_output=True)
    (repo / "tracked.txt").write_text("dirty edit must not ship")
    (repo / "untracked-secret").write_text("not shipped")
    for split in ("train", "dev", "test", "test_cueflip"):
        directory = repo / "data/v1" / split
        directory.mkdir(parents=True)
        rows = []
        for source in sorted((ROOT / "data/examples").glob("*.json")):
            row = json.loads(source.read_text())
            row["split"] = "test" if split.startswith("test") else split
            rows.append(row)
        (directory / "episodes.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
        (directory / "manifest.json").write_text("{}")
        (directory / "summary.json").write_text('{"status":"complete"}')
        if split.startswith("test"):
            _freeze_directory(directory)
    pairs = repo / "data/pairs/train"
    pairs.mkdir(parents=True)
    (pairs / "pairs.jsonl").write_text("fixture")
    bundle = tmp_path / "bundle.zip"
    result = pack.pack(repo, repo, bundle)
    return repo, bundle, result


def arguments(tmp_path, bundle, **kwargs):
    return argparse.Namespace(
        **{
            "task": "split",
            "bundle": bundle,
            "work_dir": tmp_path / "working/tg",
            "output_root": tmp_path / "working",
            "split": "dev",
            "arm": "A0",
            "model": "dev",
            "dtype": "fp16",
            "adapter_dir": None,
            "shard": None,
            "parallel_shards": 1,
            "batch_size": 2,
            "max_length": 2048,
            "resume_from": None,
            "smoke": False,
            **kwargs,
        }
    )


def test_pack_committed_only_all_data_and_reproducible(packaged, scripts, tmp_path):
    repo, bundle, result = packaged
    pack, _, helper = scripts
    with zipfile.ZipFile(bundle) as archive:
        assert archive.read("tracked.txt") == b"committed"
        assert "untracked-secret" not in archive.namelist()
        assert "data/pairs/train/pairs.jsonl" in archive.namelist()
        assert all(
            f"data/v1/{s}/episodes.jsonl" in archive.namelist() for s in helper.SPLITS
        )
        manifest = json.loads(archive.read(helper.MANIFEST))
        assert manifest["git_commit"] == result["git_commit"]
        assert all(
            helper.sha(archive.read(n)) == h for n, h in manifest["files"].items()
        )
    second = tmp_path / "second.zip"
    pack.pack(repo, repo, second)
    assert helper.sha(second.read_bytes()) == result["zip_sha256"]
    helper.unpack(bundle, tmp_path / "extracted")
    helper.unpack(bundle, tmp_path / "extracted")  # Reuse without overwriting sources.
    (tmp_path / "extracted/tracked.txt").write_text("tamper")
    with pytest.raises(ValueError, match="differs"):
        helper.unpack(bundle, tmp_path / "extracted")


def test_pack_refuses_unfrozen_or_changed_release(packaged, scripts, tmp_path):
    repo, _, _ = packaged
    pack, _, _ = scripts
    directory = repo / "data/v1/test"
    (directory / "episodes.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        pack.pack(repo, repo, tmp_path / "bad.zip")
    (directory / "FROZEN").unlink()
    with pytest.raises(ValueError, match="not frozen"):
        pack.pack(repo, repo, tmp_path / "unfrozen.zip")


@pytest.mark.parametrize("mutation", ["hash", "traversal", "duplicate", "frozen"])
def test_unpack_refuses_bad_bundles_before_install(
    packaged, scripts, tmp_path, mutation
):
    _, bundle, _ = packaged
    _, _, helper = scripts
    with zipfile.ZipFile(bundle) as archive:
        files = {n: archive.read(n) for n in archive.namelist()}
    if mutation == "hash":
        files["tracked.txt"] = b"tampered"
    elif mutation == "traversal":
        files["../escape"] = b"bad"
    elif mutation == "frozen":
        del files["data/v1/test/FROZEN"]
        manifest = json.loads(files[helper.MANIFEST])
        del manifest["files"]["data/v1/test/FROZEN"]
        files[helper.MANIFEST] = json.dumps(manifest).encode()
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        if mutation == "duplicate":
            with pytest.warns(UserWarning):
                archive.writestr("tracked.txt", "duplicate")
    with pytest.raises(ValueError):
        helper.unpack(bad, tmp_path / "target")
    assert not (tmp_path / "escape").exists()


def test_parallel_shards_launch_before_wait_and_archive_outputs(
    packaged, scripts, tmp_path, capsys
):
    _, bundle, _ = packaged
    _, runner, _ = scripts
    args = arguments(tmp_path, bundle, parallel_shards=2)
    calls, events = [], []

    class Process:
        returncode = None

        def __init__(self, cmd, **kwargs):
            events.append("start")
            calls.append((cmd, kwargs))
            kwargs["stdout"].write(b"RAW_UNSAFE_SENTINEL")
            output = Path(cmd[cmd.index("--root") + 1]) / "runs/fixture"
            output.mkdir(parents=True)
            (output / "episodes.jsonl").write_text("fixture")

        def wait(self):
            events.append("wait")
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

    installs = []
    result = runner.run(
        args,
        run_process=lambda *a, **k: installs.append((a, k)),
        popen=Process,
        token_fn=lambda: "test-token",
    )
    assert events[:2] == ["start", "start"]
    assert len(calls) == 2 and len(result["outputs"]) == 2
    assert installs[0][0][0][-1] == ".[gpu]"
    for i, (cmd, kwargs) in enumerate(calls):
        assert cmd[cmd.index("--shard") + 1] == f"{i + 1}/2"
        assert cmd[cmd.index("--dtype") + 1] == "fp16"
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == str(i)
        assert kwargs["env"]["HF_TOKEN"] == "test-token"
    assert "RAW_UNSAFE_SENTINEL" not in capsys.readouterr().out
    with zipfile.ZipFile(result["outputs"][0]) as archive:
        assert archive.read("process.log") == b"RAW_UNSAFE_SENTINEL"
        assert any(n.endswith("episodes.jsonl") for n in archive.namelist())
    # Same verified source can run a second command in the same Kaggle session.
    runner.run(
        args, run_process=lambda *a, **k: None, popen=Process, token_fn=lambda: "token"
    )


def test_commands_fp16_training_and_side_without_judge(packaged, scripts, tmp_path):
    _, bundle, _ = packaged
    _, runner, _ = scripts
    args = arguments(tmp_path, bundle, task="train-a4")
    cmd = runner.command(args, tmp_path, tmp_path / "job", None)
    assert cmd[cmd.index("--max-length") + 1] == "2048"
    assert cmd[cmd.index("--checkpoint-dir") + 1].endswith("checkpoints/A4/dev")
    assert "--run-root" in cmd and "--dtype" in cmd
    args.task = "eval-side"
    cmd = runner.command(args, tmp_path, tmp_path / "job", None)
    assert "eval-side" in cmd and "--judge" not in cmd and "judge-side" not in cmd
    args.dtype = "bf16"
    with pytest.raises(ValueError, match="fp16"):
        runner.run(args)
    args.dtype, args.parallel_shards = "fp16", 2
    with pytest.raises(ValueError, match="requires split"):
        runner.run(args)


def test_resume_from_output_zip_and_adapter_base_guard(tmp_path, scripts):
    _, runner, _ = scripts
    source = tmp_path / "previous/checkpoints/A4/dev"
    source.mkdir(parents=True)
    (source / "turnguard_training.json").write_text("{}")
    checkpoint = source / "checkpoint-20"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text("{}")
    archive = runner.archive_output(source.parent.parent.parent, tmp_path / "old.zip")
    restored = tmp_path / "checkpoints/A4/dev"
    runner.restore_checkpoints(archive, restored)
    assert (restored / "checkpoint-20/trainer_state.json").exists()
    with pytest.raises(ValueError, match="populated"):
        runner.restore_checkpoints(archive, restored)
    args = arguments(
        tmp_path, tmp_path / "unused", arm="A4", adapter_dir=tmp_path / "adapter"
    )
    args.adapter_dir.mkdir()
    (args.adapter_dir / "adapter_config.json").write_text(
        '{"base_model_name_or_path":"wrong"}'
    )
    with pytest.raises(ValueError, match="base model"):
        runner.prepare_arm(args, ROOT)


@pytest.mark.parametrize("task", ["train-a4", "eval-side"])
def test_runner_training_checkpoint_archive_and_side_extras(
    packaged, scripts, tmp_path, task
):
    _, bundle, _ = packaged
    _, runner, _ = scripts
    args = arguments(tmp_path, bundle, task=task)
    installs, calls = [], []

    class Process:
        returncode = 0

        def __init__(self, cmd, **kwargs):
            calls.append(cmd)
            if task == "train-a4":
                checkpoint = args.output_root / "checkpoints/A4/dev"
                checkpoint.mkdir(parents=True)
                (checkpoint / "turnguard_training.json").write_text("{}")
                complete = checkpoint / "checkpoint-20"
                complete.mkdir()
                (complete / "trainer_state.json").write_text("{}")
                (complete / "adapter_model.safetensors").write_bytes(b"fixture")

        def wait(self):
            return 0

        def poll(self):
            return 0

    result = runner.run(
        args,
        run_process=lambda *a, **k: installs.append(a[0]),
        popen=Process,
        token_fn=lambda: "token",
    )
    assert installs[0][-1] == (".[gpu,eval]" if task == "eval-side" else ".[gpu]")
    assert all("judge-side" not in cmd for cmd in calls)
    with zipfile.ZipFile(result["outputs"][0]) as archive:
        expected = json.loads(archive.read("OUTPUT_CHECKSUMS.json"))
        assert expected
        if task == "train-a4":
            assert "checkpoints/A4/dev/checkpoint-20/trainer_state.json" in expected
            assert json.loads(archive.read("job.json"))["arm"] == "A4"


def test_split_failure_still_archives_logs(packaged, scripts, tmp_path):
    _, bundle, _ = packaged
    _, runner, _ = scripts
    args = arguments(tmp_path, bundle)

    class Failed:
        def __init__(self, *a, **kwargs):
            kwargs["stdout"].write(b"failure details stay here")

        def wait(self):
            return 1

        def poll(self):
            return 1

    with pytest.raises(ValueError, match="job failed"):
        runner.run(
            args,
            run_process=lambda *a, **k: None,
            popen=Failed,
            token_fn=lambda: "token",
        )
    outputs = list(args.output_root.glob("out_*.zip"))
    assert len(outputs) == 1
    with zipfile.ZipFile(outputs[0]) as archive:
        assert archive.read("process.log") == b"failure details stay here"


def test_pairs_default_to_source_worktree_not_main_data_root(
    packaged, scripts, tmp_path
):
    data_root, _, _ = packaged
    pack, _, _ = scripts
    # A different committed source checkout can use main's split data but its own
    # freshly built Tiny pairs, never accidentally main's older 8B pairs.
    repo = tmp_path / "other-source"
    shutil.copytree(data_root, repo)
    pairs = repo / "data/pairs/train"
    pairs.mkdir(parents=True, exist_ok=True)
    (pairs / "pairs.jsonl").write_text("Tiny pairs")
    bundle = tmp_path / "tiny.zip"
    pack.pack(repo, data_root, bundle)
    with zipfile.ZipFile(bundle) as archive:
        assert archive.read("data/pairs/train/pairs.jsonl") == b"Tiny pairs"


def test_pack_refuses_data_changing_during_capture(
    packaged, scripts, tmp_path, monkeypatch
):
    repo, _, _ = packaged
    pack, _, _ = scripts
    changed = repo / "data/v1/dev/episodes.jsonl"
    original = Path.read_bytes
    seen = False

    def read(path):
        nonlocal seen
        data = original(path)
        if path == changed and not seen:
            seen = True
            path.write_bytes(data + b"\n")
        return data

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(ValueError, match="changed while packing"):
        pack.pack(repo, repo, tmp_path / "changing.zip")
