"""Run TurnGuard GPU jobs on Modal (backup to the H200; for checks, not headline runs).

The job runs exactly the committed HEAD of this repo (via `git archive`), like
`make colab-zip`; uncommitted changes are NOT sent. Results land in ./runs/
on your machine, as if the run had happened locally.

One-time setup (on your Mac):
    python3 -m pip install modal
    modal setup
    modal secret create huggingface HF_TOKEN=hf_...

Usage (from the repo root):
    modal run scripts/modal_run.py --task smoke      # ~5 min: tests, GPU, Aya 8B loads and replies
    modal run scripts/modal_run.py --task examples   # the 4 reference episodes
    modal run scripts/modal_run.py --task probe      # the 28-episode probe
    modal run scripts/modal_run.py --task split --split dev --model headline
    modal run scripts/modal_run.py --task split --split dev --shard 1/3
    modal run --detach scripts/modal_run.py --task train-a4 --gpu H100 --background
    modal run scripts/modal_run.py --task train-a4 --gpu H100 --model dev --smoke
    modal run scripts/modal_run.py --task all        # smoke + examples + probe in one container

Options: --model headline|dev (default headline), --gpu L40S|A100-80GB|H100 (default L40S).
Model weights are cached in the Modal Volume "turnguard-hf-cache", so only the
first run downloads them.
"""

import datetime
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import tomllib
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = Path("/cache/checkpoints")  # on the persistent volume
GPU_CHOICES = ("L40S", "A100-80GB", "H100")
TASKS = ("smoke", "examples", "probe", "all", "split", "train-a4", "steer-tune")
SPLITS = ("train", "dev", "test", "test_cueflip")


def _packages() -> list[str]:
    # Inside the Modal container this file is /root/modal_run.py with no repo
    # around it; the image is already built there, so the list is not needed.
    if not modal.is_local():
        return []
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = project["project"]["optional-dependencies"]
    return [*project["project"]["dependencies"], *extras["gpu"], *extras["dev"]]


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(*_packages())
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
)
cache = modal.Volume.from_name("turnguard-hf-cache", create_if_missing=True)
app = modal.App("turnguard", image=image)

SMOKE_PY = """
import torch
from turnguard.config import load_config
from turnguard.models import GenRequest, HFGenerator

config = load_config("configs/models.yaml")
print("GPU:", torch.cuda.get_device_name(), "| torch", torch.__version__)
gen = HFGenerator(config["models"]["{model}"], max_new_tokens=32, batch_size=1, seed=config["seed"])
out = gen.generate([GenRequest("smoke", 1, [
    {{"role": "user", "content": "Reply with exactly this line and nothing else: FINAL: ok"}}
])])[0]
print("Model reply:", repr(out))
assert out.strip(), "empty reply"
print("SMOKE OK")
"""


def _source_archive() -> tuple[bytes, dict]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    commit = git("rev-parse", "HEAD")
    uncommitted = bool(git("status", "--porcelain", "--untracked-files=no"))
    archive = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    info = {
        "git_commit": commit,
        "git_dirty": False,  # the archive is exactly HEAD
        "working_tree_had_uncommitted_changes": uncommitted,
        "built_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "runner": "modal",
    }
    return archive, info


def _steps(
    task: str,
    model: str,
    *,
    split: str = "dev",
    shard: str | None = None,
    batch_size: int = 8,
    smoke: bool = False,
    checkpoint_dir: str | None = None,
    arm: str = "A0",
) -> list[tuple[str, list[str]]]:
    if task == "train-a4":
        command = [
            "python",
            "-m",
            "turnguard",
            "train",
            "--arm",
            "A4",
            "--pairs",
            "data/pairs/train",
            "--model",
            model,
        ]
        if smoke:
            command.append("--smoke")
        if checkpoint_dir:
            command += ["--checkpoint-dir", checkpoint_dir]
        return [("train-a4", command)]
    if task == "split":
        command = [
            "python",
            "-m",
            "turnguard",
            "run-episodes",
            "--episodes",
            f"data/v1/{split}/episodes.jsonl",
            "--generator",
            "hf",
            "--model",
            model,
            "--batch-size",
            str(batch_size),
            "--arm",
            arm,
            "--split",
            split,
        ]
        if shard:
            command += ["--shard", shard]
        return [("split", command)]
    episodes = {
        "examples": ("data/examples/*.json", "8"),
        "probe": ("data/probe/*.json", "16"),
    }
    names = ["smoke", "examples", "probe"] if task == "all" else [task]
    steps = []
    for name in names:
        if name == "smoke":
            steps += [
                ("nvidia-smi", ["nvidia-smi"]),
                ("tests", ["python", "-m", "pytest", "-q"]),
                ("smoke", ["python", "-c", SMOKE_PY.format(model=model)]),
            ]
        else:
            pattern, batch = episodes[name]
            steps.append(
                (
                    name,
                    [
                        "python",
                        "-m",
                        "turnguard",
                        "run-episodes",
                        "--episodes",
                        pattern,
                        "--generator",
                        "hf",
                        "--model",
                        model,
                        "--batch-size",
                        batch,
                        "--arm",
                        "A0",
                        "--split",
                        "examples",
                    ],
                )
            )
    return steps


def _split_upload(split: str, directory: Path | None = None) -> tuple[bytes, dict]:
    """Snapshot every release file; transport checksums also cover mutable dev."""
    from turnguard.data.loaders import load_split, split_fingerprint

    if split not in SPLITS:
        raise ValueError(f"Unsupported split: {split}")
    directory = Path(directory) if directory else ROOT / "data/v1" / split
    before = split_fingerprint(directory)
    load_split(split, purpose="eval", directory=directory)
    buffer = io.BytesIO()
    hashes = {}
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Split contains a symlink: {path}")
            if not path.is_file():
                continue
            name = path.relative_to(directory).as_posix()
            content = path.read_bytes()
            hashes[name] = hashlib.sha256(content).hexdigest()
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    if split_fingerprint(directory) != before:
        raise ValueError("Dataset changed during upload; retry after build completes")
    return buffer.getvalue(), {"split": split, "files": hashes, "dataset": before}


def _unpack_split(payload: bytes, info: dict, work: Path) -> Path:
    """Verify uploaded bytes before loading a model or executing an episode."""
    from turnguard.data.loaders import load_split, split_fingerprint

    split = info["split"]
    if split not in SPLITS:
        raise ValueError("Invalid uploaded split")
    directory = work / "data/v1" / split
    # Source archives can contain an older release; only uploaded data may run.
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        for member in archive.getmembers():
            name = Path(member.name)
            if not member.isfile() or name.is_absolute() or ".." in name.parts:
                raise ValueError("Invalid split archive member")
        archive.extractall(directory, filter="data")
    actual = {
        path.relative_to(directory).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in directory.rglob("*")
        if path.is_file()
    }
    if actual != info["files"] or split_fingerprint(directory) != info["dataset"]:
        raise ValueError("Uploaded split checksum mismatch")
    load_split(split, purpose="eval", directory=directory)
    return directory


def _pairs_upload(directory: str | Path | None = None) -> tuple[bytes, dict]:
    """Snapshot training pairs independently of git archive, with every checksum."""
    from turnguard.data.pairs import load_training_pairs

    directory = Path(directory) if directory else ROOT / "data/pairs/train"
    load_training_pairs(directory)
    paths = sorted(directory.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Pairs contain a symlink")
    contents = {
        p.relative_to(directory).as_posix(): p.read_bytes()
        for p in paths
        if p.is_file()
    }
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in contents.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    after = {
        p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.rglob("*")
        if p.is_file()
    }
    if after != hashes:
        raise ValueError("Pairs changed during upload")
    return buffer.getvalue(), {"split": "train", "files": hashes}


def _checkpoint_key(model: str, smoke: bool, payload: bytes, work: Path) -> str:
    """Same pairs + same training settings + same model => same checkpoints.

    A changed config or pair set gets a fresh directory instead of resuming an
    incompatible run.
    """
    from turnguard.config import load_config

    arms = work / "configs/arms.yaml"
    settings = load_config(arms)["training"]["A4"] if arms.exists() else {}
    digest = hashlib.sha256(payload)
    digest.update(json.dumps(settings, sort_keys=True).encode())
    return f"A4-{model}{'-smoke' if smoke else ''}-{digest.hexdigest()[:16]}"


def _periodic_commit(stop: "threading.Event", every: int = 300) -> None:
    """Snapshot the volume while training runs, so checkpoints survive a stop."""
    while not stop.wait(every):
        try:
            cache.commit()
        except Exception as error:  # a failed snapshot must not kill training
            print(f"warning: volume commit failed: {error}", flush=True)


def _unpack_pairs(payload: bytes, info: dict, work: Path) -> Path:
    from turnguard.data.pairs import load_training_pairs

    directory = work / "data/pairs/train"
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        for member in archive.getmembers():
            name = Path(member.name)
            if not member.isfile() or name.is_absolute() or ".." in name.parts:
                raise ValueError("Invalid pairs archive member")
        archive.extractall(directory, filter="data")
    hashes = {
        p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.rglob("*")
        if p.is_file()
    }
    if info.get("split") != "train" or hashes != info["files"]:
        raise ValueError("Uploaded pairs checksum mismatch")
    load_training_pairs(directory)
    (work / "PAIRS_UPLOAD.json").write_text(json.dumps(info, indent=2) + "\n")
    return directory


def _tune_upload(pairs: str, dev_dir: str, baseline_run: str, model: str):
    from turnguard.arms.steer_tune import BASELINE_RUN, prepare_baseline
    from turnguard.config import load_config

    dev = Path(dev_dir) if dev_dir else ROOT / "data/v1/dev"
    baseline = Path(baseline_run) if baseline_run else ROOT / "runs" / BASELINE_RUN
    model_id = load_config(ROOT / "configs/models.yaml")["models"][model]
    snapshot = prepare_baseline(dev, baseline, model_id)
    pair_bytes, pair_info = _pairs_upload(pairs or None)
    dev_bytes, dev_info = _split_upload("dev", dev)
    if snapshot["dataset"] != dev_info["dataset"]:
        raise ValueError("Dev changed after A0 verification")
    contents = {
        "pairs.tar.gz": pair_bytes,
        "dev.tar.gz": dev_bytes,
        "baseline.json": json.dumps(snapshot, ensure_ascii=False).encode(),
    }
    info = {
        "pairs": pair_info,
        "dev": dev_info,
        "files": {
            name: hashlib.sha256(value).hexdigest() for name, value in contents.items()
        },
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, value in contents.items():
            member = tarfile.TarInfo(name)
            member.size = len(value)
            archive.addfile(member, io.BytesIO(value))
    return buffer.getvalue(), info


def _unpack_tune(payload: bytes, info: dict, work: Path):
    contents = {}
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        for member in archive.getmembers():
            if (
                not member.isfile()
                or member.name not in ("pairs.tar.gz", "dev.tar.gz", "baseline.json")
                or member.name in contents
            ):
                raise ValueError("Invalid tuning archive member")
            contents[member.name] = archive.extractfile(member).read()
    actual = {
        name: hashlib.sha256(value).hexdigest() for name, value in contents.items()
    }
    if actual != info["files"] or set(actual) != {
        "pairs.tar.gz",
        "dev.tar.gz",
        "baseline.json",
    }:
        raise ValueError("Uploaded tuning checksum mismatch")
    _unpack_pairs(contents["pairs.tar.gz"], info["pairs"], work)
    _unpack_split(contents["dev.tar.gz"], info["dev"], work)
    (work / "baseline.json").write_bytes(contents["baseline.json"])
    (work / "STEER_UPLOAD.json").write_text(json.dumps(info, indent=2) + "\n")


@app.function(
    gpu="L40S",
    volumes={"/cache": cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=6 * 60 * 60,
)
def run_steer_tune(
    model: str, source: bytes, build_info: dict, payload: bytes, info: dict
):
    work = Path(tempfile.mkdtemp(prefix="turnguard-steer-"))
    with tarfile.open(fileobj=io.BytesIO(source)) as archive:
        archive.extractall(work, filter="data")
    (work / "BUILD_INFO.json").write_text(json.dumps(build_info, indent=2) + "\n")
    sys.path.insert(0, str(work))
    _unpack_tune(payload, info, work)
    command = [
        "python",
        "-m",
        "turnguard",
        "steer-tune",
        "--model",
        model,
        "--pairs",
        "data/pairs/train",
        "--dev-dir",
        "data/v1/dev",
        "--baseline-snapshot",
        "baseline.json",
    ]
    code = None
    try:
        code = subprocess.run(command, cwd=work).returncode
    finally:
        runs, steering = work / "runs", work / "steering"
        if runs.exists():
            for run in runs.glob("*steer-tune*"):
                if steering.exists():
                    shutil.copytree(steering, run / "steering", dirs_exist_ok=True)
                shutil.copyfile(work / "STEER_UPLOAD.json", run / "STEER_UPLOAD.json")
            shutil.copytree(runs, "/cache/runs", dirs_exist_ok=True)
        if code == 0 and steering.exists():
            shutil.copytree(steering, "/cache/steering", dirs_exist_ok=True)
        cache.commit()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in ("runs", "steering"):
            if (work / name).exists():
                archive.add(work / name, arcname=name)
    return {"steps": {"steer-tune": code}, "runs_tar": buffer.getvalue()}


def _print_split_metrics(runs: Path) -> None:
    """Print only metrics, including all language cells; no raw generations."""
    for path in sorted(runs.glob("*/metrics.json")):
        metrics = json.loads(path.read_text("utf-8"))
        print(f"\nSplit results: {path.parent.name}")
        for label, summary in [
            ("all", metrics),
            *[
                (f"{factor}={language}", cell)
                for factor, cells in metrics["by_language"].items()
                for language, cell in cells.items()
            ],
        ]:
            rates = []
            for name in (
                "asr",
                "asr_relayed",
                "asr_incl_relay",
                "clean_task_success",
                "clean_answer_correct",
                "authorized_follow_rate",
                "authorized_followed",
                "authorized_followed_given_exposed",
            ):
                value = summary[name]
                rates.append(
                    f"{name}={value['k']}/{value['n']} ({value['rate']}) CI95={value['ci95']}"
                )
            print(
                f"{label}: {'; '.join(rates)}; ungraded={summary['counts']['ungraded']}",
                flush=True,
            )


@app.function(
    gpu="L40S",
    volumes={"/cache": cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3 * 60 * 60,
)
def run_task(
    task: str,
    model: str,
    source: bytes,
    info: dict,
    split_payload: bytes = b"",
    split_info: dict | None = None,
    shard: str | None = None,
    batch_size: int = 8,
    smoke: bool = False,
    arm: str = "A0",
    adapter_run: str | None = None,
) -> dict:
    work = Path(tempfile.mkdtemp(prefix="turnguard-"))
    work.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(source)) as archive:
        archive.extractall(work, filter="data")
    (work / "BUILD_INFO.json").write_text(json.dumps(info, indent=2) + "\n")
    sys.path.insert(0, str(work))
    if task == "split":
        if split_info is None:
            raise ValueError("Split payload is required")
        _unpack_split(split_payload, split_info, work)
    if adapter_run:
        # Trained adapters live beside their training run on the volume; the
        # arm config refers to adapters/A4/<model> relative to the work dir.
        source_adapters = Path("/cache/runs") / adapter_run / "adapters"
        if not source_adapters.is_dir():
            raise ValueError(f"No adapters in volume run {adapter_run}")
        shutil.copytree(source_adapters, work / "adapters")
        print(f"Adapter from runs/{adapter_run}", flush=True)
    checkpoint_dir = None
    committer = None
    if task == "train-a4":
        if split_info is None:
            raise ValueError("Pairs payload is required")
        _unpack_pairs(split_payload, split_info, work)
        key = _checkpoint_key(model, smoke, split_payload, work)
        checkpoint_dir = str(CHECKPOINT_ROOT / key)
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        print(f"Checkpoints: {checkpoint_dir} (resumes automatically)", flush=True)
        stop_commits = threading.Event()
        committer = threading.Thread(
            target=_periodic_commit, args=(stop_commits,), daemon=True
        )
        committer.start()
    results = {}
    try:
        for name, command in _steps(
            task,
            model,
            split=split_info["split"] if split_info else "dev",
            shard=shard,
            batch_size=batch_size,
            smoke=smoke,
            checkpoint_dir=checkpoint_dir,
            arm=arm,
        ):
            print(f"\n===== {name} =====", flush=True)
            code = subprocess.run(command, cwd=work).returncode
            results[name] = code
            if code != 0:
                print(f"!! step {name} failed with exit code {code}; stopping")
                break
    finally:
        if committer is not None:
            stop_commits.set()
        runs = work / "runs"
        if runs.exists():
            # Also keep results on the Volume, in case the local client
            # disconnects before it receives them.
            if task == "train-a4":
                # Every job retains its own adapter beside its logs/checkpoints.
                for run in runs.glob("*train-a4*"):
                    if (work / "adapters").exists():
                        shutil.copytree(
                            work / "adapters", run / "adapters", dirs_exist_ok=True
                        )
                    shutil.copyfile(
                        work / "PAIRS_UPLOAD.json", run / "PAIRS_UPLOAD.json"
                    )
            shutil.copytree(runs, "/cache/runs", dirs_exist_ok=True)
        cache.commit()  # keep downloaded weights (and runs) for the next run
    if task == "split" and runs.exists():
        _print_split_metrics(runs)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        if runs.exists():
            archive.add(runs, arcname="runs")
        if task == "train-a4" and (work / "adapters").exists():
            archive.add(work / "adapters", arcname="adapters")
    return {"steps": results, "runs_tar": buffer.getvalue()}


@app.local_entrypoint()
def main(
    task: str = "smoke",
    model: str = "headline",
    gpu: str = "L40S",
    split: str = "dev",
    shard: str | None = None,
    background: bool = False,
    pairs: str = "",
    smoke: bool = False,
    arm: str = "A0",
    adapter_run: str = "",
    dev_dir: str = "",
    baseline_run: str = "",
) -> None:
    if task not in TASKS:
        raise SystemExit(f"--task must be one of {', '.join(TASKS)}")
    if arm != "A0" and task != "split":
        raise SystemExit("--arm requires --task split")
    if arm == "A4" and not adapter_run:
        raise SystemExit("--arm A4 requires --adapter-run <volume run name>")
    if gpu not in GPU_CHOICES:
        raise SystemExit(f"--gpu must be one of {', '.join(GPU_CHOICES)}")
    if model not in ("headline", "dev"):
        raise SystemExit("--model must be headline or dev")
    if task == "train-a4" and gpu not in ("A100-80GB", "H100"):
        raise SystemExit("train-a4 requires --gpu A100-80GB or --gpu H100")
    if smoke and task != "train-a4":
        raise SystemExit("--smoke requires --task train-a4")
    from turnguard.config import load_config
    from turnguard.data.loaders import parse_shard

    parse_shard(shard)
    if split not in SPLITS:
        raise SystemExit(f"--split must be one of {', '.join(SPLITS)}")
    if shard and task != "split":
        raise SystemExit("--shard requires --task split")
    split_payload, split_info = _split_upload(split) if task == "split" else (b"", None)
    if task == "train-a4":
        split_payload, split_info = _pairs_upload(pairs or None)
    if task == "steer-tune":
        split_payload, split_info = _tune_upload(pairs, dev_dir, baseline_run, model)
    batch_size = load_config(ROOT / "configs/models.yaml")["inference"]["batch_size"]
    source, info = _source_archive()
    if info["working_tree_had_uncommitted_changes"]:
        print("warning: uncommitted changes exist and are NOT sent to Modal")
    print(f"Running {task} on {gpu} with {model} at commit {info['git_commit'][:7]}")
    remote_options = {"gpu": gpu}
    if task in ("train-a4", "steer-tune"):
        remote_options["timeout"] = 6 * 60 * 60
    if task == "steer-tune":
        remote = run_steer_tune.with_options(**remote_options)
        args = (model, source, info, split_payload, split_info)
        if background:
            call = remote.spawn(*args)
            print(f"Started steering grid in the background: {call.object_id}")
            print("Fetch: modal volume ls turnguard-hf-cache runs")
            return
        result = remote.remote(*args)
        with tarfile.open(fileobj=io.BytesIO(result["runs_tar"])) as archive:
            archive.extractall(ROOT, filter="data")
        if result["steps"]["steer-tune"]:
            raise SystemExit(1)
        return
    if background:
        # With `modal run --detach`, spawn() lets this terminal exit at once
        # (Ctrl+C or a closed laptop can no longer cancel the GPU job). Results
        # are copied to the volume's runs/ when the container finishes.
        call = run_task.with_options(**remote_options).spawn(
            task,
            model,
            source,
            info,
            split_payload,
            split_info,
            shard,
            batch_size,
            smoke,
            arm,
            adapter_run or None,
        )
        print(f"Started in the background: {call.object_id}")
        print("Fetch when done: modal volume ls turnguard-hf-cache runs")
        return
    result = run_task.with_options(**remote_options).remote(
        task,
        model,
        source,
        info,
        split_payload,
        split_info,
        shard,
        batch_size,
        smoke,
        arm,
        adapter_run or None,
    )
    with tarfile.open(fileobj=io.BytesIO(result["runs_tar"])) as archive:
        names = sorted(
            {
                m.name.split("/")[1]
                for m in archive.getmembers()
                if m.name.startswith("runs/") and m.name.count("/") >= 1
            }
        )
        archive.extractall(ROOT, filter="data")
    print("\n===== summary =====")
    for name, code in result["steps"].items():
        print(f"{name:10s} {'ok' if code == 0 else f'FAILED (exit {code})'}")
    for name in names:
        print(f"saved runs/{name}")
    if any(code != 0 for code in result["steps"].values()):
        raise SystemExit(1)
