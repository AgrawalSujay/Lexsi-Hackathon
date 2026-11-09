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
import tomllib
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
GPU_CHOICES = ("L40S", "A100-80GB", "H100")
TASKS = ("smoke", "examples", "probe", "all", "split")
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
) -> list[tuple[str, list[str]]]:
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
            "A0",
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


def _split_upload(split: str) -> tuple[bytes, dict]:
    """Snapshot every release file; transport checksums also cover mutable dev."""
    from turnguard.data.loaders import load_split, split_fingerprint

    if split not in SPLITS:
        raise ValueError(f"Unsupported split: {split}")
    directory = ROOT / "data/v1" / split
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
                "asr_incl_relay",
                "clean_task_success",
                "authorized_follow_rate",
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
    results = {}
    try:
        for name, command in _steps(
            task,
            model,
            split=split_info["split"] if split_info else "dev",
            shard=shard,
            batch_size=batch_size,
        ):
            print(f"\n===== {name} =====", flush=True)
            code = subprocess.run(command, cwd=work).returncode
            results[name] = code
            if code != 0:
                print(f"!! step {name} failed with exit code {code}; stopping")
                break
    finally:
        runs = work / "runs"
        if runs.exists():
            # Also keep results on the Volume, in case the local client
            # disconnects before it receives them.
            shutil.copytree(runs, "/cache/runs", dirs_exist_ok=True)
        cache.commit()  # keep downloaded weights (and runs) for the next run
    if task == "split" and runs.exists():
        _print_split_metrics(runs)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        if runs.exists():
            archive.add(runs, arcname="runs")
    return {"steps": results, "runs_tar": buffer.getvalue()}


@app.local_entrypoint()
def main(
    task: str = "smoke",
    model: str = "headline",
    gpu: str = "L40S",
    split: str = "dev",
    shard: str | None = None,
    background: bool = False,
) -> None:
    if task not in TASKS:
        raise SystemExit(f"--task must be one of {', '.join(TASKS)}")
    if gpu not in GPU_CHOICES:
        raise SystemExit(f"--gpu must be one of {', '.join(GPU_CHOICES)}")
    if model not in ("headline", "dev"):
        raise SystemExit("--model must be headline or dev")
    from turnguard.config import load_config
    from turnguard.data.loaders import parse_shard

    parse_shard(shard)
    if split not in SPLITS:
        raise SystemExit(f"--split must be one of {', '.join(SPLITS)}")
    if shard and task != "split":
        raise SystemExit("--shard requires --task split")
    split_payload, split_info = _split_upload(split) if task == "split" else (b"", None)
    batch_size = load_config(ROOT / "configs/models.yaml")["inference"]["batch_size"]
    source, info = _source_archive()
    if info["working_tree_had_uncommitted_changes"]:
        print("warning: uncommitted changes exist and are NOT sent to Modal")
    print(f"Running {task} on {gpu} with {model} at commit {info['git_commit'][:7]}")
    if background:
        # With `modal run --detach`, spawn() lets this terminal exit at once
        # (Ctrl+C or a closed laptop can no longer cancel the GPU job). Results
        # are copied to the volume's runs/ when the container finishes.
        call = run_task.with_options(gpu=gpu).spawn(
            task, model, source, info, split_payload, split_info, shard, batch_size
        )
        print(f"Started in the background: {call.object_id}")
        print("Fetch when done: modal volume ls turnguard-hf-cache runs")
        return
    result = run_task.with_options(gpu=gpu).remote(
        task, model, source, info, split_payload, split_info, shard, batch_size
    )
    with tarfile.open(fileobj=io.BytesIO(result["runs_tar"])) as archive:
        names = sorted(
            {
                m.name.split("/")[1]
                for m in archive.getmembers()
                if m.name.count("/") >= 1
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
