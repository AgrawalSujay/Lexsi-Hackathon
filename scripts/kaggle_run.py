"""Run the verified TurnGuard bundle on Kaggle; no GitHub access or API judge."""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path

from kaggle_bundle import SPLITS, sha, unpack, verify_release


class RunnerError(ValueError):
    """A runner-owned error containing no model output or credentials."""


def get_token() -> str:
    if token := os.environ.get("HF_TOKEN"):
        return token
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        raise RunnerError(
            "Set HF_TOKEN from the Kaggle notebook secret cell first"
        ) from None

    token = UserSecretsClient().get_secret("HF_TOKEN")
    if not token:
        raise RunnerError("Add and enable the Kaggle secret HF_TOKEN")
    return token


def restore_checkpoints(source: Path, target: Path) -> None:
    """Accept an extracted training output or its downloadable zip."""
    import tempfile

    if source.is_file():
        from kaggle_bundle import safe_name

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with zipfile.ZipFile(source) as archive:
                if len(archive.namelist()) != len(set(archive.namelist())):
                    raise RunnerError("Duplicate resume archive entries")
                expected = json.loads(archive.read("OUTPUT_CHECKSUMS.json"))
                actual = {
                    n: sha(archive.read(n))
                    for n in archive.namelist()
                    if n != "OUTPUT_CHECKSUMS.json"
                }
                if actual != expected:
                    raise RunnerError("Resume archive checksum mismatch")
                for item in archive.infolist():
                    safe_name(item.filename)
                    if (
                        item.is_dir()
                        or (item.external_attr >> 16) & 0o170000 == 0o120000
                    ):
                        raise RunnerError("Invalid resume archive entry")
                archive.extractall(directory)
            restore_checkpoints(directory, target)
        return
    candidates = list(source.rglob("turnguard_training.json"))
    if len(candidates) != 1:
        raise RunnerError(
            "Resume input must contain exactly one A4 checkpoint identity"
        )
    directory = candidates[0].parent
    if any(path.is_symlink() for path in (directory, *directory.rglob("*"))):
        raise RunnerError("Checkpoint input contains a symlink")
    if target.exists() and any(target.iterdir()):
        # Never overwrite a newer live checkpoint with an attached old output.
        raise RunnerError("Checkpoint directory already populated; omit --resume-from")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(directory, target, dirs_exist_ok=True)


def command(args, repo: Path, job: Path, shard: str | None) -> list[str]:
    base = [sys.executable, "-m", "turnguard"]
    common = ["--model", args.model, "--dtype", args.dtype]
    arms = ["--arms-config", str(job / "arms.json")]
    if args.task == "split":
        return (
            base
            + [
                "run-episodes",
                "--episodes",
                str(repo / "data/v1" / args.split),
                "--split",
                args.split,
                "--generator",
                "hf",
                "--arm",
                args.arm,
                "--batch-size",
                str(args.batch_size),
                "--root",
                str(job),
                *common,
                *arms,
            ]
            + (["--shard", shard] if shard else [])
        )
    if args.task == "eval-side":
        return base + [
            "eval-side",
            "--arm",
            args.arm,
            "--out",
            str(job / "run"),
            "--batch-size",
            str(args.batch_size),
            *common,
            *arms,
        ]
    return (
        base
        + [
            "train",
            "--arm",
            "A4",
            "--pairs",
            str(repo / "data/pairs/train"),
            "--output",
            str(job / "adapters/A4/dev"),
            "--run-root",
            str(job),
            "--checkpoint-dir",
            str(args.output_root / "checkpoints/A4/dev"),
            "--max-length",
            str(args.max_length),
            *common,
        ]
        + (["--smoke"] if args.smoke else [])
    )


def prepare_arm(args, repo: Path) -> dict:
    # JSON is also valid YAML; bootstrap does not need PyYAML.
    if args.arm == "A0" or args.task == "train-a4":
        return {"arms": {"A0": {"system_prompt": None}}}
    if args.adapter_dir is None:
        raise RunnerError("A4 evaluation requires --adapter-dir from Tiny Aya training")
    directory = args.adapter_dir.expanduser().resolve()
    args.adapter_dir = directory
    config = json.loads((directory / "adapter_config.json").read_text())
    # Read configured model ID without importing GPU modules (after install).
    from turnguard.config import load_config

    expected = load_config(repo / "configs/models.yaml")["models"][args.model]
    if config.get("base_model_name_or_path") != expected:
        raise RunnerError("Adapter base model does not match Tiny Aya")
    if not any(directory.glob("adapter_model.*")):
        raise RunnerError("Adapter weights missing")
    return {"arms": {"A4": {"adapter": str(directory)}}}


def archive_output(job: Path, output: Path) -> Path:
    """Output files remain local; archive them without printing any generations."""
    hashes = {}
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(job.rglob("*")):
            if path.is_symlink():
                raise RunnerError("Job output contains a symlink")
            if path.is_file():
                name = path.relative_to(job).as_posix()
                archive.write(path, name)
                hashes[name] = sha(path.read_bytes())
        archive.writestr("OUTPUT_CHECKSUMS.json", json.dumps(hashes, indent=2) + "\n")
    return output


def run(
    args, *, run_process=subprocess.run, popen=subprocess.Popen, token_fn=get_token
):
    if sys.version_info < (3, 12):  # noqa: UP036 - bootstrap may run before installation
        raise RunnerError("Python 3.12+ required; use the venv cell in notes/kaggle.md")
    if args.dtype != "fp16":
        raise RunnerError("Kaggle T4 runner requires --dtype fp16 (T4 has no bf16)")
    if args.parallel_shards == 2 and (args.task != "split" or args.shard):
        raise RunnerError("--parallel-shards 2 requires split without --shard")
    if args.shard and args.task != "split":
        raise RunnerError("--shard applies only to split")
    if args.batch_size < 1 or args.max_length < 1:
        raise RunnerError("Batch size and max length must be positive")
    args.bundle = args.bundle.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.task == "train-a4":
        args.arm = "A4"
    manifest = unpack(args.bundle, args.work_dir)
    repo = args.work_dir.resolve()
    # Source provenance works in an archive with no .git directory.
    build = {
        "git_commit": manifest["git_commit"],
        "git_dirty": False,
        "kaggle_bundle_sha256": sha(args.bundle.read_bytes()),
    }
    if "BUILD_INFO.json" in manifest["files"]:
        raise RunnerError("Reserved BUILD_INFO.json exists in bundle")
    if (repo / "BUILD_INFO.json").exists() and json.loads(
        (repo / "BUILD_INFO.json").read_text()
    ) != build:
        raise RunnerError("Checkout provenance differs; use a new --work-dir")
    # Repeated commands reuse this derived file, without modifying packed source.
    (repo / "BUILD_INFO.json").write_text(json.dumps(build) + "\n")
    env = dict(os.environ)
    env.update(
        HF_TOKEN=token_fn(),
        TOKENIZERS_PARALLELISM="false",
        HF_HOME=str(args.output_root / "hf-cache"),
    )
    extras = "gpu,eval" if args.task == "eval-side" else "gpu"
    run_process(
        [sys.executable, "-m", "pip", "install", "-e", f".[{extras}]"],
        cwd=repo,
        env=env,
        check=True,
    )
    sys.path.insert(0, str(repo))
    from turnguard.data.loaders import load_split, parse_shard

    parse_shard(args.shard)
    if args.task == "split":
        verify_release(
            repo / "data/v1" / args.split, frozen=args.split.startswith("test")
        )
        load_split(args.split, purpose="eval", root=repo)
    arm_config = prepare_arm(args, repo)
    checkpoints = args.output_root / "checkpoints/A4/dev"
    if args.resume_from:
        if args.task != "train-a4":
            raise RunnerError("--resume-from requires train-a4")
        restore_checkpoints(args.resume_from, checkpoints)
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    label = f"{args.arm if args.task != 'train-a4' else 'A4'}__dev__{args.task}"
    if args.task == "split":
        label += f"-{args.split}"
    group = f"{label}__{stamp}-{uuid.uuid4().hex[:6]}"
    processes, outputs, jobs = [], [], []
    shards = ["1/2", "2/2"] if args.parallel_shards == 2 else [args.shard]
    failed = False
    try:
        for index, shard in enumerate(shards):
            name = group + ("__shard-" + shard.replace("/", "-of-") if shard else "")
            job = args.output_root / name
            job.mkdir(parents=True)
            jobs.append(job)
            (job / "arms.json").write_text(json.dumps(arm_config) + "\n")
            (job / "BUILD_INFO.json").write_text(json.dumps(build) + "\n")
            (job / "job.json").write_text(
                json.dumps(
                    {
                        "task": args.task,
                        "arm": args.arm,
                        "dtype": args.dtype,
                        "model": args.model,
                        "shard": shard,
                        "gpu": index,
                        "bundle": build,
                        "parallel_shards": args.parallel_shards,
                    },
                    indent=2,
                )
                + "\n"
            )
            # Capture all child output locally. Never print unsafe generations,
            # provider tracebacks or training examples to the notebook output.
            stdout = (job / "process.log").open("wb")
            child_env = {**env, "CUDA_VISIBLE_DEVICES": str(index)}
            try:
                process = popen(
                    command(args, repo, job, shard),
                    cwd=repo,
                    env=child_env,
                    stdout=stdout,
                    stderr=subprocess.STDOUT,
                )
            finally:
                stdout.close()
            processes.append(process)
        for process in processes:
            failed |= process.wait() != 0
    finally:
        # If launch/wait failed, finish/stop children before snapshotting outputs.
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
        if args.task == "train-a4" and checkpoints.exists() and jobs:
            shutil.copytree(checkpoints, jobs[0] / "checkpoints/A4/dev")
        for job in jobs:
            # Capture adapter identity/content hashes alongside the resolved path.
            if args.adapter_dir and args.arm == "A4":
                (job / "adapter_hashes.json").write_text(
                    json.dumps(
                        {
                            path.name: sha(path.read_bytes())
                            for path in args.adapter_dir.glob("*")
                            if path.is_file()
                        },
                        indent=2,
                    )
                    + "\n"
                )
            outputs.append(
                str(archive_output(job, args.output_root / f"out_{job.name}.zip"))
            )
    if failed:
        raise RunnerError(
            "Kaggle job failed; inspect process.log inside its output zip"
        )
    return {"outputs": outputs, "git_commit": manifest["git_commit"], "judge_calls": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("split", "train-a4", "eval-side"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("/kaggle/working/tg"))
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--split", choices=SPLITS, default="dev")
    parser.add_argument("--arm", choices=("A0", "A4"), default="A0")
    parser.add_argument("--model", choices=("dev",), default="dev")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--shard")
    parser.add_argument("--parallel-shards", type=int, choices=(1, 2), default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args(argv)
    if args.dtype != "fp16":
        parser.error("T4 does not support bf16; use --dtype fp16")
    if sys.version_info < (3, 12):  # noqa: UP036 - bootstrap may run before installation
        parser.error("Python 3.12+ required; use the venv cell in notes/kaggle.md")
    try:
        report = run(args)
    except RunnerError as error:
        parser.exit(1, str(error) + "\n")
    except (ValueError, OSError, KeyError, subprocess.SubprocessError):
        # Do not echo exception text: a library error may contain unsafe output.
        parser.exit(
            1, "Kaggle runner failed; inspect local output files and notes/kaggle.md.\n"
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
