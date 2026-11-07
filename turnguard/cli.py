"""Expose commands for TurnGuard experiments and reports."""

import argparse
import glob
import hashlib
import json
from pathlib import Path

import yaml

from turnguard.config import load_config
from turnguard.harness import run_episodes
from turnguard.metrics import compute_metrics
from turnguard.models import StubGenerator
from turnguard.runs import new_run_dir
from turnguard.schema import EpisodeSpec, validate_episode


def _run(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    paths = [
        Path(p).resolve()
        for p in sorted(glob.glob(str(Path(args.episodes).expanduser())))
    ]
    if not paths:
        raise ValueError(f"No episode files match: {args.episodes}")
    specs = [EpisodeSpec.from_json(path.read_text(encoding="utf-8")) for path in paths]
    by_id = {spec.id: spec for spec in specs}
    if len(by_id) != len(specs):
        raise ValueError("Episode IDs must be unique")
    for spec in specs:
        if spec.split != args.split:
            raise ValueError(
                f"{spec.id}: split {spec.split!r} does not match --split {args.split!r}"
            )
        problems = validate_episode(spec, twins=by_id)
        if problems:
            raise ValueError(f"{spec.id}: {'; '.join(problems)}")
    if args.generator != "stub":
        raise ValueError("HF generator will be implemented in task 02 step 7")
    if not args.stub_script:
        raise ValueError("--stub-script is required for --generator stub")
    script_path = Path(args.stub_script).expanduser().resolve()
    script_bytes = script_path.read_bytes()
    scripts = json.loads(script_bytes)
    if not isinstance(scripts, dict) or not all(
        isinstance(k, str)
        and isinstance(v, list)
        and all(isinstance(item, str) for item in v)
        for k, v in scripts.items()
    ):
        raise ValueError("Stub scripts must map episode IDs to lists of strings")
    missing = by_id.keys() - scripts.keys()
    if missing:
        raise ValueError(f"Missing stub scripts: {sorted(missing)}")
    decoding = config["decoding"]
    extra = config.get("system_extra")
    if extra is not None and not isinstance(extra, str):
        raise ValueError("system_extra must be a string or null")
    generator = StubGenerator(scripts)
    records = run_episodes(
        specs,
        generator,
        max_steps_per_turn=decoding["max_steps_per_user_turn"],
        tool_result_role=decoding["tool_result_role"],
        system_extra=extra,
        arm=args.arm,
    )
    metrics = compute_metrics(records, specs)
    resolved = {
        **config,
        "config_path": str(config_path),
        "arm": args.arm,
        "split": args.split,
        "generator": args.generator,
        "model": generator.name,
        "system_extra": extra,
        "episode_paths": [str(path) for path in paths],
        "stub_script": str(script_path),
        "stub_script_sha256": hashlib.sha256(script_bytes).hexdigest(),
    }
    run_dir = new_run_dir(args.arm, generator.name, args.split, root=args.root)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    (run_dir / "episodes.jsonl").write_text(
        "".join(record.to_json() + "\n" for record in records), encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Run directory: {run_dir}")
    print(f"{'Episode':<25} {'Type':<12} {'Tag':<10} {'Steps':>5} {'Malformed':>9}")
    for record in records:
        print(
            f"{record.episode_id:<25} "
            f"{by_id[record.episode_id].episode_type:<12} "
            f"{record.result['tag']:<10} "
            f"{record.steps:>5} {record.malformed_turns:>9}"
        )
    for name in (
        "asr",
        "utility_under_attack",
        "clean_task_success",
        "authorized_follow_rate",
        "malformed_rate",
    ):
        value = metrics[name]
        estimate = (
            "n/a"
            if value["rate"] is None
            else (
                f"{value['rate']:.3f} (95% CI "
                f"{value['ci95'][0]:.3f}–{value['ci95'][1]:.3f})"
            )
        )
        print(f"{name}: {value['k']}/{value['n']} = {estimate}")
    print("first_violation_step: " + json.dumps(metrics["first_violation_step"]))
    print(
        f"Ungraded: {metrics['counts']['ungraded']}; "
        f"generation errors: {metrics['counts']['generation_errors']}"
    )
    return int(any("error" in record.result for record in records))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="turnguard", description="Evaluate and repair multi-turn prompt injection."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run-episodes", help="Run episodes and write traces and metrics"
    )
    run.add_argument(
        "--episodes", required=True, help="JSON episode path or quoted glob"
    )
    run.add_argument("--generator", choices=("stub", "hf"), required=True)
    run.add_argument("--stub-script")
    run.add_argument(
        "--model", help="Model key from configs/models.yaml (HF generator)"
    )
    run.add_argument("--arm", default="A0")
    run.add_argument(
        "--split", choices=("examples", "train", "dev", "test"), required=True
    )
    run.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs/models.yaml"),
    )
    run.add_argument(
        "--root", help="Override the repository root for output under <root>/runs"
    )
    for name in ("build-data", "freeze", "train", "eval-external", "report"):
        commands.add_parser(name, help="not implemented yet")
    args = parser.parse_args(argv)
    if args.command == "run-episodes":
        try:
            return _run(args)
        except (ValueError, OSError, KeyError, yaml.YAMLError) as error:
            parser.error(str(error))
    print("not implemented yet")
    return 0
