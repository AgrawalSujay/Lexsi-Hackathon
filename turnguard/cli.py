"""Expose commands for TurnGuard experiments and reports."""

import argparse
import glob
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from turnguard.arms.runtime import resolve_arm
from turnguard.config import load_config
from turnguard.harness import run_episodes
from turnguard.metrics import compute_metrics
from turnguard.models import HFGenerator, StubGenerator
from turnguard.protocol import PROTOCOL_VERSION
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
    arm_spec = resolve_arm(args.arm, load_config(args.arms_config))
    script_info = {}
    decoding = config["decoding"]
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else config.get("inference", {}).get("batch_size", 8)
    )
    if args.generator == "stub":
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
        generator = StubGenerator(scripts)
        script_info = {
            "stub_script": str(script_path),
            "stub_script_sha256": hashlib.sha256(script_bytes).hexdigest(),
        }
    else:
        if not args.model or args.model not in config["models"]:
            raise ValueError(
                f"--model must be a configured key: {', '.join(config['models'])}"
            )
        generator = HFGenerator(
            config["models"][args.model],
            max_new_tokens=decoding["max_new_tokens"],
            batch_size=batch_size,
            seed=config["seed"],
        )
    arm_spec.apply(generator)
    decoding = config["decoding"]
    extra = arm_spec.system_prompt
    if extra is None:
        extra = config.get("system_extra")
    elif config.get("system_extra") not in (None, extra):
        raise ValueError("system_extra conflicts with the configured arm prompt")
    if extra is not None and not isinstance(extra, str):
        raise ValueError("system_extra must be a string or null")
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
        "arm_config": asdict(arm_spec),
        "split": args.split,
        "generator": args.generator,
        "model": generator.name,
        "system_extra": extra,
        "episode_paths": [str(path) for path in paths],
        "model_key": args.model if args.generator == "hf" else None,
        "inference": {"batch_size": batch_size},
        "protocol_version": PROTOCOL_VERSION,
        **script_info,
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


def _eval_xstest(args: argparse.Namespace) -> int:
    from turnguard.evals.xstest import (
        CLASSIFIER_VERSION,
        auditkit_scores,
        evaluate_xstest,
        load_items,
        select_items,
        teacher_judge,
    )

    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    eval_config = config["xstest"]
    if eval_config["classifier_version"] != CLASSIFIER_VERSION:
        raise ValueError("Unsupported XSTest classifier version")
    if eval_config["max_new_tokens"] != 256:
        raise ValueError("XSTest requires max_new_tokens=256")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be a positive integer")
    arms_path = (
        Path(args.arms_config) if args.arms_config else root / config["arms_config"]
    )
    arm_spec = resolve_arm(args.arm, load_config(arms_path))
    model_config = load_config(root / config["models_config"])
    items = select_items(
        load_items(eval_config, cache_dir=root / ".cache/hf"),
        args.limit,
        seed=config["seed"],
    )
    classifier, client = (
        teacher_judge(root, eval_config) if args.judge == "teacher" else (None, None)
    )
    generator = HFGenerator(
        model_config["models"][args.model],
        max_new_tokens=256,
        batch_size=config["batch_size"],
        seed=config["seed"],
    )
    arm_spec.apply(generator)
    run_dir = new_run_dir(args.arm, args.model, "xstest", root=args.root)
    resolved = {
        **config,
        "config_path": str(config_path),
        "benchmark": "xstest",
        "model_key": args.model,
        "model": generator.name,
        "arm": args.arm,
        "arm_config": asdict(arm_spec),
        "judge": args.judge,
        "limit": args.limit,
        "selected_ids": [item.id for item in items],
        "decoding": {
            "do_sample": False,
            "max_new_tokens": 256,
            "chat_template": "tokenizer",
            "padding_side": "left",
        },
    }
    if client:
        resolved["judge_config"] = {
            "model": client.model,
            "api_base": client.api_base,
            "sampling": client.sampling,
            "prompt_version": client.template_version,
        }
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    records, metrics = evaluate_xstest(
        items,
        generator,
        system_prompt=arm_spec.system_prompt,
        judge=classifier,
        batch_size=config["batch_size"],
    )
    (run_dir / "episodes.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if client:
        (run_dir / "judge_calls.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in client.history), encoding="utf-8"
        )
    audit = auditkit_scores(records, cache_dir=run_dir / ".cache")
    (run_dir / "auditkit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Run directory: {run_dir}")
    print(json.dumps(metrics, indent=2))
    return int(metrics["generation_errors"] > 0 or metrics.get("judge_errors", 0) > 0)


def _eval_mmlu(args: argparse.Namespace) -> int:
    from turnguard.evals.global_mmlu import (
        EXTRACTION_VERSION,
        auditkit_scores,
        evaluate_mmlu,
        load_items,
        select_items,
    )

    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    benchmark = config["global_mmlu"]
    if benchmark["extraction_version"] != EXTRACTION_VERSION:
        raise ValueError("Unsupported Global-MMLU extraction version")
    if type(benchmark["max_new_tokens"]) is not int or benchmark["max_new_tokens"] < 1:
        raise ValueError("Global-MMLU token limit must be a positive integer")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be a positive integer")
    if args.judge != "none":
        raise ValueError(
            "Global-MMLU uses answer keys; --judge teacher is for XSTest only"
        )
    arms_path = (
        Path(args.arms_config) if args.arms_config else root / config["arms_config"]
    )
    arm_spec = resolve_arm(args.arm, load_config(arms_path))
    models = load_config(root / config["models_config"])
    items = select_items(
        load_items(benchmark, cache_dir=root / ".cache/hf"),
        args.limit,
        seed=config["seed"],
    )
    generator = HFGenerator(
        models["models"][args.model],
        max_new_tokens=benchmark["max_new_tokens"],
        batch_size=config["batch_size"],
        seed=config["seed"],
    )
    arm_spec.apply(generator)
    run_dir = new_run_dir(args.arm, args.model, "mmlu", root=args.root)
    resolved = {
        **config,
        "config_path": str(config_path),
        "benchmark": "global_mmlu",
        "model_key": args.model,
        "model": generator.name,
        "arm": args.arm,
        "arm_config": asdict(arm_spec),
        "judge": "none",
        "limit": args.limit,
        "selected_ids": [item.id for item in items],
        "decoding": {
            "do_sample": False,
            "max_new_tokens": benchmark["max_new_tokens"],
            "chat_template": "tokenizer",
            "padding_side": "left",
        },
    }
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    records, metrics = evaluate_mmlu(
        items,
        generator,
        instructions=benchmark["instructions"],
        system_prompt=arm_spec.system_prompt,
        batch_size=config["batch_size"],
    )
    (run_dir / "episodes.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    audit = auditkit_scores(records, cache_dir=run_dir / ".cache")
    (run_dir / "auditkit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Run directory: {run_dir}")
    print(json.dumps(metrics, indent=2))
    return int(
        any(group["generation_errors"] for group in metrics["languages"].values())
    )


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
    run.add_argument("--batch-size", type=int, help="HF microbatch size")
    run.add_argument(
        "--model", help="Model key from configs/models.yaml (HF generator)"
    )
    run.add_argument("--arm", default="A0")
    run.add_argument(
        "--arms-config",
        default=str(Path(__file__).resolve().parents[1] / "configs/arms.yaml"),
    )
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
    xstest = commands.add_parser(
        "eval-xstest", help="Evaluate plain-chat XSTest refusal behavior"
    )
    xstest.add_argument("--model", choices=("dev", "headline"), required=True)
    xstest.add_argument("--arm", default="A0")
    xstest.add_argument(
        "--limit", type=int, help="Total items, balanced safe/unsafe for smoke runs"
    )
    xstest.add_argument("--judge", choices=("none", "teacher"), default="none")
    xstest.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs/evals.yaml"),
    )
    xstest.add_argument("--arms-config", help="Override the shared arm configuration")
    xstest.add_argument("--root", help="Override output root for runs/")
    mmlu = commands.add_parser(
        "eval-mmlu", help="Evaluate English/Hindi Global-MMLU-Lite"
    )
    mmlu.add_argument("--model", choices=("dev", "headline"), required=True)
    mmlu.add_argument("--arm", default="A0")
    mmlu.add_argument(
        "--limit", type=int, help="Total smoke items, balanced across en/hi"
    )
    mmlu.add_argument(
        "--judge",
        choices=("none", "teacher"),
        default="none",
        help="MMLU uses answer keys; teacher judging is XSTest-only",
    )
    mmlu.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs/evals.yaml"),
    )
    mmlu.add_argument("--arms-config", help="Override the shared arm configuration")
    mmlu.add_argument("--root", help="Override output root for runs/")
    build = commands.add_parser("build-data", help="Build validated episode specs")
    build.add_argument("--split", choices=("train", "dev", "test"), required=True)
    build.add_argument("--teacher", choices=("stub", "litellm"), required=True)
    build.add_argument(
        "--limit", type=int, help="Number of injected bases (twins are extra)"
    )
    build.add_argument("--config", type=Path, help="Data builder YAML configuration")
    build.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate calls/tokens without calling a teacher or writing files",
    )
    build.add_argument(
        "--backtranslate",
        action="store_true",
        help="Cache and log a 10% backtranslation review sample",
    )
    freeze = commands.add_parser("freeze", help="Freeze the complete test release")
    freeze.add_argument("--split", choices=("test",), required=True)
    for name in ("train", "eval-external", "report"):
        commands.add_parser(name, help="not implemented yet")
    args = parser.parse_args(argv)
    if args.command == "eval-mmlu":
        try:
            return _eval_mmlu(args)
        except (ValueError, OSError, KeyError, ImportError, yaml.YAMLError) as error:
            parser.error(
                f"Global-MMLU setup/run failed ({type(error).__name__}); "
                "check configuration and dependencies"
            )
    if args.command == "eval-xstest":
        try:
            return _eval_xstest(args)
        except (ValueError, OSError, KeyError, ImportError, yaml.YAMLError) as error:
            # Do not echo third-party errors that might contain unsafe text.
            parser.error(
                f"XSTest setup/run failed ({type(error).__name__}); "
                "check configuration and dependencies"
            )
    if args.command == "freeze":
        from turnguard.data.freeze import freeze_split

        try:
            directory = freeze_split(args.split)
        except (ValueError, OSError, KeyError) as error:
            parser.error(str(error))
        print(f"Frozen split: {directory}")
        return 0
    if args.command == "build-data":
        from turnguard.data.builder import build_data

        try:
            report = build_data(
                args.split,
                teacher=args.teacher,
                limit=args.limit,
                config_path=args.config,
                dry_run=args.dry_run,
                backtranslate=args.backtranslate,
            )
        except (ValueError, OSError, KeyError, ImportError, yaml.YAMLError) as error:
            parser.error(str(error))
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "counts",
                        "rejected",
                        "requested_shortfall",
                        "plan_type_shortfalls",
                        "teacher_estimate",
                        "dedup",
                    )
                },
                indent=2,
            )
        )
        if args.dry_run:
            print("Dry-run: no teacher calls or output files written.")
        else:
            print(f"Output: {report['output']}/ (factor details in summary.json)")
        return 0
    if args.command == "run-episodes":
        try:
            return _run(args)
        except (ValueError, OSError, KeyError, ImportError, yaml.YAMLError) as error:
            parser.error(str(error))
    print("not implemented yet")
    return 0
