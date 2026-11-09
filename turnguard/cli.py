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
from turnguard.runs import RunTimer, new_run_dir
from turnguard.schema import EpisodeSpec, validate_episode


def _run(args: argparse.Namespace) -> int:
    timer = RunTimer()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    from turnguard.data.loaders import load_split, select_episodes, split_fingerprint

    source = Path(args.episodes).expanduser().resolve()
    dataset = None
    if source.is_dir() or source.name == "episodes.jsonl":
        directory = source if source.is_dir() else source.parent
        before = split_fingerprint(directory)
        specs = load_split(args.split, purpose="eval", directory=directory)
        if split_fingerprint(directory) != before:
            raise ValueError(
                "Dataset changed while loading; retry after build completes"
            )
        dataset = {"directory": str(directory), "split": args.split, **before}
        paths = [directory / "episodes.jsonl"]
    else:
        if args.split in ("test", "test_cueflip"):
            raise ValueError("Test evaluation requires a frozen built split directory")
        paths = [
            Path(p).resolve()
            for p in sorted(glob.glob(str(Path(args.episodes).expanduser())))
        ]
        if not paths:
            raise ValueError(f"No episode files match: {args.episodes}")
        specs = [
            EpisodeSpec.from_json(path.read_text(encoding="utf-8")) for path in paths
        ]
        by_id = {spec.id: spec for spec in specs}
        if len(by_id) != len(specs):
            raise ValueError("Episode IDs must be unique")
        for spec in specs:
            if spec.split != args.split:
                raise ValueError(
                    f"{spec.id}: split {spec.split!r} does not match "
                    f"--split {args.split!r}"
                )
            problems = validate_episode(spec, twins=by_id)
            if problems:
                raise ValueError(f"{spec.id}: {'; '.join(problems)}")
    specs = select_episodes(specs, limit=args.limit, shard=args.shard)
    if not specs:
        raise ValueError("No episodes selected for this limit/shard")
    by_id = {spec.id: spec for spec in specs}
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
    timer.start_generation()
    records = run_episodes(
        specs,
        generator,
        max_steps_per_turn=decoding["max_steps_per_user_turn"],
        tool_result_role=decoding["tool_result_role"],
        system_extra=extra,
        arm=args.arm,
    )
    timer.finish_generation()
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
        "dataset": dataset,
        "limit": args.limit,
        "shard": args.shard,
        "selected_ids": [spec.id for spec in specs],
        "model_key": args.model if args.generator == "hf" else None,
        "inference": {"batch_size": batch_size},
        "protocol_version": PROTOCOL_VERSION,
        **script_info,
    }
    label = args.split
    if args.shard:
        label += "-shard-" + args.shard.replace("/", "-of-")
    run_dir = new_run_dir(args.arm, generator.name, label, root=args.root)
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
    timer.finish(run_dir)
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
        "asr_relayed",
        "asr_incl_relay",
        "utility_under_attack",
        "clean_task_success",
        "clean_answer_correct",
        "authorized_follow_rate",
        "authorized_followed",
        "authorized_followed_given_exposed",
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
    print("by_scenario: " + json.dumps(metrics["by_scenario"], ensure_ascii=False))
    print("by_language: " + json.dumps(metrics["by_language"], ensure_ascii=False))
    print("first_violation_step: " + json.dumps(metrics["first_violation_step"]))
    print(
        f"Ungraded: {metrics['counts']['ungraded']}; "
        f"generation errors: {metrics['counts']['generation_errors']}"
    )
    return int(any("error" in record.result for record in records))


def _eval_xstest(args: argparse.Namespace) -> int:
    timer = RunTimer()
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
    timer.start_generation()
    records, metrics = evaluate_xstest(
        items,
        generator,
        system_prompt=arm_spec.system_prompt,
        judge=classifier,
        batch_size=config["batch_size"],
    )
    timer.finish_generation()
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
    timer.finish(run_dir)
    print(f"Run directory: {run_dir}")
    print(json.dumps(metrics, indent=2))
    return int(metrics["generation_errors"] > 0 or metrics.get("judge_errors", 0) > 0)


def _eval_mmlu(args: argparse.Namespace) -> int:
    timer = RunTimer()
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
    timer.start_generation()
    records, metrics = evaluate_mmlu(
        items,
        generator,
        instructions=benchmark["instructions"],
        system_prompt=arm_spec.system_prompt,
        batch_size=config["batch_size"],
    )
    timer.finish_generation()
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
    timer.finish(run_dir)
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
        "--episodes",
        required=True,
        help="Episode JSON/glob or built split directory/episodes.jsonl",
    )
    run.add_argument("--generator", choices=("stub", "hf"), required=True)
    run.add_argument("--stub-script")
    run.add_argument("--limit", type=int, help="Total episodes before sharding")
    run.add_argument("--shard", help="One-based i/k; stable SHA256 partition by ID")
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
        "--split",
        choices=("examples", "train", "dev", "test", "test_cueflip"),
        required=True,
    )
    run.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs/models.yaml"),
    )
    run.add_argument(
        "--root", help="Override the repository root for output under <root>/runs"
    )
    compare = commands.add_parser(
        "compare", help="Paired analysis of two verified agent runs"
    )
    compare.add_argument("run_a", type=Path)
    compare.add_argument("run_b", type=Path)
    compare.add_argument(
        "--specs-a", help="Original specs path/glob after relocating run A"
    )
    compare.add_argument(
        "--specs-b", help="Original specs path/glob after relocating run B"
    )
    rescore = commands.add_parser(
        "rescore", help="Regrade recorded facts with verified original specs"
    )
    rescore.add_argument("run_dir", type=Path)
    rescore.add_argument(
        "--specs", help="Original specs path/glob after relocating the run"
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
        "--limit",
        type=int,
        help="Maximum episodes per type (injected, clean and authorized)",
    )
    build.add_argument(
        "--cue-flip", action="store_true", help="Build the test cue-flipped companion"
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
        help="Cache and log a 10%% backtranslation review sample",
    )
    audit = commands.add_parser(
        "audit-cues", help="Audit observable dataset shortcut cues"
    )
    audit.add_argument(
        "--split", choices=("train", "dev", "test", "test_cueflip"), required=True
    )
    audit.add_argument(
        "--build-dir", type=Path, help="Directory containing episodes.jsonl"
    )
    crosscheck = commands.add_parser(
        "crosscheck", help="Independent GLM translation review"
    )
    crosscheck.add_argument("--split", choices=("dev", "train", "test"), required=True)
    crosscheck.add_argument(
        "--filler", action="store_true", help="Audit generated filler"
    )
    crosscheck.add_argument(
        "--root", type=Path, help="Read-only build/cache root; requires --dry-run"
    )
    crosscheck.add_argument("--n", type=int, default=150)
    crosscheck.add_argument("--dry-run", action="store_true")
    drop = commands.add_parser(
        "drop-episodes", help="Remove confirmed bad episodes and twins"
    )
    drop.add_argument(
        "--split", choices=("dev", "train", "test", "test_cueflip"), required=True
    )
    drop.add_argument("--ids-file", type=Path, required=True)
    drop.add_argument(
        "--reason", help="Reason for plain IDs without disagreement records"
    )
    drop.add_argument("--item-key", help="Cross-check item key for plain IDs")
    freeze = commands.add_parser("freeze", help="Freeze the complete test release")
    freeze.add_argument("--split", choices=("test",), required=True)
    pairs = commands.add_parser(
        "build-pairs", help="Verified on-policy preference pairs"
    )
    pairs.add_argument("--run", type=Path, nargs="+", action="extend", required=True)
    pairs.add_argument("--split", choices=("train", "dev"))
    pairs.add_argument(
        "--validate-only",
        action="store_true",
        help="Dev validation; never for training",
    )
    pairs.add_argument(
        "--specs", help="Original source split directory after relocating shard runs"
    )
    pairs.add_argument("--scenario-cap", type=int, default=200)
    train = commands.add_parser("train", help="not implemented yet")
    train.add_argument("--pairs", type=Path)
    train.add_argument("--arm", default="A4")
    train.add_argument("--model", choices=("dev", "headline"), default="headline")
    for name in ("eval-external", "report"):
        commands.add_parser(name, help="not implemented yet")
    args = parser.parse_args(argv)
    if args.command == "build-pairs":
        from turnguard.data.pairs import build_pairs

        try:
            report = build_pairs(
                args.run,
                split=args.split or ("dev" if args.validate_only else "train"),
                validate_only=args.validate_only,
                specs_path=args.specs,
                scenario_cap=args.scenario_cap,
            )
        except (ValueError, OSError, KeyError, yaml.YAMLError) as error:
            parser.error(str(error))
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    if args.command == "train" and args.pairs is not None:
        from turnguard.data.pairs import load_training_pairs

        try:
            load_training_pairs(args.pairs)
        except (ValueError, OSError, KeyError) as error:
            parser.error(str(error))
    if args.command in ("crosscheck", "drop-episodes"):
        try:
            if args.command == "crosscheck":
                from turnguard.data.crosscheck import crosscheck

                if args.root is not None and not args.dry_run:
                    raise ValueError("--root is read-only and requires --dry-run")
                kwargs = {
                    "n": args.n,
                    "dry_run": args.dry_run,
                    "split": args.split,
                    "filler": args.filler,
                }
                if args.root is not None:
                    kwargs["root"] = args.root
                report = crosscheck(**kwargs)
            else:
                from turnguard.data.drop_episodes import drop_episodes

                report = drop_episodes(
                    args.split,
                    args.ids_file,
                    reason=args.reason,
                    item_key=args.item_key,
                )
        except (ValueError, OSError, KeyError, ImportError) as error:
            parser.error(str(error))
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return int(
            args.command == "crosscheck"
            and report.get("overall", {}).get("errors", 0) > 0
        )
    if args.command in ("compare", "rescore"):
        from turnguard.analysis import compare_runs, rescore_run

        try:
            if args.command == "compare":
                report = compare_runs(
                    args.run_a, args.run_b, specs_a=args.specs_a, specs_b=args.specs_b
                )
            else:
                report = rescore_run(args.run_dir, specs_path=args.specs)
        except (ValueError, OSError, KeyError, yaml.YAMLError) as error:
            parser.error(str(error))
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
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
    if args.command == "audit-cues":
        from turnguard.data.cues import audit_build

        try:
            report = audit_build(args.split, build_dir=args.build_dir)
        except (ValueError, OSError, KeyError) as error:
            parser.error(str(error))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
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
                cue_flip=args.cue_flip,
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
