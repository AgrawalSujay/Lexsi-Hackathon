"""Runner-independent side-eval generation and local-only resumable judging."""

import copy
import hashlib
import io
import json
import logging
import time
from collections import Counter
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path

import yaml

from turnguard.arms.runtime import resolve_arm
from turnguard.config import load_config
from turnguard.data.teacher import PROMPT_VERSION
from turnguard.evals import global_mmlu as mmlu
from turnguard.evals import xstest
from turnguard.metrics import rate
from turnguard.models import HFGenerator, UnsupportedDtypeError
from turnguard.runs import RunTimer, _environment, new_run_dir
from turnguard.schema import canonical_json

VERSION = "side-v1"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _atomic(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temp.replace(path)


@contextmanager
def _quiet():
    """Responses/provider exceptions must never escape through library output."""
    disabled = logging.root.manager.disable
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        logging.disable(logging.CRITICAL)
        try:
            yield
        finally:
            logging.disable(disabled)


def generate_side(
    *,
    arm: str = "A0",
    model: str = "dev",
    dtype: str = "fp16",
    out: Path | None = None,
    batch_size: int | None = None,
    config_path: Path | None = None,
    arms_path: Path | None = None,
    root: Path | None = None,
    dry_run: bool = False,
    generator_factory=None,
) -> dict:
    """Generate both pinned datasets once, without reading CK settings/judging."""
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    config_path = Path(config_path) if config_path else root / "configs/evals.yaml"
    config = load_config(config_path)
    models_path = root / config["models_config"]
    models = load_config(models_path)
    arms_path = Path(arms_path) if arms_path else root / config["arms_config"]
    arm_spec = resolve_arm(arm, load_config(arms_path))
    if dtype not in ("fp16", "bf16", "fp32"):
        raise ValueError("dtype must be fp16, bf16 or fp32")
    batch_size = batch_size if batch_size is not None else config["batch_size"]
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if (
        config["xstest"]["max_new_tokens"] != 256
        or config["global_mmlu"]["max_new_tokens"] != 16
    ):
        raise ValueError("Side eval decoding requires XSTest 256 / MMLU 16 tokens")
    plan = {
        "version": VERSION,
        "arm": arm,
        "model_key": model,
        "model": models["models"][model],
        "dtype": dtype,
        "batch_size": batch_size,
        "judge": "local-only",
        "arm_config": asdict(arm_spec),
        "arm_system_prompt_sha256": arm_spec.system_prompt_sha256,
        "evals": config,
        "configs_sha256": {
            "evals": _hash(config_path),
            "models": _hash(models_path),
            "arms": _hash(arms_path),
        },
        "decoding": {
            "do_sample": False,
            "chat_template": "tokenizer",
            "padding_side": "left",
            "xstest_max_new_tokens": 256,
            "mmlu_max_new_tokens": 16,
        },
        "expected_counts": {
            "xstest": config["xstest"]["expected_counts"],
            "mmlu": config["global_mmlu"]["expected_counts"],
        },
    }
    if dry_run:
        return plan
    if out is not None and Path(out).exists():
        raise ValueError("Output directory already exists; choose a new --out")
    timer = RunTimer()
    if out is None:
        out = new_run_dir(arm, plan["model"], "side", root=root)
    else:
        out = Path(out).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=False)
        _atomic(out / "env.json", _environment(root))
    _atomic(out / "manifest.json", {"version": VERSION, "status": "generating"})
    stage = "dataset-load"
    try:
        xs = xstest.load_items(config["xstest"], cache_dir=root / ".cache/hf")
        ms = mmlu.load_items(config["global_mmlu"], cache_dir=root / ".cache/hf")
        if (
            dict(Counter(item.subset for item in xs))
            != config["xstest"]["expected_counts"]
        ):
            raise ValueError("XSTest counts differ from pinned configuration")
        if (
            dict(Counter(item.language for item in ms))
            != config["global_mmlu"]["expected_counts"]
        ):
            raise ValueError("MMLU counts differ from pinned configuration")
        inputs = [{"benchmark": "xstest", **asdict(item)} for item in xs]
        inputs += [{"benchmark": "mmlu", **asdict(item)} for item in ms]
        by_id = {item["id"]: item for item in inputs}
        if len(by_id) != len(inputs):
            raise ValueError("Side-eval IDs must be unique")
        (out / "inputs.jsonl").write_text(
            "".join(canonical_json(item) + "\n" for item in inputs)
        )
        stage = "model-load"
        generator = (generator_factory or HFGenerator)(
            plan["model"],
            max_new_tokens=256,
            batch_size=batch_size,
            seed=config["seed"],
            dtype=dtype,
        )
        arm_spec.apply(generator)
        revision = getattr(
            getattr(getattr(generator, "model", None), "config", None),
            "_commit_hash",
            None,
        )
        plan["model_revision"] = revision if isinstance(revision, str) else None
        (out / "config.yaml").write_text(
            yaml.safe_dump(plan, sort_keys=False, allow_unicode=True)
        )
        timings, counts = {}, {}
        timer.start_generation()
        with (out / "generations.jsonl").open("x", encoding="utf-8") as stream:
            for benchmark, items in (("xstest", xs), ("mmlu", ms)):
                stage = benchmark + "-generation"
                generator.max_new_tokens = 256 if benchmark == "xstest" else 16
                started = time.perf_counter()
                count = 0
                for offset in range(0, len(items), batch_size):
                    chunk = items[offset : offset + batch_size]
                    with _quiet():
                        if benchmark == "xstest":
                            records, _ = xstest.evaluate_xstest(
                                chunk,
                                generator,
                                system_prompt=arm_spec.system_prompt,
                                batch_size=batch_size,
                            )
                        else:
                            records, _ = mmlu.evaluate_mmlu(
                                chunk,
                                generator,
                                instructions=config["global_mmlu"]["instructions"],
                                system_prompt=arm_spec.system_prompt,
                                batch_size=batch_size,
                            )
                    for record in records:
                        record.update(
                            benchmark=benchmark,
                            input_sha256=_digest(by_id[record["id"]]),
                        )
                        stream.write(canonical_json(record) + "\n")
                        count += 1
                    stream.flush()
                timings[benchmark + "_generation_seconds"] = (
                    time.perf_counter() - started
                )
                counts[benchmark] = count
        timer.finish_generation()
        _atomic(out / "timings.json", timings)
        timer.finish(out)
        names = (
            "config.yaml",
            "inputs.jsonl",
            "generations.jsonl",
            "env.json",
            "timings.json",
        )
        _atomic(
            out / "manifest.json",
            {
                "version": VERSION,
                "status": "complete",
                "counts": counts,
                "files": {name: _hash(out / name) for name in names},
            },
        )
        return {
            "run_dir": str(out),
            "counts": counts,
            "timings": timings,
            "dtype": dtype,
            "generation_sha256": _hash(out / "generations.jsonl"),
            "judge_calls": 0,
        }
    except Exception as error:
        _atomic(
            out / "error.json", {"stage": stage, "error_type": type(error).__name__}
        )
        timer.finish(out)
        _atomic(out / "manifest.json", {"version": VERSION, "status": "failed"})
        if isinstance(error, UnsupportedDtypeError):
            raise UnsupportedDtypeError(str(error)) from None
        raise ValueError("Side generation failed; no raw responses printed") from None


def _load(run: Path):
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("version") != VERSION or manifest.get("status") != "complete":
        raise ValueError("Side run is incomplete or unsupported")
    required = {
        "config.yaml",
        "inputs.jsonl",
        "generations.jsonl",
        "env.json",
        "timings.json",
    }
    if set(manifest["files"]) != required or any(
        _hash(run / name) != digest for name, digest in manifest["files"].items()
    ):
        raise ValueError("Side run hash mismatch: refusing changed generations/config")
    config = load_config(run / "config.yaml")
    inputs = {
        item["id"]: item
        for item in map(json.loads, (run / "inputs.jsonl").read_text().splitlines())
    }
    records = [
        json.loads(line)
        for line in (run / "generations.jsonl").read_text().splitlines()
    ]
    if len({r["id"] for r in records}) != len(records) or set(inputs) != {
        r["id"] for r in records
    }:
        raise ValueError("Side run input/generation IDs differ")
    for row in records:
        if (
            row["input_sha256"] != _digest(inputs[row["id"]])
            or row["benchmark"] != inputs[row["id"]]["benchmark"]
        ):
            raise ValueError("Side record input hash mismatch")
    return config, inputs, records


def _xs_metrics(records: list[dict], field: str) -> dict:
    result = xstest.summarize(records, field)
    safe = result["safe"]
    safe["full_compliance"] = rate(safe["classes"]["full_compliance"], safe["graded"])
    return result


def judge_side(run: Path, *, root: Path | None = None, judge_factory=None) -> dict:
    """Local CK judging, atomically checkpointed by input plus judge signature."""
    run = Path(run).expanduser().resolve()
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    config, inputs, records = _load(run)
    judge_config = copy.deepcopy(config["evals"]["xstest"])
    # The cache includes the response in its full request: keep it inside the
    # run so raw unsafe outputs never escape into a shared project cache.
    judge_config["judge_cache"] = str(run / "judge_cache")
    judge_config["judge_sampling"]["extra_body"] = {"reasoning": {"enabled": False}}
    with _quiet():
        classify, client = (judge_factory or xstest.teacher_judge)(root, judge_config)
    provenance = {
        "model": client.model,
        "api_base": client.api_base,
        "sampling": client.sampling,
        "prompt_version": client.template_version,
        "teacher_prompt_version": PROMPT_VERSION,
        "classifier_source_sha256": _hash(Path(xstest.__file__)),
    }
    signature = _digest(provenance)
    checkpoint = run / "judgements.json"
    state = (
        json.loads(checkpoint.read_text())
        if checkpoint.exists()
        else {"version": VERSION, "entries": {}}
    )
    if state.get("version") != VERSION:
        raise ValueError("Unsupported judge checkpoint")
    reused, errors = 0, 0
    xs, ms = [], []
    for original in records:
        row = copy.deepcopy(original)
        if row["benchmark"] == "xstest":
            row["teacher_class"] = None
            response = row.get("response")
            if not isinstance(response, str) or not response.strip():
                row["regex_class"] = None
                row["generation_error"] = "empty_or_failed_generation"
            else:
                row["regex_class"] = xstest.regex_classify(response)
                key = _digest({"record": _digest(original), "judge": signature})
                entry = state["entries"].get(key)
                if entry and entry.get("label") in xstest.CLASSES:
                    reused += 1
                else:
                    try:
                        with _quiet():
                            label, details = classify(
                                inputs[row["id"]]["prompt"], response
                            )
                        if label not in xstest.CLASSES:
                            raise ValueError("Invalid judge label")
                        entry = {
                            "label": label,
                            "details": details,
                            "id": row["id"],
                            "judge_signature": signature,
                            "record_sha256": _digest(original),
                        }
                    except Exception:
                        entry = {
                            "label": None,
                            "error": "judge_failed_or_invalid_label",
                            "id": row["id"],
                            "judge_signature": signature,
                        }
                        # Invalid cached labels must be retryable on resume.
                        history = getattr(client, "history", [])
                        if history:
                            cache_key = history[-1]["key"]
                            bad = run / "judge_cache" / f"{cache_key}.json"
                            if bad.exists():
                                quarantine = run / "judge_cache/invalid"
                                quarantine.mkdir(parents=True, exist_ok=True)
                                bad.replace(
                                    quarantine / f"{cache_key}-{time.time_ns()}.json"
                                )
                            getattr(client, "_memory", {}).pop(cache_key, None)
                    state["entries"][key] = entry
                    _atomic(checkpoint, state)
                row["teacher_class"] = entry["label"]
                if entry["label"] is None:
                    row["judge_error"] = "judge_failed_or_invalid_label"
                    errors += 1
            xs.append(row)
        else:
            response = row.get("response")
            row["predicted"] = (
                mmlu.extract_answer(response) if isinstance(response, str) else None
            )
            row["correct"] = (
                row["predicted"] == inputs[row["id"]]["answer"]
                if isinstance(response, str)
                else None
            )
            row["unparseable"] = isinstance(response, str) and row["predicted"] is None
            ms.append(row)
    mmlu_metrics = {}
    for language in ("en", "hi"):
        group = [row for row in ms if row["language"] == language]
        graded = [row for row in group if row["correct"] is not None]
        mmlu_metrics[language] = {
            "accuracy": rate(sum(r["correct"] for r in graded), len(graded)),
            "total": len(group),
            "generation_errors": len(group) - len(graded),
            "unparseable": sum(r["unparseable"] for r in group),
        }
    metrics = {
        "version": VERSION,
        "xstest": {
            "teacher": _xs_metrics(xs, "teacher_class"),
            "regex": _xs_metrics(xs, "regex_class"),
            "judge_errors": errors,
        },
        "mmlu": mmlu_metrics,
        "judge": provenance,
        "judge_signature": signature,
        "checkpoint_reused": reused,
        "generation_sha256": _hash(run / "generations.jsonl"),
    }
    (run / "scored.jsonl").write_text(
        "".join(canonical_json(r) + "\n" for r in xs + ms)
    )
    _atomic(run / "judge_config.json", provenance)
    _atomic(run / "metrics.json", metrics)
    with _quiet():
        ak = {
            "xstest": xstest.auditkit_scores(xs, cache_dir=run / "auditkit_cache"),
            "mmlu": mmlu.auditkit_scores(ms, cache_dir=run / "auditkit_cache"),
        }
    _atomic(run / "auditkit.json", ak)
    return metrics
