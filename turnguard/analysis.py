"""Offline verified regrading and paired analysis of recorded agent runs."""

import copy
import glob
import hashlib
import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from turnguard.config import load_config
from turnguard.metrics import ANALYSIS_VERSION, compare_records, compute_metrics
from turnguard.oracles import GRADING_VERSION, evaluate_episode, with_generation_error
from turnguard.schema import EpisodeRecord, spec_sha256, validate_episode


def _read_specs(path: Path) -> list[dict]:
    text = path.read_text("utf-8")
    if path.suffix == ".jsonl":
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        value = json.loads(text)
        values = value if isinstance(value, list) else [value]
    if not all(
        isinstance(value, dict) and isinstance(value.get("id"), str) for value in values
    ):
        raise ValueError(f"Not an episode spec file: {path}")
    return values


def _spec_paths(run_dir: Path, config: dict, override: str | None) -> list[Path]:
    if override is not None:
        path = Path(override).expanduser()
        if path.is_dir():
            paths = (
                [path / "episodes.jsonl"]
                if (path / "episodes.jsonl").is_file()
                else sorted(path.glob("*.json"))
            )
        else:
            paths = [Path(name) for name in sorted(glob.glob(str(path)))]
    else:
        raw = config.get("episode_paths")
        if raw is None and isinstance(config.get("episodes_path"), str):
            raw = [config["episodes_path"]]
        if (
            not isinstance(raw, list)
            or not raw
            or not all(isinstance(name, str) for name in raw)
        ):
            raise ValueError(
                "Run config has no episode_paths; "
                "provide --specs with the original specs"
            )
        paths = [Path(name).expanduser() for name in raw]
        paths = [path if path.is_absolute() else run_dir / path for path in paths]
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError(
            "Original episode spec file is missing; use --specs to locate "
            "the originals (hashes must match)"
        )
    return [path.resolve() for path in paths]


def load_verified_run(
    run_dir: str | Path,
    *,
    specs_path: str | None = None,
) -> tuple[list[EpisodeRecord], list[dict], dict]:
    """Resolve original specs and verify EVERY recorded canonical hash before grading.

    Never silently substitute a current template/bank or a same-ID spec from a
    different file. --specs permits explicit relocation after transferring runs;
    it never bypasses hashes. Extra rows in a source JSONL are not part of a run.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    config = load_config(run_dir / "config.yaml")
    rows = [
        json.loads(line)
        for line in (run_dir / "episodes.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    try:
        records = [EpisodeRecord(**row) for row in rows]
    except TypeError as error:
        raise ValueError(
            "compare/rescore requires agent EpisodeRecord rows, not external eval rows"
        ) from error
    if len({record.episode_id for record in records}) != len(records):
        raise ValueError("Run contains duplicate episode IDs")
    specs = {}
    paths = _spec_paths(run_dir, config, specs_path)
    for path in paths:
        for value in _read_specs(path):
            if value["id"] in specs:
                raise ValueError(f"Duplicate episode spec ID: {value['id']}")
            specs[value["id"]] = value
    # Hash before schema validation: a changed AND invalid spec must still yield
    # the explicit mismatch diagnosis, with no filesystem mutation.
    for record in records:
        if record.episode_id not in specs:
            raise ValueError(
                f"Original spec is missing for episode {record.episode_id}"
            )
        if spec_sha256(specs[record.episode_id]) != record.spec_sha256:
            raise ValueError(
                f"Spec hash mismatch for {record.episode_id}: the episode spec "
                "has changed since the run; refusing to rescore/compare"
            )
    selected = [specs[record.episode_id] for record in records]
    for spec in selected:
        if problems := validate_episode(spec, twins=specs):
            raise ValueError(
                f"Invalid original spec {spec['id']}: {'; '.join(problems)}"
            )
    if (
        len(
            {
                (record.arm, record.model, specs[record.episode_id]["split"])
                for record in records
            }
        )
        > 1
    ):
        raise ValueError("A run must contain one arm, model and split")
    metadata = {"run_dir": str(run_dir), "spec_files": [str(path) for path in paths]}
    return records, selected, metadata


def regrade_records(
    records: list[EpisodeRecord], specs: list[dict]
) -> list[EpisodeRecord]:
    """Replace only results; raw messages, calls, answers, effects, hashes stay."""
    by_id = {spec["id"]: spec for spec in specs}
    updated = copy.deepcopy(records)
    for record in updated:
        result = evaluate_episode(by_id[record.episode_id], record)
        if "error" in record.result:
            result = with_generation_error(result, record.result["error"])
        record.result = result
    return updated


def compare_runs(
    run_a: str | Path,
    run_b: str | Path,
    *,
    specs_a: str | None = None,
    specs_b: str | None = None,
) -> dict:
    """Regrade both verified histories in memory; comparisons never mutate runs."""
    records_a, values_a, metadata_a = load_verified_run(run_a, specs_path=specs_a)
    records_b, values_b, metadata_b = load_verified_run(run_b, specs_path=specs_b)
    specs = {spec["id"]: spec for spec in values_a}
    for spec in values_b:
        if spec["id"] in specs and spec_sha256(specs[spec["id"]]) != spec_sha256(spec):
            raise ValueError(f"Spec hash mismatch between runs for {spec['id']}")
        specs[spec["id"]] = spec
    result = compare_records(
        regrade_records(records_a, values_a),
        regrade_records(records_b, values_b),
        list(specs.values()),
    )
    result.update(grading_version=GRADING_VERSION, run_a=metadata_a, run_b=metadata_b)
    return result


def rescore_run(run_dir: str | Path, *, specs_path: str | None = None) -> dict:
    """Verify all specs, stage new results/metrics, then back up and replace files."""
    run_dir = Path(run_dir).expanduser().resolve()
    episodes_file, metrics_file = run_dir / "episodes.jsonl", run_dir / "metrics.json"
    original_episodes = episodes_file.read_bytes()
    original_metrics = metrics_file.read_bytes() if metrics_file.exists() else None
    records, specs, metadata = load_verified_run(run_dir, specs_path=specs_path)
    updated = regrade_records(records, specs)
    metrics = compute_metrics(updated, specs)
    new_episodes = "".join(record.to_json() + "\n" for record in updated).encode()
    new_metrics = (
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode()
    report = {
        **metadata,
        "grading_version": GRADING_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "counts": metrics["counts"],
        "changed_grades": sum(
            a.result != b.result for a, b in zip(records, updated, strict=True)
        ),
    }
    if new_episodes == original_episodes and new_metrics == original_metrics:
        return {**report, "unchanged": True, "backup_dir": None}
    # Refuse concurrent writes rather than overwriting a still-running generation.
    if (
        episodes_file.read_bytes() != original_episodes
        or (metrics_file.read_bytes() if metrics_file.exists() else None)
        != original_metrics
    ):
        raise ValueError(
            "Run files changed during rescore; retry after the run has finished"
        )
    suffix = 1
    while True:
        backup = run_dir / (
            "rescore-backup" if suffix == 1 else f"rescore-backup-{suffix}"
        )
        try:
            backup.mkdir()
            break
        except FileExistsError:
            suffix += 1
    report.update(
        unchanged=False,
        backup_dir=str(backup),
        input_episodes_sha256=hashlib.sha256(original_episodes).hexdigest(),
        output_episodes_sha256=hashlib.sha256(new_episodes).hexdigest(),
    )
    original_manifest = (
        (run_dir / "rescore.json").read_bytes()
        if (run_dir / "rescore.json").exists()
        else None
    )
    for name, data in (
        ("episodes.jsonl", original_episodes),
        ("metrics.json", original_metrics),
        ("rescore.json", original_manifest),
    ):
        if data is not None:
            (backup / name).write_bytes(data)
    with TemporaryDirectory(prefix=".rescore-", dir=run_dir) as temporary:
        stage = Path(temporary)
        (stage / "episodes.jsonl").write_bytes(new_episodes)
        (stage / "metrics.json").write_bytes(new_metrics)
        (stage / "rescore.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        try:
            for name in ("episodes.jsonl", "metrics.json", "rescore.json"):
                os.replace(stage / name, run_dir / name)
        except OSError:
            for name, data in (
                ("episodes.jsonl", original_episodes),
                ("metrics.json", original_metrics),
                ("rescore.json", original_manifest),
            ):
                if data is None:
                    (run_dir / name).unlink(missing_ok=True)
                else:
                    shutil.copyfile(backup / name, run_dir / name)
            raise
    return report
