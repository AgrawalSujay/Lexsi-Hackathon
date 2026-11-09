"""Audited, twin-closed removal of confirmed bad translations before freezing."""

import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from turnguard.data.crosscheck import read_jsonl
from turnguard.data.cues import write_cue_audit
from turnguard.data.freeze import _validate_release
from turnguard.data.loaders import _load_release


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", "utf-8"
    )


def _snapshot(directory: Path) -> dict[str, bytes]:
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Cannot edit a release containing symlinks")
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = path.read_bytes()
    return result


def _read_requests(
    ids_file: Path, split: str, root: Path, reason: str | None, item_key: str | None
) -> list[dict]:
    text = ids_file.read_text("utf-8").strip()
    if not text:
        raise ValueError("IDs file is empty")
    try:
        value = json.loads(text)
        rows = value if isinstance(value, list) else [value]
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append(line.strip())
    lookup = {}
    disagreements = root / "data/crosscheck/test/disagreements.jsonl"
    if disagreements.exists():
        for item in read_jsonl(disagreements):
            field = "episode_ids" if split == "test" else "test_cueflip_episode_ids"
            for episode_id in item.get(field, []):
                lookup.setdefault(episode_id, []).append(item)
    requests = []
    for row in rows:
        if isinstance(row, str):
            row = {"episode_id": row}
        if not isinstance(row, dict):
            raise ValueError("IDs must be strings or removal JSON objects")
        field = "episode_ids" if split == "test" else "test_cueflip_episode_ids"
        ids = row.get(field, [row.get("episode_id", row.get("id"))])
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"Removal object must contain {field} or episode_id")
        for episode_id in ids:
            if not isinstance(episode_id, str) or not episode_id:
                raise ValueError("Invalid removal episode ID")
            matches = lookup.get(episode_id, [])
            reference = matches[0] if len(matches) == 1 else {}
            why = (
                row.get("reason")
                or row.get("issue")
                or reason
                or reference.get("issue")
            )
            key = (
                row.get("item_key")
                or row.get("key")
                or item_key
                or reference.get("key")
            )
            if (
                not isinstance(why, str)
                or not why.strip()
                or not isinstance(key, str)
                or not key.strip()
            ):
                raise ValueError(
                    f"{episode_id}: reason and cross-check item key required; "
                    "use JSONL objects or --reason/--item-key "
                    "(plain IDs can resolve from disagreements)"
                )
            requests.append({"episode_id": episode_id, "reason": why, "item_key": key})
    return requests


def _refresh_summary(summary: dict, kept: list[dict], trail: list[dict]) -> dict:
    summary["counts"] = {
        kind: sum(episode["episode_type"] == kind for episode in kept)
        for kind in ("injected", "clean", "authorized")
    }
    summary["dropped"] = trail
    injected = [episode for episode in kept if episode["episode_type"] == "injected"]
    authorized = [
        episode for episode in kept if episode["episode_type"] == "authorized"
    ]
    if "requested_injected" in summary:
        summary["requested_shortfall"] = max(
            0, summary["requested_injected"] - len(injected)
        )
    if "plan_targets" in summary:
        summary["plan_type_shortfalls"] = {
            kind: max(0, target - summary["counts"][kind])
            for kind, target in summary["plan_targets"].items()
        }
    for field in ("factors", "plan_factors"):
        if field not in summary:
            continue
        for factor, levels in summary[field].items():
            counts = Counter(
                episode["lang"][factor.removesuffix("_language")]
                if factor.endswith("_language")
                else episode["meta"].get("pressure")
                if factor == "pressure"
                else episode.get(factor)
                for episode in injected
            )
            for level, cell in levels.items():
                cell["actual"] = counts[level]
                cell["shortfall"] = max(0, cell["target"] - counts[level])
    if "overreach" in summary:
        summary["overreach"]["accepted"] = sum(
            bool(e["meta"].get("overreach")) for e in injected
        )
    if "authorized_coverage" in summary:
        summary["authorized_coverage"] = {
            "by_delegation_mode": dict(
                Counter(
                    (e["meta"].get("delegation") or {}).get("mode", "unknown")
                    for e in authorized
                )
            ),
            "by_carrier": dict(Counter(e["carrier"] for e in authorized)),
            "by_language": dict(Counter(e["lang"]["conversation"] for e in authorized)),
            "by_instruction_language": dict(
                Counter(e["lang"]["injection"] for e in authorized)
            ),
            "by_filler_variant": dict(
                Counter(str(e["meta"].get("filler_variant")) for e in authorized)
            ),
        }
    return summary


def _checksums(directory: Path, *, uppercase: bool) -> None:
    # CuratorKIT's checksums.txt does not hash itself. If an unfrozen build has
    # an uppercase checksum file too, hash checksums.txt there without a cycle.
    paths = sorted(path for path in directory.rglob("*") if path.is_file())
    (directory / "checksums.txt").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(directory)}\n"
            for path in paths
            if path.name not in ("checksums.txt", "CHECKSUMS.sha256")
        ),
        "utf-8",
    )
    if uppercase:
        paths = sorted(path for path in directory.rglob("*") if path.is_file())
        (directory / "CHECKSUMS.sha256").write_text(
            "".join(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
                f"{path.relative_to(directory)}\n"
                for path in paths
                if path.name != "CHECKSUMS.sha256"
            ),
            "utf-8",
        )


def drop_episodes(
    split: str,
    ids_file: Path,
    *,
    root: Path | None = None,
    reason: str | None = None,
    item_key: str | None = None,
) -> dict:
    """Stage, validate and hash the changed release; preserve original history.

    An ID can name either member of an injected/clean pair. Closure is symmetric.
    JSONL removal rows require reason/item_key; plain IDs resolve those from the
    reviewed disagreements, or explicit CLI defaults. This is never an automatic
    reaction to the judge: the operator chooses the confirmed removals.
    """
    if split not in ("test", "test_cueflip"):
        raise ValueError("drop-episodes supports test or test_cueflip only")
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    directory = root / "data/v1" / split
    if (directory / "FROZEN").exists():
        raise ValueError(f"Cannot drop episodes from frozen split {split}")
    before = _snapshot(directory)
    _validate_release(directory)
    episodes = [spec.to_dict() for spec in _load_release(split, directory)]
    by_id = {episode["id"]: episode for episode in episodes}
    requests = _read_requests(Path(ids_file), split, root, reason, item_key)
    summary = json.loads(before["summary.json"])
    previous = (
        read_jsonl(directory / "dropped.jsonl") if "dropped.jsonl" in before else []
    )
    if summary.get("dropped", []) != previous:
        raise ValueError("summary dropped history does not match dropped.jsonl")
    removed_before = {row["episode_id"] for row in previous}
    reasons = {}
    for request in requests:
        episode_id = request["episode_id"]
        if episode_id not in by_id:
            if episode_id in removed_before:
                continue
            raise ValueError(f"Unknown episode ID: {episode_id}")
        episode = by_id[episode_id]
        base_id = episode.get("twin_of") or episode_id
        group = {base_id} | {e["id"] for e in episodes if e.get("twin_of") == base_id}
        for member in group:
            if member in by_id:
                reasons.setdefault(member, request)
    if not reasons:
        return {
            "split": split,
            "removed": 0,
            "counts": summary.get("counts", {}),
            "already_dropped": True,
        }
    kept = [episode for episode in episodes if episode["id"] not in reasons]
    now = datetime.now(UTC).isoformat()
    trail = []
    for episode_id in sorted(reasons):
        episode = by_id[episode_id]
        twin_id = episode.get("twin_of") or next(
            (e["id"] for e in episodes if e.get("twin_of") == episode_id), None
        )
        trail.append(
            {
                **reasons[episode_id],
                "episode_id": episode_id,
                "twin_id": twin_id,
                "removed_at": now,
            }
        )
    summary = _refresh_summary(summary, kept, previous + trail)
    with TemporaryDirectory(
        prefix=f".{split}-drop-", dir=directory.parent
    ) as temporary:
        stage = Path(temporary) / split
        shutil.copytree(directory, stage)
        (stage / "episodes.jsonl").write_text(
            "".join(
                # Keep surviving row bytes/order exactly as originally exported.
                line + "\n"
                for line in before["episodes.jsonl"].decode("utf-8").splitlines()
                if line.strip() and json.loads(line)["id"] not in reasons
            ),
            "utf-8",
        )
        with (stage / "dropped.jsonl").open("a", encoding="utf-8") as stream:
            if before.get("dropped.jsonl") and not before["dropped.jsonl"].endswith(
                b"\n"
            ):
                stream.write("\n")
            for row in trail:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        limited = (
            summary.get("cue_audit", {}).get("status")
            == "indicative only (limited build)"
        )
        summary["cue_audit"] = write_cue_audit(kept, stage, limited=limited)
        _write_json(stage / "summary.json", summary)
        event = {
            "operation": "confirmed_translation_drop",
            "removed_at": now,
            "removed": len(trail),
            "episode_ids": sorted(reasons),
            "input_episodes_sha256": hashlib.sha256(
                before["episodes.jsonl"]
            ).hexdigest(),
            "output_episodes_sha256": hashlib.sha256(
                (stage / "episodes.jsonl").read_bytes()
            ).hexdigest(),
        }
        manifest = json.loads(before["manifest.json"])
        manifest.setdefault("turnguard", {}).update(
            counts=summary["counts"], dropped=previous + trail
        )
        # Earlier pipeline stages describe the original generation/gates; retain
        # those historical facts and explicitly add the post-build removal stage.
        manifest.setdefault("stage_counts", {})["ConfirmedTranslationDrop"] = {
            "input_count": len(episodes),
            "output_count": len(kept),
            "removed_count": len(trail),
        }
        exporter = manifest["stage_counts"].get("EpisodeExporter")
        if exporter is not None:
            exporter["exported_count"] = len(kept)
        provenance = json.loads(before["lexsi_provenance.json"])
        provenance.setdefault("postprocessing", []).append(event)
        manifest["provenance"] = provenance
        _write_json(stage / "manifest.json", manifest)
        _write_json(stage / "lexsi_provenance.json", provenance)
        with (stage / "dataset_card.md").open("a", encoding="utf-8") as stream:
            stream.write(
                f"\nConfirmed translation removals ({now}): {len(trail)} episodes. "
                f"Current counts: {json.dumps(summary['counts'])}. "
                "Reasons and cross-check keys are in dropped.jsonl.\n"
            )
        _checksums(stage, uppercase="CHECKSUMS.sha256" in before)
        _validate_release(stage)
        after = _snapshot(stage)
        if _snapshot(directory) != before or (directory / "FROZEN").exists():
            raise ValueError("Split changed or froze during removal; no edits applied")
        changed = [name for name in after if after[name] != before.get(name)]
        try:
            for name in changed:
                os.replace(stage / name, directory / name)
        except OSError:
            # No partially updated manifest or audit trail on a failed replace.
            for name in changed:
                if name in before:
                    (directory / name).write_bytes(before[name])
                else:
                    (directory / name).unlink(missing_ok=True)
            raise
    return {
        "split": split,
        "removed": len(trail),
        "counts": summary["counts"],
        "dropped": trail,
        "cue_audit": summary["cue_audit"],
    }
