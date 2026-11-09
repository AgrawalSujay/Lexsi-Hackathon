"""Load canonical episodes, with test restricted to explicit evaluation only."""

import hashlib
import json
import re
from pathlib import Path

from turnguard.schema import EpisodeSpec, validate_episode


def load_split(
    split: str,
    *,
    purpose: str,
    root: Path | None = None,
    directory: Path | None = None,
) -> list[EpisodeSpec]:
    """Test access requires exactly purpose='eval', even before filesystem access."""
    if split in ("test", "test_cueflip") and purpose != "eval":
        raise ValueError(
            f"split={split} is evaluation-only; purpose={purpose!r} is forbidden"
        )
    if split not in ("train", "dev", "test", "test_cueflip"):
        raise ValueError("split must be train, dev, test or test_cueflip")
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    directory = Path(directory) if directory is not None else root / "data/v1" / split
    if split in ("test", "test_cueflip"):
        from turnguard.data.freeze import verify_frozen

        verify_frozen(directory)
    if (directory / "FROZEN").exists():
        from turnguard.data.freeze import verify_frozen

        verify_frozen(directory)
    return _load_release(split, directory)


def _load_release(split: str, directory: Path) -> list[EpisodeSpec]:
    """Validate rows before freezing too; evaluation uses load_split instead."""
    summary_path = directory / "summary.json"
    if (
        summary_path.exists()
        and json.loads(summary_path.read_text("utf-8")).get("status") != "complete"
    ):
        raise ValueError(f"Split {split} build is not complete")
    specs = [
        EpisodeSpec.from_json(line)
        for line in (directory / "episodes.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    twins = {spec.id: spec for spec in specs}
    if len(twins) != len(specs):
        raise ValueError("Episode IDs must be unique")
    for spec in specs:
        expected_split = "test" if split == "test_cueflip" else split
        if spec.split != expected_split:
            raise ValueError(
                f"{spec.id}: expected split {expected_split}, found {spec.split}"
            )
        if spec.episode_type == "clean" and spec.twin_of not in twins:
            raise ValueError(f"{spec.id}: missing injected twin")
        problems = validate_episode(spec, twins=twins)
        if problems:
            raise ValueError(f"{spec.id}: {'; '.join(problems)}")
    return specs


def split_fingerprint(directory: Path) -> dict:
    """Hash release identity, including the complete source JSONL, not a shard."""
    files = {}
    for name in ("episodes.jsonl", "manifest.json", "CHECKSUMS.sha256", "FROZEN"):
        path = directory / name
        if path.is_file():
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if "episodes.jsonl" not in files or "manifest.json" not in files:
        raise ValueError("Built split requires episodes.jsonl and manifest.json")
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"sha256": digest, "files": files}


def parse_shard(shard: str | None) -> tuple[int, int]:
    """One-based i/k; SHA256 ID membership never depends on input order."""
    if shard is None:
        return 1, 1
    if not re.fullmatch(r"[1-9][0-9]*/[1-9][0-9]*", shard):
        raise ValueError("--shard must be i/k with 1 <= i <= k (one-based)")
    index, count = map(int, shard.split("/"))
    if index > count:
        raise ValueError("--shard must have 1 <= i <= k")
    return index, count


def _group_key(spec: EpisodeSpec) -> str:
    """A clean twin always travels with its injected base."""
    return spec.twin_of or spec.id


def _rank(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest(), 16)


def select_episodes(
    specs: list[EpisodeSpec], *, limit: int | None = None, shard: str | None = None
) -> list[EpisodeSpec]:
    """Stable pseudo-random selection over twin groups.

    Groups are ordered by SHA256 of the base ID, so a --limit run is a mixed
    sample of every episode type (sorted IDs would take only 'auth' episodes
    first), and an injected episode and its clean twin are never separated by
    a limit or a shard. The limit counts episodes; a group that would exceed it
    is skipped so the total never goes over.
    """
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    index, count = parse_shard(shard)
    if limit is None and shard is None:
        return specs
    groups: dict[str, list[EpisodeSpec]] = {}
    for spec in specs:
        groups.setdefault(_group_key(spec), []).append(spec)
    ordered = sorted(groups, key=lambda key: (_rank(key), key))
    if limit is not None:
        chosen, total = [], 0
        for key in ordered:
            if total + len(groups[key]) <= limit:
                chosen.append(key)
                total += len(groups[key])
        ordered = chosen
    return [
        spec
        for key in ordered
        if _rank(key) % count == index - 1
        for spec in sorted(groups[key], key=lambda spec: spec.id)
    ]
