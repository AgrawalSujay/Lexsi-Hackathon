"""Observable shortcut cues; only planted instructions and their targets supply URLs.

TLD shares use distinct identifiers as denominator; the other cues use episodes.
Clean twins have no planted instruction. Delegation checks the delivered sentence,
not merely a metadata flag. Older builds without that sentence report unknown.
"""

import json
import logging
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

LOG = logging.getLogger(__name__)
IDENTIFIER = re.compile(
    r"https?://[^\s<>\"']+|[\w.+%-]+@[\w.-]+|"
    r"\b(?:[a-z0-9-]+\.)+(?:invalid|com|org|net)(?![a-z0-9.-])(?:/[^\s<>\"']*)?|"
    r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}/[^\s<>\"']+",
    re.I,
)
CLASSES = (".invalid", "example.*", "other")


def _strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def tld_class(identifier: str) -> str:
    """Classify reserved example domains separately from .invalid and other."""
    host = (
        identifier.rsplit("@", 1)[-1]
        if "@" in identifier and "://" not in identifier
        else urlsplit(identifier if "://" in identifier else "//" + identifier).hostname
    ) or ""
    host = host.lower().rstrip(".,;:!?)]}")
    if host.endswith(".invalid"):
        return ".invalid"
    if any(
        host == suffix or host.endswith("." + suffix)
        for suffix in ("example.com", "example.org", "example.net")
    ):
        return "example.*"
    return "other"


def cue_audit(episodes: list[dict]) -> dict:
    """Summarize episode and identifier shares, retaining explicit denominators."""
    result = {}
    for kind in ("injected", "authorized", "clean"):
        rows = [e for e in episodes if e["episode_type"] == kind]
        counts, domains = Counter(), Counter()
        delegation_known = 0
        for episode in rows:
            block = episode.get("injection") or episode.get("authorized") or {}
            parts = block.get("parts", [])
            texts = [part["text"] for part in parts]
            if "text" in block:
                texts.append(block["text"])
            counts["instruction_language_mismatch"] += bool(texts) and (
                episode["lang"]["injection"] != episode["lang"]["conversation"]
            )
            counts["assistant_in_planted_text"] += any(
                re.search(r"\bassistant\b", text, re.I) is not None for text in texts
            )
            delegation = episode.get("meta", {}).get("delegation")
            if delegation is not None:
                delegation_known += 1
                turn, text = delegation["turn"], delegation["text"]
                counts["user_turn_delegation"] += (
                    bool(text) and text in episode["user_turns"][turn]
                )
            elif "delegation" in episode.get("meta", {}):
                delegation_known += 1  # new builder explicitly records every delegation
            identifiers = {
                match.group().rstrip(".,;:!?)]}").lower()
                for value in [*texts, *_strings(block.get("target", []))]
                for match in IDENTIFIER.finditer(value)
            }
            domains.update(tld_class(value) for value in identifiers)
        n, identifiers_n = len(rows), sum(domains.values())
        shares = {
            key: counts[key] / n if n else None
            for key in ("instruction_language_mismatch", "assistant_in_planted_text")
        }
        shares["user_turn_delegation"] = (
            counts["user_turn_delegation"] / delegation_known
            if delegation_known
            else None
        )
        result[kind] = {
            "n": n,
            "shares": shares,
            "delegation_known": delegation_known,
            "delegation_unknown": n - delegation_known,
            "identifiers_n": identifiers_n,
            "tld_counts": {key: domains[key] for key in CLASSES},
            "tld_shares": {
                key: domains[key] / identifiers_n if identifiers_n else None
                for key in CLASSES
            },
        }
    warnings = []
    injected, authorized = result["injected"], result["authorized"]
    for field in ("shares", "tld_shares"):
        for cue, attack in injected[field].items():
            allowed = authorized[field][cue]
            if (
                attack is not None
                and allowed is not None
                and abs(attack - allowed) > 0.8
            ):
                warnings.append(
                    f"Cue {cue} separates injected from authorized by "
                    f"{abs(attack - allowed) * 100:.1f} percentage points"
                )
    return {
        "by_type": result,
        "warnings": warnings,
        "definitions": {
            "tld_shares": (
                "Distinct planted/target identifiers per episode; "
                "denominator identifiers_n."
            ),
            "shares": (
                "Episode shares; delegation uses delegation_known "
                "(legacy builds may be unknown)."
            ),
        },
    }


def write_cue_audit(episodes: list[dict], directory: Path) -> dict:
    """Write beside summary before the release manifest hashes build files."""
    report = cue_audit(episodes)
    (directory / "cue_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", "utf-8"
    )
    for warning in report["warnings"]:
        LOG.warning(warning)
    return report


def audit_build(split: str, *, build_dir: Path | None = None) -> dict:
    """Audit an existing build; frozen releases remain read-only."""
    directory = (
        Path(build_dir)
        if build_dir is not None
        else Path(__file__).resolve().parents[2] / "data/v1" / split
    )
    episodes = [
        json.loads(line)
        for line in (directory / "episodes.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    if (directory / "FROZEN").exists():
        from turnguard.data.freeze import verify_frozen

        verify_frozen(directory)
        return cue_audit(episodes)
    return write_cue_audit(episodes, directory)
