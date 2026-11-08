"""Load canonical episodes, with test restricted to explicit evaluation only."""

import json
from pathlib import Path

from turnguard.schema import EpisodeSpec, validate_episode


def load_split(
    split: str, *, purpose: str, root: Path | None = None
) -> list[EpisodeSpec]:
    """Test access requires exactly purpose='eval', even before filesystem access."""
    if split in ("test", "test_cueflip") and purpose != "eval":
        raise ValueError(
            f"split={split} is evaluation-only; purpose={purpose!r} is forbidden"
        )
    if split not in ("train", "dev", "test", "test_cueflip"):
        raise ValueError("split must be train, dev, test or test_cueflip")
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    directory = root / "data/v1" / split
    if (directory / "FROZEN").exists():
        from turnguard.data.freeze import verify_frozen

        verify_frozen(directory)
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
