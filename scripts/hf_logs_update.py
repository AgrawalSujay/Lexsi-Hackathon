"""Add the larger-model runs and classifier-baseline outputs to the TurnGuard logs dataset.

Uploads only new files (the dataset stays private until it is made public), after
checking that no API key value or unexpected product mention appears in them.
Run from the repo root: .venv/bin/python scripts/hf_logs_update.py
"""

import re
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "SujayAgrawal/turnguard-eval-logs"
RUNS = [
    "A0__command-a-03-2025__test__20261113-1512",
    "A0__qwen3.8-27b__test__20261113-1708",
    "A0__qwen3.8-27b__test_cueflip__20261113-1714",
]
RUN_FILES = ("config.yaml", "episodes.jsonl", "metrics.json", "env.json")
ANALYSIS = [
    "analysis/api_subset_ids.txt",
    "analysis/fig_operating_points.png",
    "analysis/detector_Llama-Prompt-Guard-2-86M.json",
    "analysis/detector_Llama-Prompt-Guard-2-86M.md",
    "analysis/detector_Llama-Prompt-Guard-2-86M_episodes.jsonl",
    "analysis/detector_deberta-v3-base-prompt-injection-v2.json",
    "analysis/detector_deberta-v3-base-prompt-injection-v2.md",
    "analysis/detector_deberta-v3-base-prompt-injection-v2_episodes.jsonl",
]
CARD_SECTION = """
## Larger models and classifier baselines

- `runs/A0__command-a-03-2025__test__*`: Cohere Command A (111B) through the Cohere API
  on a 159-episode stratified subset of the frozen test set (IDs in
  `analysis/api_subset_ids.txt`). Same prompt, step loop and grader as the Aya runs.
- `runs/A0__qwen3.8-27b__test__*` and `__test_cueflip__*`: Qwen3.8-27B through Vultr
  Serverless Inference, thinking mode off, on the full test and cue-flip sets.
- `analysis/detector_*`: Meta Prompt Guard 2 (86M) and ProtectAI DeBERTa-v3
  prompt-injection v2 scored on the tool outputs of every test and cue-flip episode:
  summary tables (`.md`, `.json`) and per-episode scores (`_episodes.jsonl`).
- `analysis/fig_operating_points.png`: attack success vs task success and delegated
  following, including the filters simulated in front of the base model.
"""


def secrets():
    values = []
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            name, sep, value = line.partition("=")
            value = value.strip().strip("'\"")
            if sep and "KEY" in name.upper() and len(value) >= 12:
                values.append(value)
    return values


def main():
    files = []
    for run in RUNS:
        for name in RUN_FILES:
            path = Path("runs") / run / name
            if not path.exists():
                sys.exit(f"missing {path}")
            files.append(path)
    for name in ANALYSIS:
        if not Path(name).exists():
            sys.exit(f"missing {name}")
        files.append(Path(name))

    keys = secrets()
    pattern = re.compile(r"\b(claude|codex|anthropic)\b", re.I)
    problems = []
    for path in files:
        if path.suffix == ".png":
            continue
        text = path.read_text(errors="ignore")
        if any(k in text for k in keys):
            problems.append(f"API key value found in {path}")
        if re.search(r"(hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,})", text):
            problems.append(f"token-like string in {path}")
        hits = pattern.findall(text)
        if hits:
            problems.append(f"{len(hits)} mention(s) of {sorted(set(h.lower() for h in hits))} in {path}")
    if problems:
        print("Not uploading:\n- " + "\n- ".join(problems))
        sys.exit(1)

    api = HfApi()
    card = Path(hf_hub_download(REPO, "README.md", repo_type="dataset")).read_text()
    if "## Larger models and classifier baselines" not in card:
        card = card.rstrip() + "\n" + CARD_SECTION
    Path("hf_logs_README.tmp.md").write_text(card)

    from huggingface_hub import CommitOperationAdd

    ops = [CommitOperationAdd(path_in_repo=str(p), path_or_fileobj=str(p)) for p in files]
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj="hf_logs_README.tmp.md"))
    api.create_commit(
        REPO,
        operations=ops,
        repo_type="dataset",
        commit_message="Add larger-model runs and classifier baselines",
    )
    Path("hf_logs_README.tmp.md").unlink()
    print(f"uploaded {len(files)} files + README to https://huggingface.co/datasets/{REPO}")


if __name__ == "__main__":
    main()
