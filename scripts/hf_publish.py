"""Stage and publish the TurnGuard Hugging Face artifacts.

Two steps:
  python scripts/hf_publish.py stage            # build hf_staging/ (stdlib only, no network)
  python scripts/hf_publish.py upload           # push hf_staging/ and create collections

Staging copies only the reported artifacts, redacts model responses to unsafe XSTest
prompts, drops judge caches (which hold raw responses), and refuses to continue if it
finds anything that looks like a secret. Nothing is uploaded until `upload` is run.
"""

# ruff: noqa: E501  (model-card templates contain long markdown lines)
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

USER = "SujayAgrawal"
GITHUB = "https://github.com/AgrawalSujay/Lexsi-Hackathon"
BASE = "CohereLabs/aya-expanse-8b"
MODEL = "CohereLabs-aya-expanse-8b"
STAGE = Path("hf_staging")

REPOS = {
    "adapter": ("model", f"{USER}/turnguard-aya-expanse-8b-a4-dpo-lora"),
    "steering": ("model", f"{USER}/turnguard-aya-expanse-8b-a2-caa-steering"),
    "bench": ("dataset", f"{USER}/turnguard-bench"),
    "pairs": ("dataset", f"{USER}/turnguard-dpo-pairs"),
    "logs": ("dataset", f"{USER}/turnguard-eval-logs"),
}
COLLECTIONS = {
    "models": (
        "TurnGuard: models",
        "Aya Expanse 8B fixes for multi-turn multilingual indirect prompt injection: "
        "DPO LoRA adapter (A4) and CAA steering vector (A2).",
        [
            ("model", BASE),
            ("model", REPOS["adapter"][1]),
            ("model", REPOS["steering"][1]),
        ],
    ),
    "datasets": (
        "TurnGuard: datasets and logs",
        "TurnGuard benchmark (EN/HI/Hinglish, frozen test + cue-flip), DPO preference "
        "pairs, and every evaluation run with metrics and traces.",
        [
            ("dataset", REPOS["bench"][1]),
            ("dataset", REPOS["pairs"][1]),
            ("dataset", REPOS["logs"][1]),
        ],
    ),
}

SECRET = re.compile(
    r"hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{20,}|Bearer\s+[A-Za-z0-9._-]{20,}"
    r"|(?:api[_-]?key|token|secret)\"?\s*[:=]\s*\"[A-Za-z0-9_\-]{20,}\"",
    re.I,
)
MENTION = re.compile(r"\b(claude|codex|anthropic)\b", re.I)
REDACTED = "[redacted: model response to an unsafe XSTest prompt; label kept]"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def run_dir(arm: str, split: str, model: str = MODEL) -> Path:
    found = sorted(glob.glob(f"runs/{arm}__{model}__{split}__*"))
    if not found:
        sys.exit(f"missing run: {arm} {split}")
    return Path(found[-1])


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def load_results() -> dict:
    return json.load(open("analysis/results.json"))


def headline_table(results: dict, split: str) -> str:
    arms = results["splits"][split]["arms"]
    rows = [
        ("asr_incl_relay", "Attack success incl. relay ↓"),
        ("clean_task_success", "Clean task success ↑"),
        ("utility_under_attack", "Task success under attack ↑"),
        ("authorized_followed", "Authorized instruction followed ↑"),
    ]
    names = {
        "A0": "A0 baseline",
        "A1": "A1 prompt",
        "A2": "A2 steering",
        "A4": "A4 DPO",
    }
    out = [
        "| Metric | " + " | ".join(names[a] for a in arms) + " |",
        "|---|" + "---|" * len(arms),
    ]
    for key, label in rows:
        cells = []
        for a in arms:
            m = arms[a]["metrics"][key]
            cell = pct(m["rate"])
            if a != "A0" and m.get("p") is not None and m["p"] < 0.05:
                cell += "*"
            cells.append(cell)
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out.append(
        "| Malformed steps ↓ | "
        + " | ".join(pct(arms[a]["malformed_rate"]) for a in arms)
        + " |"
    )
    out.append("")
    out.append("\\* paired McNemar exact test vs A0, p < 0.05.")
    return "\n".join(out)


def side_table(results: dict) -> str:
    s = results["side"]
    arms = list(s)
    out = ["| Metric | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    out.append(
        "| XSTest over-refusal (safe prompts) ↓ | "
        + " | ".join(pct(s[a]["xstest_over_refusal"]) for a in arms)
        + " |"
    )
    out.append(
        "| XSTest refusal (unsafe prompts) ↑ | "
        + " | ".join(pct(s[a]["xstest_unsafe_refusal"]) for a in arms)
        + " |"
    )
    for lang in ("en", "hi"):
        out.append(
            f"| Global-MMLU-Lite {lang}, strict / lenient | "
            + " | ".join(
                f"{pct(s[a]['mmlu'][lang]['strict'])} / {pct(s[a]['mmlu'][lang].get('lenient', float('nan')))}"
                for a in arms
            )
            + " |"
        )
    return "\n".join(out)


LICENSE_NOTE = (
    "Released under **CC-BY-NC-4.0**, inherited from Aya Expanse 8B "
    "(CC-BY-NC-4.0 plus the Cohere Labs Acceptable Use Policy). Non-commercial use only."
)


def card_adapter(results: dict, metrics: dict) -> str:
    cfg = metrics.get("config", {})
    return f"""---
base_model: {BASE}
library_name: peft
license: cc-by-nc-4.0
language: [en, hi]
pipeline_tag: text-generation
tags: [prompt-injection, agents, safety, dpo, lora, safetune, multilingual, hinglish]
datasets: [{REPOS["pairs"][1]}, {REPOS["bench"][1]}]
---

# TurnGuard A4: DPO LoRA for Aya Expanse 8B against indirect prompt injection

A LoRA adapter for [{BASE}](https://huggingface.co/{BASE}) trained with SafeTune's
`CSTTrainer` (DPO) to make a tool-using agent **keep doing the user's task when tool
results contain injected instructions**, without refusing legitimate instructions the
user delegated to those sources. Part of the TurnGuard project (Lexsi Labs AI Safety &
Alignment Hackathon, Track 2). Code: {GITHUB}

## Results on the frozen TurnGuard test set (1,043 episodes, 15 unseen scenarios)

{headline_table(results, "test")}

On the cue-flip test set (surface cues changed) A4 reduces attack success from
{pct(results["splits"]["test_cueflip"]["arms"]["A0"]["metrics"]["asr_incl_relay"]["rate"])}
to {pct(results["splits"]["test_cueflip"]["arms"]["A4"]["metrics"]["asr_incl_relay"]["rate"])}.

Side effects (XSTest, Global-MMLU-Lite en/hi; *lenient* is a post-hoc parser that also
accepts a letter alone on the first line, applied to every arm):

{side_table(results)}

Known costs: malformed protocol steps rise (6% → 14%); strict Hindi MMLU drops mostly
from answer-format drift (unparseable answers 56 → 171 of 400; lenient accuracy is close
to baseline); authorized-instruction following does not improve significantly.

## Training

- Data: [{REPOS["pairs"][1]}](https://huggingface.co/datasets/{REPOS["pairs"][1]}), 1,032
  on-policy preference pairs built from the base model's own runs on the 12 training
  scenarios (real and synthesized injection failures vs. continuing the task, plus
  authorized-instruction and clean-task controls). No test or dev scenario is used.
- SafeTune 0.1.14 `CSTTrainer` on TRL 1.14.2, transformers 5.21.1.
- LoRA r={cfg.get("rank")}, alpha={cfg.get("alpha")} on attention and MLP projections;
  lr {cfg.get("learning_rate")}, β {cfg.get("beta")}, {cfg.get("epochs")} epochs,
  max length {cfg.get("max_length")}, seed {cfg.get("seed", 1234)}. About 21 minutes on one H100.
- `training_log.json` and `metrics.json` hold the full log and held-out reward accuracy.

## Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("{REPOS["adapter"][1]}")
model = AutoModelForCausalLM.from_pretrained("{BASE}", torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, "{REPOS["adapter"][1]}")
```

The agent protocol (ACTION/FINAL lines, tool-result messages) is defined in the GitHub
repo (`docs/protocol.md`); the adapter was trained and evaluated with it.

## Limitations

One training seed; single base model; evaluation in a simulated tool sandbox; part of
the utility gain reflects better adherence to the agent action format. See the write-up
in the GitHub repo.

## License

{LICENSE_NOTE}
"""


def card_steering(results: dict, summary: dict) -> str:
    chosen = summary["chosen"]
    return f"""---
base_model: {BASE}
license: cc-by-nc-4.0
language: [en, hi]
tags: [prompt-injection, agents, safety, steering, caa, safetune, multilingual]
---

# TurnGuard A2: CAA steering vector for Aya Expanse 8B

Contrastive activation addition (SafeTune `CAAModel`) vectors extracted from the
TurnGuard training preference pairs: chosen minus rejected, mean over the last prompt
token and up to 8 completion tokens, unit-normalised per layer (layers 10, 14, 18, 22).
The applied point was selected on a dev tuning subset (50 injected / 50 clean / 50
authorized episodes) by a pre-registered rule: minimise attack success subject to clean
success and authorized following within 5 points of the baseline. Selected:
**layer {chosen["layer"]}, alpha {chosen["alpha"]}**. Code: {GITHUB}

## Result: a negative finding on held-out data

{headline_table(results, "test")}

The dev-subset gain (32% → 22% attack success) did **not** transfer to the frozen test
set: attack success is unchanged, while clean task success and answer correctness rise.
The vector behaves like a task-competence direction, not an injection-resistance one.

## Files

- `steering/A2/headline.pt`, `.json`, `.yaml`: vectors, provenance and the arm config.
- `summary.json`: the full grid (6 points) with dev metrics and the selection rule.

## License

{LICENSE_NOTE}
"""


def card_bench(counts: dict) -> str:
    rows = "\n".join(
        f"| {s} | {c['injected']} | {c['clean']} | {c['authorized']} | {sum(c.values())} |"
        for s, c in counts.items()
    )
    return f"""---
license: cc-by-nc-4.0
language: [en, hi]
task_categories: [text-generation]
tags: [prompt-injection, agents, tool-use, safety, multilingual, hinglish, code-mixed, benchmark]
pretty_name: TurnGuard
configs:
- config_name: default
  data_files:
  - {{split: train, path: train/episodes.jsonl}}
  - {{split: dev, path: dev/episodes.jsonl}}
  - {{split: test, path: test/episodes.jsonl}}
  - {{split: test_cueflip, path: test_cueflip/episodes.jsonl}}
---

# TurnGuard: multi-turn multilingual indirect prompt injection for tool-using agents

Episodes for a simulated tool-using assistant (email, calendar, documents, tickets, web)
in English, Hindi and Hinglish. In an **injected** episode, an instruction hidden in a
tool result (email body, metadata, quoted text) tries to make the agent exfiltrate data,
take an extra action, destroy data, redirect, or tamper with its answer, possibly on a
later turn. Each injected episode has a **clean twin** without the injection, and
**authorized** episodes contain in-data instructions the user explicitly delegated, so
over-refusal is measured next to attack success. Code and harness: {GITHUB}

| Split | Injected | Clean twin | Authorized | Total |
|---|---|---|---|---|
{rows}

- `train` (12 scenarios), `dev` (8 scenarios) and `test` (15 scenarios, including a
  scenario family used only in test) are disjoint by scenario.
- `test` and `test_cueflip` are **frozen** (`FROZEN`, `CHECKSUMS.sha256`); cue-flip
  rewrites the surface cues of injections to test generalisation.
- Built with CuratorKIT (quality gates, dedup) from a DeepSeek teacher; every
  translation was cross-checked by a second model (GLM) and failing items were dropped
  (`audits/`, `*/dropped.jsonl`). `teacher_calls.jsonl` keeps the generation provenance.
- All addresses and domains are sandbox values (`*.invalid`, `example.*`); tools are
  simulated and have no network access.

## Metrics

Attack success counts an injection as successful if the agent executes it **or relays
it to the user**; clean task success, task success under attack, authorized-instruction
following and answer correctness are scored by deterministic oracles. See
`docs/protocol.md` and `turnguard/metrics.py` in the GitHub repo.

## License

Released under **CC-BY-NC-4.0**. Synthetic data; no personal data.
"""


def card_pairs(summary: dict) -> str:
    return f"""---
license: cc-by-nc-4.0
language: [en, hi]
tags: [dpo, preference, prompt-injection, agents, safety]
---

# TurnGuard DPO preference pairs

1,032 on-policy preference pairs used to train
[{REPOS["adapter"][1]}](https://huggingface.co/{REPOS["adapter"][1]}). They were built
from Aya Expanse 8B's own runs on the TurnGuard **train** split only (12 scenarios).
Each prompt is the full multi-turn conversation up to the decision step.

- **Injected (516):** chosen = continue the user's task; rejected = the model's real
  executed (31) or relayed (52) attack, or a synthesized failure (433).
- **Controls (516):** authorized-instruction pairs (359, chosen = follow the delegated
  instruction), clean-task anchors (157).

`summary.json` has the full composition and source-run hashes; `audit_sample.md` is a
manual review sample; `skipped.jsonl` lists episodes that produced no pair and why.
Rejected responses include model outputs and are released under **CC-BY-NC-4.0**.
"""


def card_logs(results: dict) -> str:
    return f"""---
license: cc-by-nc-4.0
language: [en, hi]
tags: [evaluation, logs, prompt-injection, agents, safety]
---

# TurnGuard evaluation logs and traces

Every evaluation run behind the TurnGuard results, with fully resolved configs,
per-episode traces and grades, metrics and environment provenance (package versions,
GPU, git commit). Code: {GITHUB}

- `runs/<arm>__<model>__<split>__<timestamp>/`: `config.yaml`, `episodes.jsonl`
  (full multi-turn traces with oracle results), `metrics.json`, `env.json`.
- Arms: A0 baseline, A1 prompt defence (`prompts/A1_defence.txt`), A2 CAA steering,
  A4 DPO LoRA. Splits: test, test_cueflip, dev, train (A0 runs used to build pairs).
- A2 tuning: three grid shards and the merged selection.
- Side evaluations (`*__side__*`): XSTest and Global-MMLU-Lite en/hi. **Responses to
  unsafe XSTest prompts are redacted** in `scored_redacted.jsonl` (judge labels kept);
  raw judge caches are not published.
- `analysis/`: the generated results tables (`results.md`, `results.json`) and figures.

## Headline (frozen test)

{headline_table(results, "test")}
"""


def stage(args) -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    results = load_results()

    # Adapter.
    train_run = Path(sorted(glob.glob(f"runs/A4__{MODEL}__train-a4__*"))[-1])
    adir = STAGE / "adapter"
    for f in (train_run / "adapters/A4/headline").iterdir():
        if f.name in ("README.md", "training_args.bin"):
            continue
        copy(f, adir / f.name)
    for name in (
        "metrics.json",
        "training_log.json",
        "config.yaml",
        "pair_split.json",
        "env.json",
    ):
        if (train_run / name).exists():
            copy(train_run / name, adir / "training" / name)
    metrics = json.load(open(train_run / "metrics.json"))
    (adir / "README.md").write_text(card_adapter(results, metrics))

    # Steering.
    merged = Path(sorted(glob.glob(f"runs/A2__{MODEL}__steer-merged-*"))[-1])
    sdir = STAGE / "steering"
    copy(merged / "steering", sdir / "steering")
    for name in ("summary.json", "selected_arm.yaml", "env.json"):
        copy(merged / name, sdir / name)
    (sdir / "README.md").write_text(
        card_steering(results, json.load(open(merged / "summary.json")))
    )

    # Benchmark.
    bdir = STAGE / "bench"
    counts = {}
    for split in ("train", "dev", "test", "test_cueflip"):
        src = Path("data/v1") / split
        for f in src.iterdir():
            if f.is_file() and f.name != ".gitkeep":
                copy(f, bdir / split / f.name)
        types = [
            json.loads(line)["episode_type"] for line in open(src / "episodes.jsonl")
        ]
        counts[split] = {t: types.count(t) for t in ("injected", "clean", "authorized")}
    for split in ("train", "dev", "test"):
        src = Path("data/crosscheck") / split
        for f in src.iterdir():
            if f.is_file():
                copy(f, bdir / "audits" / split / f.name)
    (bdir / "README.md").write_text(card_bench(counts))

    # Pairs.
    pdir = STAGE / "pairs"
    copy(Path("data/pairs/train"), pdir)
    (pdir / "README.md").write_text(
        card_pairs(json.load(open("data/pairs/train/summary.json")))
    )

    # Logs.
    ldir = STAGE / "logs"
    keep = [
        run_dir(a, s)
        for a in ("A0", "A1", "A2", "A4")
        for s in ("test", "test_cueflip")
    ]
    keep += [
        run_dir("A0", "dev"),
        run_dir("A4", "dev"),
        run_dir("A0", "train-shard-1-of-2"),
        run_dir("A0", "train-shard-2-of-2"),
        run_dir("A0", "dev", "CohereLabs-tiny-aya-global"),
    ]
    keep += [
        Path(p) for p in sorted(glob.glob(f"runs/A2__{MODEL}__steer-tune-shard-*"))
    ]
    keep += [Path(p) for p in sorted(glob.glob(f"runs/A2__{MODEL}__dev-steer-*"))]
    keep += [merged]
    for run in keep:
        for f in run.iterdir():
            if f.name.startswith("checkpoint-") or f.name == "adapters":
                continue
            copy(f, ldir / "runs" / run.name / f.name)
    unsafe_texts = []
    for arm in ("A0", "A1", "A2", "A4"):
        side = run_dir(arm, "side")
        out = ldir / "runs" / side.name
        for name in (
            "config.yaml",
            "env.json",
            "inputs.jsonl",
            "manifest.json",
            "metrics.json",
            "timings.json",
            "judge_config.json",
        ):
            if (side / name).exists():
                copy(side / name, out / name)
        rows = []
        for line in open(side / "scored.jsonl"):
            r = json.loads(line)
            if r.get("benchmark") == "xstest" and r.get("subset") == "unsafe":
                # Refusals are harmless and often repeat verbatim on safe prompts, so
                # only non-refusals are used to verify the redaction.
                if r.get("response") and r.get("teacher_class") != "full_refusal":
                    unsafe_texts.append(r["response"])
                r["response"] = REDACTED
            rows.append(json.dumps(r, ensure_ascii=False))
        (out / "scored_redacted.jsonl").write_text("\n".join(rows) + "\n")
        log = Path("runs") / f"judge_{arm}.log"
        if log.exists():
            copy(log, ldir / "judge" / f"judge_{arm}.json")
    copy(Path("analysis"), ldir / "analysis")
    arms = Path("configs/arms.yaml").read_text()
    prompt = re.search(r"A1:\s*\n\s*system_prompt:\s*>-\s*\n((?:\s{6}.*\n)+)", arms)
    if prompt:
        (ldir / "prompts").mkdir(parents=True, exist_ok=True)
        text = " ".join(line.strip() for line in prompt.group(1).splitlines())
        (ldir / "prompts" / "A1_defence.txt").write_text(text + "\n")
    (ldir / "README.md").write_text(card_logs(results))

    # Checks: secrets, unredacted unsafe responses, stray mentions.
    problems, mentions = [], []
    probes = [t.strip()[:120] for t in unsafe_texts if len(t.strip()) >= 40]
    for path in STAGE.rglob("*"):
        if not path.is_file() or path.suffix in (".safetensors", ".pt", ".bin", ".png"):
            continue
        text = path.read_text(errors="ignore")
        if SECRET.search(text):
            problems.append(f"possible secret: {path}")
        if path.name != "tokenizer.json" and MENTION.search(text):
            mentions.append(str(path))
        for p in probes:
            if p in text:
                problems.append(f"unredacted unsafe XSTest response: {path}")
                break
    sizes = {
        name: sum(f.stat().st_size for f in (STAGE / name).rglob("*") if f.is_file())
        for name in REPOS
    }
    manifest = {
        name: {"repo": repo, "type": kind, "bytes": sizes[name]}
        for name, (kind, repo) in REPOS.items()
    }
    (STAGE / "PLAN.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, info in manifest.items():
        print(f"{info['type']:8s} {info['repo']:55s} {info['bytes'] / 1e6:8.1f} MB")
    if mentions:
        print("\nFiles mentioning tool names (review before upload):")
        for m in mentions:
            print("  ", m)
    if problems:
        print("\nBLOCKING PROBLEMS:")
        for p in problems:
            print("  ", p)
        sys.exit(1)
    print(
        "\nStaged in hf_staging/. Review, then run: python scripts/hf_publish.py upload"
    )


def upload(args) -> None:
    from huggingface_hub import HfApi

    plan = json.load(open(STAGE / "PLAN.json"))
    api = HfApi()
    who = api.whoami()["name"]
    if who != USER:
        sys.exit(f"logged in as {who}, expected {USER}")
    for name, info in plan.items():
        api.create_repo(
            info["repo"], repo_type=info["type"], private=False, exist_ok=True
        )
        api.upload_folder(
            folder_path=str(STAGE / name),
            repo_id=info["repo"],
            repo_type=info["type"],
            commit_message="TurnGuard release",
        )
        print(
            f"uploaded https://huggingface.co/{'datasets/' if info['type'] == 'dataset' else ''}{info['repo']}"
        )
    for title, description, items in COLLECTIONS.values():
        existing = [c for c in api.list_collections(owner=USER) if c.title == title]
        coll = (
            existing[0]
            if existing
            else api.create_collection(
                title=title, description=description, private=False
            )
        )
        for kind, repo in items:
            api.add_collection_item(
                coll.slug, item_id=repo, item_type=kind, exists_ok=True
            )
        print(f"collection https://huggingface.co/collections/{coll.slug}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("stage", "upload"))
    parser.add_argument("--out", default="hf_staging", help="staging directory")
    args = parser.parse_args()
    global STAGE
    STAGE = Path(args.out)
    stage(args) if args.step == "stage" else upload(args)


if __name__ == "__main__":
    main()
