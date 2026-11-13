"""Larger models through an API on TurnGuard, compared with Aya A0/A4 on the same IDs.

  subset  -> stratified episode-ID list for call-capped keys (e.g. a trial key)
  compare -> rates on the IDs every listed run covers, with exact McNemar vs a base

Usage:
  python scripts/api_models.py subset --injected 100 --authorized 60
  python scripts/api_models.py compare runs/A0__CohereLabs-aya-expanse-8b__test__* \
      runs/A4__CohereLabs-aya-expanse-8b__test__* runs/A0__command-a-03-2025__test__*
"""

import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from results_report import episodes  # noqa: E402

KINDS = (
    ("injected", "Attack success (incl. relay)"),
    ("clean", "Clean task success"),
    ("authorized", "Authorized followed"),
)


def subset(args):
    specs = [json.loads(x) for x in open(Path(args.data) / "test" / "episodes.jsonl")]
    rng = random.Random(args.seed)
    out = []
    for kind, n, keys in (
        ("injected", args.injected, ("lang", "goal")),
        ("authorized", args.authorized, ("lang",)),
    ):
        groups = collections.defaultdict(list)
        for s in specs:
            if s["episode_type"] == kind:
                key = tuple(
                    s["lang"]["injection"] if k == "lang" else s.get(k) for k in keys
                )
                groups[key].append(s["id"])
        total = sum(len(v) for v in groups.values())
        for key in sorted(groups):
            ids = sorted(groups[key])
            take = max(1, round(n * len(ids) / total))
            out += rng.sample(ids, min(take, len(ids)))
        if args.clean_twins and kind == "injected":
            chosen = set(out)
            out += [s["id"] for s in specs if s.get("twin_of") in chosen]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(sorted(out)) + "\n")
    print(f"{len(out)} IDs -> {args.out}")


def mcnemar(b, c):
    n = b + c
    if not n:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def wilson(k, n, z=1.96):
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def compare(args):
    runs = {Path(r).name: episodes(r, args.data_split) for r in args.runs}
    names = list(runs)
    common = set.intersection(*(set(e) for e in runs.values()))
    base = names[0]
    lines = [f"Episodes covered by every run: {len(common)}", ""]
    head = (
        "| Metric | "
        + " | ".join(n.split("__")[0] + " " + n.split("__")[1] for n in names)
        + " |"
    )
    lines += [head, "|---" * (len(names) + 1) + "|"]
    result = {"common": len(common), "metrics": {}}
    for kind, label in KINDS:
        ids = sorted(i for i in common if runs[base][i]["type"] == kind)
        if not ids:
            continue
        cells = []
        for n in names:
            k = sum(runs[n][i]["value"] for i in ids)
            lo, hi = wilson(k, len(ids))
            rate = f"{100 * k / len(ids):.1f}%"
            cell = f"{rate} ({k}/{len(ids)}; {100 * lo:.0f}–{100 * hi:.0f})"
            if n != base:
                b = sum(runs[base][i]["value"] and not runs[n][i]["value"] for i in ids)
                c = sum(runs[n][i]["value"] and not runs[base][i]["value"] for i in ids)
                cell += f", p={mcnemar(b, c):.2g}"
            cells.append(cell)
            result["metrics"].setdefault(kind, {})[n] = {"k": k, "n": len(ids)}
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", f"p = exact McNemar vs {base} on the same episodes."]
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
        Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("subset")
    s.add_argument("--data", default="data/v1")
    s.add_argument("--injected", type=int, default=100)
    s.add_argument("--authorized", type=int, default=60)
    s.add_argument("--clean-twins", action="store_true")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="analysis/api_subset_ids.txt")
    c = sub.add_parser("compare")
    c.add_argument("runs", nargs="+")
    c.add_argument("--data-split", default="data/v1/test")
    c.add_argument("--out")
    args = ap.parse_args()
    subset(args) if args.cmd == "subset" else compare(args)


if __name__ == "__main__":
    main()
