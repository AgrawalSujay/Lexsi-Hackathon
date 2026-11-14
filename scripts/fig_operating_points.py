"""Operating points: attack success vs task success and delegated following.

Points come from the frozen test runs. "A0 + filter" simulates a classifier in
front of the base model: an attack succeeds only if A0 fell for it and the
filter missed it; a clean task or a delegated instruction counts only if A0
completed it and the filter did not flag it.

Usage: python scripts/fig_operating_points.py [--runs runs] [--analysis analysis]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from results_report import episodes  # noqa: E402

AYA = "CohereLabs-aya-expanse-8b"
MODELS = [
    ("A0 base", f"A0__{AYA}__test__", "#2a78d6", "o"),
    ("A1 prompt", f"A1__{AYA}__test__", "#eb6834", "o"),
    ("A2 steering", f"A2__{AYA}__test__", "#1baf7a", "o"),
    ("A4 DPO adapter", f"A4__{AYA}__test__", "#c98a00", "*"),
    ("Qwen3.8 27B", "A0__qwen3.8-27b__test__", "#7b52b8", "D"),
    ("Command A 111B (subset)", "A0__command-a-03-2025__test__", "#b8336a", "D"),
]
FILTERS = [
    ("A0 + Prompt Guard 2", "detector_Llama-Prompt-Guard-2-86M_episodes.jsonl", "s"),
    (
        "A0 + DeBERTa filter",
        "detector_deberta-v3-base-prompt-injection-v2_episodes.jsonl",
        "s",
    ),
]


def latest(root, prefix):
    found = sorted(Path(root).glob(prefix + "*"))
    return str(found[-1]) if found else None


def rates(ep):
    out = {}
    for kind in ("injected", "clean", "authorized"):
        vals = [e["value"] for e in ep.values() if e["type"] == kind]
        out[kind] = (100 * sum(vals) / len(vals), len(vals)) if vals else (None, 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--analysis", default="analysis")
    ap.add_argument("--data", default="data/v1/test")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    points = []
    base = None
    for label, prefix, color, marker in MODELS:
        run = latest(args.runs, prefix)
        if not run:
            continue
        ep = episodes(run, args.data)
        if label.startswith("A0 base"):
            base = ep
        r = rates(ep)
        points.append(
            (
                label,
                r["injected"][0],
                r["clean"][0],
                r["authorized"][0],
                color,
                marker,
                False,
            )
        )

    if base:
        for label, name, marker in FILTERS:
            path = Path(args.analysis) / name
            if not path.exists():
                continue
            flag = {}
            for line in open(path):
                row = json.loads(line)
                if row["split"] == "test":
                    flag[row["id"]] = row["p_field"] >= args.threshold
            sim = {
                eid: {**e, "value": e["value"] and not flag.get(eid, False)}
                for eid, e in base.items()
            }
            r = rates(sim)
            points.append(
                (
                    label,
                    r["injected"][0],
                    r["clean"][0],
                    r["authorized"][0],
                    "#6b6b6b",
                    marker,
                    True,
                )
            )

    # A filter that flags nothing lands exactly on A0: fold it into A0's label.
    a0 = next((p for p in points if p[0].startswith("A0 base")), None)
    kept = []
    for p in points:
        if (
            p[6]
            and a0
            and all(abs((p[i] or 0) - (a0[i] or 0)) < 0.05 for i in (1, 2, 3))
        ):
            name = p[0].replace("A0 + ", "")
            a0 = (f"A0 base (= A0 + {name})",) + a0[1:]
            continue
        kept.append(p)
    points = [a0 if p[0].startswith("A0 base") else p for p in kept]
    offsets = {
        "A0": (6, -16, "left"),
        "A1": (9, 2, "left"),
        "A2": (-6, 9, "right"),
        "A4": (10, 4, "left"),
        "Qw": (10, 4, "left"),
        "Co": (0, -20, "center"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    panels = [
        (2, "Clean task success (%) ↑"),
        (3, "Delegated instructions followed (%) ↑"),
    ]
    for ax, (idx, ylabel) in zip(axes, panels):
        for p in points:
            label, x, y = p[0], p[1], p[idx]
            if x is None or y is None:
                continue
            hollow = p[6]
            ax.scatter(
                x,
                y,
                s=170 if p[5] == "*" else 80,
                marker=p[5],
                color="white" if hollow else p[4],
                edgecolors=p[4],
                linewidths=2,
                zorder=3,
            )
            dx, dy, ha = offsets.get(label[:2], (8, -14, "left"))
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(dx, dy),
                ha=ha,
                fontsize=9,
                color="#333333",
            )
        ax.set_xlabel("Attack success incl. relay (%) ↓")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-2, 60)
        ax.set_ylim(0, 100)
        ax.grid(True, color="#e6e6e6", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.text(1, 97, "better ↖", fontsize=9, color="#888888", va="top")
    axes[0].set_title(
        "Safety vs task completion (frozen test)", fontsize=11, loc="left"
    )
    axes[1].set_title(
        "Safety vs following what the user delegated", fontsize=11, loc="left"
    )
    fig.text(
        0.01,
        0.01,
        "Hollow squares: A0 with a classifier in front (threshold 0.5), "
        "simulated per episode. Command A: 159-episode stratified subset, "
        "no clean episodes.",
        fontsize=8,
        color="#666666",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = Path(args.analysis) / "fig_operating_points.png"
    fig.savefig(out, dpi=200)

    def fmt(v):
        return "—" if v is None else f"{v:.1f}%"

    for p in points:
        print(
            f"{p[0]:26s} attack {fmt(p[1])}  clean {fmt(p[2])}  delegated {fmt(p[3])}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
