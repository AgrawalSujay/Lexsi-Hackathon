"""Build the results tables and figures from downloaded run folders.

Reads runs/<arm>__CohereLabs-aya-expanse-8b__{test,test_cueflip}__* for every arm
present, the side-effect judge logs (runs/judge_<arm>.log), the side generation runs
(for the post-hoc lenient MMLU column) and the merged A2 grid summary. Writes
analysis/results.md, analysis/results.json and analysis/fig_*.png. Never prints model
outputs.

Usage: python scripts/results_report.py [--runs runs] [--data data/v1] [--out analysis]
"""

import argparse
import collections
import glob
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mmlu_lenient import lenient_scores, side_run  # noqa: E402

from turnguard.analysis import compare_runs, load_verified_run, regrade_records

MODEL = "CohereLabs-aya-expanse-8b"
ARMS = ("A0", "A1", "A2", "A4")
NAMES = {
    "A0": "A0 baseline",
    "A1": "A1 prompt defence",
    "A2": "A2 steering (CAA)",
    "A4": "A4 DPO (SafeTune)",
}
# Validated categorical slots 1-4 (light surface); fixed per arm, never by rank.
COLORS = {"A0": "#2a78d6", "A1": "#eb6834", "A2": "#1baf7a", "A4": "#eda100"}
SPLITS = ("test", "test_cueflip")
SPLIT_NAMES = {"test": "Test (frozen)", "test_cueflip": "Cue-flip (frozen)"}
METRICS = (
    ("asr_incl_relay", "ASR incl. relay", "lower"),
    ("asr", "↳ executed", "lower"),
    ("asr_relayed", "↳ relayed", "lower"),
    ("clean_task_success", "Clean task success", "higher"),
    ("utility_under_attack", "Success under attack", "higher"),
    ("authorized_followed", "Authorized followed", "higher"),
    ("clean_answer_correct", "Clean answer correct", "higher"),
)
BOOT = 5000


def find_run(runs, arm, split):
    found = sorted(glob.glob(f"{runs}/{arm}__{MODEL}__{split}__*"))
    return found[-1] if found else None


def scenario(spec):
    body = spec["id"].split("-", 1)[1]
    marker = "-auth-" if spec["episode_type"] == "authorized" else f"-{spec['goal']}-"
    return body[: body.find(marker)]


def episodes(run, specs_dir):
    records, specs, _ = load_verified_run(run, specs_path=specs_dir)
    by_id = {s["id"]: s for s in specs}
    out = {}
    for record in regrade_records(records, specs):
        spec, res = by_id[record.episode_id], record.result
        kind = spec["episode_type"]
        if kind == "injected":
            value = bool(res["injection_executed"] or res["injection_relayed_to_user"])
        elif kind == "clean":
            value = bool(res["task_success"])
        else:
            value = bool(res["authorized_followed"])
        out[record.episode_id] = {
            "scenario": scenario(spec),
            "type": kind,
            "value": value,
            "conversation_lang": spec["lang"]["conversation"],
            "injection_lang": spec["lang"]["injection"],
            "goal": spec.get("goal"),
            "style": spec.get("style"),
            "position": spec.get("position"),
        }
    return out


def cluster_ci(base, other, kind):
    """Scenario-clustered bootstrap CI of the paired difference (other - base)."""
    groups = collections.defaultdict(list)
    for key, ep in base.items():
        if ep["type"] == kind and key in other:
            groups[ep["scenario"]].append((ep["value"], other[key]["value"]))
    keys = sorted(groups)

    def delta(chosen):
        pairs = [p for g in chosen for p in groups[g]]
        return (sum(b for _, b in pairs) - sum(a for a, _ in pairs)) / len(pairs)

    rng = random.Random(0)
    boots = sorted(delta([rng.choice(keys) for _ in keys]) for _ in range(BOOT))
    return (
        delta(keys),
        (boots[int(0.025 * BOOT)], boots[int(0.975 * BOOT) - 1]),
        len(keys),
    )


def breakdown(eps, field):
    rows = collections.defaultdict(list)
    for ep in eps.values():
        if ep["type"] == "injected":
            rows[ep[field]].append(ep["value"])
    return {k: (sum(v) / len(v), len(v)) for k, v in sorted(rows.items())}


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def fmt_p(p):
    if p is None:
        return "—"
    if p < 0.001:
        return f"{p:.0e}"
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def mmlu_cell(score, kind):
    return f"{pct(score[kind])} ({score['unparse_' + kind]})"


def collect(args):
    results = {"splits": {}, "side": {}, "a2_grid": None}
    for split in SPLITS:
        specs_dir = f"{args.data}/{split}"
        base_run = find_run(args.runs, "A0", split)
        if base_run is None:
            continue
        base_eps = episodes(base_run, specs_dir)
        split_out = {"arms": {}, "breakdowns": {}}
        for arm in ARMS:
            run = find_run(args.runs, arm, split)
            if run is None:
                continue
            metrics = json.load(open(f"{run}/metrics.json"))
            arm_out = {
                "run": Path(run).name,
                "malformed_rate": metrics["malformed_rate"]["rate"],
                "metrics": {},
            }
            eps = base_eps if arm == "A0" else episodes(run, specs_dir)
            if arm != "A0":
                cmp = compare_runs(base_run, run, specs_a=specs_dir, specs_b=specs_dir)[
                    "metrics"
                ]
            for key, _, _ in METRICS:
                rate = metrics[key]
                entry = {
                    "k": rate["k"],
                    "n": rate["n"],
                    "rate": rate["rate"],
                    "ci95": rate.get("ci95"),
                }
                if arm != "A0":
                    g = cmp[key]["shared_graded"]
                    entry.update(
                        delta=g["delta_b_minus_a"],
                        p=g["mcnemar_exact_p"],
                        improved=g["improved_in_b"],
                        regressed=g["regressed_in_b"],
                        ci95=g["b"]["ci95"],
                    )
                arm_out["metrics"][key] = entry
            if arm != "A0":
                for key, kind in (
                    ("asr_incl_relay", "injected"),
                    ("clean_task_success", "clean"),
                    ("authorized_followed", "authorized"),
                ):
                    d, ci, n = cluster_ci(base_eps, eps, kind)
                    arm_out["metrics"][key]["scenario_ci"] = ci
                    arm_out["metrics"][key]["scenarios"] = n
            split_out["arms"][arm] = arm_out
            split_out["breakdowns"][arm] = {
                f: breakdown(eps, f)
                for f in (
                    "conversation_lang",
                    "injection_lang",
                    "goal",
                    "style",
                    "position",
                )
            }
        results["splits"][split] = split_out
    for arm in ARMS:
        log = Path(args.runs) / f"judge_{arm}.log"
        if not log.exists():
            continue
        report = json.load(open(log))
        xs = report["xstest"]["teacher"]
        side = {
            "xstest_over_refusal": xs["safe"]["over_refusal"]["rate"],
            "xstest_unsafe_refusal": xs["unsafe"]["refusal"]["rate"],
            "mmlu": {
                lang: {
                    "strict": report["mmlu"][lang]["accuracy"]["rate"],
                    "unparse_strict": report["mmlu"][lang]["unparseable"],
                }
                for lang in ("en", "hi")
            },
        }
        run = side_run(arm, args.runs)
        if run and Path(f"{run}/generations.jsonl").exists():
            for lang, s in lenient_scores(run).items():
                side["mmlu"][lang].update(
                    lenient=s["lenient"], unparse_lenient=s["unparse_lenient"]
                )
        results["side"][arm] = side
    merged = sorted(glob.glob(f"{args.runs}/A2__{MODEL}__steer-merged-*/summary.json"))
    if merged:
        summary = json.load(open(merged[-1]))
        results["a2_grid"] = {
            "baseline": summary["baseline"],
            "points": [
                {
                    k: p[k]
                    for k in (
                        "layer",
                        "alpha",
                        "asr_incl_relay",
                        "clean_task_success",
                        "authorized_followed",
                    )
                }
                for p in summary["points"]
            ],
            "chosen": {k: summary["chosen"][k] for k in ("layer", "alpha")},
        }
    return results


def write_markdown(results, out):
    lines = ["# TurnGuard results (Aya Expanse 8B)", ""]
    lines.append(
        "Paired McNemar exact tests against A0 on identical episodes; "
        "CI = 95% scenario-clustered bootstrap of the paired difference "
        f"({BOOT} resamples). Ungraded episodes: none."
    )
    for split, data in results["splits"].items():
        arms = list(data["arms"])
        lines += ["", f"## {SPLIT_NAMES[split]}", ""]
        lines.append("| Metric | " + " | ".join(NAMES[a] for a in arms) + " |")
        lines.append("|---|" + "---|" * len(arms))
        for key, label, _ in METRICS:
            cells = []
            for arm in arms:
                m = data["arms"][arm]["metrics"][key]
                cell = f"{pct(m['rate'])} ({m['k']}/{m['n']})"
                if arm != "A0":
                    cell += f"<br>Δ {100 * m['delta']:+.1f}, p={fmt_p(m['p'])}"
                    if "scenario_ci" in m:
                        lo, hi = m["scenario_ci"]
                        cell += f"<br>CI [{100 * lo:+.1f}, {100 * hi:+.1f}]"
                cells.append(cell)
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append(
            "| Malformed steps | "
            + " | ".join(pct(data["arms"][a]["malformed_rate"]) for a in arms)
            + " |"
        )
        for field, title in (
            ("conversation_lang", "conversation language"),
            ("goal", "attack goal"),
            ("style", "injection style"),
        ):
            lines += ["", f"ASR incl. relay by {title}:", ""]
            values = sorted({v for a in arms for v in data["breakdowns"][a][field]})
            lines.append(f"| {title} | n | " + " | ".join(arms) + " |")
            lines.append("|---|---|" + "---|" * len(arms))
            for v in values:
                n = data["breakdowns"]["A0"][field][v][1]
                lines.append(
                    f"| {v} | {n} | "
                    + " | ".join(pct(data["breakdowns"][a][field][v][0]) for a in arms)
                    + " |"
                )
    if results["side"]:
        arms = list(results["side"])
        lines += [
            "",
            "## Side effects",
            "",
            "XSTest graded by an LLM judge (DeepSeek v4.1 flash, temperature 0). "
            "MMLU = Global-MMLU-Lite, 400 items per language. *Lenient* is a "
            "post-hoc parser added after seeing results (also accepts a letter "
            "alone on the first line), applied identically to every arm.",
            "",
        ]
        lines.append("| Metric | " + " | ".join(NAMES[a] for a in arms) + " |")
        lines.append("|---|" + "---|" * len(arms))
        s = results["side"]
        lines.append(
            "| XSTest over-refusal (safe, ↓) | "
            + " | ".join(pct(s[a]["xstest_over_refusal"]) for a in arms)
            + " |"
        )
        lines.append(
            "| XSTest refusal (unsafe, ↑) | "
            + " | ".join(pct(s[a]["xstest_unsafe_refusal"]) for a in arms)
            + " |"
        )
        for lang in ("en", "hi"):
            lines.append(
                f"| MMLU {lang} strict (unparseable) | "
                + " | ".join(mmlu_cell(s[a]["mmlu"][lang], "strict") for a in arms)
                + " |"
            )
            if all("lenient" in s[a]["mmlu"][lang] for a in arms):
                lines.append(
                    f"| MMLU {lang} lenient (unparseable) | "
                    + " | ".join(mmlu_cell(s[a]["mmlu"][lang], "lenient") for a in arms)
                    + " |"
                )
    if results["a2_grid"]:
        g = results["a2_grid"]
        b = g["baseline"]
        lines += [
            "",
            "## A2 steering grid (dev tuning subset: 50 injected, 50 clean, "
            "50 authorized)",
            "",
            "| Layer | α | ASR incl. relay | Clean success | Authorized followed |",
            "|---|---|---|---|---|",
            f"| A0 | — | {pct(b['asr_incl_relay'])} | "
            f"{pct(b['clean_task_success'])} | {pct(b['authorized_followed'])} |",
        ]
        for p in sorted(g["points"], key=lambda p: (p["layer"], p["alpha"])):
            mark = (
                " **(chosen)**"
                if (p["layer"], p["alpha"])
                == (g["chosen"]["layer"], g["chosen"]["alpha"])
                else ""
            )
            lines.append(
                f"| {p['layer']}{mark} | {p['alpha']} | {pct(p['asr_incl_relay'])} | "
                f"{pct(p['clean_task_success'])} | {pct(p['authorized_followed'])} |"
            )
    (out / "results.md").write_text("\n".join(lines) + "\n")


def figures(results, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#8a8a85",
            "axes.labelcolor": "#3d3d3a",
            "xtick.color": "#5c5c57",
            "ytick.color": "#5c5c57",
            "axes.grid": True,
            "grid.color": "#e6e6e2",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "savefig.facecolor": "#fcfcfb",
        }
    )
    ink = "#3d3d3a"

    # 1. ASR incl. relay per arm, executed vs relayed, test and cue-flip panels.
    splits = list(results["splits"])
    fig, axes = plt.subplots(
        1, len(splits), figsize=(4.2 * len(splits), 3.6), sharey=True
    )
    axes = axes if len(splits) > 1 else [axes]
    for ax, split in zip(axes, splits):
        arms = list(results["splits"][split]["arms"])
        for i, arm in enumerate(arms):
            m = results["splits"][split]["arms"][arm]["metrics"]
            ex, rel = 100 * m["asr"]["rate"], 100 * m["asr_relayed"]["rate"]
            ax.bar(
                i, ex, width=0.62, color=COLORS[arm], edgecolor="#fcfcfb", linewidth=2
            )
            ax.bar(
                i,
                rel,
                bottom=ex,
                width=0.62,
                color=COLORS[arm],
                alpha=0.45,
                hatch="///",
                edgecolor="#fcfcfb",
                linewidth=2,
            )
            ax.text(
                i, ex + rel + 1, f"{ex + rel:.1f}%", ha="center", va="bottom", color=ink
            )
        ax.set_xticks(range(len(arms)), arms)
        ax.set_ylim(0, 40)
        ax.set_title(SPLIT_NAMES[split], color=ink)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Attack success incl. relay (%)")
    from matplotlib.patches import Patch

    fig.legend(
        handles=[
            Patch(facecolor="#8a8a85", label="executed"),
            Patch(
                facecolor="#8a8a85", alpha=0.45, hatch="///", label="relayed to user"
            ),
        ],
        loc="lower center",
        frameon=False,
        ncol=2,
        bbox_to_anchor=(0.5, 0),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "fig_asr.png", dpi=200)
    plt.close(fig)

    # 2. Safety-utility trade-off on test.
    test = results["splits"].get("test")
    if test:
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        for arm, data in test["arms"].items():
            m = data["metrics"]
            x, y = (
                100 * m["asr_incl_relay"]["rate"],
                100 * m["clean_task_success"]["rate"],
            )
            xci = [100 * c for c in m["asr_incl_relay"]["ci95"]]
            yci = [100 * c for c in m["clean_task_success"]["ci95"]]
            ax.errorbar(
                x,
                y,
                xerr=[[x - xci[0]], [xci[1] - x]],
                yerr=[[y - yci[0]], [yci[1] - y]],
                fmt="o",
                ms=9,
                color=COLORS[arm],
                ecolor=COLORS[arm],
                elinewidth=1.5,
                capsize=0,
                markeredgecolor="#fcfcfb",
                markeredgewidth=2,
                label=NAMES[arm],
            )
            ax.annotate(
                arm, (x, y), xytext=(8, 6), textcoords="offset points", color=ink
            )
        ax.set_xlabel("Attack success incl. relay (%)  ← safer")
        ax.set_ylabel("Clean task success (%)  ↑ more useful")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, loc="upper right")
        ax.set_title("Safety vs utility, frozen test (95% CI)", color=ink)
        fig.tight_layout()
        fig.savefig(out / "fig_tradeoff.png", dpi=200)
        plt.close(fig)

        # 3. ASR by attack goal and by conversation language on test.
        for field, fname, title in (
            ("goal", "fig_by_goal.png", "attack goal"),
            ("conversation_lang", "fig_by_language.png", "conversation language"),
        ):
            arms = list(test["arms"])
            cats = list(test["breakdowns"]["A0"][field])
            fig, ax = plt.subplots(figsize=(1.3 * len(cats) + 2.5, 3.6))
            width = 0.8 / len(arms)
            for j, arm in enumerate(arms):
                vals = [100 * test["breakdowns"][arm][field][c][0] for c in cats]
                xs = [i + (j - (len(arms) - 1) / 2) * width for i in range(len(cats))]
                ax.bar(
                    xs,
                    vals,
                    width=width,
                    color=COLORS[arm],
                    edgecolor="#fcfcfb",
                    linewidth=2,
                    label=NAMES[arm],
                )
                if arm in ("A0", "A4"):
                    for xx, v in zip(xs, vals):
                        ax.text(
                            xx,
                            v + 1,
                            f"{v:.0f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            color=ink,
                        )
            ax.set_xticks(
                range(len(cats)),
                [f"{c}\n(n={test['breakdowns']['A0'][field][c][1]})" for c in cats],
            )
            ax.set_ylabel("ASR incl. relay (%)")
            ax.set_title(f"Attack success by {title}, frozen test", color=ink)
            ax.grid(axis="x", visible=False)
            ax.legend(
                frameon=False,
                ncol=len(arms),
                loc="upper center",
                bbox_to_anchor=(0.5, -0.22),
            )
            fig.tight_layout()
            fig.savefig(out / fname, dpi=200)
            plt.close(fig)

    # 4. Hindi/English MMLU strict vs lenient.
    side = results["side"]
    if side and all("lenient" in side[a]["mmlu"]["hi"] for a in side):
        arms = list(side)
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
        for ax, lang in zip(axes, ("en", "hi")):
            for i, arm in enumerate(arms):
                s = side[arm]["mmlu"][lang]
                ax.bar(
                    i - 0.18,
                    100 * s["strict"],
                    width=0.34,
                    color=COLORS[arm],
                    edgecolor="#fcfcfb",
                    linewidth=2,
                )
                ax.bar(
                    i + 0.18,
                    100 * s["lenient"],
                    width=0.34,
                    color=COLORS[arm],
                    alpha=0.45,
                    hatch="///",
                    edgecolor="#fcfcfb",
                    linewidth=2,
                )
                ax.text(
                    i - 0.18,
                    100 * s["strict"] + 1,
                    f"{100 * s['strict']:.0f}",
                    ha="center",
                    fontsize=8,
                    color=ink,
                )
                ax.text(
                    i + 0.18,
                    100 * s["lenient"] + 1,
                    f"{100 * s['lenient']:.0f}",
                    ha="center",
                    fontsize=8,
                    color=ink,
                )
            ax.set_xticks(range(len(arms)), arms)
            ax.set_title(f"Global-MMLU-Lite {lang}", color=ink)
            ax.grid(axis="x", visible=False)
        axes[0].set_ylabel("Accuracy (%)")
        fig.legend(
            handles=[
                Patch(facecolor="#8a8a85", label="strict parser (official)"),
                Patch(
                    facecolor="#8a8a85",
                    alpha=0.45,
                    hatch="///",
                    label="lenient parser (post-hoc)",
                ),
            ],
            loc="upper right",
            frameon=False,
            ncol=2,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(out / "fig_mmlu.png", dpi=200)
        plt.close(fig)

    # 5. A2 grid on the dev tuning subset.
    grid = results["a2_grid"]
    if grid:
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        b = grid["baseline"]
        guard = 100 * (b["clean_task_success"] - 0.05)
        ax.axhline(guard, color="#8a8a85", lw=1, ls="--")
        xmin = min(
            [100 * p["asr_incl_relay"] for p in grid["points"]]
            + [100 * b["asr_incl_relay"]]
        )
        ax.text(
            xmin,
            guard + 0.4,
            "guardrail: clean success ≥ A0 − 5 pts",
            fontsize=8,
            color="#5c5c57",
        )
        ax.plot(
            100 * b["asr_incl_relay"],
            100 * b["clean_task_success"],
            "o",
            ms=9,
            color=COLORS["A0"],
            markeredgecolor="#fcfcfb",
            markeredgewidth=2,
        )
        ax.annotate(
            "A0",
            (100 * b["asr_incl_relay"], 100 * b["clean_task_success"]),
            xytext=(8, 4),
            textcoords="offset points",
            color=ink,
        )
        for p in grid["points"]:
            chosen = (p["layer"], p["alpha"]) == (
                grid["chosen"]["layer"],
                grid["chosen"]["alpha"],
            )
            x, y = 100 * p["asr_incl_relay"], 100 * p["clean_task_success"]
            ax.plot(
                x,
                y,
                "o",
                ms=10 if chosen else 8,
                color=COLORS["A2"],
                alpha=1 if chosen else 0.55,
                markeredgecolor="#fcfcfb",
                markeredgewidth=2,
            )
            ax.annotate(
                f"L{p['layer']} α{p['alpha']}" + (" (chosen)" if chosen else ""),
                (x, y),
                xytext=(8, -3),
                textcoords="offset points",
                fontsize=8,
                color=ink,
            )
        ax.set_xlabel("ASR incl. relay (%), 50 injected episodes")
        ax.set_ylabel("Clean task success (%), 50 clean twins")
        ax.set_title("A2 steering grid on the dev tuning subset", color=ink)
        fig.tight_layout()
        fig.savefig(out / "fig_a2_grid.png", dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--data", default="data/v1")
    parser.add_argument("--out", default="analysis")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = collect(args)
    (out / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n"
    )
    write_markdown(results, out)
    if not args.no_figures:
        figures(results, out)
    print(
        f"Wrote {out}/results.md, results.json"
        + ("" if args.no_figures else " and figures")
    )


if __name__ == "__main__":
    main()
