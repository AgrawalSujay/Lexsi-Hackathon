"""Post-hoc lenient MMLU rescoring, applied identically to all arms.

Added after seeing results.

Official parser first; if it returns None, also accept a letter alone on the first line,
optionally after "Answer:"/"उत्तर:". Official reports are not modified.
Usage: python scripts/mmlu_lenient.py [A0 A2 A4]
"""

import glob
import json
import re
import sys
import unicodedata

from turnguard.evals import global_mmlu as G

v1 = G.extract_answer
_FIRST = re.compile(
    r"^[\s(*`\[]*(?:(?:answer|उत्तर|जवाब)\s*[:\-]?\s*[\s(*`\[]*)?"
    r"([ABCD]|ए|बी|सी|डी)[\s)*`\].:।]*(?:\n|$)",
    re.I,
)


def v2(response):
    a = v1(response)
    if a is not None:
        return a
    m = _FIRST.match(unicodedata.normalize("NFKC", response).strip())
    return G._letter(m.group(1)) if m else None


def side_run(arm, root="runs"):
    runs = sorted(glob.glob(f"{root}/{arm}__CohereLabs-aya-expanse-8b__side__*"))
    return runs[-1] if runs else None


def lenient_scores(run):
    """Return {lang: {strict, lenient, unparse_strict, unparse_lenient, n}}."""
    gold = {}
    for line in open(f"{run}/inputs.jsonl"):
        r = json.loads(line)
        if r.get("benchmark") == "mmlu" or str(r.get("id", "")).startswith("mmlu"):
            ans = next(
                (r[k] for k in ("answer", "gold", "label", "target") if k in r), None
            )
            if ans is None and isinstance(r.get("item"), dict):
                ans = r["item"].get("answer")
            gold[r["id"]] = "ABCD"[ans] if isinstance(ans, int) else ans
    out = []
    for line in open(f"{run}/generations.jsonl"):
        r = json.loads(line)
        if r.get("benchmark") == "mmlu":
            resp = r.get("response") or ""
            out.append((r["id"].split("-")[1], gold[r["id"]], v1(resp), v2(resp)))
    result = {}
    for lang in ("en", "hi"):
        g = [x for x in out if x[0] == lang]
        n = len(g)
        result[lang] = {
            "n": n,
            "strict": sum(p == a for _, a, p, _ in g) / n,
            "lenient": sum(q == a for _, a, _, q in g) / n,
            "unparse_strict": sum(p is None for _, _, p, _ in g),
            "unparse_lenient": sum(q is None for _, _, _, q in g),
        }
    return result


if __name__ == "__main__":
    for arm in sys.argv[1:] or ("A0", "A2", "A4"):
        run = side_run(arm)
        if run is None:
            print(f"{arm}: no side run")
            continue
        for lang, s in lenient_scores(run).items():
            strict = f"strict {s['strict']:.3f} (unparse {s['unparse_strict']})"
            lenient = f"lenient {s['lenient']:.3f} (unparse {s['unparse_lenient']})"
            print(f"{arm} {lang}: {strict}  {lenient}  n={s['n']}")
