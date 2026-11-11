"""Post-hoc lenient MMLU rescoring (added after seeing results; applied identically to all arms).

Official parser first; if it returns None, also accept a letter alone on the first line,
optionally after "Answer:"/"उत्तर:". Official reports are not modified.
Usage: python scripts/mmlu_lenient.py [A0 A2 A4]
"""
import glob, json, re, sys, unicodedata
from turnguard.evals import global_mmlu as G

v1 = G.extract_answer
_FIRST = re.compile(r"^[\s(*`\[]*(?:(?:answer|उत्तर|जवाब)\s*[:\-]?\s*[\s(*`\[]*)?([ABCD]|ए|बी|सी|डी)[\s)*`\].:।]*(?:\n|$)", re.I)


def v2(response):
    a = v1(response)
    if a is not None:
        return a
    m = _FIRST.match(unicodedata.normalize("NFKC", response).strip())
    return G._letter(m.group(1)) if m else None


for arm in sys.argv[1:] or ("A0", "A2", "A4"):
    run = sorted(glob.glob(f"runs/{arm}__CohereLabs-aya-expanse-8b__side__*"))[-1]
    gold = {}
    for line in open(f"{run}/inputs.jsonl"):
        r = json.loads(line)
        if r.get("benchmark") == "mmlu" or str(r.get("id", "")).startswith("mmlu"):
            ans = next((r[k] for k in ("answer", "gold", "label", "target") if k in r), None)
            if ans is None and isinstance(r.get("item"), dict):
                ans = r["item"].get("answer")
            gold[r["id"]] = "ABCD"[ans] if isinstance(ans, int) else ans
    out = []
    for line in open(f"{run}/generations.jsonl"):
        r = json.loads(line)
        if r.get("benchmark") == "mmlu":
            resp = r.get("response") or ""
            out.append((r["id"].split("-")[1], gold[r["id"]], v1(resp), v2(resp)))
    for lang in ("en", "hi"):
        g = [x for x in out if x[0] == lang]
        n = len(g)
        s1 = sum(p == a for _, a, p, _ in g); u1 = sum(p is None for _, _, p, _ in g)
        s2 = sum(q == a for _, a, _, q in g); u2 = sum(q is None for _, _, _, q in g)
        print(f"{arm} {lang}: strict {s1/n:.3f} (unparse {u1})  lenient {s2/n:.3f} (unparse {u2})  n={n}")
