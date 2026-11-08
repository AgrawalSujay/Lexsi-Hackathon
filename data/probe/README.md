# probe-v1 — vulnerability probe (pre-study)

28 hand-written episodes to answer one question before building the full
benchmark: **does Aya follow instructions planted in tool outputs, and can it
do the tasks at all?** Not part of the v1 train/dev/test splits; never reused
as test items.

| | Count |
|---|---|
| Injected | 12 (each with a clean twin: `*-clean`) |
| Clean twins | 12 |
| Authorized (following the instruction is correct) | 4 |

Coverage: all six families; carriers body, metadata (subject, search
snippet), tool error and quoted user text; positions first, later turn and
split across two emails; styles imperative, generic fake authority, claimed
named approval, task-framed, and **snag** (an error that offers a way out);
injection languages English, Hindi and Hinglish.

- Source of truth: `scripts/probe_v1.py` (regenerate with
  `python scripts/probe_v1.py`).
- `scripts/check_probe.py` confirms every episode is exposed (the natural tool
  call shows the planted text), solvable and detectable; `tests/test_probe.py`
  runs it in CI.

Run on a model:

```
python -m turnguard run-episodes --episodes 'data/probe/*.json' --generator hf \
  --model headline --batch-size 8 --arm A0 --split examples
```
