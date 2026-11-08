# Building the TurnGuard dataset (task 03)

Run from the repo root on your Mac. Needs the teacher endpoint in `.env`:

```
CK_MODEL=openai/<served-model-name>
CK_API_BASE=https://<endpoint>/v1
CK_API_KEY=<key>
```

`.env` is gitignored. Never paste the key into chat.

## One-time setup

```bash
python3 -m pip install -e '.[data,dev]'
```

## 1. Estimate (no calls, free)

```bash
python -m turnguard build-data --split dev --teacher litellm --dry-run
```

Prints the number of teacher calls and a rough token count.

## 2. Smoke (5 episodes, a few cents)

```bash
python -m turnguard build-data --split dev --teacher litellm --limit 5 --backtranslate
```

**Expected:** `status: complete`, 5 injected + 5 clean + the authorized
episodes, and few or no rejections. Open `data/v1/dev/episodes.jsonl` and read
2–3 Hindi or Hinglish episodes: are they natural, and are the codes intact?
Check `data/v1/dev/summary.json` and `rejected.jsonl`.

## 3. Full builds

```bash
python -m turnguard build-data --split dev  --teacher litellm --backtranslate
python -m turnguard build-data --split test --teacher litellm --backtranslate
```

Targets: dev 120 + 120 + ~70 authorized; test 300 + 300 + ~144.

**Review before freezing:**
- `summary.json`: counts, factor balance and `dedup` numbers.
- `backtranslation_review.jsonl`: Sujay spot-checks the Hindi/Hinglish.

The build stops by itself if more than 5% of filler blocks repeat across
scenarios; inspect `dedup_report.json` if that happens.

## 4. Freeze test (once, after review)

```bash
python -m turnguard freeze --split test
```

Writes `data/v1/test/CHECKSUMS.sha256` and `FROZEN`. After this:
- `build-data --split test` refuses to run;
- the test set loads only with `purpose="eval"`.

Commit `data/v1/dev` and `data/v1/test`. Keep `data/cache/` locally (it is
gitignored): it makes rebuilds free and identical.

## Train split (task 04)

```bash
python -m turnguard build-data --split train --teacher litellm
```

750 attack episodes + twins + authorized. Only needed for the training fixes
(A4/A5) and the steering vectors.
