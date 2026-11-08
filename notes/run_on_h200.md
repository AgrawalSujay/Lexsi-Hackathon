# Running TurnGuard on the H200 (Aya Expanse 8B)

For the teammate with the H200. This runs **Aya Expanse 8B** on the 4
reference episodes and the 28-episode vulnerability probe, and tells us whether
the 8B model can act as an agent well enough to be our headline model.

**Time:** ~10 min setup, ~5 min download (if not cached), ~20–30 min of runs.
**Send back:** `runs_h200.zip` (step 6) and anything odd you saw.

This is separate from the smoke test (`run_smoke.sh`). The two can run in
either order; if the smoke test already ran, step 2 reuses its environment and
its downloaded model.

---

## 0. Check the machine

```bash
nvidia-smi            # an H200 with ~40 GB free is plenty for the 8B model
python3.12 --version  # Python 3.12 or newer is required
df -h ~               # ~40 GB free for the model and packages
```

**Expected:** the GPU is listed, Python is 3.12+, enough disk.

## 1. Get the code

```bash
git clone <repo-url> turnguard
cd turnguard
git log -1 --oneline        # note this commit in your message back
```

**Expected:** the latest commit message mentions the H200 runbook or later.

## 2. Python environment

If you already ran the smoke test in `~/h200_smoke`, reuse its environment and
model cache:

```bash
PY=~/h200_smoke/.venv-smoke/bin/python
export HF_HOME=~/h200_smoke/hf_home
$PY -m pip install -e '.[gpu]'
```

Otherwise create a fresh one:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e '.[gpu]'
PY=.venv/bin/python
```

Then, in the same shell:

```bash
export HF_TOKEN=hf_...      # Read token, licence for CohereLabs/aya-expanse-8b accepted
# Only if the GPU node has no internet AND the model is already downloaded:
# export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
```

**Expected:** `pip install` ends without errors.

## 3. Quick sanity check (CPU, seconds)

```bash
$PY -m pytest -q
```

**Expected:** all tests pass (`N passed`, no `failed`). If anything fails,
stop and send the output.

## 4. Run the 4 reference episodes

```bash
$PY -m turnguard run-episodes --episodes 'data/examples/*.json' \
  --generator hf --model headline --batch-size 8 --arm A0 --split examples
```

**Expected:** after `Loading weights`, a table with 4 rows (episode, type, tag,
steps, malformed), then the metrics, ending with `Ungraded: 0; generation
errors: 0`. A few minutes.

## 5. Run the 28-episode probe

```bash
$PY -m turnguard run-episodes --episodes 'data/probe/*.json' \
  --generator hf --model headline --batch-size 16 --arm A0 --split examples
```

**Expected:** a table with 28 rows, then the metrics. About 10–20 minutes.
Out-of-memory is handled automatically (the batch is halved and retried).

### What the results mean

There is no "correct" answer: this run *measures* the model. For reference,
Tiny Aya (3.35B) on the same probe:

| Metric | Tiny Aya | What we hope to see on 8B |
|---|---|---|
| `malformed_rate` | 0/91 | stays near 0 |
| `clean_task_success` | 6/12 | **clearly higher**: 8B can do the tasks |
| `exposure → injection_exposure_rate` | 9/12 | higher: it reads before acting |
| `exposure → asr_given_exposed` | 4/9 | anything; this is the finding |
| `authorized_follow_rate` | 1/4 | higher |

**Tell Sujay right away if you see:** `Ungraded` above 0 or `generation
errors` above 0; a `malformed_rate` above 0.2; an out-of-memory crash that
stops the run; or almost every episode ending as `noop` with 1–2 steps.

## 6. Send the results

```bash
zip -qr runs_h200.zip runs
```

Send `runs_h200.zip` to Sujay with the commit from step 1. Nothing keeps
running on the GPU afterwards.
