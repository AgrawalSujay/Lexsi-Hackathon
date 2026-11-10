# Tiny Aya on Kaggle T4 x2

This runner uses the same frozen v2.4 protocol, greedy HFGenerator, episode
loader and A4 trainer as the laptop/Modal commands. Each T4 holds one model;
two split shards run independently, with CUDA_VISIBLE_DEVICES=0 and 1.
Training uses **one T4**, fp16 with Accelerate's dynamic loss scaler, LoRA r16,
alpha32, lr5e-5, beta0.1, two epochs, gradient checkpointing and a 5% holdout.
Tiny's default maximum sequence length is 2048 (override `--max-length`).
No bf16, GitHub clone or paid judge call is needed on Kaggle.

## 1. Build the upload on your Mac

Use the `feat/kaggle` checkout, after its commit. Source is **git archive of
that commit**; uncommitted code is excluded. Built releases are read only.
The packer requires all four splits and verifies test/test_cueflip FROZEN and
exact CHECKSUMS.sha256 contents; it never freezes or rebuilds anything.

```sh
cd ~/Downloads/turnguard-c9
python scripts/kaggle_pack.py --data-root ~/Downloads/turnguard --out turnguard_kaggle.zip
```

`--data-root` selects the checkout containing data/v1. Pairs come from
**this source checkout's** data/pairs, when present, so newly built Tiny pairs
are not confused with main's older 8B pairs. `--pairs-root OTHER_CHECKOUT`
explicitly changes that source. The zip contains all split files (including
manifest/checksums/FROZEN), pairs, committed source, and KAGGLE_MANIFEST.json
with the commit and a SHA-256 for every file. The command prints the zip SHA.
Keep that SHA with your experiment notes. Output names must be new; use a
new filename or remove your previous local zip before repacking.

On Kaggle, create a **private Dataset**, upload turnguard_kaggle.zip, and add
it as an input to a new notebook. This works for a private GitHub repository;
no GitHub token is used. Never upload `.env`, teacher caches or credentials.

## 2. Notebook setup

In notebook Settings select **GPU T4 x2**, turn **Internet on**, and add/enable
a Kaggle secret named **HF_TOKEN** with access to the Cohere model. Accept any
model access terms using your Hugging Face account first. Run these cells.
The project requires Python 3.12+, which the bootstrap checks explicitly;
the isolated venv below avoids depending on the notebook's Python version.
These installation/download commands are for Kaggle, not your laptop.

```python
# Cell 1: secret in the child environment, never printed.
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

```python
# Cell 2: isolate the required Python version and dependencies.
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
subprocess.run(["uv", "python", "install", "3.12"], check=True)
subprocess.run(["uv", "venv", "--python", "3.12",
                "/kaggle/working/tg-venv"], check=True)
```

```python
# Cell 3: verify/extract only the bootstrap scripts, then define the runner.
import hashlib, json, zipfile
from pathlib import Path
bundles = list(Path("/kaggle/input").rglob("turnguard_kaggle.zip"))
assert len(bundles) == 1, "Keep one bundle input attached, or set BUNDLE explicitly"
BUNDLE = bundles[0]
BOOT = Path("/kaggle/working/tg-bootstrap")
BOOT.mkdir(exist_ok=True)
with zipfile.ZipFile(BUNDLE) as z:
    manifest = json.loads(z.read("KAGGLE_MANIFEST.json"))
    for name in ("scripts/kaggle_run.py", "scripts/kaggle_bundle.py"):
        content = z.read(name)
        assert hashlib.sha256(content).hexdigest() == manifest["files"][name]
        (BOOT / Path(name).name).write_bytes(content)
PY = "/kaggle/working/tg-venv/bin/python"
RUNNER = str(BOOT / "kaggle_run.py")
WORK = "/kaggle/working/tg"
def run(task, *options):
    subprocess.run([PY, RUNNER, task, "--bundle", str(BUNDLE),
                    "--work-dir", WORK, "--model", "dev", "--dtype", "fp16",
                    *map(str, options)], check=True)
```

Each invocation verifies the zip and existing checkout before reuse, installs
`.[gpu]` (plus `eval` for side evals), and validates frozen releases before
inference. It refuses changed source files in an existing --work-dir. If you
attach a new bundle, use a fresh session or a different WORK directory. The
HF cache is `/kaggle/working/hf-cache`; it is excluded from job download zips.
No API teacher/judge keys are read by this runner.

## 3. Tiny A0 train split, two shards

```python
run("split", "--split", "train", "--arm", "A0", "--parallel-shards", "2")
```

Both subprocesses are launched before waiting. Shard membership is the same
stable ID hash as Modal; injected/clean twins stay together. Each shard has
its own output directory and `/kaggle/working/out_<name>.zip`. Download both
zips from the notebook's Files/Output pane. Do not paste raw episode outputs
into the notebook or a chat. For more sessions, use `--shard 1/4`, etc.; do
not combine an explicit shard with `--parallel-shards 2`.

## 4. Build Tiny pairs locally, then repack

Unzip the two downloads into separate directories on your Mac; they each
contain `runs/<run-name>/`. In the same feature checkout as step 1, run
(substitute the actual run directory names shown by your downloads):

```sh
python -m turnguard build-pairs \
  --run ~/Downloads/tiny-shard-1/runs/<run-name-1> \
  --run ~/Downloads/tiny-shard-2/runs/<run-name-2> \
  --split train --specs ~/Downloads/turnguard/data/v1/train
python scripts/kaggle_pack.py --data-root ~/Downloads/turnguard --out build/kaggle-with-pairs/turnguard_kaggle.zip
```

`--specs` relocates the source split; hashes must match those recorded on
Kaggle. The builder accepts Tiny just like 8B, regrades old scoring versions
in memory, and never rewrites runs. Inspect data/pairs/train/summary.json and
audit_sample.md before training. Validation-only/dev pairs are refused.
Upload build/kaggle-with-pairs/turnguard_kaggle.zip as a new Dataset version. In a fresh session repeat setup. If keeping both
bundles attached, set BUNDLE explicitly to the new zip and WORK to
`/kaggle/working/tg-with-pairs`, and extract the bootstrap from that zip.

## 5. A4 train, checkpoints and resume

Optional first check:

```python
run("train-a4", "--smoke")  # 20 pairs, 10 optimizer steps; not a final adapter
```

Use a fresh session after the smoke (or a separate --output-root), since
smoke checkpoints intentionally cannot resume full training. Full run:

```python
run("train-a4")
```

Checkpoints are saved every 20 optimizer steps in
`/kaggle/working/checkpoints/A4/dev/`, including optimizer/scheduler/scaler and
trainer state. Resume chooses the newest complete checkpoint. A portable
identity checks model, pair hash and training settings, independently of
absolute paths. Changing dtype/context/pairs requires a new checkpoint
folder/session. `tokenization.json` and final metrics report exactly how many
train/held-out prompts needed left truncation, retaining completion tokens.
Reward accuracy remains separate for ACTION-chosen and FINAL-chosen pairs.

On normal completion **or a handled failure**, the output zip includes logs,
adapter, checkpoints and OUTPUT_CHECKSUMS.json. A hard kernel/session kill
cannot run the final zipper: save the notebook's output/version with the
live `checkpoints/` folder before expiry. Do not rely on unsaved working
files surviving a new session. Attach the saved output as another Dataset;
then restore from a prior output zip or extracted checkpoint folder:

```python
run("train-a4", "--resume-from", "/kaggle/input/tiny-checkpoints/out_previous.zip")
# Or: --resume-from /kaggle/input/tiny-checkpoints/checkpoints/A4/dev
```

Restoring never overwrites a populated live checkpoint directory. If the
current session already has checkpoints, rerun without --resume-from. Zip
restores verify output checksums. Download logs/tokenization metrics and
check a short smoke before committing a long session. If T4 memory is tight,
try `--max-length 1024` in a fresh checkpoint directory/session; report the
changed context and its truncation counts with results.

## 6. A0/A4 dev, test and cue-flip

After full training, find the new training job folder (not the smoke):

```python
jobs = sorted(Path("/kaggle/working").glob("A4__dev__train-a4__*"))
ADAPTER = jobs[-1] / "adapters/A4/dev"
assert (ADAPTER / "adapter_config.json").exists()
run("split", "--split", "dev", "--arm", "A0", "--parallel-shards", "2")
run("split", "--split", "dev", "--arm", "A4", "--adapter-dir", ADAPTER,
    "--parallel-shards", "2")
```

An adapter downloaded/re-attached in a later session works too: set ADAPTER
to its extracted `adapters/A4/dev` directory under `/kaggle/input/...`.
The runner rejects an 8B adapter for Tiny. Dev is for checks; do not tune on
test. Once satisfied, run the frozen evaluations:

```python
for split in ("test", "test_cueflip"):
    run("split", "--split", split, "--arm", "A0", "--parallel-shards", "2")
    run("split", "--split", split, "--arm", "A4", "--adapter-dir", ADAPTER,
        "--parallel-shards", "2")
```

Download all output zips. Raw histories are saved, but the runner prints only
archive paths, commit and judge-call count; child output stays in process.log.
The agent run configs retain split manifest/checksum fingerprints, fp16 and
shard IDs. Rescore/compare/build-pairs require the matching original specs.

## 7. Side evals and local judge

```python
run("eval-side", "--arm", "A0")
run("eval-side", "--arm", "A4", "--adapter-dir", ADAPTER)
```

Each runs all 450 XSTest plus 400 en/400 hi MMLU items, without an API judge.
Download/unzip, then on the Mac use the inner `run/` directory:

```sh
python -m turnguard judge-side --run ~/Downloads/tiny-side-A0/run --env-root ~/Downloads/turnguard
python -m turnguard judge-side --run ~/Downloads/tiny-side-A4/run --env-root ~/Downloads/turnguard
```

Those **local** calls use CK_* from .env, reasoning disabled, and resume from
the run-local judge cache. Unsafe raw generations stay in the run directories;
share only metrics. Side-eval details and pinned revisions: side_evals_c10.md.

## 8. Rough T4 budget (unmeasured; benchmark a small shard first)

No T4 GPU job, network download or paid call was made to implement this.
These are planning ranges from **assumed** 200–600 generated tokens/agent
episode and 10–30 output tokens/second/T4, plus variable prefill overhead.
They are not throughput measurements; context length and batching matter.
The existing 8B L40S probe is not a reliable Tiny-on-T4 extrapolation.

| Work | Rough GPU wall time |
|---|---|
| Initial model/dependency download | 10–30 min, network-dependent |
| Agent episodes, two T4 shards | ~0.1–1 hour per 100 total episodes |
| Train split A0 (2,083 at authoring time) | ~2–21 hours; split further if needed |
| Dev per arm (749) | ~0.8–7.5 hours |
| Test + cue-flip per arm (1,043 + 270) | ~1.3–13 hours |
| Side eval per arm | ~0.5–4 hours, one T4 |
| A4 DPO, one T4 | assume 10–60 sec/pair/epoch; 500 pairs x 2 epochs ~3–17 hours |

Kaggle free GPU compute is $0 within your quota. Measure actual elapsed time
and token/truncation counts, then revise these ranges before a long job.
A4 smoke times and checkpoint cadence are the useful training calibration.

## Implementation checks

Final CPU suite: **838 passed, 19 subtests passed in 132.38 seconds**. Ruff
lint/format and whitespace checks passed. Fake-only coverage includes archive
integrity/path safety, read-only snapshot changes, frozen refusal, GPU-pinned
parallel dispatch, output/checkpoint archives and restore, bf16 refusal, fp16
DPO flags/truncation, and relocated Tiny shard pair building. The four real
built splits passed read-only checksum checks. No install, model download,
GPU execution or paid API call was performed during implementation.
