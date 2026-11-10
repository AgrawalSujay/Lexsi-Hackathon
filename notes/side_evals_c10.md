# C10: portable side-effect evaluations

Based on main `25c91ce`. Generation loads one HFGenerator, applies the existing
arm configuration (system prompt, hook or adapter), and uses greedy decoding.
Tiny Aya on Kaggle T4 uses **fp16**, not bf16. No judge, CK key or paid call is
needed for generation. This implementation was checked with fakes only.

## Kaggle and local generation

From an installed checkout with its model dependencies, run:

```sh
python -m turnguard eval-side --arm A0 --model dev --dtype fp16 --batch-size 2 --out /kaggle/working/runs/A0__tiny-aya__side
```

The equivalent Python entry point is
`turnguard.evals.side.generate_side(arm="A0", model="dev", dtype="fp16", out=Path(...), batch_size=2)`.
Use `--dry-run` to inspect configuration without loading datasets or a model.
The output directory must be new. A1 uses the configured defence paragraph;
A2/A4 can use `--arms-config` with local vector/adapter paths. Those artifacts
must match Tiny Aya; headline-model artifacts are not interchangeable.

All 450 XSTest prompts (250 safe, 200 unsafe) and 800 Global-MMLU-Lite items
(400 English, 400 Hindi) are generated. The pinned dataset revisions remain:

- Paul/XSTest: `f600c994b256f12867dfa5b3eb3d545a3e62f8b5`.
- CohereLabs/Global-MMLU-Lite: `36c2fd756f19ccf13a9a96c8e53ccecc02192b8b`.

Output caps are 256 tokens for XSTest and 16 for MMLU. The run contains input
snapshots, raw generations, arm/configuration hashes, resolved model revision
when available, environment and elapsed/generation timings, and a manifest
hashing immutable generation files. Raw outputs, including unsafe outputs,
are confined to the run directory and never printed. Generator/provider
logging is suppressed while processing outputs; failure reports contain only
stage and exception type. Generation is not resumable; local judging is.

## Modal remains available

```sh
modal run --detach scripts/modal_run.py --task eval-side --model dev --arm A0 --background
```

L40S is the default. Existing `--steer-run` and `--adapter-run` options stage
volume artifacts for A2 and A4. No local `.env` or judge keys are uploaded.
The same runner-independent entry point generates into the existing run volume.

## Local judging after downloading the complete run directory

```sh
python -m turnguard judge-side --run runs/A0__tiny-aya__side
```

`--env-root PATH` can select the checkout containing `.env`. The existing
CK_MODEL/CK_API_BASE/CK_API_KEY teacher settings are used, explicitly with
reasoning disabled. XSTest retains full_compliance/full_refusal/partial_refusal
labels and regex scores; MMLU uses offline letter extraction. Metrics include
safe full-compliance, unsafe full-or-partial refusal, per-class counts,
ungraded counts, English/Hindi accuracy and Wilson intervals. Unparseable
MMLU outputs count as wrong; generation failures are reported separately.
AuditKIT stores classification/assertion run records where the existing eval
bridges support them; aggregate Wilson rates use turnguard.metrics.rate.

Judge request caches and atomic checkpoints live inside the run directory,
so cached unsafe responses remain there. Successful labels resume without
new calls. Invalid labels are ungraded, quarantined and retryable. Changing
the judge settings or rubric invalidates checkpoint reuse. Immutable input,
configuration and generation hashes are checked before calling the judge.

## Budget before GPU use

Kaggle free T4 GPU compute costs $0 under its quota; Tiny Aya T4 runtime has
not been measured. For paid L40S, budget 30–60 minutes (~$1–2 GPU compute) per
arm conservatively, including loading. This is a range, not a measured side-eval
runtime: the earlier Aya-8B A0 dev run took 5,748 seconds; its trace volume gives
only a rough output-throughput proxy (~48 tokens/s at four characters/token).
The side-eval caps total 128,000 output tokens, with uncertain prompt/padding
and T4 throughput costs. No separate Tiny Aya measurement supports a tighter
estimate. L40S pricing used: $0.000542/s (~$1.95/hour), excluding storage/other
charges ([Modal pricing](https://modal.com/pricing)). No GPU or judge calls
were made while implementing C10.

## Validation

Full CPU suite: **819 passed, 19 subtests passed in 133.69 seconds**.
Ruff check, formatting and `git diff --check` passed. The eval-side dry run
resolved Tiny Aya/fp16 without dataset downloads or model initialization.
