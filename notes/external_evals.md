# Task 05: external side-effect evaluations

XSTest and Global-MMLU (English, Hindi) side-effect evaluations.

## Data and scoring

XSTest uses [Paul/XSTest](https://huggingface.co/datasets/Paul/XSTest) at commit
`f600c994b256f12867dfa5b3eb3d545a3e62f8b5`, `xstest_prompts.csv`, CC-BY-4.0.
The loader validates 250 safe + 200 unsafe items before applying a smoke limit.
`--limit N` selects N total items, alternating seeded safe/unsafe samples;
full runs use all 450. Dataset revisions came from HF repository metadata;
no dataset evaluation or model download was run during development.

Labels follow the [XSTest paper](https://aclanthology.org/2024.naacl-long.301/):
`full_compliance`, `full_refusal`, `partial_refusal`. Safe over-refusal and
unsafe refusal both count full + partial refusals. Each subset records its
class counts, total, graded and ungraded counts, rate, and Wilson 95% interval
through the existing `turnguard.metrics.rate`.

The regex baseline is always recorded. It detects refusal phrases and explicit
compliance cues, distinguishing direct mixed answers from offers of safe
alternatives where possible. It is an English heuristic, not a semantic claim:
quoted refusal phrases, empathy, indirect answers, and subtle alternatives can
be misclassified. Use the optional teacher view and human review to assess it.
An empty completion or generation error is ungraded, not compliance. A bad judge
label remains ungraded in the teacher view, without silently using the regex
label instead. Regex and teacher views keep independent denominators.

## Model and arm parity

Both `run-episodes` and `eval-xstest` use `turnguard.arms.runtime.resolve_arm`
and `ArmConfig.apply`, configured in `configs/arms.yaml` or `--arms-config`.
The arm modules themselves are still stubs; unconfigured non-A0 arms fail rather
than silently receiving the baseline. Configure A1's exact defence as
`system_prompt`, A2/A3's `intervention_factory: module:callable` plus
`intervention_kwargs`, or A4/A5's `adapter` path/ID. A factory takes
`generator=<HFGenerator>` and returns its context-manager factory. Adapters load
with `is_trainable=False`, before hooks are constructed. CPU tests fake these
integrations; no real steering or adapter checkpoint has been run here.

XSTest is plain chat: one user message, optionally preceded by the arm's exact
system prompt. A0 has no system message. There are no tools, ACTION/FINAL
instructions, or agent retries in the prompt. The agent benchmark continues to
render its protocol prompt and appends the same configured defence text as its
system extra. Both use `HFGenerator`, its tokenizer chat template, left padding,
greedy decoding, 256 new tokens, and its intervention context. Existing legacy
agent `system_extra` stays supported; conflicting arm text fails.

## Optional teacher judge

`--judge teacher` uses task 03's CK_MODEL/CK_API_BASE/CK_API_KEY settings from
worktree `.env` or environment, CuratorKIT's backend, and its full-request cache.
A dedicated classifier system prompt replaces the filler-generation instruction
for judge requests only. It requests exactly one of the three labels; invalid
labels and errors are reported. Cache keys include the full question/response,
messages, judge model/base, all sampling options and prompt versions, with no
API key. Judge cache lives at `data/cache/xstest/`. Temperature is 0; retries
are bounded. No teacher calls were made during development. PLAN's human validation of
judge labels still needs to be done on actual model responses.

## AuditKIT fit and gaps (1.1.6)

AuditKIT's own model backend would bypass our generator/hooks, so generation
happens once in TurnGuard. `auditkit.evaluate(model="precomputed")` then scores
recorded `Sample.actual_output` via custom three-class scorers, keeping its
`RunResult`/prediction records and raw text in `auditkit.json`. Separate safe
and unsafe views use minimize and maximize directions respectively. SDK means
agree with our rates; its normal standard errors are not the requested Wilson
intervals, so those come from TurnGuard. Ungraded items remain in TurnGuard's
records and counts, and are omitted from AuditKIT's measured views.

The SDK has no per-evaluate cache directory argument. A scoped XDG_CACHE_HOME
override places its cache under the run, and restores the prior environment
immediately. No AuditKIT model/LLM judge runs, telemetry, uploads, or experiment
DB is invoked. Raw outputs are saved, never sent to console reports. Backend
exceptions are not echoed because they can contain unsafe text.

## Verification and user-run smoke

From the repo root, CPU only:

```sh
.venv/bin/python -m pytest tests/test_xstest.py tests/test_cli_runs.py tests/test_cli.py -q
.venv/bin/python -m ruff check .
```

On your model machine, after installing GPU and eval extras (plus data extra for
the optional teacher) and configuring any non-A0 arm:

```sh
python -m turnguard eval-xstest --model dev --arm A0 --limit 5 --judge none
python -m turnguard eval-xstest --model headline --arm A1 --arms-config configs/arms.yaml --judge teacher
```

Each run writes `config.yaml` (including pinned dataset, resolved arm, and
256-token greedy decoding), raw `episodes.jsonl`, `metrics.json`, `env.json`,
`auditkit.json`, and optional `judge_calls.jsonl` under
`runs/<arm>__<model-key>__xstest__<timestamp>/`. Console output is aggregate-only.
No model runs, paid calls, merges or pushes have been made.


## Global-MMLU (second cowork stop)

[CohereLabs/Global-MMLU-Lite](https://huggingface.co/datasets/CohereLabs/Global-MMLU-Lite/blob/36c2fd756f19ccf13a9a96c8e53ccecc02192b8b/README.md)
is pinned to `36c2fd756f19ccf13a9a96c8e53ccecc02192b8b` (Apache-2.0).
The pinned card lists 400 test items in English and 400 in Hindi. The loader
reads both complete test sets by default and validates those counts before any
smoke sampling. It does not include the dev/few-shot partition. `--limit N`
means N total questions, alternating seeded en/hi strata; all full-run items
are retained. Each language reports its actual total and graded n. The same
source sample ID is namespaced by language so English and Hindi cannot collide.

Questions/options come directly from `question`, `option_a` through `option_d`,
with `answer` as the gold letter. The user message ends with a localized
letter-only instruction from `configs/evals.yaml`; Hindi uses Devanagari prose
with A/B/C/D option labels. A0 has no system message; other arms use the same
shared runtime as XSTest and the agent benchmark. Generation uses HFGenerator,
greedy decoding and a default 16-token cap, recorded in the resolved run config.
No agent tools or protocol text is introduced.

The versioned extractor handles standalone letters, markdown/punctuation,
English/Hindi answer markers, Hindi pronunciations ए/बी/सी/डी and Unicode
full-width letters. A clear terminal final-answer marker takes precedence over
reasoning; otherwise conflicting letters are unparseable. It does not infer an
answer from option text or the first letter of an English word/article.
Unparseable responses (including valid empty generations) count as wrong and
are reported separately. Calls failing twice remain ungraded with explicit
error counts, as required by PLAN; they are not silently treated as an answer.
Accuracy and Wilson CIs use `turnguard.metrics.rate` per language.

AuditKIT `evaluate(model="precomputed")` records each language separately with
a custom accuracy scorer using that same extractor. Unparseable answers remain
measured zeros, raw model outputs stay in prediction records, and its means
agree with TurnGuard rates. A custom scorer preserves raw responses while
supporting Hindi; the stock string exact match or English extraction would not
handle these outputs correctly. TurnGuard supplies the required Wilson interval.
As for XSTest, AuditKIT does not generate or choose a model. The scoped cache
helper is shared between evals and serialized with a lock.

MMLU is scored against answer keys and never calls a teacher. The common
`--judge` option accepts `none`; choosing `teacher` fails before loading data or
models, explaining that teacher judging applies only to XSTest.

CPU verification from the repo root:

```sh
.venv/bin/python -m pytest tests/test_global_mmlu.py tests/test_xstest.py -q
```

Model smoke (not yet run):

```sh
python -m turnguard eval-mmlu --model dev --arm A0 --limit 5
# All 800 test questions, no few-shot/dev questions:
python -m turnguard eval-mmlu --model headline --arm A0
```

Install the optional eval extra for HF dataset loading. The output layout is
`runs/<arm>__<model-key>__mmlu__<timestamp>/` with `config.yaml`, raw
`episodes.jsonl`, `metrics.json`, `env.json` and `auditkit.json`. Global-MMLU
changes remain uncommitted pending review at this checkpoint.
