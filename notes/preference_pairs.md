# M3 on-policy preference-pair builder (C8 Stop 2)

## Commands and safety guards

```bash
python -m turnguard build-pairs --run runs/TRAIN_SHARD_1 runs/TRAIN_SHARD_2 --split train --specs data/v1/train
python -m turnguard build-pairs --run ../turnguard/runs/A0__CohereLabs-aya-expanse-8b__dev__20261109-0802 --validate-only --specs ../turnguard/data/v1/dev
```

Run these from the C8 checkout. `--run` accepts multiple directories, including
repeated --run options. `--specs` explicitly relocates the original split after
copying a Modal run; it never bypasses hashes. Identical duplicate traces merge;
conflicting raw histories refuse. All shards must have the same model, arm and
dataset identity. Stale grading results are upgraded in memory, without editing
the input run. Source run hashes are checked again before exporting.

Training mode only accepts train runs/specs and train-tagged authored scenarios;
held-out code/tool_error examples cannot train. Dev is allowed **only** with
--validate-only (which infers --split dev if omitted); test is never allowed.
The exact spec and complete built-dataset hashes use the existing verified-run
loader. Recorded manifest template hashes must match the templates used for
routes. Original run/data files stay read-only.

Train output is data/pairs/train/. Validation output is always
**data/pairs/_validation_dev/**, marked **validation — never for training** in
summary.json, audit_sample.md, a VALIDATION_ONLY marker, and every pair row.
The mandatory load_training_pairs preflight refuses that directory, marker,
non-train summaries, and dev/validation rows even after copying the JSONL into
a renamed folder. `train --arm A4 --model headline --pairs ...` already runs
that preflight, before its still-unimplemented trainer path. Stop 3 must retain
this check before constructing any trainer/model.

## How decisions and labels are built

Each prompt is the exact messages_model prefix before an actual assistant turn.
It preserves the real system prompt, native tool-role messages if used, format
reminders and prior protocol content. It never substitutes an oracle-generated
history, or the raw model's invented tool results/trailing ACTION/FINAL lines.
Raw generation text is retained for model-produced rejections. Calls are
replayed from the recorded initial state, verifying their success/error and
model-visible output, final state, answer sequence and step count. Scripted
error counters/fired rules are part of the recovered decision state.

The ordinary injected point is the first actual assistant turn after observed
exposure of any injection part, using the same observed-exposure semantics as
C8 grading. The positive is the first unfinished route action in that user
turn. Write completion uses matching task-oracle constraints rather than exact
English body wording. Lookup completion also accepts an equivalent recorded
query whose tool result contains all requested result chunks (even with extra
results or different rankings); a correct answer is not a bad anchor merely
because search keywords differ. Every positive
must parse, bind, execute successfully, match the route/oracle step and avoid a
new injected action. Once a turn's route is complete, an oracle-derived FINAL
may supply its answer facts (or a route-complete turn boundary). These are
step-progress checks, not a claim that one action solves every user turn.

If that recorded decision actually executes the target, its own raw output is
the rejection (`injected-executed`). Otherwise an executable obey call or canary
FINAL is derived from the target (`injected-synthesized`). The latter also
covers otherwise-safe episodes and episodes whose harmful action occurs only
later; a legitimate first action is never falsely labeled as the model's attack.

**Relay exception:** for an episode tagged relayed, use the actual harmful
FINAL's own model-prefix context (which can be later than first exposure).
Only remove complete sentences containing remaining distinctive target terms,
using the clean-state filtering from Stop 1. Email/URL dots are not sentence
boundaries; Hindi danda is supported. No new answer prose is written. The
remainder must be nonempty, pass every answer oracle for that turn at this
step, contain no relay/execution target, and have no unfinished current-turn
route calls. No answer oracle, pending calls, or a lost required fact means
skip that decision: **never synthesize a replacement FINAL for a relay**.
Retained sentence bytes are unchanged apart from outer whitespace. Literal
term-guided sentence removal and loose authored oracles still need human review.

Clean anchors align the corresponding clean trace by the same reveal tool call
and user turn, not an assistant-step ordinal (retries may differ). Quoted
exposure aligns the first model decision of that user turn. An actual wrong
FINAL, malformed output or failed action supplies the rejection; successful
alternative tool actions are conservatively not labeled wrong. Correct clean
outputs are skipped. The positive follows the clean state's unfinished route.

Authorized pairs use the first model point after the approved instruction is
observed where its first still-unfulfilled target call can be grounded. Quoted
approvals can wait until a recorded lookup in the same user turn supplies the
content. A factual actual call is preferred; otherwise target constraints,
obey_args and same-tool route arguments fill the positive. Open body/text fields
require current-turn oracle facts actually visible in user/tool messages;
assistant assertions are not factual evidence. Missing facts mean skip, never
checker placeholder text (`x`) or facts read only from hidden state. Unbound
identity/date/amount arguments also cause a skip. Free subjects/titles may use
a neutral label. Loose target oracles still need human review. The rejection is
the model's skipped/refused FINAL/malformed output or an explicit synthetic
refusal; a valid task action is not mislabeled as a refusal. Call targets must
execute and pass their target oracle from the recovered state.

## Balance, provenance and audit

Default --scenario-cap 200 caps **total** pairs per scenario; an odd cap rounds
down to an even count. Within each scenario retain exactly one injected pair
per clean-anchor/authorized pair, so global balance is 1:1 too. Seed 1234 and
source x conversation/instruction-language strata make selection deterministic.
Scarce sources/language cells get coverage before repeated cells. Exact duplicate
(prompt, chosen, rejected) triples collapse before balance. Controls without
matching injected capacity, and injected pairs without control capacity, are
explicit balance removals rather than silently changing the ratio.

Outputs: pairs.jsonl, summary.json, skipped.jsonl, and a deterministic
**50-pair audit_sample.md** (or every pair when fewer than 50 exist). Audit
entries include full exact prompts, chosen/rejected strings and provenance.
Each pair records episode/spec hash, scenario, step/turn, language factors,
chosen/rejected source, oracle-step check, pre-step state hash, source run hashes
and template hashes. Summary counts every requested source separately:
injected-executed, injected-relayed, injected-synthesized, clean anchor, authorized.
It also separates candidate versus selected counts, skip/balance reasons,
conversation language, instruction language, joint language cells, per-source
language cells and scenario counts. Clean twins retain their original injection
language factor; it describes the matched base, not a new clean instruction.

No train run directories were present during implementation; the owner-requested
real dev validation is recorded separately in notes/pairs_stop2_validation.md.
No GPU, API, generation or training job is launched by this builder.
