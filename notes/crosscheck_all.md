# All-split translation and filler audit (C7)

## Commands and read-only inspection

```bash
python -m turnguard crosscheck --split dev --dry-run --root ~/Downloads/turnguard
python -m turnguard crosscheck --split train --dry-run --root ~/Downloads/turnguard
python -m turnguard crosscheck --split test --dry-run --root ~/Downloads/turnguard
python -m turnguard crosscheck --split dev --filler --dry-run --root ~/Downloads/turnguard
python -m turnguard crosscheck --split train --filler --dry-run --root ~/Downloads/turnguard
python -m turnguard crosscheck --split test --filler --dry-run --root ~/Downloads/turnguard
```

`--root` is a **read-only CLI option**: it requires `--dry-run`, otherwise the
command refuses before collection, client construction or output writes. Dry
runs read that checkout's builds, teacher cache and CK settings, and inspect GLM
cache hits without writing anything there. No API client is created. This lets
a development worktree estimate completed main builds while an older test
census continues writing its own results. Incomplete live GLM cache files count
as misses in estimates; real judging refuses incomplete entries instead of
replacing them. Cache-hit estimates can change as the other process progresses.

For actual audits, run from the checkout holding the builds/settings, omit
`--root` and `--dry-run`. There were **no real API calls** during C7 development.
Main's builds and existing data/crosscheck/test outputs were only read.

`--n` defaults to 150. If n >= population size, every unique item is reviewed
and the report says `selection: census`, with `census_n` and a `census` item list
in dry runs. It does not describe that result as a sample. A smaller n uses the
same seeded language x kind stratification as C6. An empty translation
population is a zero-item census. To inspect or audit a whole split, use a
large n, for example:

```bash
python -m turnguard crosscheck --split test --filler --n 1000000 --dry-run --root ~/Downloads/turnguard
python -m turnguard crosscheck --split test --filler --n 1000000
```

The second command is for the owner to run after reviewing the estimate. It
calls the API. Test filler is included in exactly the same machinery as dev
and train filler.

## Collection and cache reuse

Translation collection reads only the requested primary split (dev/train/test).
Test additionally attaches uses in test_cueflip; it never samples cue-flip-only
translations. The exact source/translation/language/kind request remains the
C6 request, with no split or episode IDs, preserving old request hashes.

The writable GLM cache is now **data/crosscheck/cache/**, shared across splits
and modes. Existing per-split caches, particularly data/crosscheck/test/cache/,
are read-only fallbacks. Nothing moves or rewrites their entries. The same
translation in multiple splits therefore uses one cached response. Filler uses
a separate prompt/version, so it cannot accidentally reuse a translation
verdict. Existing request identity checks, GLM model verification, 4000-token
budget + one empty-content 8000-token retry, <=4 workers and <30 starts/minute
remain in effect. Credentials never enter the cache or reports.

Filler collection uses surviving episode meta.filler_block references and
filler_blocks.jsonl to find the original generated fields. It joins the block's
cache keys to the exact generation payload in data/cache: filled template
`about`, `kind`, `language`, and `include_exactly_once` (including keep strings
added by the teacher). It does not regenerate briefs from potentially changed
templates, and it fails loudly if original provenance is missing. Deduped
variants already point at their surviving blocks; rejected/unused blocks do
not enter the population. The audit text is the original filler field, before
any planted injection was appended. All filler languages, **including English**,
are eligible.

Items deduplicate on exact text + brief. Identical text under a different brief
needs a separate judgment; identical text/brief in different blocks or splits
is reviewed once. Each item records every block/field and the corresponding
used episode IDs, plus matching cue-flip IDs for test. The GLM instruction asks
whether the text matches the brief, sounds natural in the specified language,
and contains an extra instruction not requested by that brief. Authorized
instructions that the brief explicitly requires are not accidental injections.

Filler schema is strictly boolean `matches_brief`, `natural`, and
`contains_instruction`, plus string `issue`; malformed results have null fields
and an error, never an audit pass. Existing handling of a surrounding JSON code
fence is retained. The derived pass requires matches_brief AND natural AND NOT
contains_instruction. All rates use Wilson CIs; errors and missing usage stay
explicit.

Each split has its own data/crosscheck/<split>/ output. Filler writes
filler_items.jsonl, filler_verdicts.jsonl, filler_disagreements.jsonl, and the
`filler` section in summary.json. Translation files and prior translation
summary values are preserved. Translation reruns preserve the filler section;
translation rates remain at the top level for C6 consumers and also appear in
an explicit `translations` section. Summaries separately account for newly
billed versus attributed cached tokens, including thinking tokens as output.

## Actual read-only dry-run estimates

The C7 worktree has no built dev/train data; the approved read-only root override
read main's completed builds on 2026-11-09. Episode counts: dev 587 (118 injected,
118 clean, 351 authorized); train 2380 (659/659/1062); test 1074 (421/421/232),
plus test_cueflip 273 (108/108/57). These are snapshots, not build targets.

Default **n=150**, each a sample. Cost assumes the full 4000-token first-attempt
output budget plus input estimated at chars/4; prices are the requested
$0.75/$3.00 per million input/output tokens. These are planning budgets, not
actual spend. A retry ceiling assumes every new first attempt needs the one
8000-token retry; input remains an approximation.

| Split / mode | Population | New calls / cache hits | Input estimate | Output budget | USD budget | With empty retries |
|---|---:|---:|---:|---:|---:|---:|
| dev translations | 186 | 149 / 1 | 35,024 | 596,000 | $1.814268 | $5.416536 |
| dev filler | 216 | 150 / 0 | 43,228 | 600,000 | $1.832421 | $5.464842 |
| train translations | 701 | 148 / 2 | 36,104 | 592,000 | $1.803078 | $5.382156 |
| train filler | 509 | 150 / 0 | 42,848 | 600,000 | $1.832136 | $5.464272 |
| test translations | 560 | 0 / 150 | 0 | 0 | $0 | $0 |
| test filler | 508 | 150 / 0 | 44,381 | 600,000 | $1.833286 | $5.466572 |

Full **census**, n=1000000, in a subsequent snapshot:

| Split / mode | Census items | New calls / cache hits | Input estimate | Output budget | USD budget | With empty retries |
|---|---:|---:|---:|---:|---:|---:|
| dev translations | 186 | 185 / 1 | 43,659 | 740,000 | $2.252744 | $6.725488 |
| dev filler | 216 | 216 / 0 | 62,262 | 864,000 | $2.638697 | $7.869393 |
| train translations | 701 | 696 / 5 | 170,037 | 2,784,000 | $8.479528 | $25.311056 |
| train filler | 509 | 509 / 0 | 146,076 | 2,036,000 | $6.217557 | $18.543114 |
| test translations | 560 | 121 / 439 | 29,839 | 484,000 | $1.474379 | $4.400759 |
| test filler | 508 | 508 / 0 | 149,996 | 2,032,000 | $6.208497 | $18.512994 |

Main's test census was running throughout inspection. Its remaining-call cost
will decrease as more entries arrive; no attempt was made to alter, pause or
restart it. The tables are independent per-command estimates; running one
split can create shared cache hits that reduce later splits' cost, so summing
these budgets is conservative rather than an exact combined bill.

## Audited removal on every split

`drop-episodes --split dev|train|test|test_cueflip --ids-file confirmed.jsonl`
now supports every split. It accepts confirmed translation or filler
disagreement objects directly; plain IDs resolve an unambiguous reason/key
from either disagreement file for that split. Cue-flip uses test's matching ID
list. As in C6, the operator must confirm bad items; judgments never delete
anything automatically. Either member of a base/clean pair removes both.

FROZEN refusal, append-only dropped.jsonl and summary.dropped, count/coverage
refresh, manifest/provenance/card updates, checksum regeneration and cue audit
remain. Postprocessing is now named `confirmed_crosscheck_drop` so a filler
removal is not mislabeled as a translation failure. Earlier generation/gating
history remains historical. Staged dev/train releases validate their own
schemas/completeness without imposing test-only freeze eligibility rules.
