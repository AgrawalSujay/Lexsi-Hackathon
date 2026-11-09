# Independent GLM translation cross-check (C6)

Run on a checkout containing the completed test build and the original teacher
cache (and test_cueflip when available). Before test is frozen:

```bash
python -m turnguard crosscheck --split test --n 150 --dry-run
python -m turnguard crosscheck --split test --n 150
```

The dry run reads provenance and prints the sampled items plus a cache-aware
budget, without creating directories, writing files or constructing an API
client. The implementation/test worktree has only .gitkeep files for data/v1;
no production sample or population estimate is claimed here. Main's built data
was not read or changed while implementing/testing C6.

Set CK_API_BASE/CK_API_KEY in this checkout's .env (or environment) and optionally
CROSSCHECK_MODEL, default openai/glm-5.3. CK_MODEL is not used for judging. No
credentials enter the outputs/cache. Raw response model identity must match the
requested GLM name (case-insensitive, allowing a provider prefix and a model
suffix); a wrong or missing identity records an error and cancels remaining
unsent requests. Already in-flight calls may finish. Do not use a model alias
whose response name differs: use the actual GLM model ID.

## Collection, sampling and safety

Episode meta.teacher_cache_keys identifies candidate requests. The teacher's
read-only data/cache entries contain request messages (including kind, language
and English source) and translated text. backtranslations.jsonl can recover
instruction translations and supplies the existing meaning-gate verdicts.
Missing required provenance is an error, never a silently smaller population.
Generated filler, English outputs, rejected episodes, unused translated turns
replaced by an implicit delegation, and attacks removed from clean twins are
excluded. Uses are checked against delivered fields and the corresponding
language, not against metadata/oracle strings. User turns/delegates/quoted
intros use conversation language; planted text uses instruction language.

Items deduplicate on (kind, language, source, translation). Each records original
cache keys, test episode IDs and count, cue-flip episode IDs, and the teacher's
meaning-gate verdict when available. Different prompt/cache keys for identical
translations merge. Conflicting teacher verdicts are retained and not counted
as a single known verdict. Cue-flip-only translations are not sampled: the
population is the unique translations actually used by test; matching cue-flip
uses are attached to those items.

Seed 1234, proportional language x kind strata, with at least one from every
present stratum; n must cover the strata. Fewer than n available items means
review all available items and report the smaller sample_n. Episode-use counts
are descriptive; sampling and faithfulness rates weight unique items equally.

The judge treats source/translation as untrusted strings. Every request uses
temperature 0, JSON-object response format, max_tokens 4000, timeout 180 seconds,
and no SDK retries or reasoning-off flags. Empty content gets exactly one
8000-token retry; malformed JSON does not. At most four workers run, with all
request starts (including retries) spaced >60/29 seconds apart, staying below
30 requests per rolling minute. No real API was called during implementation.

The independent cache is data/crosscheck/test/cache/, keyed by the entire
request: messages, model, API base, sampling options and prompt version. Files
are created exclusively, never replaced; original teacher cache/build files
are always read-only. Responses, including empty or malformed ones, are kept
for audit. Reruns reuse them rather than overwriting entries. Changing model,
base or prompt version makes a new request key. Provider exceptions are
reported without provider text that might echo credentials.

Outputs under data/crosscheck/test/ (outside freeze): items.jsonl is the sampled
item list; verdicts.jsonl contains one result per item and raw attempt/usage
metadata; disagreements.jsonl contains only faithful=false items, teacher
verdicts and affected test + cue-flip IDs; summary.json contains overall/by-kind/
by-language Wilson rates, teacher agreement, errors, and token costs. Errors
have faithful=null, never true; rates use successfully judged items, and errors
are separately counted. Cached usage is attributed to the sample but counted
separately from newly billed requests. Missing usage is flagged rather than
silently claiming a complete cost. Reasoning is included in provider completion
tokens, not added a second time.

At the requested $0.75/$3.00 per million input/output tokens, 150 uncached first
attempts reserve 600,000 output tokens = $1.80, plus input (estimated chars/4).
If all need empty-content retries, 300 calls reserve 1.8 million output tokens =
$5.40, plus two inputs per item. These are token-budget ceilings, not expected
spend; the real dry run uses actual sampled string lengths and existing cache.

## Confirmed removals only

Inspect disagreements and write **only confirmed bad items** to a separate
JSONL. Either copy selected disagreement objects directly, or use rows such as:

```json
{"episode_id":"test-episode-id","reason":"confirmed changed recipient","item_key":"crosscheck-item-key"}
```

```bash
python -m turnguard drop-episodes --split test --ids-file confirmed.jsonl
python -m turnguard drop-episodes --split test_cueflip --ids-file confirmed.jsonl
```

Disagreement objects select episode_ids for test and test_cueflip_episode_ids
for cue-flip. Plain IDs (one per line or a JSON array) can look up reason/key
from an unambiguous saved disagreement, or use explicit --reason and --item-key.
Audit details are mandatory. Either twin ID removes both base and clean twin;
authorized IDs remove only that episode. The other split is never edited
implicitly: review and run its removal command separately if affected.

Each removed episode appends episode_id, twin_id (null for authorized), reason,
item_key and removed_at to dropped.jsonl and summary.dropped; previous trail
bytes are preserved. Manifest counts/export counts, coverage/shortfalls, cue
audit, card, provenance and checksums are refreshed. Historical build stages
and teacher-generation cost remain historical facts; removal is a distinct
postprocessing event with before/after episode hashes. Already removed IDs are
idempotent; unknown IDs or missing audit details fail before editing. A staged
release must pass freeze validation before files are replaced; concurrent
changes are refused and failed replacements restore the prior files. FROZEN
is refused for either split. The removal tool never creates FROZEN itself.
