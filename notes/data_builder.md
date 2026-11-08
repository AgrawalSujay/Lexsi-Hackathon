# Task 03, step 3b: teacher and languages

Use the worktree's `.venv`; real generation needs the existing
`curatorkit[generation]` extra (inspected version 0.1.23). The actual backend is
`curatorkit.llm.litellm.LiteLLMBackend`. No API calls were made during development;
backend tests replace its completion transport and forbid socket connections.

Put `CK_MODEL=openai/<served-model>`, `CK_API_BASE=<server>/v1`, and
`CK_API_KEY=<key>` in the worktree `.env` (gitignored), or environment. All three
are required and passed explicitly; environment overrides the file. A keyless
local server can use a nonempty dummy key. Keys are excluded from cache and build
artifacts. The builder does not modify the environment.

From the repo root:

```sh
.venv/bin/python -m turnguard build-data --split dev --teacher litellm --limit 5 --dry-run --backtranslate
# Sujay runs this only after reviewing the estimate and supplying settings:
.venv/bin/python -m turnguard build-data --split dev --teacher litellm --limit 5 --backtranslate
.venv/bin/python -m pytest tests/test_builder.py tests/test_teacher.py tests/test_templates.py -q
```

Each scenario must declare `split`. `configs/data.yaml` lists required families
per split; missing families (including code in test) fail before any teacher is
constructed. The batch-1 bank is incomplete, so default builds currently fail
with the missing family names. Tests use temporary, explicitly tagged fixtures.
No authored bank YAML was changed for this implementation.

Sampling covers conversation/injection languages independently. Filler (including
subjects) is generated directly in the conversation language. User turns, quoted
intros and authorized text/delegation are translated into that language; injection
parts into their independently sampled language. Scripted error prefixes and
route args remain English. Each authorized variant is emitted once per
conversation language. Twins come from the fully translated injected episode.

Keep tokens are variable/attacker values plus identifier-like asset words
(digits, addresses, paths, filename extensions), not whole prose assets. Each
translation preserves every keep token that occurred once in its English source
exactly once. Filler include facts are also required exactly once. Checks reject
altered ASCII digits, omitted/duplicated identifiers and non-sandbox filler links.

Requests are serial (concurrency 1), temperature 0, with at most one CuratorKIT
retry and no additional LiteLLM retries. `teacher_sampling` contains all decoding
options; no seed is used for teacher reproducibility. The SHA-256 cache under
`data/cache/` covers full messages, model, base, every sampling/backend parameter,
prompt version, configured template version and actual template/bank contents.
Responses (including failed content checks) are cached: to retry bad generated
text, deliberately change the request/settings or remove its identified cache
entry. Reruns reuse cached filler/translations; credentials never enter cache keys.

Stub output stays in `build/stub/<split>/`; its language markers only exercise
plumbing, not translation quality. Real teacher output goes to `data/v1/<split>/`.
Both include episodes, rejection reasons, counts/factor shortfalls and teacher-call
lineage. `--dry-run` writes nothing and estimates unique uncached calls and tokens
(input characters/4 plus maximum output budget, excluding retries). Without CK
settings its estimates assume an empty cache. `--backtranslate` reviews 10% of
unique translations (rounded up, selected deterministically by hash), through the
same cache, logging source/translation/backtranslation and a character similarity
score in `backtranslations.jsonl`. This score is for Sujay's review, not a semantic
guarantee. CuratorKIT dedup, SecretsGate and final release artifacts follow in 3c.
