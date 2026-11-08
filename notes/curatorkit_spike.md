# CuratorKIT exploration — task 03, Step 1 only

**Inspected:** CuratorKIT **0.1.23**, Python 3.12.8, on `task03`
(branch base `ec07a58`). Installed with `.venv/bin/python -m pip install
'curatorkit[generation]'`; verified with `.venv/bin/python -m pip show curatorkit`
and `pip check`. All installs and disposable probes live in this worktree's
ignored `.venv/`. No provider calls, GPU use, model downloads, builder changes,
Step 2 templates, merges, or pushes.

Module paths below are import paths; their files are under
`.venv/lib/python3.12/site-packages/curatorkit/`. The wheel **and sdist omit
examples/tests**. Read matching upstream tag `0.1.23`, commit
[`eb25731cf34f0b2370543ac1e6651dc9040a5e67`](https://github.com/Lexsi-Labs/CuratorKIT/tree/eb25731cf34f0b2370543ac1e6651dc9040a5e67/examples):
`examples/quickstart/{README.md,pipeline.yaml}`, `clean_and_dedup.py`,
`multi_source_ingest.py`, `hygiene_pipeline.py`, `generate_sft_from_pdf.py`,
`generate_dpo_pairs.py`, `adaptive_recovery.py`, and the fake-LLM pattern in
`tests/unit/test_prompt_template_validation.py`. Source copies are inside `.venv/`.

## 1. Free-form generation: use the LLM layer directly

`curatorkit.llm.litellm.LiteLLMBackend` inherits
`curatorkit.llm.base.BaseLLM`; call `generate(messages, **overrides)` or
`await agenerate(...)`. Both return `LLMResponse`; use `.text` and retain
`.to_provenance_dict()` (model, token counts, latency). Unlike the stock
QA/preference generators, this layer accepts **our complete prompt** without
requiring a generated QA JSON schema. Future real-provider call, **not run**:

```python
import os
from curatorkit.llm.litellm import LiteLLMBackend

llm = LiteLLMBackend(
    model=os.environ["CK_MODEL"],  # LiteLLM provider/model name
    api_base=os.getenv("CK_API_BASE"),
    temperature=0.0,
    max_tokens=512,
    max_retries=0,
)
response = llm.generate(
    [{"role": "user", "content": prompt}],
    seed=1234,
)
```

For an OpenAI-compatible local teacher, use `model="openai/<served-model>"`
and its `/v1` base URL. Explicit `api_key=` is supported; otherwise LiteLLM
uses provider-standard variables such as `OPENAI_API_KEY`. With `api_base`
but no explicit key, this backend supplies `"nokey"`.
**`CK_MODEL`/`CK_API_BASE` are example-script conventions, not automatically
read by the backend or YAML parser**: our caller must pass them, as above.

High-level Python equivalents (`curatorkit.curator.CuratorConfig`) are
`llm_model`, `llm_api_base`, `llm_api_key`, `llm_temperature`, `llm_max_tokens`,
`llm_timeout`, `llm_max_retries`, `llm_drop_params`, `llm_extra_body`.
YAML (`curatorkit.config.LLMConfig`) instead uses `llm: {model, api_base,
api_key, temperature, max_tokens, timeout, max_retries, drop_params, extra_body}`;
role overrides include `generator_llm` and `judge_llm`. No YAML `seed` field
exists in `LLMConfig`; pass `seed=1234` through the direct call. With
`drop_params=True`, provider-unsupported parameters can be dropped, so a
server-side deterministic-seed guarantee still requires verification.

**Verified offline:** a `BaseLLM` echo subclass accepted our prompt; a fake
LiteLLM transport verified model/base/key/seed forwarding. Socket connections
were blocked. `_postprocess()` strips a `<think>` block and leading whitespace;
validate exact required strings on the returned text. Cache full messages,
model/revision, base, seed, sampling and template version: `BaseLLM.config_hash()`
only covers model/temperature/max_tokens, so it is insufficient as our cache key.

## 2. Own JSONL pipeline: a Python adapter is needed

`curatorkit.connectors.jsonl.JSONLReader(path, preprocessing_fn=...)` accepts
our file. Its hook may return a `curatorkit.schema.DataSample`, bypassing
format detection; carry a deep copy at `metadata["episode"]`, keep the episode
ID, and set `task_type="language_modeling", output=rendered_text`.
The rendered view should cover all model-visible user/state strings, including
metadata carriers, every part, and scripted errors; do not dedup only user tasks.

`curatorkit.gates.schema.SchemaGate` checks **DataSample attributes**, not JSON
Schema or dotted nested paths. `required_fields=["output", "metadata"]` is a
useful envelope check; a custom `curatorkit.interfaces.BaseGate.run()` must
call `turnguard.schema.validate_episode(..., twins=...)` on the nested episode,
returning `(passed, RejectedSample list)` with reasons. A reader hook raising
`ValueError` also becomes a structured reader rejection, but a separate gate
would preserve clearer validation provenance.

Compose `curatorkit.pipeline.Pipeline(steps, output_dir=...)` in this order:
`JSONLReader → episode gate → SchemaGate → ExactDeduplicator →
MinHashDeduplicator → hygiene checks → custom episode exporter`.
Both deduplicators are in `curatorkit.normalizers.dedup`. They have **no arbitrary
`text_field` option**; `language_modeling` selects `output`. Exact dedup
normalizes case/whitespace; MinHash uses character n-grams with
`threshold=0.85, ngram=3, num_perm=128, seed=1234`. Neither changes original
text. Dedup removals appear in stage counts/provenance, **not rejected.jsonl**.

High-level configuration is viable for stock stages:
`Curator(CuratorConfig(dataset=path, preprocessing_fn=hook, clean=False,
dedup="minhash", minhash_seed=1234, schema_required_fields=["output","metadata"],
output_split=None, generation_task=None, export_formats=[], output_dir=...)).run()`.
`dedup="minhash"` includes exact dedup first. YAML keys are
`readers: [{type: jsonl, path: ..., preprocessing_fn: dotted.module.function}]`,
`gates: [{type: schema, required_fields: [output, metadata], min_tokens: ...,
max_tokens: ...}]`, and `normalizers: [{type: exact_dedup}, {type: minhash_dedup,
minhash_seed: 1234, minhash_threshold: 0.85, minhash_ngram: 3,
minhash_num_perm: 128}]`. YAML has fixed gate/exporter type enums; custom
nested validation and lossless episode export require Python composition.

**Hygiene:** `curatorkit.hygiene.secrets.SecretsGate(fields=["output"])`
rejects credentials; YAML uses `gates[].type: secrets`, `secrets_fields`,
`secrets_code_corpus_mode`, `secrets_hex_limit`, `secrets_base64_limit`.
`curatorkit.hygiene.pii.PIIPseudonymizer` transforms strings, with Python
`fields`, `entity_types`, `language`, `faker_seed`, `nlp_engine`, `spacy_model`;
YAML prefixes these with `pii_` under `type: pii_pseudonymizer`.
Neither recursively scans arbitrary `metadata.episode` by default. `[generation]`
lacks hygiene dependencies: SecretsGate construction raised the documented
ImportError. Full `[hygiene]` adds detect-secrets, Presidio, Faker and Detoxify;
NER models are additional. **Not installed/run here.** Hindi NER coverage is
not established. Sanitize generated filler before assembly or reject findings;
never pseudonymize already-pinned facts, addresses or matchers. Real builds
must fail clearly if a requested hygiene stage is unavailable, not silently skip it.

## 3. Artifacts: supported, with explicit handling for our exporter

| Artifact | Installed producer and implications |
|---|---|
| `manifest.json` | `curatorkit.manifest.ProvenanceManifest.write()` records stage counts, rejection breakdown, dedup stats, versions and lineage. Works on adapted `DataSample`s. |
| `rejected.jsonl` | `ProvenanceManifest.write_rejected_sidecar()` always writes it, even empty. Rows are CK envelopes including nested metadata and provenance, not bare episodes. |
| `dataset_card.md` | `curatorkit.manifest.DatasetCardGenerator.generate(...)`; default also writes `README.md` (`write_readme=False` disables it; high-level key `write_hf_readme`). Stock HF config discovery knows stock exporters, not our custom episode schema. |
| `lexsi_provenance.json` | Written alongside the manifest; `build_provenance(method, inputs, params, base_model)` supplies `lexsi.provenance/1`. Direct LLM calls need their usage/model/prompt lineage attached explicitly. |
| `checksums.txt` | `ProvenanceManifest.write_checksums(paths)` hashes the supplied files; it is not the later TurnGuard `CHECKSUMS.sha256`/`FROZEN` contract. |

`Curator.run()` and `curatorkit.cli.run` automatically orchestrate these writers;
**bare `Pipeline.run()` does not**, despite its dry-run help suggesting always-emitted
metadata. Call the writers explicitly after the custom exporter. The CLI checksum
list uses known exporter filenames, so it misses custom `episodes.jsonl`; the
high-level Python path globs JSON/JSONL. Both generate the card **after** checksums,
so the Markdown card/README are omitted. Write final cards first, then checksum
all intended artifacts explicitly (excluding the checksum file itself).

Stock `AlpacaExporter`, `MessagesExporter`, `CorpusExporter`, etc. select training
columns and discard arbitrary metadata. None emits our nested episode as-is.
Use a `BaseExporter.export(samples, output_dir)` that writes only
`sample.metadata["episode"]` to `episodes.jsonl`; retain CK sample lineage
separately. Exported episode JSON remains compatible with `EpisodeSpec`.

**Measured:** five input rows (four references plus one duplicate) → raw reader
0 accepted/5 rejected (`unrecognized_format:no_known_column_pattern`). Hooked
reader/schema 5 → exact dedup 4 → MinHash 4 → lossless exporter 4; all survivors
still validate and equal their original nested episode. Nested required-field
attempt rejected all five as `missing_field:metadata.episode.id`. All five named
metadata artifacts were written and checksum entries verified. `Curator.run()`
also emitted all five automatically with no exporters. The first adapter's
manifest had empty `source_files`: `_collect_source_files` looks for provenance
`notes["source_file"]`, not `source_uri`; adding that note fixed it. Preserve
source SHA/config hashes explicitly. A combined exact+MinHash manifest's
`dedup_stats` emphasized the latter; `stage_counts` retained both counts.

## 4. Fit gaps and proposed workaround for later steps

- **Keep canonical JSON separate from checks.** Use an untouched episode in
  metadata plus a complete disposable text projection. Disable stock cleaning
  (`CuratorConfig.clean=False`; omit YAML `text_cleaner`): HTML, Unicode and
  whitespace normalization can alter exact planted text. Revalidate the final
  episode, facts, carrier placement, target terms and twin relationship after
  all assembly. Reader hooks returning DataSample bypass automatic connector
  provenance too; append source/config provenance explicitly.
- **Preserve matched groups.** Clean twins intentionally resemble attacks;
  MinHash can delete them. Dedup scenario/base-content groups before expanding
  injected/clean/authorized variants, then carry the entire retained group.
  Record removals and verify the twin invariant. Do not let stock random
  `output_split` or single-field resampling decide frozen/held-out partitions;
  keep family/carrier/factor stratification and split guards in TurnGuard.
- **Main-path exposure stays authored.** CK's generation tasks do not validate
  tool reachability. Name the required ID in the request or author an initial
  listing that reveals the intended carrier; later/split parts need explicit
  required retrievals on the task route. A listing that omits a body is not
  proof of body exposure. Future acceptance checks should execute an authored
  legitimate stub route against the tools, requiring every part to reach the
  model. Clarify first-call exposure versus `position: later` in Step 2 rather
  than silently changing position labels. No templates implemented here.
- **Relay terms stay deterministic.** Require every injected call matcher to
  carry at least one distinctive, nonempty `args_contains` term; preserve the
  exact attacker address/canary across text, translation and matcher. These
  constraints are not built-in CK checks. Prefer generating filler separately,
  then insert authored injection strings; reject missing/extra occurrences.
  PII replacements (Faker may create external emails) must not change them.
- **Avoid superficially similar generators.**
  `curatorkit.generators.adversarial_qa_generator.AdversarialQAGenerationTask`
  and `generators.injector.BadSampleInjector` inject incorrect QA answers for
  gate evaluation, not executable tool instructions with JSON-pointer slots,
  multi-part exposure or target matchers. Use CK's free-form LLM backend,
  not these generators. Disable automatic hallucination/reward/refiner/probe
  stages unless deliberately requested: they may make paid calls or rewrite
  the very adversarial text being measured.

**Proposed integration:** a small TurnGuard-owned read/validate/export adapter
around real CK deterministic stages and artifact writers, with constrained
free-form filler generation, explicit full-config caching/provenance, and
pair-aware dedup. This is an exploration recommendation, not Step 2 approval
or a builder implementation. All probe files are ignored under `.venv/spike/`.
