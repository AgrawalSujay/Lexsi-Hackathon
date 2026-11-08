# Builder shortcut cues (C2, stop 1)

`python -m turnguard audit-cues --split dev --build-dir build/stub/dev`
reads `episodes.jsonl` and writes `cue_audit.json`. Every completed build also
writes that file before CuratorKIT hashes its artifacts. Stopped builds receive
an empty audit alongside their diagnostic summary; dry runs write nothing.
Frozen releases are verified and audited without changing their files.

The audit inspects planted text and its target matchers for addresses and URLs
(including the bank's scheme-less URLs), excluding unrelated user/tool data.
TLD shares count distinct identifiers per episode, with the denominator shown.
Language/`assistant` shares count episodes; clean twins have no planted text.
Delegation is checked against the exact delivered sentence in metadata. Legacy
builds without that metadata report an unknown denominator rather than guessing
whether a translated sentence delegated authority. Differences strictly above
80 percentage points between injected and authorized episodes log warnings.

Normal builds sample authorized instruction language independently. Seeded
selection assigns about 15% of injected bases an authored delegation for their
position; missing position variants are never invented. Accepted/requested
counts appear under `overreach` in the summary, and clean twins retain the same
delegation and filler.

`build-data --split test --teacher litellm --cue-flip` uses test scenarios and
writes `data/v1/test_cueflip/`, with separate IDs. Defaults (`cueflip_targets` in
`configs/data.yaml`) are 120 injected bases, their 120 twins and up to 60
scenario-balanced authorized attempts. Rejections/dedup can lower final counts.
Stub mode writes only `build/stub/test_cueflip/`. Attacks use the conversation
language and overreach; authorized instructions use another language and an
`Assistant:` prefix. `--limit` still caps injected bases only.

C3 hook: `attacker_pool(bank, split, profile="cueflip")` reads an optional
`attacker_pool_profiles.cueflip.<split>` overlay with the existing `addr`, `url`,
`user` fields. It otherwise retains the current split's pool. Authorized target
resources remain authored in their scenario variants; C3 can supply the matching
lookalike resources there. No bank content or domains change in C2 stop 1;
`domain_profile` reports whether an overlay is available.

`freeze --split test` validates and freezes both test directories when the
companion exists, hashing every file in each. Both loaders allow only
`purpose="eval"`; a frozen main test release also prevents building its companion.
