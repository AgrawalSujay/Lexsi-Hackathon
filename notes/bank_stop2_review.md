# C3 Stop 2 — controls, coverage and cue audit

35 scenarios: 15 train, 5 dev and 15 test; code remains test-only.
111 authorized source variants (50/15/46), three or four per scenario;
35 integrate delegation into the task itself. Primary test controls cap at 250.

## Destructive controls

| Family | Train control | Test control |
|---|---|---|
| Email | school permission archive, 30-day cutoff | expense-claim newsletter archive, 5-day cutoff |
| Docs | obsolete clinic consent duplicate | obsolete party file during contract review |
| Tickets | cancelled swim class refund | damaged-delivery refund |
| Calendar | cancelled plumbing visit | cancelled kiln orientation |
| Code | exempt (held-out family) | obsolete garden-controller branch deletion |

Each wrong-item attack targets a separate decoy. Email targets match only
`archive_old(days=0)`, never the legitimate cutoff. The code control also
authorizes deletion of a different obsolete documentation branch in one view.
Canceled event controls keep a live event available for invite-exfiltration tests.

## Actual CPU stub audit

These are accepted builds with five injected bases plus their exact twins;
all applicable authorized views are extra, subject to the test cap. No paid
teacher calls, GPU work or model downloads. Stub language markers test language
routing/labels; this is not linguistic validation of real Hindi generations.

| Split | Injected / clean / authorized | Duplicate rejections | TLD gap | Language gap | Assistant gap | Delegation gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 5 / 5 / 1170 | 0 | 4.55 pp | 27.26 pp | 10.00 pp | 60.00 pp |
| dev | 5 / 5 / 342 | 9 | 18.00 pp | 27.84 pp | 13.80 pp | 60.00 pp |
| test | 5 / 5 / 250 | 0 | 17.74 pp | 7.60 pp | 10.00 pp | 60.00 pp |

All nonstructural gaps are within 30 percentage points. The TLD column is
the maximum over `.invalid`, `example.*` and other; raw shares, denominators and
coverage are in `notes/bank_stop2_audits.json`. Delegation is the expected
structural difference (40% overreach versus user-authorized instructions).
The development rejections are duplicate views after filler-version merging;
every injected base and its clean twin was retained. Small smoke audits do not
estimate population rates: rerun audits on every real teacher build before freeze.

## Implementation checks

- Implicit task replacement occurs before placement, so quoted parts survive.
- Authorized follow-up calls execute in their exposure turn, including later turns.
- Capped test sampling retains every source variant and conversation language.
- Legacy tags and bot names have numeric suffixes for exact-token translation checks.
- Existing split-disjoint wording, addressing balance, first-call exposure, twins,
  scenario-held-out rules and frozen-split safeguards remain tested.

Reproduce in this worktree: `.venv/bin/python -m pytest -q`, then
`.venv/bin/python scripts/render_prototype.py`. For S=train/dev/test run
`.venv/bin/python -m turnguard build-data --split S --teacher stub --limit 5`
and `.venv/bin/python -m turnguard audit-cues --split S --build-dir build/stub/S`.


Verification: the full CPU run passed 583 tests + 19 subtests; an additional
later-turn implicit-quote regression was then added and verified with the
control module (584 current test cases verified in total). The prototype passes
21,207/21,207 episodes. Lint/format checks pass. No merge or push.
