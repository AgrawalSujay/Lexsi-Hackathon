# Built-split runs and L40S planning (C5)

## Commands and provenance

From a checkout with a completed local `data/v1/dev/` build:

```bash
python -m turnguard run-episodes --episodes data/v1/dev/episodes.jsonl --split dev --generator hf --model headline --arm A0
modal run scripts/modal_run.py --task split --split dev --model headline --gpu L40S
modal run scripts/modal_run.py --task split --split dev --model dev --gpu L40S --shard 1/2
modal run scripts/modal_run.py --task split --split dev --model dev --gpu L40S --shard 2/2
```

A directory works in place of `episodes.jsonl`. Existing JSON files and quoted
JSON globs still work for examples/train/dev. Test and test_cueflip require a
complete frozen release, loaded with `purpose="eval"`; individual test JSONs
cannot bypass that guard. The freeze command still validates before writing
FROZEN; its internal validation does not grant evaluation access.

Shards are **one-based** i/k: SHA256(episode ID) modulo k selects membership.
IDs sort before selection. `--limit N` caps the total set **before** sharding,
so the same limit across all k shards gives a disjoint, complete partition of
that limited set. A shard can contain a clean twin without its injected base;
the full source release is validated before selection. Shard labels appear in
run directory names, preventing simultaneous containers overwriting each
other's results. Rerunning a shard gives the same IDs and creates a fresh run;
it does not automatically skip episodes from an older run. Very small limits
can leave a shard empty; that is rejected before model loading.

The resolved run config records selected IDs, limit/shard, and SHA256 hashes of
the complete source JSONL, manifest, and checksums/FROZEN when present. Dataset
identity covers the **whole** source, even for a one-episode run. Rescore and
compare verify that identity, plus every recorded canonical spec hash. Use
`rescore RUN --specs data/v1/dev/` (or compare's --specs-a/--specs-b) when moving
runs back from Modal's temporary container paths. Different dataset identities
cannot be compared silently.

Modal archives committed source, then uploads **every file** from the requested
local split separately, rejecting symlinks. The receiver checks transport
checksums for the exact file set, release identity, schema, and frozen-release
checksums before model loading. It uses A0 and the model config's inference
batch size (currently 8), not the old probe's hard-coded 16. No build or cache
files are written by upload. Results go to the existing volume and back to
local runs/. Metrics print overall and by conversation language, instruction
language, and their cross-product; rates have Wilson intervals and ungraded
counts stay explicit. Raw generations remain in the run artifacts.

Successful agent, XSTest and MMLU runs now write UTC `started_at`/`finished_at`,
monotonic `elapsed_seconds` (setup/model loading through artifact writing), and
`generation_seconds` in env.json. Generation time is the harness/evaluator
window, including tool execution and any evaluator judge calls; it excludes
model loading and scoring performed after that window. Modal image building,
queue time, upload/download and volume copy are outside the CLI timer.

## What the existing probes actually record

Read-only inspection of `~/Downloads/turnguard/runs/` found the two v2.4 A0
28-episode L40S probes (commit b872fa0):

| Model | Directory suffix | Batch | Assistant turns | Output characters |
|---|---|---:|---:|---:|
| Aya Expanse 8B | `examples__20261108-0422` | 16 | 78 | 25,796 |
| Tiny Aya | `examples__20261108-0423` | 16 | 122 | 13,393 |

Neither run's env/config, records, calls, nor raw outputs has start/end or
per-step timestamps. All four artifact mtimes differ by milliseconds because
they were written **after** inference. The minute in a directory name also
marks post-inference creation; differences between directories are not run
durations. There is therefore **no usable measured wall-clock reference**.
The H200 note supplies scores, not elapsed timings. No GPU run was launched
for this estimate, and main's data/v1 was not inspected or modified.

## Conditional estimate for the 587-episode dev set

Reweighting the reference output character means to 118 injected + 118 clean
+ 351 authorized (rather than multiplying the probe's type mix):

| Model | Mean chars injected / clean / authorized | Dev output chars | At 3–5 chars/token |
|---|---|---:|---:|
| Aya Expanse 8B | 1022.0 / 884.8 / 728.5 | ~480,710 | ~96k–160k tokens (~164–273/episode) |
| Tiny Aya | 493.6 / 479.9 / 427.8 | ~265,013 | ~53k–88k tokens (~90–151/episode) |

Character/token conversion is a planning proxy, **not Aya tokenizer counting**.
Hindi, longer tasks and three-language controls can exceed it. The probe has
only four authorized controls, so its authorized mean is especially uncertain.
The 256-token limit applies per assistant turn, not per episode. The new build
also has two user turns and can use up to twelve assistant steps.

Because throughput is unmeasured, use the following **assumed effective
aggregate output-token rates**, including padding/prefill/tool overhead during
generation. They are sensitivity scenarios, not claims about either model's
L40S speed. Divide output tokens by measured generation tokens/second when a
new timed run is available.

| Effective aggregate tokens/s | 8B generation window | Tiny generation window |
|---:|---:|---:|
| 50 | 32–53 min | 18–29 min |
| 100 | 16–27 min | 9–15 min |
| 200 | 8–13 min | 4–7 min |

Thus the conditional planning envelope is **8–53 min for 8B, 4–29 min for
Tiny**, plus unmeasured loading/download/queue/artifact overhead. This is not a
measured total runtime prediction. Double output volume if multilingual or
longer tasks double tokens; generation time and cost double too.

Modal's [published price](https://modal.com/pricing), checked 2026-11-09:
L40S $0.000542/GPU-second = $1.9512/GPU-hour. That gives roughly **$0.26–$1.74
for 8B, $0.13–$0.96 for Tiny** over the generation windows above. Loading adds
$0.03252 per GPU-minute per container. CPU ($0.0000131/core-second), memory
($0.00000222/GiB-second), storage and any other billable overhead are extra;
these are GPU-only estimates, not total invoice quotes.

Two equal shards ideally halve generation wall time while preserving total
GPU work; three ideally divide it by three. Each container loads its own model
and pays a separate warm-up cost; hash shards can differ in difficulty and
size, and smaller batches/tails reduce efficiency. **Start with one container
(or a timed small limit); use two when turnaround matters after calibration.
Three are not justified by the current timing evidence for only 587 episodes.**
Once timings exist, compare `generation_seconds` with `elapsed_seconds` before
paying for more parallelism. Do not average shard rates: pool numerator and
denominator counts (and recompute Wilson CIs), or recompute metrics on the
combined disjoint records/specs.
