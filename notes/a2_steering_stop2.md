# C9 Stop 2: A2 CAA steering and recorded-baseline dev tuning

## SafeTune inspection

The laptop CPU environment has no installed GPU stack. Inspected the pinned **SafeTune 0.1.14 distribution source** (the dependency-free wheel inspected for C8), particularly:

- `safetune.interventions.steer.caa.CAAConfig`, `extract_caa_vectors`, `CAAModel`;
- `safetune.core.runtime.inference.vector_extraction.SteeringVectorExtractor`;
- `safetune._refusal_helpers._get_decoder_layers`.

`CAAModel` provides additive residual hooks and a reusable `__enter__`/`__exit__` lifecycle that fits `HFGenerator.intervention`. Its original hook adds to every sequence position, so `CurrentTokenCAA` subclasses it and overrides only `_make_hook`: the last prompt position at prefill (to affect the first generated token), and the current token on cached decode. Earlier prompt positions are untouched; batch placement, dtype/device conversion, tuple outputs and cleanup remain supported. No alternate runtime backend is used.

The public extractor offers last-token or whole-sequence-mean pooling, not the requested last-prompt-plus-completion window. Extraction is therefore direct forward-hook pooling, using SafeTune's decoder-layer locator and the existing `HFGenerator` model/tokenizer loading path. This is the intentional API adaptation.

## Extraction and artifacts

`python -m turnguard steer-vectors --pairs ... --model headline --layers 10,14,18,22`:

1. Use the existing mandatory training-pair loader: dev/test/validation pairs are refused before model loading, including copied validation artifacts. Read and hash the input, checking that it stays unchanged.
2. Render each exact on-policy prompt with the model chat template and generation prompt. Append raw completion token IDs without special tokens. Use up to **8** completion tokens, taking the same count on both sides of a pair (the shorter side caps both).
3. At each **zero-based decoder block output**, pool the last prompt token and those first completion tokens. Average chosen-minus-rejected windows equally across pairs, then unit-normalize per layer. The shared last-prompt contribution cancels under deterministic causal evaluation. Identical completion prefixes contribute zero; their count and unnormalized mean-vector norms are reported.
4. Run the base decoder under inference mode, without caching or the LM-head logits; retain only pooled residuals. Remove capture hooks even on failure.
5. Save `steering/A2/headline.pt` with train-only provenance (model, pair-file hash/count/source mix, window/index convention, package/GPU/git information, extraction token count and elapsed seconds), plus a SHA-256 JSON sidecar.

Runtime `make_hook(generator=..., vector=..., layer=L, alpha=a)` loads tensors with `weights_only=True`, verifies model/train/version/layer/shape/finite nonzero norm, normalizes v, and returns a fresh SafeTune context. Selected settings additionally bind the exact vector SHA-256. This avoids loading an old selection against a newly extracted vector. OOM/retry and normal exit remove all hooks. No protocol text changes; A2 uses no A1 paragraph.

The generated `steering/A2/headline.yaml` / parent `selected_arm.yaml` contains the requested `intervention_factory: turnguard.arms.steer:make_hook` and vector/layer/alpha kwargs (plus vector checksum). Use it via `--arms-config`; no untuned default is silently declared as the selected A2 arm.

## Fixed subset and A0

Selection is SHA-256 ID ordering, the same `_rank` rule as existing split selection, independently per type: first 50 injected, their exact 50 clean twins, first 50 authorized. Input order does not matter. Missing types/twins, non-dev data, wrong model/decoding/protocol, changed specs/dataset, missing baseline records or ungraded baseline records are errors.

`prepare_baseline` reads `runs/A0__CohereLabs-aya-expanse-8b__dev__20261109-0802` against the original verified dev specs, selects the same IDs, and regrades their records **in memory with c8-v1**. Config/record hashes are checked before/after; it never rewrites the run or dataset. Each recorded base system prompt is also checked against the frozen current protocol. The snapshot is checked again on the remote side. **No A0 generation is performed.**

Real read-only CPU preflight:

| Metric | Recorded A0 on selected 150 |
|---|---:|
| Execution ASR | 2/50 = 4% |
| Relay-only ASR | 14/50 = 28% |
| ASR including relay | 16/50 = 32% |
| Clean task success | 22/50 = 44% |
| Authorized followed (target, independent of task success) | 8/50 = 16% |
| Ungraded | 0 |

No relay labels changed on this already-upgraded subset. All 150 records were nevertheless regraded. Their exact IDs, spec hashes, original run hashes and baseline metrics travel in the snapshot. Test/test_cueflip are never tuning inputs.

## Six-point tuning and selection

The grid is **layers 14/18/22 × alpha 4/8**, with the configured inference batch size and frozen greedy/256-token/six-step protocol. A single model instance is reused for extraction and all six points. Every point sees the same 150 IDs. Existing `metrics.py` reports execution, relay-only, combined ASR, clean success, authorized followed, full language/scenario breakdowns and CIs.

Predeclared rule: choose lowest combined ASR among fully graded points with clean task success and authorized followed each at least recorded A0 minus **0.05 absolute**. If none qualifies, minimize the maximum guardrail shortfall beyond 0.05, then total shortfall, then combined ASR. Remaining ties: lower alpha, then lower layer. Missing denominators/ungraded points cannot win. The exact rule is stored before generating results. No best configuration is claimed here: the six GPU points have not been run.

Each point exports ordinary re-scorable run config/episodes/metrics/environment timing. The parent run keeps the baseline snapshot/metrics, selection rule, point grid, summary, selected arm and whole-job elapsed time. Failures keep diagnostics. The configured model must match the recorded A0; `--model dev` needs a corresponding Tiny Aya A0 run, not the provided Aya-8B run.

## Modal and owner commands

New task `steer-tune`: independently uploads checksummed train pairs, the dev release and the verified recorded-A0 snapshot alongside `git archive HEAD`. The container verifies all payloads before model loading. The job persists per-point/parent runs, vector/provenance/selection and transport checksums to `turnguard-hf-cache`; successful selections also appear under `steering/A2/` on the volume. Failed jobs cannot replace that working selection. Per-job partial artifacts remain available. Timeout: six hours. Use global `--detach` with `--background` to survive client disconnects.

From `/Users/sujayagrawal/Downloads/turnguard-c9`, owner-run GPU command (not executed here):

```bash
modal run --detach scripts/modal_run.py --task steer-tune --model headline --gpu L40S --background \
  --pairs ../turnguard/data/pairs/train \
  --dev-dir ../turnguard/data/v1/dev \
  --baseline-run ../turnguard/runs/A0__CohereLabs-aya-expanse-8b__dev__20261109-0802
modal volume ls turnguard-hf-cache runs
modal volume ls turnguard-hf-cache steering/A2
```

Standalone extraction (GPU):

```bash
python -m turnguard steer-vectors --pairs ../turnguard/data/pairs/train --model headline --layers 10,14,18,22
```

The local CPU dry runs of both new commands passed against **1,032 real train pairs**, the original dev release and the verified A0 snapshot. No models/tokenizers were loaded, no API/GPU jobs ran, and main's data/run files were read only. Extraction's character/4 approximation is **1,525,250 forward tokens** (actual tokenizer count is recorded during extraction).

## Time/cost: six grid points plus extraction

Measured source: A0 env.json records **NVIDIA L40S**, **5,748.12 generation seconds** for the full 749 episodes. The selected 150 contain 722/3,602 assistant steps (20.04%) and 213,493/1,099,106 generated characters (19.42%). Scaling that measured workload gives **18.6–19.2 minutes per point**, **~112–115 minutes for six points**. There is **no additional A0 GPU point**.

Extraction over ~1.53M estimated forward tokens is unmeasured; allow **10–40 minutes**. Together this suggests roughly 2–2.6 hours before output-length/steering and startup variation; budget **2–4 hours total on L40S**, or **~$3.90–$7.80 GPU cost**, plus separately billed host resources/cold setup. This is a planning range, not a steering benchmark. [Modal's pricing](https://modal.com/pricing) lists L40S at $0.000542/s ($1.9512/h); the six-hour timeout would cap GPU time at ~$11.71. No unmeasured H100 speedup is assumed.

## Verification / integration boundary

Full final CPU suite: **778 passed, 19 subtests passed in 132.30 seconds**. Ruff check/format and git diff checks passed. Focused steering/Modal/A1/A4 regressions: **70 passed**. Tests use NumPy tensors, fake models/tokenizers/SafeTune contexts, recorded local fixtures and a fake Modal SDK; no network or GPU.

A4 checkpoint/resume code is preserved relative to the requested base `22056a4`: `harden.py` is unchanged; `_steps`, `_pairs_upload`, `_checkpoint_key`, `_periodic_commit`, `_unpack_pairs` and `run_task` have identical ASTs. The new job is separate. Main advanced to `a62f653` during this stop (runner work); this branch remains based on the explicitly requested `22056a4` plus rebased A1. Integration of that newer main and C9 Stop 3 is for the next stop.
