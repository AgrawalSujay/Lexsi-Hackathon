# C8 Stop 3: A4 preference trainer

## SafeTune source inspection and API

SafeTune, TRL and Torch were absent from the existing laptop venv (`pip show`), so no claim of an installed GPU-stack check is made. Inspected the **SafeTune 0.1.14 distribution source**, downloaded without dependencies or installation; wheel SHA-256 `64829cc07ad29f8aa28bde6094e9c9401676c67df536074e7c34e8cdd3e55897`. Inspected its compatible **TRL 1.14.2** source the same way. Both versions are pinned in the GPU extra so Modal installs the inspected APIs. No model, tokenizer, GPU job or paid API was run.

| Source module | Finding / use |
|---|---|
| `safetune.interventions.harden.cst` | **`CSTTrainer` / `CSTConfig`**: lower-level DPO trainer/config, subclassing TRL. This is the A4 training API; no direct TRL fallback was needed. |
| `safetune.core.data_compiler.cst` | `CSTFormatter` optionally adds opposite system prompts and reverses preferences. Do not call it for our already-approved on-policy pairs. |
| `safetune.runner.harden._dpo_adversarial` | Higher-level CST runner can replace data with contamination/refusal examples and saves a merged model; bypass it to preserve our pairs and adapter-only output. |
| `safetune.runner.utils.model_utils` | `load_tok(model_id)` and `load_model(model_id, dtype=torch.bfloat16, device="cuda")`, shared model-loading path with the agent benchmark. |
| `trl.trainer.dpo_trainer` | `peft_config` + `ref_model=None` uses the base with its adapter disabled. No second 8B reference is allocated. Named eval dictionaries are supported, but reward logging ignores the subset prefix; a small subclass fixes that. |

`trainer_api()` permits direct `trl.DPOTrainer` only if an installed SafeTune lacks the CST module; a missing GPU dependency or missing SafeTune itself is an error, not a silent backend switch.

## Configuration and data handling

`configs/arms.yaml` has `training.A4`: rank 16, alpha 32, learning_rate 5e-5, warmup_ratio 0.1, beta 0.1, epochs 2, max_length 4096, gradient_checkpointing true, gradient_accumulation_steps 8, holdout_fraction 0.05; target_modules q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj. Attention and MLP modules must actually exist. bf16 is required; training/eval batch size is 1, dropout 0, seed 1234, external reporting disabled.

The LoRA DPO learning rate is 5e-5, with linear warmup over the first 10% of optimizer steps; both values are passed explicitly to SafeTune’s CSTConfig.

The pair loader rejects dev/validation/test material before optional GPU imports, including copied/renamed validation data. Original model-format messages remain conversational, rendered with the model tokenizer; no extra system prompt, response inversion or replacement dataset. Adapter output refuses an existing path.

Holdout is a deterministic kind-stratified 5%, rounded up. On the real **1,032** pairs: **980 train / 52 held out**, with **7 FINAL / 45 ACTION** held out. All chosen responses: **134 FINAL / 898 ACTION**; FINAL comprises the noted **82 terse oracle answers** (72 synthesized + 10 executed-source pairs) plus **52 edited relay remnants**. Held-out loss and reward accuracy are logged for `all`, `FINAL`, and `ACTION`, at each epoch and at the end; n=0 is explicit for a missing kind in tiny smokes. Eval batch size 1 avoids unequal last-batch weighting in TRL's reward aggregation.

The 4096-token budget preserves every chosen/rejected completion token. If needed, the *same* prompt is left-truncated for both completions, retaining recent context; counts are logged per training/eval subset. A completion that alone exhausts the budget is rejected. No pairs are silently dropped or completion labels truncated. Actual tokenizer budgeting only happens on the GPU job, not this CPU dry run.

`--smoke`: seeded 20-pair subset, 10 optimizer steps, accumulation 1, separate default `*-smoke` adapter. It still holds out 5% (one pair for twenty) and reports missing kinds with n=0. The ordinary train command uses all pairs and two epochs.

## Outputs and Modal

Default headline adapter: `adapters/A4/headline` (already referenced by `arms.A4.adapter`); Tiny Aya: `adapters/A4/dev`. `--output` can select another directory. Each `runs/A4__<model>__train-a4__<timestamp>/` has resolved config, pair hash and split IDs, environment/elapsed seconds, named held-out metrics, training log/state, checkpoints, and an `arms.yaml` override. For Tiny Aya or a custom output, use the generated override for inference. Adapters remain unmerged PEFT adapters.

Modal task `train-a4` accepts `--model`, `--pairs`, `--smoke`, `--background` and either H100 or A100-80GB. Source remains `git archive HEAD`; **pairs are uploaded separately**, with all file checksums verified in the container before training and validation-data checks repeated. Each job's adapter, logs/checkpoints and `PAIRS_UPLOAD.json` are persisted under its own `runs/.../` in `turnguard-hf-cache`. Foreground jobs also return `adapters/` locally. Training timeout is six hours. Use Modal's global `--detach` with `--background` so a client disconnect does not stop the app.

From `/Users/sujayagrawal/Downloads/turnguard-c8` (commands below are for the owner; not executed here):

```bash
# GPU smoke, Tiny Aya, ten optimizer steps
modal run --detach scripts/modal_run.py --task train-a4 --model dev --gpu H100 --smoke --background

# Full headline job; pairs directory defaults to data/pairs/train
modal run --detach scripts/modal_run.py --task train-a4 --model headline --gpu H100 --background

# Inspect/download the job's adapter and logs after completion
modal volume ls turnguard-hf-cache runs
modal volume get turnguard-hf-cache runs/<run-name> ./runs/<run-name>
# Adapter is in ./runs/<run-name>/adapters/A4/headline; configure adapter: to that path,
# or copy it to ./adapters/A4/headline for the default arm config.
```

Local GPU alternative:

```bash
python -m turnguard train --arm A4 --pairs data/pairs/train --model headline
```

CPU preflight executed here:

```bash
../turnguard/.venv/bin/python -B -m turnguard train --arm A4 --pairs data/pairs/train --model headline --dry-run
```

## Time and cost estimate (unmeasured training range)

The preflight estimates **1,496,882 policy pair tokens per epoch**, ~2.99M for two epochs, using characters/4; Hindi/tokenizer differences can increase this. Reference forwards, backward passes, checkpoint recomputation and eval add work. No training throughput was measured; do not substitute the autoregressive probe's generation speed for DPO throughput.

Budget **1–3 hours on H100** (~$4–12 GPU) or **1.5–4 hours on A100-80GB** (~$4–10 GPU), excluding cold image/weight downloads, queue time and separate CPU/RAM/storage charges. These are planning ranges, not a GPU benchmark; calibrate with the ten-step smoke and its elapsed/trainer logs. At the checked [Modal rates](https://modal.com/pricing), H100 is $0.001097/s (~$3.95/h) and A100-80GB $0.000694/s (~$2.50/h). Six-hour timeout caps one job's GPU time at ~$23.70 / ~$15 respectively.

## Verification

**738 passed, 19 subtests passed in 125.42 seconds**; focused trainer/Modal tests: 30 passed. Ruff check/format and git diff checks passed. Fake-only tests cover a 20-pair training/export, holdout determinism and kind stratification, exact prompt preservation, all configured LoRA/DPO settings, bf16/reference contract, 4096 budgeting without completion loss, separate FINAL/ACTION reward logs, missing-metric refusal, copied validation refusal, ten-step smoke, CLI wiring, upload integrity, background spawning, and adapter/log/checksum persistence. No tests use network or GPU. Actual CUDA compatibility, peak memory and throughput remain for the owner-run smoke.
