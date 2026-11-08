# Running TurnGuard on Modal

Modal is the backup GPU: quick checks and decisions, not the headline results
(every arm in the final tables runs on the H200). The account has $30/month of
credit with a $30 usage limit, so a run can stop but cannot charge a card.

`scripts/modal_run.py` sends the **committed** code (uncommitted changes are
not sent) to a Modal GPU, runs it, and copies the results into `runs/` here.

## One-time setup

```bash
python3 -m pip install modal
modal setup
modal secret create huggingface HF_TOKEN=hf_your_token_here
```

The token needs read access and the licence accepted on
`CohereLabs/aya-expanse-8b`. Type it in your terminal, never in chat.

## Runs (from the repo root)

| Command | What it does | Time | Cost on L40S |
|---|---|---|---|
| `modal run scripts/modal_run.py --task smoke` | GPU check, tests, Aya 8B loads and replies | ~5–10 min the first time (image build and 16 GB download), ~3 min after | < $0.50 |
| `modal run scripts/modal_run.py --task examples` | The 4 reference episodes | ~5 min | ~$0.20 |
| `modal run scripts/modal_run.py --task probe` | The 28-episode probe | ~15–25 min | ~$0.80 |
| `modal run scripts/modal_run.py --task all` | All three in one container | ~30 min | ~$1 |

Options: `--model dev` runs Tiny Aya instead of Aya 8B; `--gpu A100-80GB` or
`--gpu H100` picks a faster, pricier GPU.

**Expected:** the smoke step prints the GPU name, `N passed`, a model reply,
and `SMOKE OK`. The run steps print the same table and metrics as on the H200
(see `notes/run_on_h200.md` for what the numbers mean). The final summary
lists each step as `ok` and the saved `runs/...` folders.

## If something goes wrong

- `Secret 'huggingface' not found`: run the `modal secret create` line above.
- `401` / `gated repo`: the token lacks access or the licence is not accepted.
- A step shows `FAILED`: copy the lines above the summary and send them.
- Check spend at modal.com → Settings → Usage & billing.
