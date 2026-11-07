# First real-model smoke run (Colab)

Run this yourself on a Colab GPU runtime with Python 3.12 or newer, from the
TurnGuard checkout root. Copy the current code into that runtime first.
The `dev` key in `configs/models.yaml` selects `CohereLabs/tiny-aya-global`.

```sh
python -c 'import sys; assert sys.version_info >= (3, 12), sys.version'
python -m pip install -e '.[gpu]'
python -m turnguard run-episodes \
  --episodes 'data/examples/*.json' \
  --generator hf --model dev --batch-size 1 \
  --arm A0 --split examples
```

In a notebook, use `%%bash` at the top of the cell and `set -e` before these
commands, with the working directory set to the checkout root.

Expected: a four-row episode table and core rates with Wilson intervals;
`runs/A0__CohereLabs-tiny-aya-global__examples__<timestamp>/` contains
`config.yaml`, `episodes.jsonl`, `metrics.json`, and `env.json`.
The actual tags and malformed rate are observations, not predetermined.
An ungraded run or generation error produces a nonzero exit code and retains
the trace and error. Please return the table and any errors for review.

No real model was loaded or downloaded while implementing this adapter.
CPU tests replace torch, the tokenizer, and SafeTune loaders with Python doubles.

The SafeTune loader source was inspected before implementation:
https://github.com/Lexsi-Labs/SafeTune/blob/main/src/safetune/runner/utils/model_utils.py
`load_tok(name)` defaults to right padding; the adapter switches to left padding.
`load_model(name)` loads through SafeTune and selects device/dtype from its runtime
configuration. The adapter preserves those shared objects for intervention hooks.
