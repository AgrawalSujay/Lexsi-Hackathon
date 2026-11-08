# Running TurnGuard on the H200

For the teammate with the H200. Runs Aya Expanse 8B on the 4 reference
episodes, then the 28-episode probe. About 20–30 minutes plus the model
download if it isn't cached yet.

## 1. Get the code onto the box

From the laptop: `scp ~/Downloads/turnguard_src.zip <user>@<h200-host>:~/`

On the H200:

```bash
cd ~ && rm -rf turnguard && unzip -q turnguard_src.zip -d turnguard && cd turnguard
cat BUILD_INFO.json          # note the git_commit
```

## 2. Python environment

If you already ran the smoke test (`~/h200_smoke`), reuse its environment and
model cache:

```bash
PY=~/h200_smoke/.venv-smoke/bin/python
export HF_HOME=~/h200_smoke/hf_home
$PY -m pip install -e '.[gpu]'
```

Otherwise create one (Python 3.12+ required):

```bash
python3.12 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -e '.[gpu]'
PY=.venv/bin/python
```

Then:

```bash
export HF_TOKEN=hf_...       # Read token; Aya Expanse 8B licence accepted
# Compute node without internet (model already downloaded): also
# export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
```

## 3. Run

```bash
$PY -m turnguard run-episodes --episodes 'data/examples/*.json' \
  --generator hf --model headline --batch-size 8 --arm A0 --split examples

$PY -m turnguard run-episodes --episodes 'data/probe/*.json' \
  --generator hf --model headline --batch-size 16 --arm A0 --split examples
```

Each prints a table of episodes and the metrics. Out-of-memory is handled
automatically (the batch is halved and retried).

## 4. Send back

```bash
zip -qr runs_h200.zip runs
```

Send `runs_h200.zip` to Sujay. When done, nothing keeps running on the GPU.
