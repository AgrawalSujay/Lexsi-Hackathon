# TurnGuard

**Sujay Agrawal, Ratan Kokal** · Lexsi Labs AI Safety & Alignment Hackathon, Track 2

We built TurnGuard to see what happens when a tool-using assistant reads an email, document, or ticket with an instruction hidden inside it. Does it follow the instruction, pass it along to the user, or carry on with the task the user asked for?

The benchmark runs multi-turn tasks in English, Hindi, and Hinglish. Each attacked episode has a clean twin. We also included cases where the user explicitly asks the assistant to follow instructions from a tool result, since ignoring those instructions can break the task too.

We compared three changes to Aya Expanse 8B for Track 2 of the Lexsi Labs AI Safety & Alignment Hackathon. We used CuratorKIT to build the benchmark, SafeTune for activation steering and DPO, and AuditKIT to score the side evaluations.

- [Full write-up](WRITEUP.md)
- [Hugging Face models](https://huggingface.co/collections/SujayAgrawal/turnguard-models-6af5680cfeefa5c5291959dd)
- [Hugging Face datasets and logs](https://huggingface.co/collections/SujayAgrawal/turnguard-datasets-and-logs-6af5680eeb29a7da63da3311)

## What we found

The frozen test set contains 1,043 episodes across 15 scenarios held out from training and tuning.

| | A0 baseline | A1 prompt defence | A2 CAA steering | **A4 DPO LoRA** |
|---|---|---|---|---|
| Attack success incl. relay ↓ | 31.9% | 36.6% | 30.7% | **2.5%** |
| Clean task success ↑ | 20.6% | 21.6% | 29.0% | **51.8%** |
| Task success under attack ↑ | 10.3% | 10.1% | 15.2% | **44.7%** |
| XSTest over-refusal ↓ | 8.8% | 11.2% | 8.4% | 8.4% |

The DPO adapter helped the most. It brought attack success down to 2.5% and completed more of the user's tasks under attack. Our defensive prompt raised attack success on this test, and the improvement we saw from steering on dev didn't carry over.

The baseline often passed the injected request along to the user. Our scorer flagged 99 of these relays before training and zero after DPO, though it can miss paraphrases. The adapter still followed only 24.5% of authorized instructions, and malformed steps rose from 6% to 14%.

See the [full results](analysis/results.md) for breakdowns, paired tests, and figures.

## Repository layout

```
turnguard/            Python package
  protocol.py         agent system prompt, ACTION/FINAL parser (protocol v2.4)
  tools.py            deterministic simulated tools (email, calendar, docs, tickets, web, code)
  harness.py          batched multi-turn episode runner
  oracles.py          task / injection / relay / authorized checks (grading c8-v1)
  metrics.py          metric definitions, Wilson CIs, paired tests, scenario bootstrap
  data/               CuratorKIT data builder, teacher client, cross-check, preference pairs
  arms/               A1 prompt defence, A2 steering (SafeTune CAA), A4 training (SafeTune CST)
  evals/              XSTest and Global-MMLU side evaluations (AuditKIT scoring)
  auditkit_bridge.py  TurnGuard traces -> AuditKIT agent_eval episodes
  cli.py              python -m turnguard <command>
configs/              models, arms (incl. the A1 prompt), data builder, evals
docs/protocol.md      the agent protocol every arm is evaluated with
scripts/              modal_run.py (GPU jobs), results_report.py, mmlu_lenient.py, hf_publish.py
analysis/             generated results tables and figures
tests/                CPU-only tests (stub model, no downloads)
notes/                design notes per development stage
```

Download the datasets, run logs, and adapters from the Hugging Face collections above.

## Setup

From the repository root, create a Python 3.12 environment and run the CPU tests. The tests use a stub model, so you can run them before downloading weights. On a GPU machine, install `".[dev,data,eval,gpu]"` instead.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev,data,eval]"
make test
```

- To download Aya Expanse 8B, accept its licence on Hugging Face and use a token with read access.
- If you rebuild the data, set `CK_MODEL`, `CK_API_BASE`, and `CK_API_KEY` in `.env`. We generated the data with DeepSeek V4.1 Flash through an OpenAI-compatible endpoint, with reasoning disabled. Set `CROSSCHECK_MODEL` to use a second model family for the translation check. We used GLM-5.3.
- We ran the GPU jobs on [Modal](https://modal.com) with `scripts/modal_run.py`. The script sends the committed code, so commit any local changes you want the jobs to use. Set up Modal once with these commands.
  ```bash
  .venv/bin/modal setup
  .venv/bin/modal secret create huggingface HF_TOKEN=<token>
  ```
  Replace `<token>` with your Hugging Face token in your terminal. Keep the token out of shared logs.

## Reproduce the results

These are the commands we used to produce the published artifacts. The L40S runs cost about $1.95 per hour, and we trained the DPO adapter on an H100. Replace placeholders such as `<timestamp>` and `<A4 train run>` with your own run identifiers.

### Start with the published data

```bash
.venv/bin/hf download SujayAgrawal/turnguard-bench --repo-type dataset --local-dir hf_bench
mkdir -p data/v1 data/pairs
for s in train dev test test_cueflip; do cp -r hf_bench/$s data/v1/; done
.venv/bin/hf download SujayAgrawal/turnguard-dpo-pairs --repo-type dataset --local-dir data/pairs/train
```

This downloads the exact frozen test set and the preference pairs. Look for the `FROZEN` marker and checksums in `data/v1/test`. You can skip step 1. Run step 2 if you plan to tune the steering configuration in step 4.

### Or rebuild the data

```bash
.venv/bin/python -m turnguard build-data --split train --teacher litellm
.venv/bin/python -m turnguard build-data --split dev   --teacher litellm
.venv/bin/python -m turnguard build-data --split test  --teacher litellm
.venv/bin/python -m turnguard build-data --split test  --teacher litellm --cue-flip
.venv/bin/python -m turnguard audit-cues --split test
.venv/bin/python -m turnguard crosscheck --split test
.venv/bin/python -m turnguard crosscheck --split test --filler
.venv/bin/python -m turnguard crosscheck --split train
.venv/bin/python -m turnguard crosscheck --split dev
.venv/bin/python -m turnguard drop-episodes --split test --ids-file data/crosscheck/test/drop_test.jsonl
.venv/bin/python -m turnguard drop-episodes --split test_cueflip --ids-file data/crosscheck/test/drop_cueflip.txt --reason "confirmed translation problem"
.venv/bin/python -m turnguard freeze --split test
```

The `crosscheck` commands have GLM-5.3 review every translation and every filler block; train and dev were reviewed the same way as test. The rebuild is deterministic with the original teacher outputs in `data/cache/`. Fresh generation can produce different episodes. Use the Hugging Face download for the exact frozen test set.

### 1. Run the baseline on train and build preference pairs

We split the train run into two shards. Each took about 1.5 hours on an L40S.

```bash
for k in 1 2; do
  .venv/bin/modal run --detach scripts/modal_run.py --task split --split train --shard $k/2 \
      --model headline --gpu L40S --background
done
```

After both shards finish, download each run with `.venv/bin/modal volume get turnguard-hf-cache runs/<run> runs/`, then build the pairs:

```bash
.venv/bin/python -m turnguard build-pairs --split train \
    --run runs/A0__CohereLabs-aya-expanse-8b__train-shard-1-of-2__* runs/A0__CohereLabs-aya-expanse-8b__train-shard-2-of-2__*
```

### 2. Run the baseline on dev

You’ll use this run to select the steering configuration in step 4.

```bash
.venv/bin/modal run --detach scripts/modal_run.py --task eval-arm --arm A0 --split dev \
    --model headline --gpu L40S --background --data-root data/v1
```

### 3. Train the DPO adapter (A4)

Training took about 21 minutes on an H100.

```bash
.venv/bin/modal run --detach scripts/modal_run.py --task train-a4 --model headline \
    --pairs data/pairs/train --gpu H100 --background
```

### 4. Tune activation steering (A2)

The three grid shards took about 50 minutes each on L40S GPUs. The merge runs on CPU.

```bash
for g in 1 2 3; do
  .venv/bin/modal run --detach scripts/modal_run.py --task steer-tune --model headline --gpu L40S \
      --background --grid-shard $g/3 --pairs data/pairs/train --dev-dir data/v1/dev \
      --baseline-run runs/A0__CohereLabs-aya-expanse-8b__dev__<timestamp>
done
.venv/bin/modal run scripts/modal_run.py --task steer-merge --grid-runs <shard1>,<shard2>,<shard3>
```

### 5. Evaluate on the frozen test and cue-flip splits

Each configuration took about three hours on an L40S across both splits.

```bash
.venv/bin/modal run --detach scripts/modal_run.py --task eval-suite --arm A0 --model headline --gpu L40S --background --data-root data/v1
.venv/bin/modal run --detach scripts/modal_run.py --task eval-suite --arm A1 --model headline --gpu L40S --background --data-root data/v1
.venv/bin/modal run --detach scripts/modal_run.py --task eval-suite --arm A2 --steer-run <merged A2 run> --model headline --gpu L40S --background --data-root data/v1
.venv/bin/modal run --detach scripts/modal_run.py --task eval-suite --arm A4 --adapter-run <A4 train run> --model headline --gpu L40S --background --data-root data/v1
```

`--task eval-arm --split test` (or `--split test_cueflip`) runs a single split.

### 6. Run XSTest and Global-MMLU-Lite

Generating responses took about 18 minutes per configuration on an L40S. We scored the responses locally afterward.

```bash
.venv/bin/modal run --detach scripts/modal_run.py --task eval-side --arm A4 --adapter-run <A4 train run> --model headline --gpu L40S --background
```

Download the side run, then judge XSTest locally with the LLM judge (temperature 0) and score MMLU offline:

```bash
.venv/bin/python -m turnguard judge-side --run runs/A4__CohereLabs-aya-expanse-8b__side__<timestamp> > runs/judge_A4.log
```

For the other configurations, use `--arm A0`, `--arm A1`, or `--arm A2 --steer-run <merged A2 run>` in the GPU command.

### 7. Generate the tables, statistics, and figures

This takes about a minute on CPU.

`results_report.py` writes `analysis/results.md`, `results.json` and the `fig_*.png` figures. `mmlu_lenient.py` applies the post-hoc lenient MMLU parser to every arm. `compare` runs the paired tests between any two runs.

```bash
.venv/bin/python scripts/results_report.py
.venv/bin/python scripts/mmlu_lenient.py
.venv/bin/python -m turnguard compare <run A> <run B> --specs-a data/v1/test --specs-b data/v1/test
```

### 8. Publish (optional)

`stage` builds the upload folders, redacts responses to unsafe XSTest prompts and scans for secrets; `upload` pushes them and creates the collections.

```bash
.venv/bin/python scripts/hf_publish.py stage
.venv/bin/python scripts/hf_publish.py upload
```

## Evaluation settings

- We used greedy decoding with at most 256 new tokens per step and six steps per user turn. Every configuration used the tokenizer’s chat template, left padding, and tool results delivered as user messages.
- We used seed 1234 throughout. The data loaders block the test split during training, steering-vector extraction, and tuning.
- The tools run in a sandbox and never touch the network. Addresses and domains use sandbox values such as `*.invalid` and `example.*`.

## What it cost

We used about $60 in Modal GPU credits to train the adapter, tune steering, evaluate all four configurations on test and cue-flip, and run the side evaluations. We used free cloud credits for the DeepSeek endpoint that generated the data.

## License

- Code: MIT.
- Adapter, vectors, data and logs: CC-BY-NC-4.0 (inherited from Aya Expanse 8B; also subject to the Cohere Labs Acceptable Use Policy).
