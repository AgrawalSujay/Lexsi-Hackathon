# C9 Stop 3: sharded tuning, volume selection and final-arm evaluation

Rebased `feat/arms-a1-a2` onto the requested **a62f653**. Resolved conflicts by retaining main's existing split `--arm`/`--adapter-run` functionality and C9 steering. Evaluation tasks reuse that split command and its existing loader, frozen-release verification, stable episode/twin sharding, generator and metrics. Protocol v2.4 is unchanged. `harden.py` and A4 checkpoint helpers remain intact; checkpoint/resume regressions pass.

## Grid sharding and merge

`steer-tune --gpu H100` uses the same existing GPU option dispatch as L40S. `--grid-shard i/k` supports k=1,2,3, assigning `GRID[i-1::k]`; three shards each run two points. Every shard independently extracts the four candidate layer vectors from the same train pairs, and uses the same 50 injected + 50 clean + 50 authorized dev IDs and current-regraded recorded A0 snapshot. A0 is never generated. Partial shards do not select a winner or replace the volume's working selection. Run and point names include unique nonces, preventing concurrent containers from overwriting one another.

The new **CPU-only** `steer-merge` task reloads the volume and reads one to three completed parent shard run directories. It requires all six distinct points, the expected shard partition, no duplicate shards, and matching:

- pairs hash, model ID and **resolved model revision** (`model.config._commit_hash`);
- candidate layers, completion window, pooling/index/direction/normalization convention;
- extraction **code commit**, with committed-code provenance;
- dev release fingerprint and ordered selected IDs, composition, grading/protocol/decoding;
- recorded A0 source hashes, baseline rates and the predeclared selection rule.

**Cross-shard vectors are not compared by bytes or numerical equality.** Their metadata establishes input equivalence; each vector's own checksum detects corruption and binds it to the points it actually evaluated. No cross-vector cosine comparison is performed. Thus harmless GPU arithmetic differences do not prevent merging.

The original guardrail/ASR selection rule is reused unchanged. The merged run copies **the winning point's own shard vector bytes and JSON sidecar**, preserves that checksum in its selected A2 config, and records `vector_source_run`. It never substitutes the first shard's vector or re-averages vectors. Inference checks the loaded base model's resolved revision against the vector's provenance.

## Owner commands (not executed here)

Run from `/Users/sujayagrawal/Downloads/turnguard-c9`. This launches three background H100 containers; use `1/2` and `2/2` instead for two containers, or omit `--grid-shard` for one full grid.

```bash
for grid_index in 1 2 3
do
  modal run --detach scripts/modal_run.py --task steer-tune --model headline --gpu H100 --background \
    --grid-shard "$grid_index/3" \
    --pairs ../turnguard/data/pairs/train \
    --dev-dir ../turnguard/data/v1/dev \
    --baseline-run ../turnguard/runs/A0__CohereLabs-aya-expanse-8b__dev__20261109-0802
done
```

After all shards finish, list the volume and pass their **parent** `steer-tune-shard-...` run names to the CPU merge (not the individual `dev-steer-L...` point runs):

```bash
modal volume ls turnguard-hf-cache runs
modal run scripts/modal_run.py --task steer-merge --grid-runs 'SHARD_1_RUN,SHARD_2_RUN,SHARD_3_RUN'
```

It prints the new merged run name, usable as `--steer-run`. No `--gpu` option is forwarded to the merge function. Local equivalent after downloading the shard directories:

```bash
python -m turnguard steer-merge --runs runs/SHARD_1_RUN runs/SHARD_2_RUN runs/SHARD_3_RUN
```

## Volume arm loading and final evaluations

`eval-arm --arm A0|A1|A2|A4 --split dev|test|test_cueflip` delegates to the existing split runner. `eval-suite` runs **test then test_cueflip** for one arm in one container/background job, applying the same `--shard i/k` independently to each release. Both test releases must pass frozen verification locally and again inside the container. Separate raw records, configs, metrics and timing are retained for each split. A failure in one suite step still lets the other split run; both exit statuses are reported. Suite/eval-arm jobs have a six-hour timeout.

`--steer-run <volume-run-name>` loads a merged or complete unsharded A2 selection, its model-specific vector and checksum sidecar from `runs/<name>/steering/A2/`. Unmerged partial shards are refused. The vector checksum, model and selected layer/alpha must agree before generation. The config is rewritten to the staged vector's actual container path.

`--adapter-run <volume-run-name>` reuses main's A4 mechanism, selecting `runs/<name>/adapters/A4/<model>/` specifically. It checks the adapter configuration/weights and declared base model when present, stages the files and writes the actual adapter path. This also fixes Tiny Aya evaluation accidentally pointing to the headline adapter. Run names cannot escape the volume root and symlink artifacts are refused. Each evaluation run includes `ARM_ARTIFACT.json` with its source run and staged file checksums. The existing `adapter:` / `intervention_factory:` APIs remain the model-loading path.

One background command per final arm (replace the run-name placeholders). `--data-root` is a **read-only upload source**; nothing is written to main's data directory. These commands refuse unfinished or unfrozen test releases.

```bash
modal run --detach scripts/modal_run.py --task eval-suite --arm A0 --model headline --gpu H100 --background --data-root ../turnguard/data/v1
modal run --detach scripts/modal_run.py --task eval-suite --arm A1 --model headline --gpu H100 --background --data-root ../turnguard/data/v1
modal run --detach scripts/modal_run.py --task eval-suite --arm A2 --steer-run MERGED_RUN --model headline --gpu H100 --background --data-root ../turnguard/data/v1
modal run --detach scripts/modal_run.py --task eval-suite --arm A4 --adapter-run TRAINING_RUN --model headline --gpu H100 --background --data-root ../turnguard/data/v1
```

For two containers per arm, issue the same suite command with `--shard 1/2` and `--shard 2/2`. To run one split only, use `--task eval-arm --split test` (or dev/test_cueflip) with the same arm/artifact/background/shard flags. Printed results retain separate execution, relay-only and combined ASR, clean success, authorized metrics, ungraded counts, and language breakdowns.

## Timing and verification

Independent shard extraction duplicates calibration work. The existing measured L40S reference suggests ~38 minutes for each two-point shard plus the previously estimated 10–40 minute extraction: roughly **50–80 minutes wall time with three L40S containers**, before startup and response-length variation. It uses three extractions rather than one. H100 is supported but has **not been benchmarked here**; no unmeasured speedup or new GPU result is claimed. The merge consumes CPU only. Actual per-job/point elapsed times and extraction token counts are recorded for later calibration.

**807 tests passed, 19 subtests passed in 126.77 seconds**; 99 focused regressions passed. Ruff check/format and git diff checks passed. Tests use fake models, NumPy tensors, tiny local releases, fake Modal calls and fake adapter/vector bytes: no test uses network or GPU. They cover disjoint/complete grid sharding, two-point shard execution, changed input provenance and revisions, different vector bytes across shards, winning-vector retention, CPU volume merge, H100/background dispatch, frozen suite transport, all four arms, model-specific adapter staging, vector mismatch refusal, suite failure persistence, and A4 checkpoint behavior.

The real-data **CPU dry run** of shard 2/3 passed: points `(14, 8)` and `(22, 4)`, composition 50/50/50, 1,032 train pairs, zero A0 generation. No GPU extraction, tuning or final evaluation was run, and original main datasets/runs were read only.
