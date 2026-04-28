# Rec System

This repo now contains a small but clean experiment stack for the synthetic recommendation dataset in this directory.

## Layout

- `actual.csv` / `actual_matrix.csv`: full latent ground truth for every user-item pair
- `reported.csv` / `reported_matrix.csv`: partially observed ratings available to the recommender
- `recsys/`: package with loaders, metrics, models, and training entrypoints
- `recsys/federated/`: federated MovieLens benchmark, attack generation, and evaluation code
- `tests/`: smoke tests for data loading and model training

## Implemented Models

- `bpr_mf`: matrix factorization trained with a BPR pairwise ranking loss
- `two_tower`: user-history tower plus item tower, still trained with the same pairwise loss
- `neural_cf`: an MLP scorer over user/item embeddings for a stronger nonlinear baseline

## Evaluation Protocol

Training only uses `reported.csv`.

Evaluation ranks each user's three unobserved items and compares those rankings against the hidden truth in `actual.csv`. The main metrics are:

- `top1_accuracy`: whether the top unseen recommendation is one of the user's best unseen items
- `ndcg@3`: ranking quality on the three hidden items
- `pairwise_accuracy`: agreement with pairwise order among the hidden items
- `mrr`: reciprocal rank of the best hidden item

The CLI also reports a `popularity` baseline for sanity checking.

## Federated MovieLens Benchmark

The repo also includes a separate federated benchmark for real MovieLens data:

- `BPR MF` only
- shard-local user embeddings
- server-global item embeddings and item bias
- synchronous round-based aggregation
- one malicious shard that injects synthetic users for a segment-targeted push attack

Supported dataset variants:

- `ml-1m`
- `ml-10m`
- `ml-25m`
- `ml-32m`

Recommended “full MovieLens” target:

- `ml-32m` if you want the largest standard MovieLens ratings benchmark supported by this repo
- `ml-25m` if you already have that dataset extracted or want a slightly lighter run

Expected extracted files:

- `ml-1m` / `ml-10m`: `movies.dat`, `ratings.dat`
- `ml-25m` / `ml-32m`: `movies.csv`, `ratings.csv`

The `--data-root` flag can point either to:

- the parent directory containing `ml-32m/` or `ml-25m/`
- the extracted dataset directory itself

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run A Single Model

```bash
python -m recsys.train --model bpr_mf --epochs 200 --device cuda
```

## Compare All Models

```bash
python -m recsys.train --compare-all --epochs 200 --device cuda --output artifacts/model_report.json
```

The generated JSON report is useful for future poisoning/robustness experiments because it already includes overall metrics, per-profile metrics, and the sampled training loss curve.

## Run The Federated Benchmark

```bash
python -m recsys.federated.benchmark \
  --data-root data/movielens \
  --dataset-variant ml-25m \
  --num-shards 8 \
  --malicious-shard-id 0 \
  --target-genre Action \
  --local-epochs 1 \
  --federated-rounds 10 \
  --batch-size 8192 \
  --embedding-dim 64 \
  --eval-every 2 \
  --eval-batch-size 8192 \
  --num-gpus 8 \
  --max-positive-interactions 200 \
  --attack-filler-items-per-user 8 \
  --attack-neutral-items-per-user 0 \
  --attack-target-weight 20 \
  --attack-filler-weight 0.5 \
  --attack-budgets 0,0.01,0.05,0.1 \
  --top-k 10 \
  --num-eval-negatives 100 \
  --output artifacts/federated_movielens_report.json
```

The federated CLI now shows progress for both the MovieLens load/preprocess phase and the training phase. You will see bars for reading the ratings file, filtering/splitting users, the clean target-selection pass, the clean baseline, each attack-budget run, and shard-by-shard progress within each federated round. If you want to suppress that output, add `--disable-progress`.

When you do not pass `--target-item-id`, the benchmark now uses a heuristic clean-model target selector instead of blindly picking the least popular item in the genre. It ranks candidates by a vulnerability score that favors:

- larger reachable benign audiences
- stronger support from the target segment's existing train-history items
- smaller average margin to the current `top-k` boundary
- lower clean `target_hitrate@k`, so already-exposed items are penalized

Performance-oriented knobs:

- `--eval-every`: only run held-out evaluation every N federated rounds; the final round is always evaluated
- `--eval-batch-size`: user batch size for GPU-batched evaluation
- `--num-gpus 8`: convenience shorthand for using `cuda:0` through `cuda:7` when those devices are visible
- `--devices cuda:0,cuda:1,...`: explicit device list; use this instead of `--num-gpus` if you want a custom subset/order
- `--max-positive-interactions`: drop long-history users before train/val/test splitting to reduce padded shard tensors; useful when a few extreme users dominate GPU memory
- `--save-attack-bundle path`: save the exact shard partition, target selection, attack setup, and cached clean/naive attack results for later defense sweeps
- `--load-attack-bundle path`: reload that exact attack setup; if the training signature still matches, the benchmark reuses the cached clean and naive attack results and only reruns defended attacks
- attack-strength knobs: `--attack-target-weight`, `--attack-filler-weight`, `--attack-filler-items-per-user`, `--attack-neutral-items-per-user`, and the candidate-pool flags
- defense knobs: `--defense-method focus_clip_mean` or `focus_clip_trimmed_mean` to downweight shards whose item-level update mass is overly concentrated in their top pushed items; tune with `--focus-top-k` and `--focus-factor`

## Run On The Full MovieLens Dataset

Example using the full `ml-32m` release from GroupLens:

```bash
mkdir -p data/movielens
cd data/movielens
curl -O https://files.grouplens.org/datasets/movielens/ml-32m.zip
unzip ml-32m.zip
cd /path/to/rec_system
```

This should leave you with:

```text
data/movielens/
  ml-32m/
    movies.csv
    ratings.csv
    README.html
    links.csv
    tags.csv
```

Then run:

```bash
python -m recsys.federated.benchmark \
  --data-root data/movielens \
  --dataset-variant ml-32m \
  --num-shards 16 \
  --malicious-shard-id 0 \
  --target-genre Action \
  --local-epochs 1 \
  --federated-rounds 10 \
  --batch-size 8192 \
  --embedding-dim 64 \
  --eval-every 2 \
  --eval-batch-size 4096 \
  --num-gpus 8 \
  --max-positive-interactions 200 \
  --attack-filler-items-per-user 8 \
  --attack-neutral-items-per-user 0 \
  --attack-target-weight 20 \
  --attack-filler-weight 0.5 \
  --attack-budgets 0,0.01,0.05,0.1 \
  --top-k 10 \
  --num-eval-negatives 200 \
  --device cuda \
  --output artifacts/federated_movielens_ml32m_report.json
```

If you want to point directly at the extracted dataset folder instead of its parent directory, this also works:

```bash
python -m recsys.federated.benchmark \
  --data-root data/movielens/ml-32m \
  --dataset-variant ml-32m \
  --num-shards 16 \
  --malicious-shard-id 0 \
  --target-genre Action \
  --local-epochs 1 \
  --federated-rounds 10 \
  --batch-size 8192 \
  --embedding-dim 64 \
  --eval-every 2 \
  --eval-batch-size 4096 \
  --num-gpus 8 \
  --max-positive-interactions 200 \
  --attack-filler-items-per-user 8 \
  --attack-neutral-items-per-user 0 \
  --attack-target-weight 20 \
  --attack-filler-weight 0.5 \
  --attack-budgets 0,0.01,0.05,0.1 \
  --top-k 10 \
  --num-eval-negatives 200 \
  --device cuda \
  --output artifacts/federated_movielens_ml32m_report.json
```

Notes for full-dataset runs:

- the benchmark keeps only users with at least `3` positive interactions at or above the `--min-positive-rating` threshold
- if a few extreme users are blowing up shard memory, set `--max-positive-interactions` to something like `200` or `300` to drop those outliers at load time
- the current default positive threshold is `4.0`
- larger `--num-shards`, `--batch-size`, and `--num-eval-negatives` will increase runtime noticeably
- use `--eval-every` greater than `1` if you want less frequent held-out evaluation during long runs
- use `--num-gpus 8` if the node exposes eight visible CUDA devices; otherwise lower that count or pass an explicit `--devices` list
- for a more aggressive data-poisoning baseline, increase `--attack-target-weight`, keep `--attack-neutral-items-per-user 0`, and raise `--attack-budgets`
- for attacks that spread the push across 2-3 items, prefer `--defense-method focus_clip_mean` over plain norm clipping
- start with fewer `--federated-rounds` if you are validating the pipeline before a longer benchmark
- after `pip install -e .`, the benchmark uses `tqdm` progress bars automatically; before refreshing the environment it falls back to plain-text round progress

The federated JSON report contains:

- clean baseline metrics and per-round traces
- attacked metrics for each attack budget
- target-segment `target_hitrate@3`, `target_hitrate@5`, `target_hitrate@10`, and `target_mean_rank`
- target-segment uplift versus clean
- the chosen target item metadata

## Exact Defense Sweeps

To save the exact attack setup once and reuse it for later defense runs:

```bash
python -m recsys.federated.benchmark \
  ... \
  --defense-method none \
  --save-attack-bundle artifacts/ml25m_attack_bundle.json \
  --output artifacts/ml25m_attack_baseline.json
```

Then sweep a defense against that exact same shard partition, target item, attack budgets, and cached naive attack results:

```bash
python -m recsys.federated.benchmark \
  ... \
  --load-attack-bundle artifacts/ml25m_attack_bundle.json \
  --defense-method focus_clip_mean \
  --focus-top-k 3 \
  --focus-factor 2.0 \
  --output artifacts/ml25m_focus_defense.json
```

This does not reuse the final attacked model checkpoint, because exact defended runs need to retrain from round 0 under the new aggregation rule. Instead, it reuses the exact attack setup and, when the training signature matches, skips the expensive clean target-selection and naive attack reruns.
