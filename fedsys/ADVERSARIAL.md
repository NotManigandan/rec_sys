# ADVERSARIAL.md — Federated Learning Attack & Defense Layer

This document covers the complete adversarial module (`fedsys/adversarial/`) including the theory behind each mechanism, the code design, all configurable parameters, and step-by-step usage instructions for both attack-only, defense-only, and combined scenarios.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Module Structure](#2-module-structure)
3. [The Attack — Data Poisoning](#3-the-attack--data-poisoning)
   - 3.1 [Threat Model](#31-threat-model)
   - 3.2 [How a Poisoned Profile is Built](#32-how-a-poisoned-profile-is-built)
   - 3.3 [Embedding-Table Design Decision](#33-embedding-table-design-decision)
   - 3.4 [Target Selection](#34-target-selection)
   - 3.5 [AttackConfig Parameters](#35-attackconfig-parameters)
   - 3.6 [Code Walkthrough — `poison.py`](#36-code-walkthrough--poisonpy)
   - 3.7 [Code Walkthrough — `target.py`](#37-code-walkthrough--targetpy)
4. [The Defense — Robust Aggregation](#4-the-defense--robust-aggregation)
   - 4.1 [Why Plain FedAvg is Vulnerable](#41-why-plain-fedavg-is-vulnerable)
   - 4.2 [Defense Methods Explained](#42-defense-methods-explained)
   - 4.3 [Focus Score — Detecting Poisoned Updates](#43-focus-score--detecting-poisoned-updates)
   - 4.4 [Suppressed-Shard Detection](#44-suppressed-shard-detection)
   - 4.5 [RobustAggregator Parameters](#45-robustaggregator-parameters)
   - 4.6 [Code Walkthrough — `aggregation.py`](#46-code-walkthrough--aggregationpy)
5. [Adversarial Evaluation](#5-adversarial-evaluation)
   - 5.1 [Metrics Explained](#51-metrics-explained)
   - 5.2 [Code Walkthrough — `eval.py`](#52-code-walkthrough--evalpy)
6. [How to Run — CLI Quick-Start](#6-how-to-run--cli-quick-start)
   - 6.1 [Attack Only (no defense)](#61-attack-only-no-defense)
   - 6.2 [Defense Only (no malicious node)](#62-defense-only-no-malicious-node)
   - 6.3 [Attack + Defense Together](#63-attack--defense-together)
   - 6.4 [All Available Flags](#64-all-available-flags)
7. [Programmatic API](#7-programmatic-api)
8. [Tests](#8-tests)
9. [Design Trade-offs and Limitations](#9-design-trade-offs-and-limitations)

---

## 1. Overview

The adversarial layer adds two independent capabilities on top of the core fedsys gRPC Federated Learning framework:

| Capability | Where it runs | Purpose |
|---|---|---|
| **Attack** | Training node (malicious) | Inject synthetic user profiles to push a target item into top-K recommendations for a victim user segment |
| **Defense** | Coordinator (aggregation step) | Detect and dampen the influence of poisoned gradient updates using robust aggregation |
| **Adversarial Eval** | Coordinator (evaluation step) | Measure whether the attack is succeeding by tracking the target item's rank in the victim segment |

The three capabilities are **fully independent** — you can run any combination:
- Attack only → observe attack impact with no defense
- Defense only → harden a clean federated system against future attacks
- Both → benchmark defense effectiveness against a live attacker
- Neither → standard FL behavior (equivalent to the original `FedAvgAggregator`)

---

## 2. Module Structure

```
fedsys/adversarial/
├── __init__.py                      # Package-level docstring
├── eval.py                          # Target-exposure evaluation metrics
├── attack/
│   ├── __init__.py                  # Re-exports key symbols
│   ├── poison.py                    # PoisonedBPRPairDataset + AttackConfig
│   └── target.py                    # Target item / genre selection helpers
└── defense/
    ├── __init__.py                  # Re-exports RobustAggregator
    └── aggregation.py               # All 5 robust aggregation methods

tests/adversarial/
├── test_poison.py                   # Unit tests for poison.py
├── test_defense.py                  # Unit tests for aggregation.py
├── test_target.py                   # Unit tests for target.py
├── test_eval.py                     # Unit tests for eval.py
└── test_integration.py              # Full gRPC: 1 coord + 1 clean + 1 malicious node
```

**Integration points in the existing codebase:**

| Existing file | What was added |
|---|---|
| `fedsys/config.py` | `CoordinatorConfig`: defense fields; `NodeConfig`: attack fields |
| `fedsys/coordinator/aggregator.py` | `build_aggregator()` factory function |
| `fedsys/coordinator/server.py` | Uses `build_aggregator()`; runs adversarial eval each round |
| `scripts/run_coordinator.py` | `--defense`, `--adv-target-item`, `--adv-target-genre` flags |
| `scripts/run_node.py` | `--attack` and all `--attack-*` flags |

---

## 3. The Attack — Data Poisoning

### 3.1 Threat Model

The attacker controls **one or more training nodes** out of the N total participants. The attacker's goal is a **targeted push attack**: make a specific item (the *target item*) appear in the top-K recommendation list for users who belong to the *target genre segment* (users whose dominant genre is the target genre), without noticeably degrading overall recommendation quality for the rest of the system.

This is a **white-box** attack with respect to the local training procedure — the attacker knows the model architecture and loss function — but a **grey-box** attack with respect to the global model (they receive the global model each round just like any honest node).

### 3.2 How a Poisoned Profile is Built

Each round, before local training starts, the malicious node generates:

```
N_synth = ceil(|shard_users| × attack_budget)   synthetic user profiles
```

Each synthetic profile contains:

| Item type | Source | Rating | Purpose |
|---|---|---|---|
| **Target item** × 1 | Specified by attacker | 5.0 × `target_weight` | Direct positive signal for the target item |
| **Filler items** × F | Same genre as target, most popular | 4.0 | Makes profile look like a plausible genre fan |
| **Neutral items** × M | Unrelated genre (e.g., Comedy) | 3.5 | Provides statistical cover, reduces anomaly detection risk |

The BPR pairwise loss on these profiles creates gradients that push the **item embedding** of the target item toward the synthetic-user direction. Since item embeddings are shared globally, this influences the model's recommendations for **all real users** in the victim segment — not just the synthetic ones.

### 3.3 Embedding-Table Design Decision

In the original `recsys` implementation, user embeddings are **local to each shard** (stored in `ServerState`), so synthetic users with indices beyond `num_real_users` never cause index-out-of-range errors. In fedsys, both user *and* item embeddings live in a single globally shared `BPRModel` whose `nn.Embedding` sizes are fixed at initialisation time.

**Solution:** The coordinator reserves a block of user indices above the real-user range for synthetic profiles:

```
total_embedding_rows = num_real_users + max_synthetic_users_per_coord
```

- The coordinator calls `poisoned_num_users()` at startup and uses the result for `ModelConfig.num_users`.
- Every node (clean and malicious) sees the same embedding-table shape — no state-dict mismatch.
- Only the malicious node ever generates training samples that index into the reserved block.
- Clean nodes never touch those rows; the corresponding embedding vectors simply receive zero gradient.

### 3.4 Target Selection

Two strategies are provided in `target.py`:

**Strategy 1 — Simple (heuristic):** `select_target_item()`

Selects the item with the **lowest training-popularity** among items in the target genre that have at least `min_popularity` interactions. A low-popularity item is nearly invisible in current recommendations, so any rank improvement is clearly attributable to the attack.

**Strategy 2 — Model-based (vulnerability scoring):** `select_target_from_clean_model()`

Evaluates each candidate item by computing a **vulnerability score** against the current global model:

```
vulnerability = fraction of segment users for whom the item is
                in rank positions [k, 2k)  (close but not yet in top-k)
```

An item with a high vulnerability score is on the cusp of being recommended — a small push can cross the threshold. This is the most efficient target for the attacker.

**Automatic genre selection:** `choose_target_genre()`

Scores all genres by `1 / (avg_item_popularity + 1)`, filtered to genres with at least `min_segment_users` dominant-genre users. Picks the genre whose items are least popular overall — the easiest genre to attack.

### 3.5 AttackConfig Parameters

```python
@dataclass
class AttackConfig:
    enabled: bool = False

    # What to attack
    target_item_index: int = -1    # Contiguous item index (0-based) to push.
                                   # Set to -1 then call select_target_item().
    target_genre: str = ""         # Genre the target item belongs to.

    # Budget
    attack_budget: float = 0.30    # Fraction of shard users to add as fakes.
                                   # 0.30 = 30 % extra synthetic profiles.

    # Profile shape
    num_filler_items: int = 30     # Filler items from same genre.
    num_neutral_items: int = 20    # Neutral items from a different genre.
    filler_from_top: bool = True   # If True, take most-popular filler items.
    neutral_genre: str = "Comedy"  # Genre used for neutral items.
    target_weight: float = 1.0     # Multiplier on the target item's rating.
                                   # Higher = stronger gradient push.

    # Embedding-table sizing (must match coordinator)
    max_synthetic_users_per_coord: int = 200

    seed: int = 42                 # RNG seed for reproducibility.
```

### 3.6 Code Walkthrough — `poison.py`

#### `poisoned_num_users(real_num_users, attack_cfg) -> int`
Returns `real_num_users + attack_cfg.max_synthetic_users_per_coord` when the attack is enabled, otherwise returns `real_num_users` unchanged. The coordinator and all nodes must use this value for `ModelConfig.num_users`.

#### `_build_synthetic_profiles(dataset, num_profiles, attack_cfg, base_user_index) -> List[UserSplit]`
Creates `num_profiles` synthetic `UserSplit` objects. Each profile:
1. Picks filler items from `dataset.items_by_genre[target_genre]`, sorted by `train_item_popularity` descending (if `filler_from_top=True`).
2. Picks neutral items from `dataset.items_by_genre[neutral_genre]`, shuffled.
3. Assembles `train_ratings` with the target item rated `5.0 × target_weight`, fillers at 4.0, neutrals at 3.5.
4. Sets `val_item = test_item = target_item_index` so evaluators see the target as the held-out item.

#### `PoisonedBPRPairDataset`
A `torch.utils.data.Dataset` that concatenates:
- The **clean** `BPRPairDataset` (real users, real training data)
- **Synthetic** BPR triplets `(synthetic_user_id, target_item, random_neg_item)`

The first `len(clean_dataset)` indices return clean samples; indices beyond that return synthetic ones. Negative items are sampled online at `__getitem__` time (same as `BPRPairDataset`) with rejection sampling against the profile's `known_items`.

#### `build_poisoned_dataloader(dataset, shard_user_indices, attack_cfg, model_num_users, ...) -> DataLoader`
The main entry point for the malicious node. When `attack_cfg.enabled=False` it returns a plain clean dataloader — so the same call works for both honest and malicious nodes.

---

### 3.7 Code Walkthrough — `target.py`

#### `benign_segment_users(dataset, target_genre) -> List[int]`
Returns all user indices whose `dominant_genre_by_user` equals `target_genre`. These are the "victim" users the attack is trying to influence.

#### `select_target_item(dataset, target_genre, min_popularity, max_popularity) -> int`
Filters items in `target_genre` by popularity bounds, sorts ascending, returns the item with the minimum popularity count. Fast, no model required.

#### `select_target_from_clean_model(model, dataset, target_genre, k, ...) -> int`
1. Filters candidates by popularity bounds and takes the `top_n_candidates` least popular.
2. Calls `_score_item_vulnerability()` for each candidate.
3. Returns the item with the highest vulnerability score.

#### `_score_item_vulnerability(model, item_index, segment_users, ...) -> float`
Scores all `num_items` items for each segment user in batches. Computes the rank of `item_index`. Returns the fraction of users where `k <= rank < 2k`.

#### `choose_target_genre(dataset, min_segment_users, exclude_genres) -> str`
Calls `_genre_vulnerability_score()` for every genre and returns the winner.

#### `_genre_vulnerability_score(dataset, genre, min_segment_users) -> float`
Returns `1 / (avg_item_popularity + 1)` for the genre, or `-inf` if too few segment users.

---

## 4. The Defense — Robust Aggregation

### 4.1 Why Plain FedAvg is Vulnerable

In standard Federated Averaging the coordinator computes a weighted mean of all received updates:

```
new_global = Σ (n_i / N_total) × local_state_i
```

A poisoned update that concentrates large gradients on the target item's embedding vector is treated the same as any other update. With enough attack budget, the target item embedding is shifted far enough to appear in top-K for the victim segment.

### 4.2 Defense Methods Explained

All five methods operate on the **delta space**: `Δ_i = local_state_i - global_state`. Defenses constrain the magnitude and direction of these deltas before combining them.

#### `"none"` — Plain FedAvg
No defense. Identical to the original `FedAvgAggregator`. Use this as the baseline to measure attack impact.

#### `"clip_mean"` — Norm Clipping + Mean
For each client update:
```
if ||Δ_i||_2 > θ:
    Δ_i ← Δ_i × (θ / ||Δ_i||_2)
```
Then take the (sample-count-weighted) mean of the clipped deltas. The threshold `θ` (`--defense-clip-thresh`, default 5.0) limits how much any single node can shift the global model in one round.

**Good for:** Bounding the absolute influence of any one node regardless of attack type.

**Limitation:** Does not distinguish between an attacker sending a large focused update and an honest node that trained on a large, noisy shard.

#### `"clip_trimmed_mean"` — Norm Clipping + Trimmed Mean
Same norm clipping as above, but instead of a weighted mean, applies **coordinate-wise trimmed mean**: for each scalar parameter, sort the values across all clients, remove the top and bottom `trim_fraction` fraction, and average the rest.

```
For each parameter p:
    sort values [v_1, ..., v_n] across clients
    discard bottom k = floor(n × trim_fraction) and top k values
    result[p] = mean(remaining values)
```

**Good for:** Removing outlier values at each coordinate independently, even when the attack only corrupts a specific region of the parameter space (e.g., a single item embedding row).

#### `"focus_clip_mean"` — Focus Score + Norm Clip + Mean
Before clipping, computes a **focus score** for each update (see Section 4.3). Updates with high focus scores (concentrated on a small set of item embeddings — the attack signature) are **down-weighted**:

```
weight_i = (1 - focus_score_i) × n_samples_i
```

After re-normalising weights, applies norm clipping, then takes the weighted mean.

**Good for:** Directly targeting the concentrated-gradient pattern that characterises targeted push attacks.

#### `"focus_clip_trimmed_mean"` — Focus Score + Norm Clip + Trimmed Mean
Combines all three mechanisms: focus-score re-weighting, then norm clipping, then coordinate-wise trimmed mean. This is the strongest defense but also the most aggressive — it can discard genuine signal from honest nodes in low-data regimes.

### 4.3 Focus Score — Detecting Poisoned Updates

A poisoned update concentrates gradient mass on just a few item embeddings (the target item and its genre neighbours), while benign updates spread gradient across the full item catalog.

The focus score measures this concentration:

```python
def _focus_score(delta, global_item_emb, k_frac=0.05):
    emb_delta = delta["item_embedding.weight"]   # shape: (num_items, dim)
    row_norms = emb_delta.norm(dim=1)            # per-item L2 norm
    total_norm = row_norms.sum() + 1e-12

    k = max(1, int(num_items × k_frac))          # top-k% items
    top_k_sum = row_norms.topk(k).values.sum()

    return top_k_sum / total_norm
```

- A **benign** update spreads gradient roughly uniformly: `focus_score ≈ k_frac` (e.g., 0.05)
- A **poisoned** update concentrates on ~1–5 items: `focus_score → 1.0`

The `k_frac` parameter (`--defense-focus-k-frac`, default 0.05) controls what "top fraction" means. Lower values make the score more sensitive to single-item concentration.

### 4.4 Suppressed-Shard Detection

When a focus-based defense is active, the coordinator automatically logs any node whose update satisfies both:
- `||Δ_i||_2 > 2 × clip_threshold` (update was heavily clipped)
- `focus_score_i >= focus_threshold` (update was concentrated on few items)

These events are logged with `"event": "DEFENSE_SUSPICIOUS_NODES"` in the telemetry JSONL and SQLite files. You can monitor them in real time with:

```bash
# Tail the coordinator's telemetry log
python -c "
import json, sys
for line in open('logs/telemetry.jsonl'):
    e = json.loads(line)
    if e.get('event') == 'DEFENSE_SUSPICIOUS_NODES':
        print(e)
"
```

### 4.5 RobustAggregator Parameters

```python
RobustAggregator(
    logger,
    method="clip_mean",     # One of DEFENSE_METHODS
    device="cpu",
    clip_threshold=5.0,     # L2 norm threshold θ for clipping
    trim_fraction=0.10,     # Fraction to trim from each tail (trimmed-mean methods)
    focus_k_frac=0.05,      # Top-item fraction for focus score (focus methods)
    focus_threshold=0.50,   # Focus score above which a node is flagged suspicious
)
```

The `aggregate()` method has the same signature as `FedAvgAggregator.aggregate()` plus an optional `global_state` keyword argument:

```python
new_state, serialized_bytes = aggregator.aggregate(
    updates,          # {node_id: serialized_state_dict_bytes}
    sample_counts,    # {node_id: num_local_samples}
    epoch,            # current FL round
    global_state=..., # current global model state_dict (required for all except "none")
)
```

### 4.6 Code Walkthrough — `aggregation.py`

#### `_delta(local, global_state) -> StateDict`
Element-wise subtraction of two state dicts. Returns `{key: local[key] - global[key]}` for all keys present in both.

#### `_l2_norm(delta) -> float`
Flattens all tensors in the delta dict and computes `||concat(values)||_2`.

#### `_scale_delta(delta, factor) -> StateDict`
Multiplies all tensors by a scalar. Used by norm clipping: `scale = min(1.0, θ / ||Δ||)`.

#### `_trimmed_mean_tensor(tensors, trim_fraction) -> Tensor`
Stacks tensors into a `(n, ...)` tensor, sorts along the client axis, removes `k = floor(n × trim_fraction)` from each tail, returns the mean of the remainder.

#### `_focus_score(delta, global_item_emb, k_frac) -> float`
Described in Section 4.3 above.

#### `_detect_suppressed_shards(norms, focus_scores, ...) -> List[str]`
Compares each node's norm and focus score against thresholds. Returns a list of suspicious node IDs and logs them to telemetry.

#### `RobustAggregator._robust_aggregate(updates, sample_counts, epoch, global_state)`
Full pipeline:
1. Deserialize all local state dicts.
2. Compute deltas vs. global state.
3. Compute L2 norms for all deltas.
4. If focus method: compute focus scores; re-weight clients; flag suspicious nodes.
5. Clip each delta to `θ`.
6. If trimmed method: apply `_trimmed_mean_tensor` per-key. Otherwise: apply `_add_deltas` with weights.
7. Reconstruct new global state: `new_state[k] = global[k] + aggregated_delta[k]`.

#### `build_aggregator(logger, defense_method, ...) -> FedAvgAggregator | RobustAggregator`
Factory function in `fedsys/coordinator/aggregator.py`. Returns `FedAvgAggregator` for `"none"` (zero overhead), `RobustAggregator` for everything else. This is what `serve()` calls automatically based on `CoordinatorConfig.defense_method`.

---

## 5. Adversarial Evaluation

### 5.1 Metrics Explained

The adversarial evaluation runs at the coordinator after each round (when `adv_target_item >= 0` and an `ml_dataset` is loaded). It computes the following on the **validation split** (and the **test split** at the end):

| Metric | Description |
|---|---|
| `hit@K` | Standard Hit@K over all users |
| `ndcg@K` | Standard NDCG@K over all users |
| `segment_hit@K` | Hit@K restricted to users in the victim segment (dominant genre = target genre) |
| `segment_ndcg@K` | NDCG@K for victim segment |
| `target_hit@K` | Fraction of victim-segment users who have the **target item** in their personal top-K |
| `target_ndcg@K` | NDCG@K contribution from the target item appearing in victim-segment top-K |

`target_hit@K` is the primary attack metric. Without an attack it should be near 0 (the target item is obscure). Under a successful attack it rises significantly.

To **compare attack vs. baseline**, use `compare_attack_vs_clean()`:

```python
from fedsys.adversarial.eval import compare_attack_vs_clean

diff = compare_attack_vs_clean(attack_metrics, clean_metrics, cutoffs=(10, 20))
# e.g.: {"delta_hit@10": -0.002, "delta_target_hit@10": +0.143, ...}
```

Positive `delta_target_hit@K` with small `delta_hit@K` degradation is the attacker's goal.

### 5.2 Code Walkthrough — `eval.py`

#### `evaluate_with_target_exposure(model, dataset, target_item_index, target_genre, split, cutoffs, device, ...)`
Top-level function called by the coordinator each round. Internally calls:
- `_ranking_metrics()` — standard Hit/NDCG over all users in the split.
- `_ranking_metrics()` again — same, but restricted to `segment_users`.
- `_target_exposure()` — computes rank of `target_item_index` specifically for each segment user.

Returns a flat dict with all metrics prefixed appropriately.

#### `_ranking_metrics(model, user_indices, dataset, split, cutoffs, device, batch_size)`
For each user:
1. Score all items with the model (`model(u_rep, all_items)`).
2. Mask known positives (except the held-out item for this split).
3. Compute rank of the held-out item.
4. Accumulate Hit@K and NDCG@K.

#### `_target_exposure(model, user_indices, dataset, target_item_index, cutoffs, device, batch_size)`
Same as above except the "item to evaluate" is always `target_item_index`, and known positives are masked leaving the target unmasked. Returns `(hit_dict, ndcg_dict)` keyed by cutoff.

#### `compare_attack_vs_clean(attack_metrics, clean_metrics, cutoffs)`
Simple dict difference: `{f"delta_{k}": attack[k] - clean[k]}` for all keys.

---

## 6. How to Run — CLI Quick-Start

### 6.1 Attack Only (no defense)

Run one malicious node alongside one clean node, with no defense at the coordinator:

```bash
# Terminal 1 — Coordinator (standard FedAvg, no defense)
python scripts/run_coordinator.py \
    --port 50051 \
    --total-nodes 2 --min-nodes 2 \
    --num-rounds 10 \
    --model-type bpr \
    --ml-data-root data/ --ml-variant ml-1m \
    --defense none \
    --attack-max-synth 200 \
    --adv-target-item 42 \
    --adv-target-genre Action

# Terminal 2 — Clean node (shard 0)
python scripts/run_node.py \
    --port 50051 \
    --movielens data/ --ml-variant ml-1m \
    --partition 0 --num-partitions 2 \
    --attack-max-synth 200

# Terminal 3 — Malicious node (shard 1, with attack)
python scripts/run_node.py \
    --port 50051 \
    --movielens data/ --ml-variant ml-1m \
    --partition 1 --num-partitions 2 \
    --attack \
    --attack-target-item 42 \
    --attack-target-genre Action \
    --attack-budget 0.3 \
    --attack-max-synth 200
```

**Automatic target selection** (instead of hardcoding item 42):

```bash
# Omit --attack-target-item; provide only --attack-target-genre.
# The node calls select_target_item() automatically.
python scripts/run_node.py ... --attack --attack-target-genre Action
```

---

### 6.2 Defense Only (no malicious node)

All nodes are honest, but the coordinator uses a robust aggregation method as a precaution:

```bash
# Terminal 1 — Coordinator with norm-clipping defense
python scripts/run_coordinator.py \
    --port 50051 \
    --total-nodes 3 --min-nodes 3 \
    --num-rounds 5 \
    --model-type bpr \
    --ml-data-root data/ --ml-variant ml-1m \
    --defense clip_mean \
    --defense-clip-thresh 5.0

# Terminals 2, 3, 4 — Clean nodes (standard)
python scripts/run_node.py --port 50051 --movielens data/ --partition 0 --num-partitions 3
python scripts/run_node.py --port 50051 --movielens data/ --partition 1 --num-partitions 3
python scripts/run_node.py --port 50051 --movielens data/ --partition 2 --num-partitions 3
```

---

### 6.3 Attack + Defense Together

Benchmark a specific defense against a live attacker:

```bash
# Coordinator — focus_clip_trimmed_mean defense + adversarial evaluation
python scripts/run_coordinator.py \
    --port 50051 \
    --total-nodes 3 --min-nodes 3 \
    --num-rounds 10 \
    --model-type bpr \
    --ml-data-root data/ --ml-variant ml-1m \
    --defense focus_clip_trimmed_mean \
    --defense-clip-thresh 5.0 \
    --defense-trim-frac 0.1 \
    --defense-focus-k-frac 0.05 \
    --attack-max-synth 200 \
    --adv-target-item 42 \
    --adv-target-genre Action

# 2 clean nodes
python scripts/run_node.py --port 50051 --movielens data/ --partition 0 --num-partitions 3 --attack-max-synth 200
python scripts/run_node.py --port 50051 --movielens data/ --partition 1 --num-partitions 3 --attack-max-synth 200

# 1 malicious node (shard 2)
python scripts/run_node.py \
    --port 50051 \
    --movielens data/ --ml-variant ml-1m \
    --partition 2 --num-partitions 3 \
    --attack \
    --attack-target-item 42 \
    --attack-target-genre Action \
    --attack-budget 0.30 \
    --attack-num-filler 30 \
    --attack-num-neutral 20 \
    --attack-neutral-genre Comedy \
    --attack-max-synth 200
```

The coordinator will print adversarial metrics each round, e.g.:

```
[coordinator] Adv epoch   4  target_hit@10=0.0312  target_ndcg@10=0.0089
[coordinator] Adv epoch   5  target_hit@10=0.0298  target_ndcg@10=0.0081
```

Low and stable `target_hit@K` indicates the defense is working.

---

### 6.4 All Available Flags

#### Coordinator flags

| Flag | Default | Description |
|---|---|---|
| `--defense` | `none` | Aggregation method: `none`, `clip_mean`, `clip_trimmed_mean`, `focus_clip_mean`, `focus_clip_trimmed_mean` |
| `--defense-clip-thresh` | `5.0` | L2 norm clipping threshold θ |
| `--defense-trim-frac` | `0.10` | Trimmed-mean tail fraction (each side) |
| `--defense-focus-k-frac` | `0.05` | Top-item fraction for focus score |
| `--adv-target-item` | `-1` | Target item index to track; `-1` disables adversarial eval |
| `--adv-target-genre` | `""` | Genre name matching the target item |

#### Node flags (attack mode)

| Flag | Default | Description |
|---|---|---|
| `--attack` | off | Enable data-poisoning attack on this node |
| `--attack-target-item` | `-1` | Target item index; auto-selected from genre if `-1` |
| `--attack-target-genre` | `""` | Genre of the target item (required with `--attack`) |
| `--attack-budget` | `0.30` | Fraction of shard users to add as synthetic profiles |
| `--attack-num-filler` | `30` | Number of filler items per synthetic profile |
| `--attack-num-neutral` | `20` | Number of neutral items per synthetic profile |
| `--attack-neutral-genre` | `Comedy` | Genre from which neutral items are drawn |
| `--attack-target-weight` | `1.0` | Rating weight multiplier for target item |
| `--attack-max-synth` | `0` | Synthetic-user row reservation on this node; use same value as coordinator in adversarial runs |
| `--attack-seed` | `42` | RNG seed for reproducibility |

---

## 7. Programmatic API

Use the modules directly without any CLI for custom experiments:

```python
from fedsys.adversarial.attack.target import (
    choose_target_genre,
    select_target_item,
    select_target_from_clean_model,
)
from fedsys.adversarial.attack.poison import (
    AttackConfig,
    build_poisoned_dataloader,
    poisoned_num_users,
)
from fedsys.adversarial.defense.aggregation import RobustAggregator, DEFENSE_METHODS
from fedsys.adversarial.eval import evaluate_with_target_exposure, compare_attack_vs_clean

# 1. Auto-select what to attack
genre  = choose_target_genre(ml_dataset)
target = select_target_item(ml_dataset, genre)
# Or using model-based selection (more compute):
target = select_target_from_clean_model(global_model, ml_dataset, genre, k=10)

# 2. Configure the attack
attack_cfg = AttackConfig(
    enabled=True,
    target_item_index=target,
    target_genre=genre,
    attack_budget=0.30,
    num_filler_items=30,
    num_neutral_items=20,
)

# 3. Size the model to include synthetic user slots
extended_users = poisoned_num_users(ml_dataset.num_users, attack_cfg)
model_cfg.num_users = extended_users  # coordinator and ALL nodes use this

# 4. Build poisoned dataloader for the malicious node
dl = build_poisoned_dataloader(
    dataset=ml_dataset,
    shard_user_indices=my_shard,
    attack_cfg=attack_cfg,
    model_num_users=extended_users,
    batch_size=256,
)

# 5. Build a robust aggregator at the coordinator
from fedsys.logging_async import AsyncTelemetryLogger
logger = AsyncTelemetryLogger("logs/telemetry.jsonl", "logs/telemetry.db").start()
aggregator = RobustAggregator(
    logger,
    method="focus_clip_mean",
    clip_threshold=5.0,
    focus_k_frac=0.05,
)

# 6. In the aggregation loop (pass global_state for delta-space operations)
new_state, serialized = aggregator.aggregate(
    updates, sample_counts, epoch,
    global_state=dict(global_model.state_dict()),
)

# 7. Evaluate attack impact
metrics = evaluate_with_target_exposure(
    model=global_model,
    dataset=ml_dataset,
    target_item_index=target,
    target_genre=genre,
    split="val",
    cutoffs=(10, 20),
)
print(f"target_hit@10 = {metrics['target_hit@10']:.4f}")

# 8. Compare to clean baseline
diff = compare_attack_vs_clean(metrics, clean_metrics, cutoffs=(10, 20))
print(f"Attack lifted target_hit@10 by {diff['delta_target_hit@10']:+.4f}")
```

---

## 8. Tests

Run all adversarial tests at once:

```bash
# Unit tests (no coordinator process needed, runs in ~5s each)
python tests/adversarial/test_poison.py
python tests/adversarial/test_defense.py
python tests/adversarial/test_target.py
python tests/adversarial/test_eval.py

# Integration test (3 scenarios: clip_mean, focus_clip_trimmed_mean, none)
# Spins up in-process gRPC coordinator + 2 nodes, completes in ~20s
python tests/adversarial/test_integration.py
```

| Test file | What it covers |
|---|---|
| `test_poison.py` | `poisoned_num_users`, synthetic profile content, `PoisonedBPRPairDataset` length and index validity, `build_poisoned_dataloader` |
| `test_defense.py` | Math helpers (`_l2_norm`, `_delta`, `_trimmed_mean_tensor`, `_focus_score`), all 5 methods round-trip shape preservation, clipping actually reduces large updates |
| `test_target.py` | `benign_segment_users`, `select_target_item` correctness, `choose_target_genre`, `select_target_from_clean_model` with a tiny BPR model |
| `test_eval.py` | `evaluate_with_target_exposure` returns all expected keys and values in `[0,1]`, `compare_attack_vs_clean` arithmetic |
| `test_integration.py` | Full gRPC pipeline: 1 coordinator + 1 clean node + 1 malicious node, 2 rounds each, for `clip_mean`, `focus_clip_trimmed_mean`, and `none` |

---

## 9. Design Trade-offs and Limitations

### Attack

| Consideration | Detail |
|---|---|
| **Embedding-table extension** | All nodes (clean and malicious) must use `poisoned_num_users()` for `ModelConfig.num_users` when an attack is possible. The coordinator must pre-allocate the extended table at startup, before any node registers. |
| **Budget cap** | `max_synthetic_users_per_coord` is a hard cap on synthetic users. It must be set to `>= attack_budget × max_shard_size` for the attack to run at full budget. |
| **Attack is per-round** | Synthetic profiles are injected fresh every local training round, not just once. The malicious node re-generates `N_synth` profiles each time `FederatedNode` starts local training. |
| **Multi-target attacks** | The current implementation supports one target item per node. Running multiple malicious nodes with different targets is possible but not natively coordinated. |
| **Pointwise models** | The attack is designed for BPR (pairwise loss). For `simple`/`neural_cf`/`two_tower` models the synthetic profiles will still be injected but the pairwise BPR loss won't fire — only BCELoss on the target item's direct positives. Effectiveness is lower. |

### Defense

| Consideration | Detail |
|---|---|
| **Clipping threshold tuning** | `clip_threshold` should be set based on the typical L2 norm of honest updates. Too low degrades model quality; too high allows attacks through. A common heuristic is to set it to the median norm of clean updates. |
| **Trimmed mean and few clients** | `clip_trimmed_mean` requires enough clients to trim. With `n=2` clients and `trim_fraction=0.1`, trimming has no effect (fewer than 1 client per tail). Increase client count or lower `trim_fraction` for very small federations. |
| **Focus score and sparse catalogs** | For very small item catalogs or highly skewed popularity, even benign updates can have high focus scores. Calibrate `focus_k_frac` and `focus_threshold` for your dataset. |
| **Global state requirement** | All methods except `"none"` require `global_state` to compute deltas. This is always available inside `run_aggregation_loop` but must be passed explicitly when using `RobustAggregator` directly. |
| **Non-BPR models** | The focus score specifically looks at `item_embedding.weight`. For `simple`/`neural_cf` models that use different parameter names, the focus score will always be 0 and the `focus_*` methods will degrade to plain `clip_*`. |
