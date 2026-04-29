# BPR + MovieLens Integration and Fixes

This document explains the BPR/MovieLens integration work in this codebase, the runtime failures we hit, exactly what was changed, and why the current implementation now runs end-to-end.

---

## 1) What was the goal?

The goal was to move the federated recommender path from the older/undesired implementations to:

- `BPRModel` as the recommender model,
- MovieLens as the dataset,
- the existing gRPC FL system (`fedsys/coordinator`, `fedsys/node`, `proto/federated.proto`),
- proper validation + test evaluation,
- model checkpointing for **best** and **final** models,
- and a reproducible test script that actually executes coordinator + nodes together.

In short: keep the FL orchestration and networking layer from this repo, but make the training/evaluation stack BPR + MovieLens.

---

## 2) Main failures that were fixed

### Failure A: state_dict size mismatch on node startup

Symptom (earlier run): node failed to load global model due to tensor shape mismatch in `BPRModel`.

Root causes:

1. Coordinator and node had different `num_users` / `num_items` (coordinator used CLI values; node derived dimensions from loaded MovieLens data).
2. Coordinator and node used different `embedding_dim` defaults (16 vs 32 in different scripts).

Fixes:

- Coordinator now loads MovieLens early (when `--ml-data-root` is set) to derive the **actual** dimensions used in `ModelConfig`.
- Defaults aligned so both launcher scripts use `--embedding-dim` default `32`.
- Node path explicitly ensures `model_cfg.model_type = "bpr"` when MovieLens mode is selected.

### Failure B: coordinator crash in `save_checkpoint`

Symptom from your log:

- `ValueError: Unknown format code 'f' for object of type 'str'` in `fedsys/coordinator/server.py`.
- Then node RPCs fail with `StatusCode.UNAVAILABLE` / `Cancelling all calls` because the coordinator process/thread crashed.

Root cause:

- Best-checkpoint log line assumed pointwise metrics (`loss`, `accuracy`, `auc_roc`) and formatted fallback `'?'` with `:.4f`.
- In BPR flow, validation metrics are ranking metrics (`hit@10`, `ndcg@10`, etc.), so `val_metrics.get('loss', '?')` returned string `'?'`, causing formatting failure.

Fix:

- Replaced hardcoded formatting with a generic formatter that prints all available float metrics from `val_metrics`.

Result:

- No formatting exception.
- Coordinator remains alive.
- Node `UNAVAILABLE` cascade disappears.

---

## 3) File-by-file code changes

## `fedsys/models/bpr.py` (new)

Purpose: Define `BPRModel` used by coordinator and nodes.

Key behavior:

- User embedding table
- Item embedding table
- Optional user/item biases
- Score function: dot(user_emb, item_emb) + biases

Why it matters:

- BPR is a ranking model optimized with pairwise preference constraints instead of pointwise BCE labels.

---

## `fedsys/data/movielens_dataset.py` (new)

Purpose: End-to-end MovieLens support for federated BPR.

Key behavior:

- Load dataset variants (`ml-1m`, `ml-10m`, `ml-25m`, `ml-32m`) from root path.
- Parse ratings and map external IDs to compact contiguous IDs.
- Build leave-one-out style splits for ranking evaluation.
- Construct `BPRPairDataset` with online negative sampling.
- Partition users into deterministic shards (`partition_users`) for federated clients.
- Build shard-specific training dataloaders (`build_movielens_train_dataloader`).

Why it matters:

- FL requires per-node data partitioning and stable indexing.
- BPR requires sampled (user, positive item, negative item) tuples.

---

## `fedsys/models/recommendation.py` (updated)

Purpose: Model factory.

Changes:

- Removed old large NCF path.
- Added `"bpr"` branch in `build_model()`.
- Kept `"simple"` branch for synthetic smoke testing.

Why it matters:

- Coordinator and nodes construct model objects through the same factory path.
- This prevents architecture drift when switching model type.

---

## `fedsys/node/trainer.py` (updated)

Purpose: Local training loop on each node.

Changes:

- Added dynamic batch-mode handling:
  - If batch contains BPR fields (`pos_item_id`, etc.), run pairwise BPR objective.
  - Otherwise run existing pointwise BCE path.

Why it matters:

- One trainer supports both synthetic simple model and BPR ranking without duplicated node runtime logic.

---

## `fedsys/coordinator/evaluator.py` (updated)

Purpose: Validation/test metrics computed at coordinator.

Changes:

- Added ranking evaluator (`evaluate_ranking`) for BPR:
  - Hit@K
  - NDCG@K
- Kept pointwise evaluator for synthetic/simple flow.

Why it matters:

- BPR quality is measured by ranking metrics, not BCE accuracy/AUC.

---

## `fedsys/coordinator/server.py` (updated)

Purpose: Coordinator service + aggregation loop + checkpointing.

Important changes:

1. Aggregation loop now supports both evaluation modes:
   - BPR ranking mode when MovieLens dataset is configured.
   - Pointwise mode when CSV val/test loaders are configured.

2. Checkpoint policy:
   - Save `model_epoch_<n>.pt` each round.
   - Save/overwrite `model_best.pt` based on validation criterion.
   - Save `model_final.pt` at final round.

3. **Crash fix in `save_checkpoint`** (latest fix):
   - Replaced hardcoded BCE metric print with generic float metric formatter:
     - Works for both `loss/accuracy/auc_roc` and `hit@10/ndcg@10/...`.

Why it matters:

- This is the exact fix for your runtime exception.
- It prevents coordinator shutdown and subsequent node gRPC failures.

---

## `scripts/run_coordinator.py` (updated)

Purpose: Coordinator launcher.

Changes:

- Added MovieLens flags (`--ml-data-root`, `--ml-variant`).
- For BPR mode + MovieLens path, load dataset before creating `ModelConfig` so `num_users`/`num_items` are taken from actual data.
- Default embedding dim aligned to `32`.

Why it matters:

- Removes manual, error-prone dimension syncing.
- Prevents model parameter shape mismatch between coordinator and node.

---

## `scripts/run_node.py` (updated)

Purpose: Node launcher.

Changes:

- Added MovieLens flags (`--movielens`, `--ml-variant`).
- Load MovieLens shard and build BPR training dataloader.
- Set `model_cfg.model_type = "bpr"` in MovieLens path.
- Default embedding dim aligned to `32`.

Why it matters:

- Node model and coordinator model now agree on architecture and dimensions.

---

## `scripts/_test_bpr_movielens.py` (new)

Purpose: Reproducible integration test for this exact pipeline.

What it does:

1. Loads MovieLens once and derives canonical dimensions.
2. Creates one shared `ModelConfig` used by coordinator and all nodes.
3. Starts coordinator in a background thread.
4. Starts N nodes, each with its own user shard.
5. Runs configurable rounds/epochs.
6. Fails fast if any node raises exception.
7. Validates checkpoint artifacts (`model_best.pt`, `model_final.pt`).

Why it matters:

- Gives a one-command way to verify no regression in coordinator-node compatibility and checkpoint flow.

---

## 4) Why `UNAVAILABLE` happened after the ValueError

This is important operationally:

- The node stack trace (`grpc StatusCode.UNAVAILABLE`) was not the primary bug.
- It was a downstream effect of coordinator failure in aggregation/checkpoint path.
- Once the formatting crash was fixed, RPC streaming stayed healthy and the node completed rounds.

---

## 5) Validation of the fix (actual run)

We ran the test script directly:

`python scripts/_test_bpr_movielens.py --ml-data-root data/ --ml-variant ml-1m --num-rounds 3 --num-nodes 2 --local-epochs 1`

Observed:

- Coordinator started and stayed alive.
- Two nodes trained and uploaded updates across 3 rounds.
- Ranking metrics printed every round.
- `model_best.pt` and `model_final.pt` were present.
- Test-set ranking metrics printed at the end.
- Process ended with `PASSED`.

This verifies both the functional BPR integration and the specific checkpoint-formatting crash fix.

---

## 6) Current behavior summary

With the current code:

- BPR + MovieLens is first-class in the existing gRPC FL system.
- Coordinator and node model shapes are synchronized from dataset-derived dimensions.
- Best and final model checkpointing works for ranking workflows.
- Both ranking validation and ranking test evaluation are integrated.
- There is a dedicated reproducible integration test script.

---

## 7) Known next work (not yet implemented)

Planned from your requirements:

- Adversarial scenario integration (after stable BPR/MovieLens baseline).

That can now be layered on top of this baseline with fewer confounders, because model/dataset/eval/checkpoint plumbing is stable.
