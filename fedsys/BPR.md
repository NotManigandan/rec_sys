# BPR + MovieLens in `fedsys`

This document describes the **current** BPR + MovieLens implementation in the gRPC federated learning framework, including architecture, data flow, configuration, training/evaluation behavior, and how to run it.

---

## 1) Scope

The BPR + MovieLens path uses:

- the existing FL runtime (`fedsys/coordinator`, `fedsys/node`, gRPC transport),
- MovieLens data loading and user-sharded partitioning,
- pairwise BPR local training,
- ranking-based validation/test evaluation,
- checkpointing for epoch, best, and final models.

The same core runtime also supports other models and synthetic data, but this document focuses on the BPR + MovieLens path.

---

## 2) Core Components

### Model: `fedsys/models/bpr.py`

`BPRModel` is a ranking model with:

- user embeddings,
- item embeddings,
- optional bias terms,
- scoring via user-item compatibility (embedding interactions).

Local training optimizes pairwise preferences (positive item > sampled negative item), which is appropriate for implicit-feedback recommendation.

### Dataset: `fedsys/data/movielens_dataset.py`

MovieLens support includes:

- loading `ml-1m`, `ml-10m`, `ml-25m`, `ml-32m`,
- ID remapping to contiguous user/item indices,
- user-level train/val/test preparation,
- `BPRPairDataset` for online negative sampling,
- deterministic user partitioning across FL nodes.

### Model Factory: `fedsys/models/recommendation.py`

The central `build_model()` factory builds the configured model type (`bpr`, `simple`, `neural_cf`, `two_tower`). This keeps coordinator and node model construction aligned.

### Node Trainer: `fedsys/node/trainer.py`

The node trainer supports both:

- pairwise BPR batches (MovieLens path), and
- pointwise batches (synthetic path).

For MovieLens + BPR, it runs pairwise loss over `(user, pos_item, neg_item)` samples.

### Coordinator Evaluation: `fedsys/coordinator/evaluator.py`

MovieLens evaluation uses ranking metrics, primarily:

- `hit@10`, `ndcg@10`,
- `hit@20`, `ndcg@20`.

At round end and final test, coordinator computes and prints ranking metrics.

### Coordinator Runtime: `fedsys/coordinator/server.py`

Coordinator responsibilities:

- serve global model via chunked gRPC streaming,
- receive and reconstruct node updates,
- aggregate updates (FedAvg or robust aggregator when enabled),
- evaluate on validation/test splits,
- save checkpoints (`model_epoch_<n>.pt`, `model_best.pt`, `model_final.pt`).

---

## 3) End-to-End BPR MovieLens Flow

1. Coordinator starts with MovieLens config and derives canonical dimensions.
2. Nodes register and get shard assignments.
3. Each node loads MovieLens shard and builds BPR dataloader.
4. Each round:
   - node fetches global model,
   - node trains locally with pairwise BPR objective,
   - node sends model update to coordinator,
   - coordinator aggregates once K updates are collected.
5. Coordinator evaluates and checkpoints.
6. Final model is saved and tested.

---

## 4) Dimension Consistency Rules

Correct BPR runs depend on consistent model dimensions across coordinator and all nodes:

- `model_type` must match (`bpr` for this doc’s path),
- `num_users` and `num_items` come from loaded MovieLens dataset,
- `embedding_dim` must match everywhere.

For adversarial runs with synthetic-user reservation, the same `--attack-max-synth` value must be passed to the coordinator and every node (clean + malicious) so model shapes remain consistent.

---

## 5) CLI Usage

### Coordinator

```bash
python scripts/run_coordinator.py \
  --host 127.0.0.1 --port 50051 \
  --total-nodes 2 --min-nodes 2 \
  --num-rounds 3 --round-timeout 120 \
  --model-type bpr \
  --ml-data-root data/ --ml-variant ml-1m \
  --embedding-dim 32 \
  --checkpoint-dir checkpoints/bpr_ml
```

### Node 0

```bash
python scripts/run_node.py \
  --node-id node0 \
  --coordinator 127.0.0.1:50051 \
  --movielens data/ --ml-variant ml-1m \
  --partition 0 --num-partitions 2 \
  --model-type bpr \
  --device cuda:0 \
  --local-epochs 1 --batch-size 512 --lr 0.001 \
  --num-rounds 3
```

### Node 1

```bash
python scripts/run_node.py \
  --node-id node1 \
  --coordinator 127.0.0.1:50051 \
  --movielens data/ --ml-variant ml-1m \
  --partition 1 --num-partitions 2 \
  --model-type bpr \
  --device cuda:0 \
  --local-epochs 1 --batch-size 512 --lr 0.001 \
  --num-rounds 3
```

---

## 6) Metrics and Checkpoints

### Validation/Test Metrics

MovieLens ranking mode reports:

- `hit@10`, `ndcg@10`,
- `hit@20`, `ndcg@20`.

When adversarial tracking is enabled, additional target/segment metrics are printed (documented in `ADVERSARIAL.md`).

### Checkpoints

Coordinator writes:

- `model_epoch_<n>.pt` each round,
- `model_best.pt` for best validation performance,
- `model_final.pt` after final round.

---

## 7) Testing

Recommended coverage:

- unit tests for model and dataset helpers,
- integration tests with 1 coordinator + 2 nodes on MovieLens,
- GPU matrix benchmark for normal and adversarial modes across available models.

You can use:

```bash
python scripts/run_gpu_matrix_benchmark.py
```

This runs all configured models in normal and adversarial modes and writes logs + `logs/gpu_full/report.json`.

---

## 8) Practical Notes

- Use the same `--model-type` on coordinator and nodes.
- Keep `--embedding-dim` consistent across all participants.
- For adversarial experiments, ensure `--attack-max-synth` is explicitly synchronized across coordinator and all nodes.
- Start with `ml-1m` for fast iteration, then scale to larger variants.

---

## 9) Summary

The BPR + MovieLens path is fully integrated into the gRPC FL runtime with:

- consistent coordinator/node model construction,
- MovieLens-native sharded BPR training,
- ranking-focused evaluation,
- robust checkpoint lifecycle,
- reproducible benchmark/test workflows.
