# FedSys — gRPC Federated Learning Framework

Synchronous federated learning framework with clean separation between networking (gRPC), computation (PyTorch), and observability (async telemetry). Supports multiple recommender models, both synthetic and MovieLens pipelines, and a full adversarial layer (data-poisoning attack + robust aggregation defense).

## Requirements

- Python 3.10+
- PyTorch 2.2+
- Optional CUDA

```bash
pip install -r requirements.txt
python scripts/generate_proto.py
```

## Supported Models

`--model-type` currently supports:

- `simple` (two-layer baseline, `fedsys/models/simple.py`)
- `bpr` (pairwise BPR-MF, `fedsys/models/bpr.py`)
- `neural_cf` (MLP recommender, `fedsys/models/neural_cf.py`)
- `two_tower` (dual-tower scorer, `fedsys/models/two_tower.py`)

Model construction is centralized in `fedsys/models/recommendation.py`.

---

## Start Training (Multiple Models)

### A) Synthetic dataset

Generate data once:

```bash
python scripts/generate_synthetic_data.py --num-partitions 2 --val-fraction 0.1 --test-fraction 0.1
```

Start coordinator:

```bash
python scripts/run_coordinator.py ^
  --host 127.0.0.1 ^
  --port 50051 ^
  --total-nodes 2 ^
  --min-nodes 2 ^
  --num-rounds 3 ^
  --model-type simple ^
  --val-data data/synthetic/val.csv ^
  --test-data data/synthetic/test.csv
```

Start nodes in two terminals:

```bash
python scripts/run_node.py --node-id node-0 --coordinator 127.0.0.1:50051 --data-dir data/synthetic --partition 0 --num-partitions 2 --model-type simple
python scripts/run_node.py --node-id node-1 --coordinator 127.0.0.1:50051 --data-dir data/synthetic --partition 1 --num-partitions 2 --model-type simple
```

To switch models, change `--model-type` on coordinator + all nodes to one of:
`simple | bpr | neural_cf | two_tower`.

### B) MovieLens dataset

Expected structure:

- `data/ml-1m/ratings.dat`
- `data/ml-1m/movies.dat`

Start coordinator (example `bpr`):

```bash
python scripts/run_coordinator.py ^
  --host 127.0.0.1 ^
  --port 50051 ^
  --total-nodes 2 ^
  --min-nodes 2 ^
  --num-rounds 3 ^
  --model-type bpr ^
  --ml-data-root data/ ^
  --ml-variant ml-1m
```

Start nodes:

```bash
python scripts/run_node.py --node-id node-0 --coordinator 127.0.0.1:50051 --movielens data/ --ml-variant ml-1m --partition 0 --num-partitions 2 --model-type bpr
python scripts/run_node.py --node-id node-1 --coordinator 127.0.0.1:50051 --movielens data/ --ml-variant ml-1m --partition 1 --num-partitions 2 --model-type bpr
```

You can run `simple`, `neural_cf`, and `two_tower` on MovieLens too by changing `--model-type` on both coordinator and nodes.

---

## Adversarial Scenarios

The `fedsys/adversarial/` module adds an optional attack-and-defense layer on top of the core FL pipeline. All three capabilities are independent — enable any combination.

> See **ADVERSARIAL.md** for the complete guide.

### Attack only — observe impact with no defense

```bash
# Coordinator (plain FedAvg + adversarial eval tracking target item 42)
python scripts/run_coordinator.py ^
  --port 50051 --total-nodes 2 --min-nodes 2 --num-rounds 10 ^
  --model-type bpr --ml-data-root data/ --ml-variant ml-1m ^
  --defense none ^
  --adv-target-item 42 --adv-target-genre Action

# Clean node
python scripts/run_node.py --coordinator 127.0.0.1:50051 ^
  --movielens data/ --partition 0 --num-partitions 2

# Malicious node (shard 1, poisoning enabled)
python scripts/run_node.py --coordinator 127.0.0.1:50051 ^
  --movielens data/ --partition 1 --num-partitions 2 ^
  --attack --attack-target-item 42 --attack-target-genre Action --attack-budget 0.3
```

### Defense only — harden an honest system

```bash
python scripts/run_coordinator.py ^
  --port 50051 --total-nodes 2 --min-nodes 2 --num-rounds 5 ^
  --model-type bpr --ml-data-root data/ --ml-variant ml-1m ^
  --defense clip_mean --defense-clip-thresh 5.0

python scripts/run_node.py --coordinator 127.0.0.1:50051 --movielens data/ --partition 0 --num-partitions 2
python scripts/run_node.py --coordinator 127.0.0.1:50051 --movielens data/ --partition 1 --num-partitions 2
```

### Attack + Defense — benchmark defense effectiveness

```bash
python scripts/run_coordinator.py ^
  --port 50051 --total-nodes 3 --min-nodes 3 --num-rounds 10 ^
  --model-type bpr --ml-data-root data/ --ml-variant ml-1m ^
  --defense focus_clip_trimmed_mean ^
  --adv-target-item 42 --adv-target-genre Action

# 2 clean nodes + 1 malicious node
python scripts/run_node.py --coordinator 127.0.0.1:50051 --movielens data/ --partition 0 --num-partitions 3
python scripts/run_node.py --coordinator 127.0.0.1:50051 --movielens data/ --partition 1 --num-partitions 3
python scripts/run_node.py --coordinator 127.0.0.1:50051 --movielens data/ --partition 2 --num-partitions 3 ^
  --attack --attack-target-item 42 --attack-target-genre Action --attack-budget 0.3
```

---

## Important CLI Flags

### Coordinator (`scripts/run_coordinator.py`)

| Flag | Description |
|---|---|
| `--host`, `--port` | Server bind address |
| `--total-nodes`, `--min-nodes` | N and K for K-of-N aggregation |
| `--num-rounds`, `--round-timeout` | Training rounds and straggler timeout |
| `--model-type` | `simple\|bpr\|neural_cf\|two_tower` |
| `--num-users`, `--num-items`, `--embedding-dim`, `--hidden-dim` | Model dimensions |
| `--val-data`, `--test-data` | Synthetic evaluation CSVs |
| `--ml-data-root`, `--ml-variant` | MovieLens evaluation |
| `--checkpoint-dir`, `--log-dir`, `--chunk-size-mb` | I/O settings |
| `--defense` | `none\|clip_mean\|clip_trimmed_mean\|focus_clip_mean\|focus_clip_trimmed_mean` |
| `--defense-clip-thresh` | L2 norm clip threshold θ (default 5.0) |
| `--defense-trim-frac` | Trimmed-mean tail fraction (default 0.10) |
| `--defense-focus-k-frac` | Focus-score top-item fraction (default 0.05) |
| `--adv-target-item` | Item index to track for attack measurement (-1 = off) |
| `--adv-target-genre` | Genre of the tracked target item |

### Node (`scripts/run_node.py`)

| Flag | Description |
|---|---|
| `--node-id`, `--coordinator` | Node identity and server address |
| `--device`, `--local-epochs`, `--batch-size`, `--lr`, `--num-rounds` | Training settings |
| `--data-dir` | Synthetic CSV partitions |
| `--movielens` + `--ml-variant` | MovieLens data source |
| `--partition`, `--num-partitions` | Shard assignment |
| `--model-type` | Must match coordinator |
| `--embedding-dim`, `--hidden-dim`, `--chunk-size-mb`, `--log-dir` | Model / I/O settings |
| `--attack` | Enable data-poisoning (malicious node mode) |
| `--attack-target-item` | Item to push (-1 = auto-select from genre) |
| `--attack-target-genre` | Genre of target item |
| `--attack-budget` | Fraction of shard users to replicate as synthetic profiles (default 0.30) |
| `--attack-num-filler`, `--attack-num-neutral` | Profile shape (default 30, 20) |
| `--attack-neutral-genre`, `--attack-target-weight` | Profile tuning |
| `--attack-max-synth` | Max synthetic users; must match coordinator pre-allocation (default 200) |

---

## Checkpoints and Evaluation

Coordinator saves:

- `checkpoints/model_epoch_<n>.pt` (each round)
- `checkpoints/model_best.pt` (best validation metric)
- `checkpoints/model_final.pt` (final round)

Evaluation mode:

- Synthetic: pointwise metrics (`loss`, `accuracy`, `auc_roc`)
- MovieLens: ranking metrics (`hit@10`, `ndcg@10`, `hit@20`, `ndcg@20`)
- Adversarial (when `--adv-target-item` set): `target_hit@10`, `target_ndcg@10`, `segment_hit@10`

---

## Tests

```
tests/
├── synthetic/          # 4 models × synthetic data (test_simple, test_bpr, test_neural_cf, test_two_tower)
├── movielens/          # 4 models × MovieLens data
├── adversarial/        # attack, defense, target-selection, eval unit tests + full gRPC integration
└── common.py           # shared helpers
```

Run examples:

```bash
# Standard integration tests
python tests/synthetic/test_simple.py
python tests/movielens/test_bpr.py

# Adversarial unit tests
python tests/adversarial/test_poison.py
python tests/adversarial/test_defense.py
python tests/adversarial/test_target.py
python tests/adversarial/test_eval.py

# Adversarial integration (1 coord + 1 clean + 1 malicious, 3 defense scenarios)
python tests/adversarial/test_integration.py
```

---

## Project Structure

```text
fedsys/
├── fedsys/
│   ├── adversarial/
│   │   ├── eval.py                  # target-exposure evaluation
│   │   ├── attack/
│   │   │   ├── poison.py            # PoisonedBPRPairDataset + AttackConfig
│   │   │   └── target.py            # target item / genre selection
│   │   └── defense/
│   │       └── aggregation.py       # RobustAggregator (5 methods)
│   ├── coordinator/
│   │   ├── server.py
│   │   ├── aggregator.py            # FedAvgAggregator + build_aggregator()
│   │   ├── evaluator.py
│   │   └── registry.py
│   ├── node/
│   ├── data/
│   │   ├── synthetic_dataset.py
│   │   └── movielens_dataset.py
│   └── models/
│       ├── simple.py
│       ├── bpr.py
│       ├── neural_cf.py
│       ├── two_tower.py
│       └── recommendation.py
├── scripts/
│   ├── run_coordinator.py
│   ├── run_node.py
│   └── generate_synthetic_data.py
├── tests/
│   ├── synthetic/
│   ├── movielens/
│   ├── adversarial/
│   └── common.py
├── ADVERSARIAL.md       # full adversarial guide
├── BPR.md               # BPR + MovieLens integration details
├── FLOW.md              # runtime flow and sequence diagram
└── CODE.md              # per-function codebase reference
```
