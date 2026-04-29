# FedSys — gRPC Federated Learning Framework

Synchronous federated learning framework with clean separation between networking (gRPC), computation (PyTorch), and observability (async telemetry). Supports multiple recommender models and both synthetic and MovieLens pipelines.

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

## Important CLI Flags

### Coordinator (`scripts/run_coordinator.py`)

- `--host`, `--port`
- `--total-nodes`, `--min-nodes`, `--num-rounds`, `--round-timeout`
- `--model-type` = `simple|bpr|neural_cf|two_tower`
- `--num-users`, `--num-items`, `--embedding-dim`, `--hidden-dim`
- `--val-data`, `--test-data` (synthetic evaluation)
- `--ml-data-root`, `--ml-variant` (MovieLens evaluation)
- `--checkpoint-dir`, `--log-dir`, `--chunk-size-mb`

### Node (`scripts/run_node.py`)

- `--node-id`, `--coordinator`
- `--device`, `--local-epochs`, `--batch-size`, `--lr`, `--num-rounds`
- Data source (mutually exclusive):
  - `--data-dir` (synthetic CSV partitions)
  - `--movielens` + `--ml-variant`
- Partitioning: `--partition`, `--num-partitions`
- `--model-type` = `simple|bpr|neural_cf|two_tower`
- `--embedding-dim`, `--hidden-dim`, `--chunk-size-mb`, `--log-dir`

## Checkpoints and Evaluation

Coordinator saves:

- `checkpoints/model_epoch_<n>.pt` (each round)
- `checkpoints/model_best.pt` (best validation metric)
- `checkpoints/model_final.pt` (final round)

Evaluation mode:

- Synthetic: pointwise metrics (`loss`, `accuracy`, `auc_roc`)
- MovieLens: ranking metrics (`hit@10`, `ndcg@10`, `hit@20`, `ndcg@20`)

## Tests

Integration tests are under `tests/`:

- `tests/synthetic/` → all 4 models on synthetic data
- `tests/movielens/` → all 4 models on MovieLens

Run examples:

```bash
python tests/synthetic/test_simple.py
python tests/movielens/test_bpr.py
```

## Project Structure

```text
fedsys/
├── fedsys/
│   ├── coordinator/
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
└── tests/
    ├── synthetic/
    ├── movielens/
    └── common.py
```
