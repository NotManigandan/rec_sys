# Architecture Notes

This note summarizes the local benchmark run on April 27, 2026 using:

```bash
python -m recsys.train --compare-all --epochs 200 --device cuda --output artifacts/model_report.json
```

## Current Results

| Model | Top-1 | NDCG@3 | Pairwise Acc. | MRR |
| --- | ---: | ---: | ---: | ---: |
| `bpr_mf` | 0.8340 | 0.9562 | 0.8212 | 0.9101 |
| `neural_cf` | 0.8095 | 0.9493 | 0.8308 | 0.8962 |
| `two_tower` | 0.6900 | 0.9120 | 0.7182 | 0.8281 |
| `popularity` | 0.7260 | 0.9274 | 0.7777 | 0.8545 |

## Interpretation

### `bpr_mf`

This is the strongest overall baseline for the current dataset.

- The data is tiny: 6 items and 3 observed ratings per user.
- The signal is mostly collaborative, not feature-rich.
- A simple latent-factor model is enough to recover most of the hidden ranking structure.

### `neural_cf`

This is the best nonlinear comparison model in the current codebase.

- It slightly improves pairwise ordering.
- It does not beat `bpr_mf` on the top recommendation metrics.
- It is more expressive, but that extra capacity does not buy much on this small synthetic setup.

### `two_tower`

This is the right shape if the project later adds side features or cold-start constraints, but it is not the best fit for the current data.

- The user tower only sees the 3 reported ratings.
- There are no extra user or item features yet.
- It performs well for the majority profile but degrades sharply on smaller or less common profiles.

## Recommended Next Models

### `LightGCN`

This is the best next architecture to try if the dataset remains interaction-only.

- It matches the user-item bipartite graph structure directly.
- It is a standard strong baseline for implicit or ranking-focused recommendation.
- It is still simpler and easier to analyze than sequence transformers.

### `TorchRec` / `DLRM`-style models

This is the best path if the project later adds feature-rich users/items and scales to larger item vocabularies.

- Good fit for production-style sparse embeddings.
- Useful if you want the multi-GPU part of the project to highlight sharded embedding tables.

### `SASRec` or other sequence models

Do not prioritize these unless the simulator starts modeling time, exposure order, or session effects.

- The current data is not sequential.
- A sequence model would add complexity without matching the problem structure.

## Recommendation For The Project

Use `bpr_mf` as the primary baseline for the clean/poisoned/robust comparison.

Keep `neural_cf` as a stronger nonlinear check.

Only invest in `two_tower` or more complex architectures after the team decides to add richer side information or a more realistic sequential simulator.
