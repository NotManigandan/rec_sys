# Evaluation Metrics Guide

This file explains **all metrics currently computed in this codebase**, how they are calculated, and what to expect in:

- benign (no attack),
- adversarial attack,
- adversarial attack + defense.

---

## 1) Where metrics come from

- Pointwise (synthetic/BCE): `fedsys/coordinator/evaluator.py` -> `evaluate()`
- Ranking (MovieLens/BPR and other ranking models): `fedsys/coordinator/evaluator.py` -> `evaluate_ranking()`
- Adversarial extensions: `fedsys/adversarial/eval.py` -> `evaluate_with_target_exposure()`
- Attack-vs-clean deltas: `fedsys/adversarial/eval.py` -> `compare_attack_vs_clean()`

---

## 2) Pointwise metrics (synthetic path)

These are used when coordinator evaluates with regular labeled samples (`label` in batch).

## `loss` (BCE loss)

- **What**: Mean binary cross-entropy with logits.
- **How**:
  - For each sample: `BCEWithLogitsLoss(logit, label)`
  - Summed over all samples, then divided by number of samples.
- **Range**: `[0, +inf)` (lower is better).

## `accuracy`

- **What**: Fraction of correct binary decisions.
- **How**:
  - Probability = `sigmoid(logit)`
  - Prediction = `1` if probability `>= 0.5`, else `0`
  - Accuracy = `correct / total`
- **Range**: `[0, 1]` (higher is better).

## `auc_roc`

- **What**: Ranking quality of positive vs negative classes, threshold-independent.
- **How**:
  - Implemented via Mann-Whitney style pairwise comparison (no sklearn).
  - Counts how often positive examples receive higher predicted probability than negative ones.
  - Ties count as `0.5`.
- **Range**: `[0, 1]` (higher is better).

---

## 3) Ranking metrics (MovieLens path)

For each user, model scores all items. Known positive items are masked out except the held-out target (`val_item` or `test_item`). Rank is then computed for that held-out target.

If target rank is `r` (1-based rank in coordinator evaluator):

- rank 1 means best possible recommendation for that held-out item.

## `hit@K`

- **What**: Fraction of users where held-out target item appears in top-K.
- **How**:
  - `hit@K = (# users with rank <= K) / (# evaluated users)`
- **Range**: `[0, 1]` (higher is better).

## `ndcg@K`

- **What**: Position-sensitive top-K quality for held-out target.
- **How**:
  - If rank `r <= K`, contribution = `1 / log2(r + 1)`.
  - Else contribution = `0`.
  - Average over users.
- **Range**: `[0, 1]` (higher is better).

## `top1_accuracy`

- **What**: Fraction of users where held-out target is ranked first.
- **How**:
  - `top1_accuracy = (# users with rank == 1) / (# evaluated users)`
- **Range**: `[0, 1]` (higher is better).

## `ndcg@3`

- **What**: NDCG at cutoff 3 for held-out target.
- **How**:
  - Same rule as `ndcg@K` with `K=3`.
- **Range**: `[0, 1]` (higher is better).

## `mrr`

- **What**: Mean reciprocal rank of held-out target.
- **How**:
  - For each user: contribution = `1 / rank`
  - `mrr = average(1 / rank)`
- **Range**: `(0, 1]` (higher is better).

## `pairwise_accuracy` (ranking adaptation)

- **What**: Fraction of pairwise target-vs-negative comparisons that are correctly ordered.
- **How in this codebase**:
  - With one held-out target and many negatives:
  - For each user, `pairwise_accuracy_user = (# negatives ranked below target) / (# negatives)`
  - Averaged across users.
  - Equivalent formula used:
    - `pairwise_accuracy_user = (num_items - rank) / (num_items - 1)`
- **Range**: `[0, 1]` (higher is better).

---

## 4) Adversarial metrics

When `evaluate_with_target_exposure()` is enabled, it returns:

- Overall ranking metrics (same family as above):  
  `hit@K`, `ndcg@K`, `top1_accuracy`, `ndcg@3`, `mrr`, `pairwise_accuracy`
- Segment metrics (target-genre users):  
  `segment_hit@K`, `segment_ndcg@K`, `segment_top1_accuracy`, `segment_ndcg@3`, `segment_mrr`, `segment_pairwise_accuracy`
- Target-exposure metrics:  
  `target_hit@K`, `target_ndcg@K`

## `segment_*`

- **What**: Same ranking metrics, but computed only on users in `users_by_genre[target_genre]`.
- **Why**: Attack is usually targeted; segment behavior is often more sensitive than global averages.

## `target_hit@K`

- **What**: Fraction of segment users where the **attack target item** appears in top-K.
- **How**:
  - Score all items for each segment user.
  - Mask known positives except target item.
  - Compute rank of target item.
  - `target_hit@K = (# segment users with target rank <= K) / (# segment users)`
- **Range**: `[0, 1]` (higher means stronger target exposure).

## `target_ndcg@K`

- **What**: Position-sensitive version of target exposure.
- **How**:
  - If target rank `r <= K`: contribution = `1 / log2(r + 1)`
  - Else `0`
  - Average over segment users.
- **Range**: `[0, 1]` (higher means stronger target exposure).

---

## 5) Delta metrics (attack vs clean)

`compare_attack_vs_clean(attack_metrics, clean_metrics)` returns:

- `delta_<metric> = attack_metric - clean_metric`

Examples:

- `delta_hit@10 < 0`: benign recommendation quality dropped under attack.
- `delta_target_hit@10 > 0`: target item is being pushed successfully.

---

## 6) What to expect by scenario

These are expected trends, not strict guarantees.

## A) Benign (no attack)

- `hit@K`, `ndcg@K`, `top1_accuracy`, `mrr`, `pairwise_accuracy`: stable baseline.
- `target_hit@K`, `target_ndcg@K` (if tracked): typically low for hard/rare targets.
- `segment_*`: usually close to overall metrics (can differ by genre skew).

## B) Attack, no defense

Typical signature:

- `target_hit@K`, `target_ndcg@K`: **increase** (attacker success).
- `segment_*` for target genre: often shifts more than overall metrics.
- Overall benign quality (`hit@K`, `ndcg@K`, `mrr`, etc.): may degrade, stay similar, or occasionally improve slightly by chance.
- Most diagnostic pattern:
  - `delta_target_*` strongly positive
  - while `delta_hit@K` / `delta_ndcg@K` is flat or negative.

## C) Attack + defense

Expected defense behavior:

- `target_hit@K`, `target_ndcg@K`: reduced versus attack-only run.
- Benign ranking quality recovers partially (or fully) versus attack-only.
- Very strong defenses can slightly hurt benign metrics while suppressing target exposure.

Operationally, a good defense tends to produce:

- lower `target_*` than attack-only,
- acceptable `hit@K` / `ndcg@K` / `mrr` tradeoff versus benign baseline.

---

## 7) Practical interpretation checklist

For each model, compare three runs: clean, attack, attack+defense.

1. Attack success:
   - check `delta_target_hit@10`, `delta_target_ndcg@10`
2. Collateral damage:
   - check `delta_hit@10`, `delta_ndcg@10`, `delta_mrr`
3. Defense value:
   - attack+defense should lower `target_*` vs attack-only
   - while keeping benign metrics near clean baseline.

---

## 8) Notes

- Metric values depend on dataset split, target item difficulty, and model type.
- Single-run comparisons can be noisy; repeat with multiple seeds for robust conclusions.
- Segment-level metrics are important in targeted attacks; global metrics alone can hide attack impact.
