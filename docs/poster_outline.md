# Poster Outline

## 1. Working Title Options

- Robust Federated Recommendation Under Targeted Data Poisoning
- Detecting Malicious Shards in Federated Recommender Training
- Targeted Shilling Attacks and Shard-Level Defenses for Federated Recommendation

## 2. One-Sentence Poster Story

A naive federated recommender can be manipulated by a malicious shard to push a chosen item into the recommendation lists of a chosen user segment, and a simple shard-level concentration defense can substantially reduce that manipulation without severely harming normal recommendation quality.

## 3. Poster Structure

### 3.1 Motivation

- Recommenders can be attacked through fake interactions.
- Federated training makes client or shard trust a central problem.
- Aggregate quality can stay high while a targeted attack still succeeds.

### 3.2 Problem Setup

- Dataset: MovieLens
- Model: federated MF-BPR, optionally Neural BPR
- Users are partitioned into shards
- One shard is malicious
- Goal: push one target item to one target segment

### 3.3 Attack

- Synthetic users are injected into the malicious shard.
- Profiles contain the target item and genre-aligned filler items.
- The attack budget controls the number of synthetic users.
- A heuristic target selector chooses an attackable genre-item pair from the clean model.

### 3.4 Defense

- Compare shard updates at the server.
- Measure how concentrated each shard’s item-level update is.
- Downweight or clip shards that are abnormally focused on a small number of items.
- Main defense: `focus_clip_mean`

### 3.5 Metrics

- `overall HR@10`
- `target_hitrate@3`
- `target_hitrate@5`
- `target_hitrate@10`
- `target_mean_rank`

### 3.6 Main Result

- Clean model: target item has near-zero exposure.
- Attacked model: target exposure rises significantly.
- Defended model: target exposure drops relative to the attacked model.
- Overall HR changes much less than target exposure, showing why targeted metrics are necessary.

### 3.7 Systems Angle

- Multi-GPU shard-parallel training
- Long-history filtering to reduce memory outliers
- Exact attack-bundle caching for fast defense sweeps

## 4. Required Poster Figures and Tables

### Figure 1. Framework Diagram

Show:

- MovieLens data
- federated shard partition
- one malicious shard injecting synthetic users
- server aggregation
- defended aggregation path

### Figure 2. Clean vs Attacked vs Defended Line Plot

Use a line plot for the chosen main model, with:

- x-axis: attack budget
- y-axis: `target_hitrate@10`
- three lines:
  - clean reference
  - attacked with no defense
  - defended with `focus_clip_mean`

Optional second panel:

- x-axis: attack budget
- y-axis: `overall HR@10`

This figure should show that the attack materially raises target exposure while the defense pushes it back down.

### Figure 3. Training and Systems Plot

Add at least one training graph that supports the systems angle.

Best options:

- throughput versus optimization components
- round time versus optimization components
- GPU throughput before and after runtime optimizations

Recommended x-axis categories:

- baseline runtime
- multi-GPU shard parallelism
- long-history filtering
- exact attack-bundle reuse for defense sweeps

### Table 3. Throughput Gain from ML Systems Optimizations

Purpose:

- quantify the speedup from engineering work, not just the modeling work

Suggested columns:

- optimization setting
- defense mode used for benchmark
- samples per second or updates per second
- end-to-end runtime
- peak GPU memory
- relative speedup

Suggested rows:

- baseline single-GPU or early runtime
- multi-GPU shard parallel
- long-history filter enabled
- exact attack-bundle reuse enabled
- final optimized pipeline

### Table 4. Model Comparison

Purpose:

- compare model robustness and standard performance
- justify why one model is used for the main defense experiments

Recommended model rows:

- MF-BPR
- Neural BPR
- one additional comparison model or sanity-check baseline

Recommended columns:

- model
- clean `HR@10`
- attacked `target_hit@10`
- defended `target_hit@10`
- runtime or throughput
- qualitative notes

The qualitative notes should explain why MF-BPR is the best main defense-test model:

- strong enough to be meaningful
- simple enough to interpret
- fast enough for repeated attack and defense ablations

### Table 5. Defense Method Comparison

Purpose:

- compare how different defenses trade off attack suppression against recommendation utility

Suggested defense rows:

- none
- `clip_mean`
- `clip_trimmed_mean`
- `focus_clip_mean`
- `focus_clip_trimmed_mean`

Suggested columns:

- defense
- attacked `target_hit@3`
- attacked `target_hit@5`
- attacked `target_hit@10`
- defended `target_mean_rank`
- `overall HR@10`
- notes on aggressiveness or stability

## 5. Core Ablations to Include

### 5.1 Attack Budget

- `0.1`
- `0.25`
- `0.5`
- `0.8`

Purpose:

- show how attack strength scales with attacker resources

This ablation feeds directly into Figure 2 and Table 5.

### 5.2 Defense Mode

- `none`
- `clip_mean`
- `focus_clip_mean`
- optional `focus_clip_trimmed_mean`

Purpose:

- show that abnormal item-focus detection beats or complements simpler clipping

This ablation feeds directly into Table 5.

### 5.3 Target Selection Strategy

- naive low-exposure selector
- heuristic vulnerability selector

Purpose:

- show that smarter attacker targeting matters

### 5.4 Number of Shards

- `4`
- `8`
- `16`

Purpose:

- show how federated partitioning changes attack strength and defense behavior

### 5.5 Long-History Filtering

- no filter
- `max_positive_interactions = 200`

Purpose:

- show a systems optimization that improves memory behavior without changing the story too much

This ablation feeds directly into Figure 3 and Table 3.

### 5.6 Model Type

- `mf_bpr`
- `neural_bpr`

Purpose:

- show the framework is not limited to only one recommender model

This ablation feeds directly into Table 4.

## 6. Key Claims We Can Safely Make

- A federated recommender can be vulnerable to targeted poisoning from a malicious shard.
- Aggregate quality metrics can hide successful targeted manipulation.
- Comparing shard updates can reveal suspicious concentration patterns.
- A simple shard-level focus defense can reduce attack success with limited utility loss.

## 7. Claims We Should Avoid Overstating

- We should not claim universal robustness to adaptive attackers.
- We should not claim our heuristic defense solves all poisoning settings.
- We should not claim this benchmark is identical to production federated deployment.

## 8. Limitations Section

- one malicious shard
- dataset-aware attacker
- one-item targeting in the current main experiments
- heuristic defense rather than learned detector
- simulation benchmark rather than real-world deployment

## 9. Presenter Notes

### If someone asks why global HR barely changes

Say:

- this is a targeted manipulation attack
- the attacker cares about one item and one audience
- global quality can stay similar while a specific segment is manipulated

### If someone asks why target hit matters

Say:

- it directly measures whether the attacker’s item enters the top-K list seen by the intended audience

### If someone asks why the defense is server-side

Say:

- in federated training, the server naturally sees shard updates
- that makes shard-level update screening a practical first defense

## 10. Poster Build Priority

If space is limited, prioritize:

1. framework diagram
2. Figure 2 attack-budget line plot
3. Table 5 defense comparison
4. Table 4 model comparison
5. Figure 3 systems or training graph
