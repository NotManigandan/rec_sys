# Project Description

## 1. Executive Summary

This project studies whether a federated recommender system can be manipulated by a malicious shard and whether a lightweight server-side defense can reduce that manipulation without destroying normal recommendation quality.

The current benchmark uses MovieLens data and a federated Bayesian Personalized Ranking (MF-BPR) recommender. One shard is treated as malicious. That shard injects synthetic users designed to push a chosen target item into the recommendation lists of a chosen user segment, such as users whose dominant genre is Action.

The main story we want to show is:

- a naive federated recommender is vulnerable to targeted poisoning
- the attack can noticeably increase exposure of the attacker’s chosen item
- overall recommendation quality can remain fairly stable, which makes the attack harder to notice
- a shard-level defense based on abnormal update concentration can suppress the attack

## 2. Why This Project Matters

Recommenders are often evaluated only with aggregate quality metrics such as hit rate or NDCG. That can hide a more subtle failure mode: the system may still look healthy overall while being manipulated for a specific audience.

Federated training makes this more interesting because the server does not directly see all raw user data. Instead, it aggregates shard or client updates. That means an attacker may be able to poison one shard and influence the final global model through the aggregation step.

This project therefore combines three themes:

- recommender systems
- federated learning
- adversarial robustness

## 3. High-Level Framework

The full pipeline is:

1. Load and preprocess MovieLens.
2. Split users into federated shards.
3. Train a clean federated recommender.
4. Use a heuristic target selector to choose an attackable genre-item target.
5. Inject synthetic users into one malicious shard.
6. Retrain under attack and measure how much target exposure increases.
7. Apply a defense during shard aggregation and compare clean, attacked, and defended outcomes.

At a high level, the benchmark asks:

- Can one bad shard make the recommender push a chosen item to a chosen audience?
- Can the server detect or suppress the abnormal shard update using only update statistics?

## 4. Dataset and Preprocessing

The current federated benchmark uses MovieLens, with the main large-scale experiments run on `ml-25m` and some earlier debugging runs on `ml-1m` and `ml-10m`.

Important preprocessing choices:

- only positive interactions above a threshold are kept for the federated ranking task
- each user’s interactions are sorted chronologically
- the last interactions are held out for validation and test
- each user is assigned a dominant genre based on their training-history preferences
- users are partitioned into shards for federated training

There is also an optional `max_positive_interactions` filter, which removes extreme long-history users. This is mainly a systems optimization because a few very long histories can create padded tensor outliers and highly uneven GPU memory use.

## 5. Federated Training Setup

The current federated simulation is shard-based rather than true cross-machine production federated learning.

Each shard contains a subset of users. In each federated round:

- the server sends the current global item-side state to each shard
- each shard trains locally for a small number of epochs
- each shard returns an update
- the server aggregates those updates into the next global state

We currently support multi-GPU shard-parallel training. Different shards can be assigned to different GPUs, so the benchmark has a genuine systems angle in addition to the modeling and robustness angle.

## 6. Models

### 6.1 MF-BPR

This is the main model and the primary baseline.

MF-BPR learns:

- a user embedding
- an item embedding
- an item bias

Training uses a pairwise ranking loss. Intuitively, for a sampled user, the model is trained so that a positive item should score above a sampled negative item.

Why it is a good main baseline:

- simple and interpretable
- directly optimized for ranking
- fast enough to support many attack and defense sweeps
- strong enough to be meaningful

### 6.2 Neural BPR

This is a heavier neural variant that keeps the same overall federated setup but uses a larger neural scorer. It is useful as a second model because it helps us show that the attack and defense story is not limited to the simplest matrix-factorization baseline.

Why we are not prioritizing SASRec yet:

- our immediate goal is a clean vulnerability and defense story
- MF-BPR and Neural BPR are easier to interpret and iterate on
- transformer-based sequential models are still an option for future extension

## 7. Attack Formulation

### 7.1 Threat Model

The current threat model assumes:

- one shard is malicious
- the malicious shard injects synthetic users
- the attacker wants to increase recommendation exposure of one target item
- the target audience is a specific benign user segment, usually defined by dominant genre

This is currently a mixed shard, not a pure malicious-only shard. That means the malicious shard contains both benign and synthetic users.

### 7.2 Synthetic User Attack

Each synthetic user includes:

- the target item
- multiple target-genre filler items
- optionally neutral filler items

The attack is intentionally aggressive. The target item is given a large positive sampling weight, and the filler items are chosen to make the profile look aligned with the target audience.

The attack budget controls how many synthetic users are added relative to the benign users inside the malicious shard.

### 7.3 Heuristic Target Selector

We no longer select the target item simply by low popularity. Instead, we use a vulnerability heuristic based on the clean model.

The selector looks for a target that has:

- a reasonably large reachable audience
- enough support from that audience’s training behavior
- low current top-K exposure
- a plausible path to crossing the top-K boundary under attack

This matches a fixed-resource attacker better. Instead of attacking blindly, the attacker tries to spend its budget on the most attackable item-audience pair.

## 8. Defense Formulation

### 8.1 Why Shard-Level Defense

A learned detector would require a large amount of attack/no-attack training data. That is expensive and also unnecessary for a first defense.

Instead, we use the key assumption that an effective malicious shard should look different from other shards in its update behavior. In particular, it may push a small number of items much harder than normal benign shards do.

### 8.2 Main Defense: Focus-Based Clipping

The main defense compares shard updates and looks for abnormal concentration in the item-level update mass.

Intuition:

- a benign shard spreads its learning across many items
- a malicious shard trying to push one or a few items tends to create a more concentrated update

The current defense modes include:

- `none`
- `clip_mean`
- `clip_trimmed_mean`
- `focus_clip_mean`
- `focus_clip_trimmed_mean`

The most important ones for our poster are the naive baseline with `none` and the abnormal-shard defense with `focus_clip_mean`.

### 8.3 What the Defense Is Doing

The defense computes shard-level concentration scores such as whether the top few pushed items take up an abnormally large share of the shard’s item update. Suspicious shards are clipped or downweighted before the server averages the updates.

This is intentionally simple:

- no extra detector training
- no access to raw user data
- directly aligned with the federated aggregation point

## 9. Evaluation Metrics

We care about two kinds of outcomes.

### 9.1 Normal Recommendation Quality

- `overall HR@10`
- optionally `NDCG@10`

These tell us whether the recommender is still doing its normal job well for benign users.

### 9.2 Targeted Attack Success

- `target_hitrate@3`
- `target_hitrate@5`
- `target_hitrate@10`
- `target_mean_rank`

These tell us whether the attacker’s chosen item is being pushed into the recommendation lists of the target audience.

This distinction is important. A targeted attack can succeed even when global quality barely changes.

## 10. Current Experimental Story

The clean model starts with near-zero exposure of the chosen target item to the chosen segment.

Under attack:

- target hit rates increase substantially
- the target item’s mean rank improves
- overall HR@10 often changes only slightly

This supports the story that targeted manipulation can be real and meaningful even when aggregate recommender quality still looks healthy.

Under defense:

- target hit rates should fall back toward the clean baseline
- target mean rank should worsen for the attacker
- overall benign quality should remain relatively stable

## 11. Key Ablations We Should Run

### 11.1 Attack Budget

Vary the number of synthetic users added to the malicious shard.

Suggested values:

- `0.1`
- `0.25`
- `0.5`
- `0.8`

This shows how attack strength scales with attacker resources.

### 11.2 Defense Mode

Compare:

- `none`
- `clip_mean`
- `focus_clip_mean`
- optionally `focus_clip_trimmed_mean`

This tests whether update concentration is a better signal than generic norm clipping.

### 11.3 Target Selection Strategy

Compare:

- older naive low-exposure targeting
- heuristic vulnerability-based targeting

This shows that attacker resource allocation matters.

### 11.4 Number of Shards

Suggested values:

- `4`
- `8`
- `16`

This matters because more shards can dilute the influence of one malicious shard, while fewer shards give each shard more aggregation weight.

### 11.5 Long-History Filtering

Compare:

- no max-history filter
- `max_positive_interactions = 200`

This is mostly a systems ablation, but we should verify that it does not change the attack story too much.

### 11.6 Model Comparison

If time permits, compare:

- `mf_bpr`
- `neural_bpr`

This helps show that the results are not specific to only one recommender architecture.

## 12. Planned Poster Assets

The poster should not only report the final attack and defense result. It should also make the framework easy to understand visually.

### 12.1 Figure 1: Framework Diagram

This figure should illustrate the full benchmark flow:

- MovieLens preprocessing
- shard partitioning
- clean federated training
- heuristic target selection
- malicious synthetic-user injection into one shard
- attacked federated training
- defended aggregation at the server

This is the most important explanatory figure for teammates and for the poster audience.

### 12.2 Figure 2: Clean vs Attacked vs Defended Line Plot

This should be a line plot over attack budgets for the model we choose as the main benchmark, which is currently MF-BPR.

Recommended x-axis:

- attack budget

Recommended y-axis:

- `target_hitrate@10`

Recommended lines:

- clean reference
- attacked with no defense
- defended with `focus_clip_mean`

Optional second panel:

- `overall HR@10` versus attack budget

This figure directly shows the project’s main story.

### 12.3 Figure 3: Training or Throughput Graph

We also want one systems-oriented plot, such as:

- throughput before and after optimization components
- round time before and after optimization components
- end-to-end runtime for baseline versus optimized pipeline

This figure supports the claim that the project also includes ML systems optimization work rather than only attack and defense modeling.

### 12.4 Table 3: Throughput Gain from ML Systems Optimizations

This table should summarize the performance effect of the main engineering components.

Suggested rows:

- baseline pipeline
- multi-GPU shard parallelism
- long-history filtering
- exact attack-bundle reuse for defense sweeps
- final optimized runtime

Suggested columns:

- setting
- defense mode used during benchmark
- throughput
- end-to-end runtime
- peak GPU memory
- relative speedup

### 12.5 Table 4: Model Comparison

This table should compare about three model choices in terms of both robustness and cost.

Recommended rows:

- MF-BPR
- Neural BPR
- one additional comparison model or sanity-check baseline

Recommended columns:

- clean `HR@10`
- attacked `target_hit@10`
- defended `target_hit@10`
- runtime or throughput
- interpretation

The purpose of this table is not just to compare numbers. It should also explain why MF-BPR is the best model for the main defense experiments:

- it is strong enough to be meaningful
- it is simple enough to interpret
- it is cheap enough to support many attack and defense sweeps

### 12.6 Table 5: Defense Comparison

This table should compare the main defense options directly.

Recommended rows:

- none
- `clip_mean`
- `clip_trimmed_mean`
- `focus_clip_mean`
- `focus_clip_trimmed_mean`

Recommended columns:

- `target_hit@3`
- `target_hit@5`
- `target_hit@10`
- `target_mean_rank`
- `overall HR@10`
- short notes

This table is where we show whether the focus-based abnormal-shard defense is actually better than simpler baselines.

## 13. Systems Contributions and Practical Engineering

This project also has a clear systems component:

- multi-GPU shard-parallel training
- aggressive attack parameterization for meaningful stress tests
- long-history filtering to reduce padded tensor outliers
- exact attack-bundle caching so the same attacked setup can be reused for multiple defense runs

The attack-bundle cache is especially useful because it ensures exact reuse of:

- the shard partition
- the selected target genre and target item
- the poisoning parameters
- the clean baseline
- the naive attacked baseline

This makes defense comparisons much faster and much more reproducible.

## 14. Main Assumptions and Limitations

We should be explicit about what this project currently assumes.

- one shard is malicious
- the attack is dataset-aware
- the attack currently targets one item at a time
- the defense is heuristic rather than learned
- the benchmark is a simulation, not a deployment

These are reasonable assumptions for a first stage, but they should be described honestly on the poster.

## 15. What We Want Teammates to Understand Quickly

If someone knows nothing about the project, the most important points are:

- we are training a recommender in a federated setting
- one shard can poison the training data
- the attacker is trying to get one item recommended to one audience
- global quality metrics alone may not reveal the problem
- our defense looks for shards whose updates are abnormally concentrated on a few items
- the main evidence comes from comparing clean, attacked, and defended runs

## 16. Recommended Talking Track for Team Meetings

1. Start with the problem: targeted manipulation in federated recommendation.
2. Explain the benchmark: MovieLens, shards, MF-BPR, one malicious shard.
3. Explain the attack goal: push a chosen item to a chosen audience.
4. Explain why overall HR is not enough.
5. Show target hit metrics under clean versus attacked training.
6. Show how the focus-based defense suppresses abnormal shard updates.
7. End with ablations and next steps.

## 17. Immediate Next Steps

- run defended experiments from exact saved attack bundles
- compare `none` versus `focus_clip_mean`
- prepare the core clean/attacked/defended plots
- run the main ablations for the poster
- optionally add `neural_bpr` as a second model family
