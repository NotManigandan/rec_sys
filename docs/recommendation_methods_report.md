# Recommendation System Methods and Evaluation Report

## 1. Overview

This report summarizes the recommendation methods currently implemented for the synthetic recommendation-system dataset in `rec_system/` and explains how model performance is measured.

The current dataset contains:

- `2,000` users
- `6` items
- `3` reported ratings per user in `reported.csv`
- full hidden ground-truth ratings in `actual.csv`

The training setup uses the partially observed ratings in `reported.csv` to learn a recommender. The learned model is then evaluated on the `3` unseen items for each user using the hidden truth in `actual.csv`.

This setup is useful because it gives a known ground truth. That makes it possible to measure whether the recommender is learning the users' actual latent preferences rather than only matching noisy observed behavior.

## 2. Methods

### 2.1 Popularity Baseline

The popularity baseline is the simplest method. It does not personalize recommendations by user.

Instead, it computes which items have the highest average reported rating across the whole dataset and recommends those items first.

Strengths:

- very simple
- fast to compute
- useful as a sanity-check baseline

Weaknesses:

- ignores individual user differences
- cannot adapt to different preference profiles
- often over-recommends items that are globally popular but not optimal for a specific user

### 2.2 BPR Matrix Factorization (`bpr_mf`)

This is the main baseline and currently the strongest overall model.

BPR stands for Bayesian Personalized Ranking. In this approach:

- each user gets a learned embedding vector
- each item gets a learned embedding vector
- the user-item score is computed mainly by a dot product between those embeddings, plus user and item bias terms

The model is trained using pairwise ranking comparisons. For a given user, if one reported item has a higher rating than another reported item, the model is trained to score the higher-rated item above the lower-rated one.

Strengths:

- directly optimized for ranking
- simple and interpretable
- a strong fit for collaborative-filtering data
- works well when there are user-item interactions but not many side features

Weaknesses:

- limited expressiveness compared with deeper nonlinear models
- depends mostly on interaction structure rather than rich metadata

### 2.3 Neural Collaborative Filtering (`neural_cf`)

This method is a more flexible nonlinear version of collaborative filtering.

Like matrix factorization, it learns user embeddings and item embeddings. However, instead of using only a dot product, it concatenates the user and item embeddings and passes them through a small multilayer perceptron (MLP) to produce a score.

Strengths:

- more expressive than standard matrix factorization
- can model more complex user-item interactions
- useful as a stronger nonlinear comparison model

Weaknesses:

- can overfit more easily on small datasets
- more parameters and slightly less interpretable than matrix factorization
- extra complexity does not always improve top recommendation quality

### 2.4 Two-Tower Model (`two_tower`)

The two-tower model separates the user representation and item representation into two different encoders.

In the current implementation:

- the user tower converts the user's reported rating history into a user vector
- the item tower uses a learned embedding for each item
- the final score is the similarity between the user vector and the item vector

Two-tower models are common in industrial retrieval systems because they scale well and can use richer user or item features.

Strengths:

- good architecture for large-scale retrieval
- easy to extend with side features
- useful if the project later includes richer metadata or cold-start settings

Weaknesses:

- less effective on this dataset because there are only 6 items and only 3 reported ratings per user
- there are currently no rich side features to justify the architecture
- it underperforms the simpler BPR approach in this setup

## 3. Evaluation Metrics

The evaluation process ranks the `3` unseen items for each user and compares that ranking against the user's hidden true ratings in `actual.csv`.

### 3.1 Top-1 Accuracy

Top-1 accuracy measures whether the model's highest-ranked unseen item is actually one of the user's best unseen items.

Interpretation:

- `1.0` means the model always places a truly best hidden item at the top
- `0.0` means it never does

This is the most intuitive "did we recommend the best item?" metric.

### 3.2 NDCG@3

NDCG stands for Normalized Discounted Cumulative Gain.

This metric evaluates the quality of the full ranking of the `3` hidden items, not just the top choice. Higher-relevance items are rewarded more, and items ranked earlier in the list receive more credit.

Interpretation:

- `1.0` means a perfect ranking
- values closer to `1.0` indicate that the ordering is close to the ideal hidden ranking

### 3.3 Pairwise Accuracy

Pairwise accuracy checks whether the model correctly orders pairs of hidden items when those two items have different true ratings.

Interpretation:

- `1.0` means the model always orders unequal pairs correctly
- `0.5` would be close to random guessing in many cases

This metric is especially useful for ranking models such as BPR because it measures ranking consistency directly.

### 3.4 MRR

MRR stands for Mean Reciprocal Rank.

For each user, it looks at where the best hidden item appears in the ranked list:

- if the best item is ranked first, the score is `1.0`
- if it is ranked second, the score is `0.5`
- if it is ranked third, the score is `0.333...`

The metric then averages this over all users.

Interpretation:

- higher is better
- values close to `1.0` mean the best hidden item usually appears near the top

## 4. Results

Table 1. Overall model performance on hidden-item ranking.

| Method | Top-1 Accuracy | NDCG@3 | Pairwise Accuracy | MRR |
| --- | ---: | ---: | ---: | ---: |
| Popularity Baseline | 0.7260 | 0.9274 | 0.7777 | 0.8545 |
| BPR Matrix Factorization (`bpr_mf`) | 0.8340 | 0.9562 | 0.8212 | 0.9101 |
| Neural Collaborative Filtering (`neural_cf`) | 0.8095 | 0.9493 | 0.8308 | 0.8962 |
| Two-Tower Model (`two_tower`) | 0.6900 | 0.9120 | 0.7182 | 0.8281 |

## 5. Interpretation of Results

The main conclusions are:

1. `bpr_mf` is the strongest overall model on this dataset.
2. `neural_cf` is competitive and slightly better on pairwise ordering, but it does not beat `bpr_mf` on the main top recommendation metrics.
3. `two_tower` is the weakest learned model in this setup, which suggests that the dataset is too small and too feature-poor for that architecture to pay off.
4. all learned personalized models should be compared against the popularity baseline, and both `bpr_mf` and `neural_cf` clearly outperform it on most metrics.

Why BPR performs best here:

- the dataset is small
- the task is inherently ranking-based
- there are no rich user/item features
- the hidden preference structure is well matched by a latent-factor ranking model

Why the two-tower model underperforms:

- it is designed for richer feature settings
- each user only has `3` observed ratings
- there are only `6` items total
- the architecture has less useful structure to exploit than a direct collaborative-filtering model

## 6. Recommendation

For the next phase of the project, the recommended modeling plan is:

1. Use `bpr_mf` as the primary baseline recommender.
2. Keep `neural_cf` as a stronger nonlinear comparison model.
3. Treat the popularity model as the non-personalized sanity-check baseline.
4. Do not prioritize the two-tower model unless the dataset is expanded with richer user or item features.

This gives a clean experimental story for later robustness experiments:

- clean baseline with a strong but simple recommender
- attacked version of the recommender
- defended or robustified version of the recommender

## 7. Limitations

The current results should be interpreted in the context of the dataset design:

- only `6` items are available
- each user reports only `3` observed ratings
- the dataset is synthetic
- no timestamps, text, or metadata features are included

Because of this, the current benchmark is best understood as a controlled proof of concept rather than a realistic production recommendation problem.

## 8. Appendix: One-Sentence Summary of Each Method

- Popularity Baseline: recommend globally popular items to everyone.
- BPR Matrix Factorization: learn user and item latent vectors and rank higher-rated observed items above lower-rated observed items.
- Neural Collaborative Filtering: learn user and item embeddings and use an MLP to model nonlinear interactions.
- Two-Tower Model: encode user history and item identity separately and score them by similarity.
