from __future__ import annotations

import math
from typing import Dict, List

import torch

from .bpr import FederatedConfig, FederatedShard, ServerState, build_scorer_module, score_candidate_items
from .movielens import MovieLensFederatedDataset

TARGET_HIT_CUTOFFS = (3, 5, 10)


def _dcg(rank_position: int) -> float:
    return 1.0 / math.log2(rank_position + 2.0)


def evaluate_federated_model(
    dataset: MovieLensFederatedDataset,
    server_state: ServerState,
    config: FederatedConfig,
    user_states: Dict[int, torch.Tensor],
    shards: Dict[int, FederatedShard],
    target_genre: str,
    target_item_index: int,
    malicious_shard_id: int,
    top_k: int,
    num_eval_negatives: int,
    seed: int,
    eval_device: torch.device,
    eval_batch_size: int = 1024,
) -> Dict[str, object]:
    user_to_shard: Dict[int, int] = {}
    shard_local_lookup: Dict[int, Dict[int, int]] = {}
    for shard_id, shard in shards.items():
        shard_local_lookup[shard_id] = {
            user_index: local_index for local_index, user_index in enumerate(shard.user_indices)
        }
        for user_index in shard.user_indices:
            user_to_shard[user_index] = shard_id

    benign_users = [
        user_index
        for user_index in range(dataset.num_users)
        if user_to_shard.get(user_index) != malicious_shard_id
    ]
    target_segment_user_set = {
        user_index
        for user_index in benign_users
        if dataset.dominant_genre_by_user.get(user_index) == target_genre
    }

    per_segment: Dict[str, Dict[str, float]] = {}
    for genre, users in dataset.users_by_genre.items():
        benign_segment_users = [user_index for user_index in users if user_index in benign_users]
        if not benign_segment_users:
            continue
        segment_metrics = {
            "users": float(len(benign_segment_users)),
            "eligible_users": 0.0,
            "target_hitrate@k": 0.0,
            "target_mean_rank": 0.0,
        }
        for cutoff in TARGET_HIT_CUTOFFS:
            segment_metrics[f"target_hitrate@{cutoff}"] = 0.0
        per_segment[genre] = segment_metrics

    server_state_on_device = server_state.to(eval_device)
    scorer_module = build_scorer_module(server_state_on_device, config, eval_device)
    generator_device = "cpu" if eval_device.type == "cpu" else str(eval_device)
    eval_generator = torch.Generator(device=generator_device)
    eval_generator.manual_seed(seed)

    overall_hits = 0.0
    overall_ndcg = 0.0
    target_hits = 0.0
    target_exposure = 0.0
    target_rank_total = 0.0
    target_rank_users = 0
    target_hits_by_cutoff = {cutoff: 0.0 for cutoff in TARGET_HIT_CUTOFFS}

    for start in range(0, len(benign_users), max(eval_batch_size, 1)):
        chunk_users = benign_users[start : start + max(eval_batch_size, 1)]
        known_item_rows: List[List[int]] = []
        include_target_column: List[bool] = []
        user_embeddings_rows: List[torch.Tensor] = []
        test_items: List[int] = []

        for user_index in chunk_users:
            shard_id = user_to_shard[user_index]
            local_index = shard_local_lookup[shard_id][user_index]
            user_embeddings_rows.append(user_states[shard_id][local_index].detach().cpu())
            split = dataset.splits_by_user[user_index]
            known_item_rows.append(list(split.known_items) or [-1])
            include_target_column.append(target_item_index not in split.known_items and target_item_index != split.test_item)
            test_items.append(split.test_item)

        max_known_items = max(len(row) for row in known_item_rows)
        known_item_matrix = torch.full((len(chunk_users), max_known_items), -1, dtype=torch.long)
        for row_index, known_items in enumerate(known_item_rows):
            known_item_matrix[row_index, : len(known_items)] = torch.tensor(known_items, dtype=torch.long)

        user_embeddings = torch.stack(user_embeddings_rows).to(eval_device)
        known_item_matrix_device = known_item_matrix.to(eval_device)
        test_items_device = torch.tensor(test_items, dtype=torch.long, device=eval_device)

        negative_items = torch.randint(
            high=dataset.num_items,
            size=(len(chunk_users), num_eval_negatives),
            device=eval_device,
            generator=eval_generator,
        )
        invalid_negatives = (
            (known_item_matrix_device.unsqueeze(1) == negative_items.unsqueeze(2)).any(dim=2)
            | (negative_items == target_item_index)
            | (negative_items == test_items_device.unsqueeze(1))
        )
        while invalid_negatives.any():
            replacement = torch.randint(
                high=dataset.num_items,
                size=(int(invalid_negatives.sum().item()),),
                device=eval_device,
                generator=eval_generator,
            )
            negative_items[invalid_negatives] = replacement
            invalid_negatives = (
                (known_item_matrix_device.unsqueeze(1) == negative_items.unsqueeze(2)).any(dim=2)
                | (negative_items == target_item_index)
                | (negative_items == test_items_device.unsqueeze(1))
            )

        candidate_width = 1 + num_eval_negatives + 1
        candidate_matrix_device = torch.full(
            (len(chunk_users), candidate_width),
            0,
            dtype=torch.long,
            device=eval_device,
        )
        valid_mask_device = torch.zeros(
            (len(chunk_users), candidate_width),
            dtype=torch.bool,
            device=eval_device,
        )
        candidate_matrix_device[:, 0] = test_items_device
        valid_mask_device[:, 0] = True
        candidate_matrix_device[:, 1 : 1 + num_eval_negatives] = negative_items
        valid_mask_device[:, 1 : 1 + num_eval_negatives] = True
        if any(include_target_column):
            include_target_tensor = torch.tensor(include_target_column, dtype=torch.bool, device=eval_device)
            candidate_matrix_device[include_target_tensor, -1] = target_item_index
            valid_mask_device[include_target_tensor, -1] = True

        scores = score_candidate_items(
            user_embeddings=user_embeddings,
            item_indices=candidate_matrix_device,
            server_state=server_state_on_device,
            config=config,
            scorer_module=scorer_module,
        )
        scores = scores.masked_fill(~valid_mask_device, float("-inf"))
        ranked_indices = torch.argsort(scores, dim=1, descending=True)
        ranked_items = candidate_matrix_device.gather(1, ranked_indices).cpu()

        for row_index, user_index in enumerate(chunk_users):
            ranked_items_row = ranked_items[row_index].tolist()
            split = dataset.splits_by_user[user_index]
            test_rank = ranked_items_row.index(test_items[row_index])
            if test_rank < top_k:
                overall_hits += 1.0
                overall_ndcg += _dcg(test_rank)

            genre = dataset.dominant_genre_by_user.get(user_index)
            target_is_eligible = target_item_index not in split.known_items
            if genre in per_segment and target_is_eligible:
                target_rank = ranked_items_row.index(target_item_index) + 1
                per_segment[genre]["eligible_users"] += 1.0
                per_segment[genre]["target_mean_rank"] += target_rank
                if target_rank <= top_k:
                    per_segment[genre]["target_hitrate@k"] += 1.0
                for cutoff in TARGET_HIT_CUTOFFS:
                    if target_rank <= cutoff:
                        per_segment[genre][f"target_hitrate@{cutoff}"] += 1.0

            if user_index in target_segment_user_set and target_is_eligible:
                target_rank = ranked_items_row.index(target_item_index) + 1
                target_rank_total += target_rank
                target_rank_users += 1
                if target_rank <= top_k:
                    target_hits += 1.0
                    target_exposure += 1.0
                for cutoff in TARGET_HIT_CUTOFFS:
                    if target_rank <= cutoff:
                        target_hits_by_cutoff[cutoff] += 1.0

    benign_user_count = max(len(benign_users), 1)
    target_user_count = max(target_rank_users, 1)
    segment_summary = {}
    for genre, metrics in per_segment.items():
        eligible_users = float(max(metrics["eligible_users"], 1.0))
        summary = {
            "users": int(metrics["users"]),
            "eligible_users": int(metrics["eligible_users"]),
            "target_hitrate@k": metrics["target_hitrate@k"] / eligible_users,
            "target_mean_rank": metrics["target_mean_rank"] / eligible_users,
        }
        for cutoff in TARGET_HIT_CUTOFFS:
            summary[f"target_hitrate@{cutoff}"] = metrics[f"target_hitrate@{cutoff}"] / eligible_users
        segment_summary[genre] = summary

    target_segment_metrics = {
        "genre": target_genre,
        "users": target_rank_users,
        "target_hitrate@k": target_hits / target_user_count,
        "target_exposure_rate@k": target_exposure / target_user_count,
        "target_mean_rank": target_rank_total / target_user_count,
    }
    for cutoff in TARGET_HIT_CUTOFFS:
        target_segment_metrics[f"target_hitrate@{cutoff}"] = target_hits_by_cutoff[cutoff] / target_user_count

    return {
        "overall_benign": {
            "hr@k": overall_hits / benign_user_count,
            "ndcg@k": overall_ndcg / benign_user_count,
            "users": benign_user_count,
        },
        "target_segment": target_segment_metrics,
        "per_segment": segment_summary,
    }
