from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import random
from pathlib import Path
from threading import Lock
from typing import Dict, List

import torch

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - exercised only when tqdm is absent.
    tqdm = None

from .attack import build_poisoned_shards
from .bpr import (
    FederatedConfig,
    FederatedShard,
    ShardUpdate,
    ShardRuntime,
    ServerState,
    aggregate_server_states,
    build_clean_shards,
    build_initial_server_state,
    build_initial_user_states,
    build_prepared_shards,
    build_scorer_module,
    build_shard_runtimes,
    score_candidate_items,
    train_local_shard_runtime,
)
from ..device import resolve_runtime_device
from .eval import evaluate_federated_model
from .movielens import MovieLensFederatedDataset, benign_segment_users, load_movielens_dataset, partition_users

TARGET_SELECTION_CANDIDATE_POOL_SIZE = 128
TARGET_SELECTION_MAX_CLEAN_HITRATE = 0.02
TARGET_SELECTION_MARGIN_EPS = 1e-2
TARGET_SELECTION_TOP_CANDIDATES = 5
ATTACK_BUNDLE_VERSION = 1


class RoundProgress:
    def __init__(self, total: int, desc: str, enabled: bool) -> None:
        self.total = total
        self.desc = desc
        self.enabled = enabled
        self.current = 0
        self._bar = None
        if enabled and tqdm is not None:
            self._bar = tqdm(total=total, desc=desc, dynamic_ncols=True, position=0)

    def update(self, metrics: Dict[str, object] | None = None) -> None:
        self.current += 1
        if not self.enabled:
            return
        postfix = None
        if metrics is not None:
            postfix = {}
            overall_benign = metrics.get("overall_benign")
            if isinstance(overall_benign, dict) and "hr@k" in overall_benign:
                postfix["hr"] = f"{overall_benign['hr@k']:.4f}"
            target_segment = metrics.get("target_segment")
            if isinstance(target_segment, dict) and "target_hitrate@k" in target_segment:
                postfix["target_hit"] = f"{target_segment['target_hitrate@k']:.4f}"
            training = metrics.get("training")
            if isinstance(training, dict) and "mean_loss" in training:
                postfix["loss"] = f"{training['mean_loss']:.4f}"
            if not postfix:
                postfix = None
        if self._bar is not None:
            self._bar.update(1)
            if postfix is not None:
                self._bar.set_postfix(postfix, refresh=False)
            return
        if postfix is None:
            print(f"{self.desc}: round {self.current}/{self.total}", flush=True)
        else:
            detail_parts = []
            if "hr" in postfix:
                detail_parts.append(f"hr={postfix['hr']}")
            if "target_hit" in postfix:
                detail_parts.append(f"target_hit={postfix['target_hit']}")
            if "loss" in postfix:
                detail_parts.append(f"loss={postfix['loss']}")
            details = " ".join(detail_parts)
            print(f"{self.desc}: round {self.current}/{self.total} {details}".rstrip(), flush=True)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


class ShardProgress:
    def __init__(self, total: int, desc: str, enabled: bool) -> None:
        self.total = total
        self.desc = desc
        self.enabled = enabled
        self.current = 0
        self._lock = Lock()
        self._bar = None
        if enabled and tqdm is not None:
            self._bar = tqdm(
                total=total,
                desc=desc,
                dynamic_ncols=True,
                position=1,
                leave=False,
            )

    def update(self, shard_id: int, sample_count: int) -> None:
        with self._lock:
            self.current += 1
            if not self.enabled:
                return
            postfix = {
                "shard": shard_id,
                "samples": sample_count,
            }
            if self._bar is not None:
                self._bar.update(1)
                self._bar.set_postfix(postfix, refresh=False)
                return
            print(
                f"{self.desc}: shard {self.current}/{self.total} "
                f"id={shard_id} samples={sample_count}",
                flush=True,
            )

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def set_seed(seed: int, device: torch.device | None = None) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if device is not None and device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _parse_attack_budgets(text: str) -> List[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def _attack_params_from_args(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "filler_items_per_user": args.attack_filler_items_per_user,
        "neutral_items_per_user": args.attack_neutral_items_per_user,
        "target_item_weight": args.attack_target_weight,
        "filler_item_weight": args.attack_filler_weight,
        "neutral_item_weight": args.attack_neutral_weight,
        "filler_candidate_pool_size": args.attack_filler_pool_size,
        "neutral_candidate_pool_size": args.attack_neutral_pool_size,
    }


def _serialize_shard_to_users(shard_to_users: Dict[int, tuple[int, ...]]) -> Dict[str, List[int]]:
    return {
        str(shard_id): list(user_indices)
        for shard_id, user_indices in sorted(shard_to_users.items())
    }


def _deserialize_shard_to_users(payload: Dict[str, List[int]]) -> Dict[int, tuple[int, ...]]:
    return {
        int(shard_id): tuple(int(user_index) for user_index in user_indices)
        for shard_id, user_indices in payload.items()
    }


def _benchmark_signature(
    args: argparse.Namespace,
    training_devices: List[torch.device],
    *,
    attack_budgets: List[float] | None = None,
    attack_params: Dict[str, object] | None = None,
    runtime_seed: int | None = None,
    num_shards: int | None = None,
) -> Dict[str, object]:
    return {
        "dataset_variant": args.dataset_variant,
        "model": args.model,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "mlp_layers": args.mlp_layers,
        "dropout": args.dropout,
        "local_epochs": args.local_epochs,
        "federated_rounds": args.federated_rounds,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "eval_every": args.eval_every,
        "eval_batch_size": args.eval_batch_size,
        "top_k": args.top_k,
        "num_eval_negatives": args.num_eval_negatives,
        "min_positive_rating": args.min_positive_rating,
        "max_positive_interactions": args.max_positive_interactions,
        "num_shards": args.num_shards if num_shards is None else num_shards,
        "devices": [str(training_device) for training_device in training_devices],
        "attack_budgets": _parse_attack_budgets(args.attack_budgets) if attack_budgets is None else attack_budgets,
        "attack_params": _attack_params_from_args(args) if attack_params is None else attack_params,
        "seed": args.seed if runtime_seed is None else runtime_seed,
    }


def _build_attack_bundle(
    *,
    dataset: MovieLensFederatedDataset,
    args: argparse.Namespace,
    shard_to_users: Dict[int, tuple[int, ...]],
    target_genre: str,
    target_item_index: int,
    target_selection_report: Dict[str, object] | None,
    clean_section: Dict[str, object],
    attack_results: List[Dict[str, object]],
    training_devices: List[torch.device],
    attack_budgets: List[float],
    attack_params: Dict[str, object],
    runtime_seed: int,
) -> Dict[str, object]:
    return {
        "version": ATTACK_BUNDLE_VERSION,
        "dataset": {
            "variant": dataset.variant,
            "num_users": dataset.num_users,
            "num_items": dataset.num_items,
            "min_positive_rating": args.min_positive_rating,
            "max_positive_interactions": args.max_positive_interactions,
        },
        "shard_to_users": _serialize_shard_to_users(shard_to_users),
        "attack_setup": {
            "malicious_shard_id": args.malicious_shard_id,
            "target_genre": target_genre,
            "target_item_index": target_item_index,
            "target_movie_id": dataset.movie_id_for_item(target_item_index),
            "target_title": dataset.title_for_item(target_item_index),
            "target_selection": target_selection_report,
            "attack_budgets": attack_budgets,
            "attack_params": attack_params,
            "seed": runtime_seed,
        },
        "benchmark_signature": _benchmark_signature(
            args,
            training_devices,
            attack_budgets=attack_budgets,
            attack_params=attack_params,
            runtime_seed=runtime_seed,
            num_shards=len(shard_to_users),
        ),
        "clean": clean_section,
        "attacks": attack_results,
    }


def _load_attack_bundle(path: Path) -> Dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("version") != ATTACK_BUNDLE_VERSION:
        raise ValueError(f"Unsupported attack bundle version: {payload.get('version')}")
    return payload


def _validate_attack_bundle(
    bundle: Dict[str, object],
    dataset: MovieLensFederatedDataset,
    args: argparse.Namespace,
) -> None:
    bundle_dataset = bundle["dataset"]
    expected = {
        "variant": dataset.variant,
        "num_users": dataset.num_users,
        "num_items": dataset.num_items,
        "min_positive_rating": args.min_positive_rating,
        "max_positive_interactions": args.max_positive_interactions,
    }
    mismatches = [
        key for key, value in expected.items()
        if bundle_dataset.get(key) != value
    ]
    if mismatches:
        mismatch_text = ", ".join(
            f"{key}=bundle:{bundle_dataset.get(key)!r} current:{expected[key]!r}"
            for key in mismatches
        )
        raise ValueError(f"Attack bundle is incompatible with the current dataset settings: {mismatch_text}")


def _target_item_candidates(
    dataset: MovieLensFederatedDataset,
    target_genre: str,
    target_segment_users: List[int],
) -> List[int]:
    candidates = []
    for item_index in dataset.items_by_genre.get(target_genre, ()):
        if all(item_index not in dataset.splits_by_user[user_index].known_items for user_index in target_segment_users):
            candidates.append(item_index)
    if candidates:
        return candidates
    return list(dataset.items_by_genre.get(target_genre, ()))


def select_target_item(
    dataset: MovieLensFederatedDataset,
    target_genre: str,
    target_segment_users: List[int],
) -> int:
    candidates = _target_item_candidates(dataset, target_genre, target_segment_users)
    if not candidates:
        raise ValueError(f"No candidate items were found for target genre {target_genre}.")
    ranked_candidates = sorted(
        candidates,
        key=lambda item_index: (dataset.train_item_popularity.get(item_index, 0), item_index),
    )
    return ranked_candidates[0]


def select_target_item_from_clean_model(
    dataset: MovieLensFederatedDataset,
    clean_server_state: ServerState,
    config: FederatedConfig,
    clean_user_states: Dict[int, torch.Tensor],
    clean_shards: Dict[int, FederatedShard],
    target_genre: str,
    target_segment_users: List[int],
    malicious_shard_id: int,
    top_k: int,
) -> int:
    _, target_item_index, _ = select_target_from_clean_model(
        dataset=dataset,
        clean_server_state=clean_server_state,
        config=config,
        clean_user_states=clean_user_states,
        clean_shards=clean_shards,
        requested_target_genre=target_genre,
        malicious_shard_id=malicious_shard_id,
        top_k=top_k,
    )
    return target_item_index


def _segment_train_support(
    dataset: MovieLensFederatedDataset,
    target_segment_users: List[int],
) -> Dict[int, int]:
    support: Dict[int, int] = {}
    for user_index in target_segment_users:
        for item_index, _ in dataset.splits_by_user[user_index].train_ratings:
            support[item_index] = support.get(item_index, 0) + 1
    return support


def _build_user_lookup(
    clean_shards: Dict[int, FederatedShard],
) -> tuple[Dict[int, int], Dict[int, Dict[int, int]]]:
    user_to_shard: Dict[int, int] = {}
    shard_local_lookup: Dict[int, Dict[int, int]] = {}
    for shard_id, shard in clean_shards.items():
        shard_local_lookup[shard_id] = {
            user_index: local_index for local_index, user_index in enumerate(shard.user_indices)
        }
        for user_index in shard.user_indices:
            user_to_shard[user_index] = shard_id
    return user_to_shard, shard_local_lookup


def _evaluate_target_genre_vulnerability(
    dataset: MovieLensFederatedDataset,
    clean_server_state: ServerState,
    config: FederatedConfig,
    clean_user_states: Dict[int, torch.Tensor],
    clean_shards: Dict[int, FederatedShard],
    target_genre: str,
    target_segment_users: List[int],
    malicious_shard_id: int,
    top_k: int,
) -> Dict[str, object]:
    candidates = _target_item_candidates(dataset, target_genre, target_segment_users)
    if not candidates:
        raise ValueError(f"No candidate items were found for target genre {target_genre}.")

    user_to_shard, shard_local_lookup = _build_user_lookup(clean_shards)
    benign_target_users = [
        user_index for user_index in target_segment_users if user_to_shard.get(user_index) != malicious_shard_id
    ]
    if not benign_target_users:
        raise ValueError(f"No benign target users were available for target genre {target_genre}.")

    target_support = _segment_train_support(dataset, benign_target_users)
    ranked_pool = sorted(
        candidates,
        key=lambda item_index: (
            -target_support.get(item_index, 0),
            dataset.train_item_popularity.get(item_index, 0),
            item_index,
        ),
    )
    candidate_pool = ranked_pool[: min(TARGET_SELECTION_CANDIDATE_POOL_SIZE, len(ranked_pool))]
    candidate_pool_set = set(candidate_pool)

    candidate_stats = {
        item_index: {
            "eligible_users": 0,
            "clean_hits": 0,
            "margin_sum": 0.0,
            "segment_support": float(target_support.get(item_index, 0)),
            "global_popularity": float(dataset.train_item_popularity.get(item_index, 0)),
        }
        for item_index in candidate_pool
    }

    selector_device = clean_server_state.item_embeddings.device
    scorer_module = build_scorer_module(clean_server_state, config, selector_device)
    for user_index in benign_target_users:
        split = dataset.splits_by_user[user_index]
        shard_id = user_to_shard[user_index]
        local_index = shard_local_lookup[shard_id][user_index]
        user_embedding = clean_user_states[shard_id][local_index].to(selector_device)
        candidate_items = [
            item_index for item_index in range(dataset.num_items) if item_index not in split.known_items
        ]
        candidate_tensor = torch.tensor(candidate_items, dtype=torch.long, device=selector_device)
        scores = score_candidate_items(
            user_embeddings=user_embedding,
            item_indices=candidate_tensor,
            server_state=clean_server_state,
            config=config,
            scorer_module=scorer_module,
        )
        top_count = min(max(top_k, 1), int(scores.numel()))
        top_scores, top_indices = torch.topk(scores, k=top_count)
        boundary_score = float(top_scores[-1].item())
        top_items = {candidate_items[index] for index in top_indices.detach().cpu().tolist()}
        position_lookup = {
            item_index: position
            for position, item_index in enumerate(candidate_items)
            if item_index in candidate_pool_set
        }
        for item_index, position in position_lookup.items():
            score_value = float(scores[position].item())
            candidate_stats[item_index]["eligible_users"] += 1
            candidate_stats[item_index]["margin_sum"] += max(boundary_score - score_value, 0.0)
            if item_index in top_items:
                candidate_stats[item_index]["clean_hits"] += 1

    candidate_summaries = []
    for item_index in candidate_pool:
        stats = candidate_stats[item_index]
        eligible_users = max(int(stats["eligible_users"]), 1)
        clean_hit_rate = stats["clean_hits"] / eligible_users
        mean_margin = stats["margin_sum"] / eligible_users
        vulnerability_score = (
            eligible_users
            * (stats["segment_support"] + 1.0)
            / max(mean_margin, TARGET_SELECTION_MARGIN_EPS)
            / (1.0 + 25.0 * clean_hit_rate)
        )
        candidate_summaries.append(
            {
                "item_index": item_index,
                "movie_id": dataset.movie_id_for_item(item_index),
                "title": dataset.title_for_item(item_index),
                "eligible_users": int(stats["eligible_users"]),
                "clean_hit_rate@k": clean_hit_rate,
                "mean_margin_to_topk": mean_margin,
                "segment_support": int(stats["segment_support"]),
                "global_popularity": int(stats["global_popularity"]),
                "vulnerability_score": vulnerability_score,
                "is_low_exposure": clean_hit_rate <= TARGET_SELECTION_MAX_CLEAN_HITRATE,
            }
        )

    if not candidate_summaries:
        raise ValueError(f"No candidate summaries were generated for target genre {target_genre}.")

    ranked_candidates = sorted(
        candidate_summaries,
        key=lambda summary: (
            0 if summary["is_low_exposure"] else 1,
            -summary["vulnerability_score"],
            summary["mean_margin_to_topk"],
            -summary["segment_support"],
            summary["global_popularity"],
            summary["item_index"],
        ),
    )
    selected = ranked_candidates[0]
    return {
        "genre": target_genre,
        "segment_users": len(benign_target_users),
        "candidate_pool_size": len(candidate_pool),
        "selected_item_index": selected["item_index"],
        "selected_movie_id": selected["movie_id"],
        "selected_title": selected["title"],
        "selected_vulnerability_score": selected["vulnerability_score"],
        "selected_clean_hit_rate@k": selected["clean_hit_rate@k"],
        "selected_mean_margin_to_topk": selected["mean_margin_to_topk"],
        "top_candidates": ranked_candidates[:TARGET_SELECTION_TOP_CANDIDATES],
    }


def select_target_from_clean_model(
    dataset: MovieLensFederatedDataset,
    clean_server_state: ServerState,
    config: FederatedConfig,
    clean_user_states: Dict[int, torch.Tensor],
    clean_shards: Dict[int, FederatedShard],
    requested_target_genre: str | None,
    malicious_shard_id: int,
    top_k: int,
) -> tuple[str, int, Dict[str, object]]:
    user_to_shard, _ = _build_user_lookup(clean_shards)
    target_genres = (
        [requested_target_genre]
        if requested_target_genre is not None
        else sorted(dataset.users_by_genre)
    )
    genre_summaries: List[Dict[str, object]] = []
    for target_genre in target_genres:
        target_segment_users = list(benign_segment_users(dataset, target_genre, user_to_shard, malicious_shard_id))
        if not target_segment_users:
            continue
        genre_summary = _evaluate_target_genre_vulnerability(
            dataset=dataset,
            clean_server_state=clean_server_state,
            config=config,
            clean_user_states=clean_user_states,
            clean_shards=clean_shards,
            target_genre=target_genre,
            target_segment_users=target_segment_users,
            malicious_shard_id=malicious_shard_id,
            top_k=top_k,
        )
        genre_summaries.append(genre_summary)

    if not genre_summaries:
        raise ValueError("No viable target genre/item pair was found from the clean model.")

    ranked_genres = sorted(
        genre_summaries,
        key=lambda summary: (
            -summary["selected_vulnerability_score"],
            summary["selected_mean_margin_to_topk"],
            -summary["segment_users"],
            summary["selected_item_index"],
        ),
    )
    selected_summary = ranked_genres[0]
    selection_report = {
        "strategy": "heuristic_v1",
        "candidate_pool_size": TARGET_SELECTION_CANDIDATE_POOL_SIZE,
        "max_clean_hitrate@k": TARGET_SELECTION_MAX_CLEAN_HITRATE,
        "margin_epsilon": TARGET_SELECTION_MARGIN_EPS,
        "selected_genre": selected_summary["genre"],
        "selected_item_index": selected_summary["selected_item_index"],
        "selected_movie_id": selected_summary["selected_movie_id"],
        "selected_title": selected_summary["selected_title"],
        "selected_vulnerability_score": selected_summary["selected_vulnerability_score"],
        "selected_clean_hit_rate@k": selected_summary["selected_clean_hit_rate@k"],
        "selected_mean_margin_to_topk": selected_summary["selected_mean_margin_to_topk"],
        "selected_segment_users": selected_summary["segment_users"],
        "top_genres": [
            {
                "genre": summary["genre"],
                "segment_users": summary["segment_users"],
                "selected_item_index": summary["selected_item_index"],
                "selected_movie_id": summary["selected_movie_id"],
                "selected_title": summary["selected_title"],
                "selected_vulnerability_score": summary["selected_vulnerability_score"],
                "selected_clean_hit_rate@k": summary["selected_clean_hit_rate@k"],
                "selected_mean_margin_to_topk": summary["selected_mean_margin_to_topk"],
            }
            for summary in ranked_genres[:TARGET_SELECTION_TOP_CANDIDATES]
        ],
        "top_candidates": selected_summary["top_candidates"],
    }
    return selected_summary["genre"], selected_summary["selected_item_index"], selection_report


def choose_target_genre(
    dataset: MovieLensFederatedDataset,
    malicious_shard_id: int,
    user_to_shard: Dict[int, int],
    requested_genre: str | None,
) -> str:
    if requested_genre is not None:
        if requested_genre not in dataset.users_by_genre:
            raise ValueError(f"Unknown target genre: {requested_genre}")
        return requested_genre
    benign_counts = {
        genre: len(benign_segment_users(dataset, genre, user_to_shard, malicious_shard_id))
        for genre in dataset.users_by_genre
    }
    filtered_counts = {genre: count for genre, count in benign_counts.items() if count > 0}
    if not filtered_counts:
        raise ValueError("No benign target segment could be derived from genre affinity.")
    return sorted(filtered_counts.items(), key=lambda pair: (-pair[1], pair[0]))[0][0]


def resolve_training_devices(
    primary_device: torch.device,
    requested_devices: str | None,
    num_gpus: int | None = None,
) -> List[torch.device]:
    if requested_devices is not None and requested_devices.strip():
        return [resolve_runtime_device(text.strip()) for text in requested_devices.split(",") if text.strip()]
    if num_gpus is not None:
        if num_gpus <= 0:
            raise ValueError("--num-gpus must be positive when provided.")
        if primary_device.type != "cuda":
            raise ValueError("--num-gpus can only be used when the primary device is CUDA.")
        return [resolve_runtime_device(f"cuda:{index}") for index in range(num_gpus)]
    if requested_devices is None or not requested_devices.strip():
        return [primary_device]
    return [resolve_runtime_device(text.strip()) for text in requested_devices.split(",") if text.strip()]


def assign_shards_to_devices(
    shards: Dict[int, FederatedShard],
    devices: List[torch.device],
) -> Dict[str, tuple[int, ...]]:
    device_keys = [str(device) for device in devices]
    assignment: Dict[str, List[int]] = {device_key: [] for device_key in device_keys}
    device_loads = {device_key: 0 for device_key in device_keys}
    for shard_id, shard in sorted(shards.items(), key=lambda pair: (-pair[1].sample_count, pair[0])):
        target_device_key = min(device_keys, key=lambda device_key: (device_loads[device_key], device_key))
        assignment[target_device_key].append(shard_id)
        device_loads[target_device_key] += max(shard.sample_count, 1)
    return {device_key: tuple(shard_ids) for device_key, shard_ids in assignment.items() if shard_ids}


def select_eval_device(
    training_devices: List[torch.device],
    shard_assignments: Dict[str, tuple[int, ...]],
    shards: Dict[int, FederatedShard],
    round_index: int,
) -> torch.device:
    if len(training_devices) == 1:
        return training_devices[0]
    device_loads = {
        device_key: sum(max(shards[shard_id].sample_count, 1) for shard_id in shard_ids)
        for device_key, shard_ids in shard_assignments.items()
    }
    ranked_devices = sorted(
        training_devices,
        key=lambda device: (device_loads.get(str(device), 0), str(device)),
    )
    return ranked_devices[round_index % len(ranked_devices)]


def summarize_round_training(shard_updates: List[ShardUpdate], round_index: int) -> Dict[str, float]:
    if not shard_updates:
        return {
            "round_index": float(round_index),
            "mean_loss": 0.0,
            "last_loss": 0.0,
            "loss_std": 0.0,
            "total_steps": 0.0,
            "mean_steps_per_shard": 0.0,
            "mean_shard_weight": 0.0,
        }

    total_steps = sum(max(shard_update.steps, 0) for shard_update in shard_updates)
    total_weight = sum(max(shard_update.weight, 1) for shard_update in shard_updates)
    weighted_mean_loss_numerator = sum(
        shard_update.mean_loss * max(shard_update.steps, 1)
        for shard_update in shard_updates
    )
    weighted_last_loss_numerator = sum(
        shard_update.last_loss * max(shard_update.steps, 1)
        for shard_update in shard_updates
    )
    mean_loss = weighted_mean_loss_numerator / max(total_steps, 1)
    last_loss = weighted_last_loss_numerator / max(total_steps, 1)
    shard_losses = torch.tensor([shard_update.mean_loss for shard_update in shard_updates], dtype=torch.float32)
    return {
        "round_index": float(round_index),
        "mean_loss": float(mean_loss),
        "last_loss": float(last_loss),
        "loss_std": float(shard_losses.std(unbiased=False).item()) if shard_losses.numel() > 1 else 0.0,
        "total_steps": float(total_steps),
        "mean_steps_per_shard": float(total_steps / max(len(shard_updates), 1)),
        "mean_shard_weight": float(total_weight / max(len(shard_updates), 1)),
    }


def _run_device_shard_group(
    shard_ids: tuple[int, ...],
    shard_runtimes: Dict[int, ShardRuntime],
    server_state: ServerState,
    dataset: MovieLensFederatedDataset,
    config: FederatedConfig,
    round_index: int,
    seed: int,
    shard_progress: ShardProgress | None,
) -> list[ShardUpdate]:
    if not shard_ids:
        raise ValueError("Device shard group cannot be empty.")
    device = shard_runtimes[shard_ids[0]].device
    base_device_server_state = server_state.to(device)
    shard_updates: list[ShardUpdate] = []
    for shard_id in shard_ids:
        runtime = shard_runtimes[shard_id]
        local_server_state, training_stats = train_local_shard_runtime(
            runtime=runtime,
            num_items=dataset.num_items,
            server_state=base_device_server_state,
            config=config,
            seed=seed + round_index * 1009 + shard_id * 53,
        )
        weight = max(runtime.sample_count, 1)
        shard_updates.append(
            ShardUpdate(
                shard_id=shard_id,
                weight=weight,
                server_state=ServerState(
                    item_embeddings=local_server_state.item_embeddings.detach(),
                    item_bias=local_server_state.item_bias.detach(),
                    scorer_state=None
                    if local_server_state.scorer_state is None
                    else {name: tensor.detach() for name, tensor in local_server_state.scorer_state.items()},
                ),
                mean_loss=float(training_stats["mean_loss"]),
                last_loss=float(training_stats["last_loss"]),
                steps=int(training_stats["steps"]),
            )
        )
        if shard_progress is not None:
            shard_progress.update(shard_id=shard_id, sample_count=runtime.sample_count)
    return shard_updates


def move_shard_updates_to_device(
    shard_updates: List[ShardUpdate],
    device: torch.device,
) -> List[ShardUpdate]:
    return [shard_update.to(device) for shard_update in shard_updates]


def run_federated_training(
    dataset: MovieLensFederatedDataset,
    shards: Dict[int, FederatedShard],
    config: FederatedConfig,
    target_genre: str,
    target_item_index: int,
    malicious_shard_id: int,
    device: torch.device,
    seed: int,
    top_k: int,
    num_eval_negatives: int,
    progress_desc: str | None = None,
    show_progress: bool = False,
    devices: List[torch.device] | None = None,
) -> tuple[ServerState, Dict[int, torch.Tensor], List[Dict[str, object]], List[Dict[str, float]]]:
    server_state = build_initial_server_state(dataset.num_items, config.embedding_dim, seed, config=config).to(device)
    initial_user_states = build_initial_user_states(shards, config.embedding_dim, seed)
    training_devices = devices or [device]
    shard_assignments = assign_shards_to_devices(shards, training_devices)
    shard_to_device = {
        shard_id: torch.device(device_key)
        for device_key, shard_ids in shard_assignments.items()
        for shard_id in shard_ids
    }
    prepared_shards = build_prepared_shards(shards)
    shard_runtimes = build_shard_runtimes(prepared_shards, initial_user_states, shard_to_device)
    round_metrics: List[Dict[str, object]] = []
    training_traces: List[Dict[str, float]] = []
    progress = RoundProgress(
        total=config.federated_rounds,
        desc=progress_desc or "federated training",
        enabled=show_progress,
    )
    executor = ThreadPoolExecutor(max_workers=len(shard_assignments)) if len(shard_assignments) > 1 else None

    try:
        for round_index in range(config.federated_rounds):
            shard_progress = ShardProgress(
                total=len(shards),
                desc=f"{progress.desc} round {round_index + 1}/{config.federated_rounds}",
                enabled=show_progress,
            )
            try:
                if executor is None:
                    updated_shard_groups = [
                        _run_device_shard_group(
                            shard_ids=next(iter(shard_assignments.values())),
                            shard_runtimes=shard_runtimes,
                            server_state=server_state,
                            dataset=dataset,
                            config=config,
                            round_index=round_index,
                            seed=seed,
                            shard_progress=shard_progress,
                        )
                    ]
                else:
                    futures = [
                        executor.submit(
                            _run_device_shard_group,
                            shard_ids=shard_ids,
                            shard_runtimes=shard_runtimes,
                            server_state=server_state,
                            dataset=dataset,
                            config=config,
                            round_index=round_index,
                            seed=seed,
                            shard_progress=shard_progress,
                        )
                        for shard_ids in shard_assignments.values()
                    ]
                    updated_shard_groups = [future.result() for future in futures]
            finally:
                shard_progress.close()

            aggregation_device = select_eval_device(training_devices, shard_assignments, shards, round_index)
            shard_updates = [
                shard_update
                for shard_group in updated_shard_groups
                for shard_update in move_shard_updates_to_device(shard_group, aggregation_device)
            ]
            training_summary = summarize_round_training(shard_updates, round_index + 1)
            training_traces.append(training_summary)
            server_state, aggregation_metrics = aggregate_server_states(
                shard_updates,
                base_server_state=server_state.to(aggregation_device),
                config=config,
            )
            metrics = None
            should_evaluate = (
                (round_index + 1) % max(config.eval_every_rounds, 1) == 0
                or round_index + 1 == config.federated_rounds
            )
            if should_evaluate:
                cpu_user_states = {
                    shard_id: runtime.user_state.detach().cpu()
                    for shard_id, runtime in shard_runtimes.items()
                }
                metrics = evaluate_federated_model(
                    dataset=dataset,
                    server_state=server_state,
                    config=config,
                    user_states=cpu_user_states,
                    shards=shards,
                    target_genre=target_genre,
                    target_item_index=target_item_index,
                    malicious_shard_id=malicious_shard_id,
                    top_k=top_k,
                    num_eval_negatives=num_eval_negatives,
                    seed=seed + round_index * 17,
                    eval_device=aggregation_device,
                    eval_batch_size=config.eval_batch_size,
                )
                metrics["round_index"] = round_index + 1
                metrics["training"] = training_summary
                metrics["aggregation"] = aggregation_metrics
                round_metrics.append(metrics)
            progress.update(metrics if metrics is not None else {"training": training_summary})
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        progress.close()

    final_user_states = {
        shard_id: runtime.user_state.detach().cpu()
        for shard_id, runtime in shard_runtimes.items()
    }
    return server_state, final_user_states, round_metrics, training_traces


def summarize_final_metrics(round_metrics: List[Dict[str, object]]) -> Dict[str, object]:
    if not round_metrics:
        raise ValueError("No round metrics were produced.")
    return round_metrics[-1]


def _trace_view(round_metrics: List[Dict[str, object]]) -> List[Dict[str, float]]:
    traces = []
    for metrics in round_metrics:
        aggregation = metrics.get("aggregation", {})
        trace = {
            "round_index": metrics["round_index"],
            "overall_hr@k": metrics["overall_benign"]["hr@k"],
            "overall_ndcg@k": metrics["overall_benign"]["ndcg@k"],
            "target_hitrate@k": metrics["target_segment"]["target_hitrate@k"],
            "target_mean_rank": metrics["target_segment"]["target_mean_rank"],
            "suppressed_shards": float(len(aggregation.get("suppressed_shards", []))),
        }
        training = metrics.get("training", {})
        if isinstance(training, dict):
            if "mean_loss" in training:
                trace["training_mean_loss"] = float(training["mean_loss"])
            if "last_loss" in training:
                trace["training_last_loss"] = float(training["last_loss"])
            if "loss_std" in training:
                trace["training_loss_std"] = float(training["loss_std"])
            if "total_steps" in training:
                trace["training_total_steps"] = float(training["total_steps"])
        for cutoff in (3, 5, 10):
            trace[f"target_hitrate@{cutoff}"] = metrics["target_segment"][f"target_hitrate@{cutoff}"]
        traces.append(trace)
    return traces


def _training_trace_view(training_traces: List[Dict[str, float]]) -> List[Dict[str, float]]:
    return [dict(trace) for trace in training_traces]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated targeted-push benchmark on MovieLens.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-variant", choices=["ml-1m", "ml-10m", "ml-25m", "ml-32m"], default="ml-1m")
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--malicious-shard-id", type=int, default=0)
    parser.add_argument("--target-genre", type=str)
    parser.add_argument("--target-item-id", type=int)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--federated-rounds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--model", choices=["mf_bpr", "neural_bpr"], default="mf_bpr")
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--mlp-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument(
        "--defense-method",
        choices=["none", "clip_mean", "clip_trimmed_mean", "focus_clip_mean", "focus_clip_trimmed_mean"],
        default="none",
    )
    parser.add_argument("--clip-factor", type=float, default=1.5)
    parser.add_argument("--trim-ratio", type=float, default=0.25)
    parser.add_argument("--focus-top-k", type=int, default=3)
    parser.add_argument("--focus-factor", type=float, default=2.0)
    parser.add_argument("--attack-budgets", type=str, default="0,0.01,0.05,0.1")
    parser.add_argument("--attack-filler-items-per-user", type=int, default=8)
    parser.add_argument("--attack-neutral-items-per-user", type=int, default=0)
    parser.add_argument("--attack-target-weight", type=float, default=20.0)
    parser.add_argument("--attack-filler-weight", type=float, default=0.5)
    parser.add_argument("--attack-neutral-weight", type=float, default=0.0)
    parser.add_argument("--attack-filler-pool-size", type=int, default=64)
    parser.add_argument("--attack-neutral-pool-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-eval-negatives", type=int, default=100)
    parser.add_argument("--min-positive-rating", type=float, default=4.0)
    parser.add_argument("--max-positive-interactions", type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--devices", type=str)
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--save-attack-bundle", type=Path)
    parser.add_argument("--load-attack-bundle", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_runtime_device(args.device)
    set_seed(args.seed, device)
    show_progress = not args.disable_progress
    training_devices = resolve_training_devices(device, args.devices, args.num_gpus)
    attack_budgets = _parse_attack_budgets(args.attack_budgets)
    attack_params = _attack_params_from_args(args)
    runtime_seed = args.seed

    if show_progress and tqdm is None:
        print("tqdm is not installed; using plain-text round progress. Run `pip install -e .` to enable bars.", flush=True)
    print(f"loading {args.dataset_variant} from {args.data_root}...", flush=True)
    dataset = load_movielens_dataset(
        data_root=args.data_root,
        variant=args.dataset_variant,
        min_positive_rating=args.min_positive_rating,
        max_positive_interactions=args.max_positive_interactions,
        show_progress=show_progress,
    )
    target_selection_report: Dict[str, object] | None = None
    cached_clean_section: Dict[str, object] | None = None
    cached_attack_results: List[Dict[str, object]] = []

    if args.load_attack_bundle is not None:
        bundle = _load_attack_bundle(args.load_attack_bundle)
        _validate_attack_bundle(bundle, dataset, args)
        shard_to_users = _deserialize_shard_to_users(bundle["shard_to_users"])
        user_to_shard = {
            user_index: shard_id
            for shard_id, user_indices in shard_to_users.items()
            for user_index in user_indices
        }
        clean_shards = build_clean_shards(dataset, shard_to_users)
        attack_setup = bundle["attack_setup"]
        target_genre = str(attack_setup["target_genre"])
        target_item_index = int(attack_setup["target_item_index"])
        target_selection_report = attack_setup.get("target_selection")
        attack_budgets = [float(value) for value in attack_setup["attack_budgets"]]
        attack_params = dict(attack_setup["attack_params"])
        runtime_seed = int(attack_setup["seed"])
        target_segment_users = list(
            benign_segment_users(dataset, target_genre, user_to_shard, args.malicious_shard_id)
        )
        if bundle.get("benchmark_signature") == _benchmark_signature(
            args,
            training_devices,
            attack_budgets=attack_budgets,
            attack_params=attack_params,
            runtime_seed=runtime_seed,
            num_shards=len(shard_to_users),
        ):
            cached_clean_section = bundle.get("clean")
            cached_attack_results = list(bundle.get("attacks", []))
            print(
                f"loaded attack bundle {args.load_attack_bundle}; "
                "reusing cached clean and naive attack results.",
                flush=True,
            )
        else:
            print(
                f"loaded attack bundle {args.load_attack_bundle}; "
                "training signature changed, rerunning clean and naive attack baselines from saved setup.",
                flush=True,
            )
    else:
        shard_to_users, user_to_shard = partition_users(dataset.num_users, args.num_shards, args.seed)
        clean_shards = build_clean_shards(dataset, shard_to_users)
        target_genre = choose_target_genre(dataset, args.malicious_shard_id, user_to_shard, args.target_genre)
        target_segment_users = list(
            benign_segment_users(dataset, target_genre, user_to_shard, args.malicious_shard_id)
        )
    print(
        f"loaded {dataset.num_users} users, {dataset.num_items} items, "
        f"model={args.model}, target_genre={target_genre}, shards={len(shard_to_users)}, "
        f"devices={[str(training_device) for training_device in training_devices]}",
        flush=True,
    )

    clean_config = FederatedConfig(
        model_name=args.model,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        mlp_layers=args.mlp_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
        federated_rounds=args.federated_rounds,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        min_positive_rating=args.min_positive_rating,
        eval_every_rounds=args.eval_every,
        eval_batch_size=args.eval_batch_size,
        aggregation_method="mean",
        clip_factor=args.clip_factor,
        trim_ratio=args.trim_ratio,
        focus_top_k=args.focus_top_k,
        focus_factor=args.focus_factor,
    )
    defended_config = FederatedConfig(
        model_name=args.model,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        mlp_layers=args.mlp_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
        federated_rounds=args.federated_rounds,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        min_positive_rating=args.min_positive_rating,
        eval_every_rounds=args.eval_every,
        eval_batch_size=args.eval_batch_size,
        aggregation_method="mean" if args.defense_method == "none" else args.defense_method,
        clip_factor=args.clip_factor,
        trim_ratio=args.trim_ratio,
        focus_top_k=args.focus_top_k,
        focus_factor=args.focus_factor,
    )

    if cached_clean_section is None:
        if args.load_attack_bundle is None:
            provisional_target_item = select_target_item(dataset, target_genre, target_segment_users)
            print("running clean target-selection pass...", flush=True)
            clean_server_state, clean_user_states, _, _ = run_federated_training(
                dataset=dataset,
                shards=clean_shards,
                config=clean_config,
                target_genre=target_genre,
                target_item_index=provisional_target_item,
                malicious_shard_id=args.malicious_shard_id,
                device=device,
                seed=runtime_seed,
                top_k=args.top_k,
                num_eval_negatives=args.num_eval_negatives,
                progress_desc="clean target selection",
                show_progress=show_progress,
                devices=training_devices,
            )

            if args.target_item_id is not None:
                if args.target_item_id not in dataset.item_id_to_index:
                    raise ValueError(f"Unknown target MovieLens item id: {args.target_item_id}")
                target_item_index = dataset.item_id_to_index[args.target_item_id]
                target_selection_report = {
                    "strategy": "manual_item_id",
                    "selected_genre": target_genre,
                    "selected_item_index": target_item_index,
                    "selected_movie_id": args.target_item_id,
                    "selected_title": dataset.title_for_item(target_item_index),
                }
            else:
                target_genre, target_item_index, target_selection_report = select_target_from_clean_model(
                    dataset=dataset,
                    clean_server_state=clean_server_state,
                    config=clean_config,
                    clean_user_states=clean_user_states,
                    clean_shards=clean_shards,
                    requested_target_genre=args.target_genre,
                    malicious_shard_id=args.malicious_shard_id,
                    top_k=args.top_k,
                )
                target_segment_users = list(
                    benign_segment_users(dataset, target_genre, user_to_shard, args.malicious_shard_id)
                )

        print(
            f"selected target item {dataset.movie_id_for_item(target_item_index)}:{dataset.title_for_item(target_item_index)}; "
            "running clean baseline...",
            flush=True,
        )
        _, _, clean_round_metrics, clean_training_traces = run_federated_training(
            dataset=dataset,
            shards=clean_shards,
            config=clean_config,
            target_genre=target_genre,
            target_item_index=target_item_index,
            malicious_shard_id=args.malicious_shard_id,
            device=device,
            seed=runtime_seed,
            top_k=args.top_k,
            num_eval_negatives=args.num_eval_negatives,
            progress_desc="clean baseline",
            show_progress=show_progress,
            devices=training_devices,
        )
        clean_final = summarize_final_metrics(clean_round_metrics)
        cached_clean_section = {
            "final_metrics": clean_final,
            "round_traces": _trace_view(clean_round_metrics),
            "training_traces": _training_trace_view(clean_training_traces),
            "final_aggregation": clean_final.get("aggregation"),
        }
    else:
        clean_final = cached_clean_section["final_metrics"]
        print(
            f"selected target item {dataset.movie_id_for_item(target_item_index)}:{dataset.title_for_item(target_item_index)}; "
            "reusing cached clean baseline...",
            flush=True,
        )

    clean_section = cached_clean_section
    attack_results = list(cached_attack_results)
    defended_results = []
    attack_results_by_budget = {float(result["attack_budget"]): result for result in attack_results}

    if not attack_results:
        for attack_budget in attack_budgets:
            poisoned_shards = build_poisoned_shards(
                clean_shards=clean_shards,
                dataset=dataset,
                malicious_shard_id=args.malicious_shard_id,
                target_genre=target_genre,
                target_item_index=target_item_index,
                attack_budget=attack_budget,
                seed=runtime_seed,
                filler_items_per_user=int(attack_params["filler_items_per_user"]),
                neutral_items_per_user=int(attack_params["neutral_items_per_user"]),
                target_item_weight=float(attack_params["target_item_weight"]),
                filler_item_weight=float(attack_params["filler_item_weight"]),
                neutral_item_weight=float(attack_params["neutral_item_weight"]),
                filler_candidate_pool_size=int(attack_params["filler_candidate_pool_size"]),
                neutral_candidate_pool_size=int(attack_params["neutral_candidate_pool_size"]),
            )
            print(
                f"running attack budget={attack_budget:.4f} "
                f"with {poisoned_shards[args.malicious_shard_id].synthetic_user_count} synthetic users...",
                flush=True,
            )
            _, _, attack_round_metrics, attack_training_traces = run_federated_training(
                dataset=dataset,
                shards=poisoned_shards,
                config=clean_config,
                target_genre=target_genre,
                target_item_index=target_item_index,
                malicious_shard_id=args.malicious_shard_id,
                device=device,
                seed=runtime_seed,
                top_k=args.top_k,
                num_eval_negatives=args.num_eval_negatives,
                progress_desc=f"attack budget={attack_budget:.4f}",
                show_progress=show_progress,
                devices=training_devices,
            )
            attack_final = summarize_final_metrics(attack_round_metrics)
            attack_result = {
                "attack_budget": attack_budget,
                "synthetic_user_count": poisoned_shards[args.malicious_shard_id].synthetic_user_count,
                "final_metrics": attack_final,
                "round_traces": _trace_view(attack_round_metrics),
                "training_traces": _training_trace_view(attack_training_traces),
                "final_aggregation": attack_final.get("aggregation"),
                "uplift_vs_clean": {
                    "target_hitrate@k": attack_final["target_segment"]["target_hitrate@k"]
                    - clean_final["target_segment"]["target_hitrate@k"],
                    "target_hitrate@3": attack_final["target_segment"]["target_hitrate@3"]
                    - clean_final["target_segment"]["target_hitrate@3"],
                    "target_hitrate@5": attack_final["target_segment"]["target_hitrate@5"]
                    - clean_final["target_segment"]["target_hitrate@5"],
                    "target_hitrate@10": attack_final["target_segment"]["target_hitrate@10"]
                    - clean_final["target_segment"]["target_hitrate@10"],
                    "target_mean_rank": attack_final["target_segment"]["target_mean_rank"]
                    - clean_final["target_segment"]["target_mean_rank"],
                    "overall_hr@k": attack_final["overall_benign"]["hr@k"]
                    - clean_final["overall_benign"]["hr@k"],
                    "overall_ndcg@k": attack_final["overall_benign"]["ndcg@k"]
                    - clean_final["overall_benign"]["ndcg@k"],
                },
            }
            attack_results.append(attack_result)
            attack_results_by_budget[attack_budget] = attack_result

    if args.defense_method != "none":
        for attack_budget in attack_budgets:
            if attack_budget <= 0:
                continue
            poisoned_shards = build_poisoned_shards(
                clean_shards=clean_shards,
                dataset=dataset,
                malicious_shard_id=args.malicious_shard_id,
                target_genre=target_genre,
                target_item_index=target_item_index,
                attack_budget=attack_budget,
                seed=runtime_seed,
                filler_items_per_user=int(attack_params["filler_items_per_user"]),
                neutral_items_per_user=int(attack_params["neutral_items_per_user"]),
                target_item_weight=float(attack_params["target_item_weight"]),
                filler_item_weight=float(attack_params["filler_item_weight"]),
                neutral_item_weight=float(attack_params["neutral_item_weight"]),
                filler_candidate_pool_size=int(attack_params["filler_candidate_pool_size"]),
                neutral_candidate_pool_size=int(attack_params["neutral_candidate_pool_size"]),
            )
            print(
                f"running defended attack budget={attack_budget:.4f} "
                f"with aggregation={args.defense_method}...",
                flush=True,
            )
            _, _, defended_round_metrics, defended_training_traces = run_federated_training(
                dataset=dataset,
                shards=poisoned_shards,
                config=defended_config,
                target_genre=target_genre,
                target_item_index=target_item_index,
                malicious_shard_id=args.malicious_shard_id,
                device=device,
                seed=runtime_seed,
                top_k=args.top_k,
                num_eval_negatives=args.num_eval_negatives,
                progress_desc=f"defended attack budget={attack_budget:.4f}",
                show_progress=show_progress,
                devices=training_devices,
            )
            defended_final = summarize_final_metrics(defended_round_metrics)
            attack_final = attack_results_by_budget[attack_budget]["final_metrics"]
            defended_results.append(
                {
                    "attack_budget": attack_budget,
                    "synthetic_user_count": poisoned_shards[args.malicious_shard_id].synthetic_user_count,
                    "defense_method": args.defense_method,
                    "final_metrics": defended_final,
                    "round_traces": _trace_view(defended_round_metrics),
                    "training_traces": _training_trace_view(defended_training_traces),
                    "final_aggregation": defended_final.get("aggregation"),
                    "uplift_vs_clean": {
                        "target_hitrate@k": defended_final["target_segment"]["target_hitrate@k"]
                        - clean_final["target_segment"]["target_hitrate@k"],
                        "target_hitrate@3": defended_final["target_segment"]["target_hitrate@3"]
                        - clean_final["target_segment"]["target_hitrate@3"],
                        "target_hitrate@5": defended_final["target_segment"]["target_hitrate@5"]
                        - clean_final["target_segment"]["target_hitrate@5"],
                        "target_hitrate@10": defended_final["target_segment"]["target_hitrate@10"]
                        - clean_final["target_segment"]["target_hitrate@10"],
                        "target_mean_rank": defended_final["target_segment"]["target_mean_rank"]
                        - clean_final["target_segment"]["target_mean_rank"],
                        "overall_hr@k": defended_final["overall_benign"]["hr@k"]
                        - clean_final["overall_benign"]["hr@k"],
                        "overall_ndcg@k": defended_final["overall_benign"]["ndcg@k"]
                        - clean_final["overall_benign"]["ndcg@k"],
                    },
                    "improvement_vs_attack": {
                        "target_hitrate@k": attack_final["target_segment"]["target_hitrate@k"]
                        - defended_final["target_segment"]["target_hitrate@k"],
                        "target_hitrate@3": attack_final["target_segment"]["target_hitrate@3"]
                        - defended_final["target_segment"]["target_hitrate@3"],
                        "target_hitrate@5": attack_final["target_segment"]["target_hitrate@5"]
                        - defended_final["target_segment"]["target_hitrate@5"],
                        "target_hitrate@10": attack_final["target_segment"]["target_hitrate@10"]
                        - defended_final["target_segment"]["target_hitrate@10"],
                        "target_mean_rank": defended_final["target_segment"]["target_mean_rank"]
                        - attack_final["target_segment"]["target_mean_rank"],
                        "overall_hr@k": defended_final["overall_benign"]["hr@k"]
                        - attack_final["overall_benign"]["hr@k"],
                        "overall_ndcg@k": defended_final["overall_benign"]["ndcg@k"]
                        - attack_final["overall_benign"]["ndcg@k"],
                    },
                }
            )

    report = {
        "dataset": {
            "variant": dataset.variant,
            "num_users": dataset.num_users,
            "num_items": dataset.num_items,
            "target_genre": target_genre,
            "target_item_index": target_item_index,
            "target_movie_id": dataset.movie_id_for_item(target_item_index),
            "target_title": dataset.title_for_item(target_item_index),
        },
        "config": {
            "num_shards": len(shard_to_users),
            "malicious_shard_id": args.malicious_shard_id,
            "local_epochs": args.local_epochs,
            "federated_rounds": args.federated_rounds,
            "batch_size": args.batch_size,
            "model": args.model,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "mlp_layers": args.mlp_layers,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "eval_every": args.eval_every,
            "eval_batch_size": args.eval_batch_size,
            "defense_method": args.defense_method,
            "clip_factor": args.clip_factor,
            "trim_ratio": args.trim_ratio,
            "focus_top_k": args.focus_top_k,
            "focus_factor": args.focus_factor,
            "top_k": args.top_k,
            "num_eval_negatives": args.num_eval_negatives,
            "max_positive_interactions": args.max_positive_interactions,
            "attack_budgets": attack_budgets,
            "attack_filler_items_per_user": attack_params["filler_items_per_user"],
            "attack_neutral_items_per_user": attack_params["neutral_items_per_user"],
            "attack_target_weight": attack_params["target_item_weight"],
            "attack_filler_weight": attack_params["filler_item_weight"],
            "attack_neutral_weight": attack_params["neutral_item_weight"],
            "attack_filler_pool_size": attack_params["filler_candidate_pool_size"],
            "attack_neutral_pool_size": attack_params["neutral_candidate_pool_size"],
            "seed": runtime_seed,
            "devices": [str(training_device) for training_device in training_devices],
            "loaded_attack_bundle": None if args.load_attack_bundle is None else str(args.load_attack_bundle),
        },
        "target_selection": target_selection_report,
        "clean": clean_section,
        "attacks": attack_results,
        "defended_attacks": defended_results,
    }

    if args.save_attack_bundle is not None:
        bundle_payload = _build_attack_bundle(
            dataset=dataset,
            args=args,
            shard_to_users=shard_to_users,
            target_genre=target_genre,
            target_item_index=target_item_index,
            target_selection_report=target_selection_report,
            clean_section=clean_section,
            attack_results=attack_results,
            training_devices=training_devices,
            attack_budgets=attack_budgets,
            attack_params=attack_params,
            runtime_seed=runtime_seed,
        )
        args.save_attack_bundle.parent.mkdir(parents=True, exist_ok=True)
        with args.save_attack_bundle.open("w", encoding="utf-8") as handle:
            json.dump(bundle_payload, handle, indent=2)
        print(f"wrote attack bundle {args.save_attack_bundle}")

    print(
        f"clean hr@{args.top_k}={clean_final['overall_benign']['hr@k']:.4f} "
        f"target_hit@{args.top_k}={clean_final['target_segment']['target_hitrate@k']:.4f} "
        f"target_item={dataset.movie_id_for_item(target_item_index)}:{dataset.title_for_item(target_item_index)}"
    )
    for attack_result in attack_results:
        final_metrics = attack_result["final_metrics"]
        print(
            f"attack budget={attack_result['attack_budget']:.4f} "
            f"hr@{args.top_k}={final_metrics['overall_benign']['hr@k']:.4f} "
            f"target_hit@{args.top_k}={final_metrics['target_segment']['target_hitrate@k']:.4f}"
        )
    for defended_result in defended_results:
        final_metrics = defended_result["final_metrics"]
        suppressed = defended_result["final_aggregation"].get("suppressed_shards", [])
        print(
            f"defended budget={defended_result['attack_budget']:.4f} "
            f"method={defended_result['defense_method']} "
            f"hr@{args.top_k}={final_metrics['overall_benign']['hr@k']:.4f} "
            f"target_hit@{args.top_k}={final_metrics['target_segment']['target_hitrate@k']:.4f} "
            f"suppressed={suppressed}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
