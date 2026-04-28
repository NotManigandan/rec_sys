from pathlib import Path

import torch

from recsys.federated.attack import build_poisoned_shards
from recsys.federated.benchmark import run_federated_training
from recsys.federated.bpr import FederatedConfig, build_clean_shards
from recsys.federated.movielens import load_movielens_dataset
from tests.federated_fixture import write_movielens_fixture


def _manual_shards():
    return {
        0: (2, 3, 4),
        1: (0, 1, 5),
    }


def test_poison_injection_changes_only_malicious_shard(tmp_path: Path) -> None:
    write_movielens_fixture(tmp_path, "ml-25m")
    dataset = load_movielens_dataset(tmp_path, "ml-25m")
    clean_shards = build_clean_shards(dataset, _manual_shards())
    target_item_index = dataset.item_id_to_index[9]
    poisoned_shards = build_poisoned_shards(
        clean_shards=clean_shards,
        dataset=dataset,
        malicious_shard_id=0,
        target_genre="Action",
        target_item_index=target_item_index,
        attack_budget=0.5,
        seed=7,
    )

    assert poisoned_shards[1].user_indices == clean_shards[1].user_indices
    assert poisoned_shards[1].train_ratings_by_user == clean_shards[1].train_ratings_by_user
    assert poisoned_shards[0].synthetic_user_count > 0

    synthetic_user_ids = [
        user_index for user_index in poisoned_shards[0].user_indices if user_index >= dataset.num_users
    ]
    assert synthetic_user_ids
    for synthetic_user_id in synthetic_user_ids:
        synthetic_profile = poisoned_shards[0].train_ratings_by_user[synthetic_user_id]
        synthetic_items = {item_index for item_index, _ in synthetic_profile}
        assert target_item_index in synthetic_items
        assert any("Action" in dataset.genres_for_item(item_index) for item_index in synthetic_items if item_index != target_item_index)


def test_zero_budget_matches_clean_training(tmp_path: Path) -> None:
    write_movielens_fixture(tmp_path, "ml-25m")
    dataset = load_movielens_dataset(tmp_path, "ml-25m")
    clean_shards = build_clean_shards(dataset, _manual_shards())
    target_item_index = dataset.item_id_to_index[9]
    config = FederatedConfig(embedding_dim=8, batch_size=4, local_epochs=1, federated_rounds=2, learning_rate=0.05)

    _, _, clean_rounds, _ = run_federated_training(
        dataset=dataset,
        shards=clean_shards,
        config=config,
        target_genre="Action",
        target_item_index=target_item_index,
        malicious_shard_id=0,
        device=torch.device("cpu"),
        seed=17,
        top_k=3,
        num_eval_negatives=4,
    )
    zero_budget_shards = build_poisoned_shards(
        clean_shards=clean_shards,
        dataset=dataset,
        malicious_shard_id=0,
        target_genre="Action",
        target_item_index=target_item_index,
        attack_budget=0.0,
        seed=17,
    )
    _, _, zero_budget_rounds, _ = run_federated_training(
        dataset=dataset,
        shards=zero_budget_shards,
        config=config,
        target_genre="Action",
        target_item_index=target_item_index,
        malicious_shard_id=0,
        device=torch.device("cpu"),
        seed=17,
        top_k=3,
        num_eval_negatives=4,
    )

    assert clean_rounds == zero_budget_rounds


def test_attack_improves_target_segment_uplift_on_fixture(tmp_path: Path) -> None:
    write_movielens_fixture(tmp_path, "ml-25m")
    dataset = load_movielens_dataset(tmp_path, "ml-25m")
    clean_shards = build_clean_shards(dataset, _manual_shards())
    target_item_index = dataset.item_id_to_index[9]
    config = FederatedConfig(embedding_dim=16, batch_size=8, local_epochs=2, federated_rounds=4, learning_rate=0.05)

    _, _, clean_rounds, _ = run_federated_training(
        dataset=dataset,
        shards=clean_shards,
        config=config,
        target_genre="Action",
        target_item_index=target_item_index,
        malicious_shard_id=0,
        device=torch.device("cpu"),
        seed=23,
        top_k=3,
        num_eval_negatives=4,
    )
    poisoned_shards = build_poisoned_shards(
        clean_shards=clean_shards,
        dataset=dataset,
        malicious_shard_id=0,
        target_genre="Action",
        target_item_index=target_item_index,
        attack_budget=1.0,
        seed=23,
    )
    _, _, attacked_rounds, _ = run_federated_training(
        dataset=dataset,
        shards=poisoned_shards,
        config=config,
        target_genre="Action",
        target_item_index=target_item_index,
        malicious_shard_id=0,
        device=torch.device("cpu"),
        seed=23,
        top_k=3,
        num_eval_negatives=4,
    )

    clean_final = clean_rounds[-1]["target_segment"]
    attacked_final = attacked_rounds[-1]["target_segment"]
    assert attacked_final["target_hitrate@3"] <= attacked_final["target_hitrate@5"] <= attacked_final["target_hitrate@10"]
    assert (
        attacked_final["target_hitrate@k"] > clean_final["target_hitrate@k"]
        or attacked_final["target_mean_rank"] < clean_final["target_mean_rank"]
    )


def test_clip_mean_defense_reduces_attack_uplift_on_fixture(tmp_path: Path) -> None:
    write_movielens_fixture(tmp_path, "ml-25m")
    dataset = load_movielens_dataset(tmp_path, "ml-25m")
    clean_shards = build_clean_shards(dataset, _manual_shards())
    target_item_index = dataset.item_id_to_index[9]
    attacked_config = FederatedConfig(
        embedding_dim=16,
        batch_size=8,
        local_epochs=2,
        federated_rounds=4,
        learning_rate=0.05,
    )
    defended_config = FederatedConfig(
        embedding_dim=16,
        batch_size=8,
        local_epochs=2,
        federated_rounds=4,
        learning_rate=0.05,
        aggregation_method="clip_mean",
        clip_factor=1.0,
    )

    poisoned_shards = build_poisoned_shards(
        clean_shards=clean_shards,
        dataset=dataset,
        malicious_shard_id=0,
        target_genre="Action",
        target_item_index=target_item_index,
        attack_budget=1.0,
        seed=23,
    )
    _, _, attacked_rounds, _ = run_federated_training(
        dataset=dataset,
        shards=poisoned_shards,
        config=attacked_config,
        target_genre="Action",
        target_item_index=target_item_index,
        malicious_shard_id=0,
        device=torch.device("cpu"),
        seed=23,
        top_k=3,
        num_eval_negatives=4,
    )
    _, _, defended_rounds, _ = run_federated_training(
        dataset=dataset,
        shards=poisoned_shards,
        config=defended_config,
        target_genre="Action",
        target_item_index=target_item_index,
        malicious_shard_id=0,
        device=torch.device("cpu"),
        seed=23,
        top_k=3,
        num_eval_negatives=4,
    )

    attacked_final = attacked_rounds[-1]
    defended_final = defended_rounds[-1]
    assert defended_final["aggregation"]["suppressed_shards"]
    assert defended_final["target_segment"]["target_hitrate@3"] <= defended_final["target_segment"]["target_hitrate@5"] <= defended_final["target_segment"]["target_hitrate@10"]
    assert (
        defended_final["target_segment"]["target_hitrate@k"] <= attacked_final["target_segment"]["target_hitrate@k"]
        or defended_final["target_segment"]["target_mean_rank"] > attacked_final["target_segment"]["target_mean_rank"]
    )
