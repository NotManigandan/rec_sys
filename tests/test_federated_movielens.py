from pathlib import Path
import json

import torch

import recsys.federated.benchmark as benchmark_module
from recsys.federated.benchmark import (
    assign_shards_to_devices,
    resolve_training_devices,
    run_federated_training,
)
from recsys.federated.bpr import (
    FederatedConfig,
    ServerState,
    ShardUpdate,
    aggregate_server_states,
    build_clean_shards,
    build_initial_server_state,
    build_initial_user_states,
    train_local_shard,
)
from recsys.federated.movielens import load_movielens_dataset, partition_users
from tests.federated_fixture import write_movielens_fixture


def _manual_shards(dataset):
    return {
        0: (2, 3, 4),
        1: (0, 1, 5),
    }


def test_movielens_csv_loader_and_split(tmp_path: Path) -> None:
    write_movielens_fixture(tmp_path, "ml-25m")
    dataset = load_movielens_dataset(tmp_path, "ml-25m")

    assert dataset.num_users == 6
    assert dataset.num_items == 10
    assert dataset.item_id_to_index[9] >= 0
    assert dataset.dominant_genre_by_user[0] == "Action"
    assert dataset.dominant_genre_by_user[1] == "Action"
    assert dataset.dominant_genre_by_user[2] == "Comedy"

    split = dataset.splits_by_user[0]
    train_movie_ids = [dataset.movie_id_for_item(item_index) for item_index, _ in split.train_ratings]
    assert train_movie_ids == [1, 2]
    assert dataset.movie_id_for_item(split.val_item) == 5
    assert dataset.movie_id_for_item(split.test_item) == 6


def test_movielens_dat_loader(tmp_path: Path) -> None:
    write_movielens_fixture(tmp_path, "ml-1m")
    dataset = load_movielens_dataset(tmp_path, "ml-1m")
    assert dataset.num_users == 6
    assert dataset.num_items == 10
    assert dataset.dominant_genre_by_user[5] == "Action"


def test_max_positive_interactions_filters_long_history_users(tmp_path: Path) -> None:
    write_movielens_fixture(tmp_path, "ml-25m")
    dataset = load_movielens_dataset(tmp_path, "ml-25m", max_positive_interactions=4)
    assert dataset.num_users == 6

    try:
        load_movielens_dataset(tmp_path, "ml-25m", max_positive_interactions=3)
    except ValueError as exc:
        assert "No MovieLens users remained" in str(exc)
    else:
        raise AssertionError("Expected long-history filtering to remove all fixture users.")


def test_partition_users_is_disjoint_and_reproducible() -> None:
    first_shards, first_lookup = partition_users(6, 2, 7)
    second_shards, second_lookup = partition_users(6, 2, 7)
    assert first_shards == second_shards
    assert first_lookup == second_lookup
    shard_users = set(first_shards[0]) | set(first_shards[1])
    assert set(first_shards[0]).isdisjoint(set(first_shards[1]))
    assert shard_users == set(range(6))


def test_resolve_training_devices_expands_num_gpus(monkeypatch) -> None:
    monkeypatch.setattr(benchmark_module, "resolve_runtime_device", lambda text: torch.device(text))
    devices = resolve_training_devices(torch.device("cuda"), None, num_gpus=3)
    assert [str(device) for device in devices] == ["cuda:0", "cuda:1", "cuda:2"]


def test_resolve_training_devices_prefers_explicit_devices_over_num_gpus(monkeypatch) -> None:
    monkeypatch.setattr(benchmark_module, "resolve_runtime_device", lambda text: torch.device(text))
    devices = resolve_training_devices(torch.device("cuda"), "cuda:2,cuda:4", num_gpus=8)
    assert [str(device) for device in devices] == ["cuda:2", "cuda:4"]


def test_assign_shards_to_devices_balances_by_sample_count() -> None:
    shards = {
        0: benchmark_module.FederatedShard(user_indices=(0,), train_ratings_by_user={0: ((1, 1.0),) * 10}, known_items_by_user={0: frozenset({1})}),
        1: benchmark_module.FederatedShard(user_indices=(1,), train_ratings_by_user={1: ((1, 1.0),) * 8}, known_items_by_user={1: frozenset({1})}),
        2: benchmark_module.FederatedShard(user_indices=(2,), train_ratings_by_user={2: ((1, 1.0),) * 3}, known_items_by_user={2: frozenset({1})}),
        3: benchmark_module.FederatedShard(user_indices=(3,), train_ratings_by_user={3: ((1, 1.0),) * 2}, known_items_by_user={3: frozenset({1})}),
    }
    assignments = assign_shards_to_devices(shards, [torch.device("cuda:0"), torch.device("cuda:1")])
    device_loads = {
        device_key: sum(shards[shard_id].sample_count for shard_id in shard_ids)
        for device_key, shard_ids in assignments.items()
    }
    assert set(assignments.keys()) == {"cuda:0", "cuda:1"}
    assert max(device_loads.values()) - min(device_loads.values()) <= 1


def test_attack_bundle_roundtrip_uses_serialized_shard_partition(tmp_path: Path) -> None:
    write_movielens_fixture(tmp_path, "ml-25m")
    dataset = load_movielens_dataset(tmp_path, "ml-25m")
    shard_to_users = _manual_shards(dataset)
    args = type(
        "Args",
        (),
        {
            "dataset_variant": "ml-25m",
            "model": "mf_bpr",
            "embedding_dim": 8,
            "hidden_dim": 32,
            "mlp_layers": 2,
            "dropout": 0.0,
            "local_epochs": 1,
            "federated_rounds": 2,
            "batch_size": 4,
            "learning_rate": 0.05,
            "weight_decay": 1e-4,
            "eval_every": 1,
            "eval_batch_size": 2,
            "top_k": 3,
            "num_eval_negatives": 4,
            "min_positive_rating": 4.0,
            "max_positive_interactions": None,
            "num_shards": 2,
            "malicious_shard_id": 0,
            "attack_budgets": "0.1,0.5",
            "attack_filler_items_per_user": 8,
            "attack_neutral_items_per_user": 0,
            "attack_target_weight": 20.0,
            "attack_filler_weight": 0.5,
            "attack_neutral_weight": 0.0,
            "attack_filler_pool_size": 64,
            "attack_neutral_pool_size": 32,
            "seed": 7,
        },
    )()
    bundle = benchmark_module._build_attack_bundle(
        dataset=dataset,
        args=args,
        shard_to_users=shard_to_users,
        target_genre="Action",
        target_item_index=dataset.item_id_to_index[9],
        target_selection_report={"strategy": "test"},
        clean_section={"final_metrics": {"foo": 1}},
        attack_results=[{"attack_budget": 0.1}],
        training_devices=[torch.device("cpu")],
        attack_budgets=[0.1, 0.5],
        attack_params={
            "filler_items_per_user": 8,
            "neutral_items_per_user": 0,
            "target_item_weight": 20.0,
            "filler_item_weight": 0.5,
            "neutral_item_weight": 0.0,
            "filler_candidate_pool_size": 64,
            "neutral_candidate_pool_size": 32,
        },
        runtime_seed=7,
    )
    bundle_path = tmp_path / "attack_bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    loaded = benchmark_module._load_attack_bundle(bundle_path)
    restored_shards = benchmark_module._deserialize_shard_to_users(loaded["shard_to_users"])

    assert restored_shards == shard_to_users
    assert loaded["attack_setup"]["target_genre"] == "Action"
    assert loaded["attack_setup"]["attack_budgets"] == [0.1, 0.5]


def test_target_selection_heuristic_prefers_closer_low_exposure_item(tmp_path: Path) -> None:
    write_movielens_fixture(tmp_path, "ml-25m")
    dataset = load_movielens_dataset(tmp_path, "ml-25m")
    clean_shards = build_clean_shards(dataset, _manual_shards(dataset))

    item_embeddings = torch.zeros((dataset.num_items, 1), dtype=torch.float32)
    item_embeddings[dataset.item_id_to_index[8], 0] = 1.0
    item_embeddings[dataset.item_id_to_index[3], 0] = 0.5
    item_embeddings[dataset.item_id_to_index[4], 0] = 0.3
    item_embeddings[dataset.item_id_to_index[7], 0] = 0.2
    item_embeddings[dataset.item_id_to_index[9], 0] = 0.9
    clean_server_state = ServerState(
        item_embeddings=item_embeddings,
        item_bias=torch.zeros(dataset.num_items, dtype=torch.float32),
    )
    clean_user_states = {
        0: torch.zeros((len(clean_shards[0].user_indices), 1), dtype=torch.float32),
        1: torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float32),
    }

    summary = benchmark_module._evaluate_target_genre_vulnerability(
        dataset=dataset,
        clean_server_state=clean_server_state,
        config=FederatedConfig(embedding_dim=1, model_name="mf_bpr"),
        clean_user_states=clean_user_states,
        clean_shards=clean_shards,
        target_genre="Action",
        target_segment_users=[0],
        malicious_shard_id=0,
        top_k=1,
    )

    assert summary["selected_movie_id"] == 9
    assert summary["top_candidates"][0]["movie_id"] == 9
    assert summary["top_candidates"][0]["mean_margin_to_topk"] < summary["top_candidates"][1]["mean_margin_to_topk"]


def test_server_aggregation_is_weighted_average() -> None:
    state_a = ServerState(
        item_embeddings=torch.tensor([[1.0], [1.0]], dtype=torch.float32),
        item_bias=torch.tensor([1.0, 1.0], dtype=torch.float32),
    )
    state_b = ServerState(
        item_embeddings=torch.tensor([[3.0], [3.0]], dtype=torch.float32),
        item_bias=torch.tensor([3.0, 3.0], dtype=torch.float32),
    )
    aggregated, diagnostics = aggregate_server_states(
        [
            ShardUpdate(shard_id=0, weight=1, server_state=state_a),
            ShardUpdate(shard_id=1, weight=3, server_state=state_b),
        ]
    )
    assert torch.allclose(aggregated.item_embeddings, torch.tensor([[2.5], [2.5]]))
    assert torch.allclose(aggregated.item_bias, torch.tensor([2.5, 2.5]))
    assert diagnostics["method"] == "mean"


def test_clip_mean_suppresses_outlier_delta() -> None:
    base_state = ServerState(
        item_embeddings=torch.zeros((2, 1), dtype=torch.float32),
        item_bias=torch.zeros(2, dtype=torch.float32),
    )
    benign_state = ServerState(
        item_embeddings=torch.tensor([[1.0], [1.0]], dtype=torch.float32),
        item_bias=torch.tensor([1.0, 1.0], dtype=torch.float32),
    )
    malicious_state = ServerState(
        item_embeddings=torch.tensor([[100.0], [100.0]], dtype=torch.float32),
        item_bias=torch.tensor([100.0, 100.0], dtype=torch.float32),
    )
    aggregated, diagnostics = aggregate_server_states(
        [
            ShardUpdate(shard_id=0, weight=1, server_state=benign_state),
            ShardUpdate(shard_id=1, weight=1, server_state=malicious_state),
        ],
        base_server_state=base_state,
        config=FederatedConfig(aggregation_method="clip_mean", clip_factor=1.0),
    )

    assert torch.allclose(aggregated.item_embeddings, torch.tensor([[1.0], [1.0]]), atol=1e-5)
    assert diagnostics["suppressed_shards"] == [1]


def test_clip_trimmed_mean_discards_extreme_coordinate_updates() -> None:
    base_state = ServerState(
        item_embeddings=torch.zeros((1, 1), dtype=torch.float32),
        item_bias=torch.zeros(1, dtype=torch.float32),
    )
    updates = [
        ShardUpdate(
            shard_id=index,
            weight=1,
            server_state=ServerState(
                item_embeddings=torch.tensor([[value]], dtype=torch.float32),
                item_bias=torch.tensor([value], dtype=torch.float32),
            ),
        )
        for index, value in enumerate([1.0, 2.0, 3.0, 100.0])
    ]
    aggregated, diagnostics = aggregate_server_states(
        updates,
        base_server_state=base_state,
        config=FederatedConfig(aggregation_method="clip_trimmed_mean", clip_factor=1000.0, trim_ratio=0.25),
    )

    assert torch.allclose(aggregated.item_embeddings, torch.tensor([[2.5]]), atol=1e-5)
    assert diagnostics["trim_count"] == 1


def test_focus_clip_mean_suppresses_multi_item_concentrated_update() -> None:
    base_state = ServerState(
        item_embeddings=torch.zeros((10, 1), dtype=torch.float32),
        item_bias=torch.zeros(10, dtype=torch.float32),
    )
    benign_state = ServerState(
        item_embeddings=torch.ones((10, 1), dtype=torch.float32),
        item_bias=torch.zeros(10, dtype=torch.float32),
    )
    malicious_embeddings = torch.zeros((10, 1), dtype=torch.float32)
    malicious_embeddings[0] = 5.0
    malicious_embeddings[1] = 5.0
    malicious_embeddings[2] = 5.0
    malicious_state = ServerState(
        item_embeddings=malicious_embeddings,
        item_bias=torch.zeros(10, dtype=torch.float32),
    )
    aggregated, diagnostics = aggregate_server_states(
        [
            ShardUpdate(shard_id=0, weight=1, server_state=benign_state),
            ShardUpdate(shard_id=1, weight=1, server_state=malicious_state),
        ],
        base_server_state=base_state,
        config=FederatedConfig(
            aggregation_method="focus_clip_mean",
            clip_factor=1000.0,
            focus_top_k=3,
            focus_factor=2.0,
        ),
    )

    assert diagnostics["method"] == "focus_clip_mean"
    assert diagnostics["suppressed_shards"] == [1]
    assert diagnostics["shards"][1]["focus_score"] > diagnostics["shards"][0]["focus_score"]
    assert diagnostics["shards"][1]["suppression_reasons"] == ["focus"]
    assert diagnostics["shards"][1]["focus_scale"] < 1.0
    assert aggregated.item_embeddings[0, 0] < malicious_state.item_embeddings[0, 0]


def test_local_training_updates_user_and_item_parameters(tmp_path: Path) -> None:
    write_movielens_fixture(tmp_path, "ml-25m")
    dataset = load_movielens_dataset(tmp_path, "ml-25m")
    clean_shards = build_clean_shards(dataset, _manual_shards(dataset))
    server_state = build_initial_server_state(dataset.num_items, embedding_dim=8, seed=5)
    user_states = build_initial_user_states(clean_shards, embedding_dim=8, seed=5)
    initial_user_state = user_states[1].clone()
    initial_item_state = server_state.item_embeddings.clone()

    updated_user_state, local_server_state = train_local_shard(
        shard=clean_shards[1],
        dataset=dataset,
        server_state=server_state,
        user_state=user_states[1],
        config=FederatedConfig(embedding_dim=8, batch_size=4, local_epochs=1, federated_rounds=1, learning_rate=0.05),
        device=torch.device("cpu"),
        seed=11,
    )

    assert not torch.allclose(updated_user_state, initial_user_state)
    assert not torch.allclose(local_server_state.item_embeddings, initial_item_state)


def test_eval_every_keeps_final_round_metrics(tmp_path: Path) -> None:
    write_movielens_fixture(tmp_path, "ml-25m")
    dataset = load_movielens_dataset(tmp_path, "ml-25m")
    clean_shards = build_clean_shards(dataset, _manual_shards(dataset))
    target_item_index = dataset.item_id_to_index[9]
    config = FederatedConfig(
        embedding_dim=8,
        batch_size=4,
        local_epochs=1,
        federated_rounds=3,
        learning_rate=0.05,
        eval_every_rounds=2,
        eval_batch_size=2,
    )

    _, _, round_metrics, training_traces = run_federated_training(
        dataset=dataset,
        shards=clean_shards,
        config=config,
        target_genre="Action",
        target_item_index=target_item_index,
        malicious_shard_id=0,
        device=torch.device("cpu"),
        seed=19,
        top_k=3,
        num_eval_negatives=4,
    )

    assert [metrics["round_index"] for metrics in round_metrics] == [2, 3]
    assert [trace["round_index"] for trace in training_traces] == [1.0, 2.0, 3.0]
    assert training_traces[-1]["mean_loss"] > 0.0
    target_segment = round_metrics[-1]["target_segment"]
    assert target_segment["target_hitrate@3"] <= target_segment["target_hitrate@5"] <= target_segment["target_hitrate@10"]
    assert "target_hitrate@10" in round_metrics[-1]["per_segment"]["Action"]
    assert round_metrics[-1]["training"]["mean_loss"] > 0.0


def test_neural_bpr_training_smoke(tmp_path: Path) -> None:
    write_movielens_fixture(tmp_path, "ml-25m")
    dataset = load_movielens_dataset(tmp_path, "ml-25m")
    clean_shards = build_clean_shards(dataset, _manual_shards(dataset))
    target_item_index = dataset.item_id_to_index[9]
    config = FederatedConfig(
        model_name="neural_bpr",
        embedding_dim=8,
        hidden_dim=32,
        mlp_layers=2,
        dropout=0.0,
        batch_size=4,
        local_epochs=1,
        federated_rounds=2,
        learning_rate=0.05,
        eval_batch_size=2,
    )

    _, _, round_metrics, training_traces = run_federated_training(
        dataset=dataset,
        shards=clean_shards,
        config=config,
        target_genre="Action",
        target_item_index=target_item_index,
        malicious_shard_id=0,
        device=torch.device("cpu"),
        seed=29,
        top_k=3,
        num_eval_negatives=4,
    )

    assert round_metrics[-1]["overall_benign"]["users"] == 3
    assert len(training_traces) == 2
    assert training_traces[-1]["mean_loss"] > 0.0
    target_segment = round_metrics[-1]["target_segment"]
    assert target_segment["target_hitrate@3"] <= target_segment["target_hitrate@5"] <= target_segment["target_hitrate@10"]
