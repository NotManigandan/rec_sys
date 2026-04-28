from pathlib import Path

from recsys.data import build_pairwise_preferences, load_ratings_dataset


def test_dataset_shape_and_observations() -> None:
    dataset = load_ratings_dataset(Path(__file__).resolve().parent.parent)
    assert dataset.num_users == 2000
    assert dataset.num_items == 6
    assert len(dataset.item_names) == 6
    assert len(dataset.reported_ratings[0]) == 3
    assert len(dataset.actual_ratings[0]) == 6


def test_pairwise_preferences_exist() -> None:
    dataset = load_ratings_dataset(Path(__file__).resolve().parent.parent)
    samples = build_pairwise_preferences(dataset)
    assert len(samples) > 0
    sample = samples[0]
    assert sample.weight > 0.0
    assert sample.positive_item_id != sample.negative_item_id
