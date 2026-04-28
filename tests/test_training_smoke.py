from pathlib import Path

import torch

from recsys.data import build_pairwise_preferences, load_ratings_dataset
from recsys.train import run_experiment, set_seed


def test_bpr_training_smoke() -> None:
    set_seed(11)
    dataset = load_ratings_dataset(Path(__file__).resolve().parent.parent)
    samples = build_pairwise_preferences(dataset)
    result = run_experiment(
        model_name="bpr_mf",
        dataset=dataset,
        samples=samples,
        device=torch.device("cpu"),
        epochs=5,
        batch_size=512,
        embedding_dim=8,
        hidden_dim=32,
        learning_rate=0.03,
        weight_decay=1e-4,
    )
    assert "top1_accuracy" in result.metrics
    assert len(result.train_losses) == 5
