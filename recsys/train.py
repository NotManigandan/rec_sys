from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import PreferenceSample, RatingsDataset, build_item_popularity, build_pairwise_preferences, load_ratings_dataset
from .device import resolve_runtime_device
from .metrics import evaluate_rankings
from .models import BPRMatrixFactorization, NeuralCollaborativeFiltering, RecommenderModel, TwoTowerHistoryModel


MODEL_REGISTRY = {
    "bpr_mf": BPRMatrixFactorization,
    "two_tower": TwoTowerHistoryModel,
    "neural_cf": NeuralCollaborativeFiltering,
}


@dataclass
class ExperimentResult:
    model: str
    metrics: Dict[str, object]
    train_losses: List[float]


class PreferenceDataset(Dataset[PreferenceSample]):
    def __init__(self, samples: List[PreferenceSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PreferenceSample:
        return self.samples[index]


def collate_samples(samples: List[PreferenceSample]) -> Dict[str, torch.Tensor]:
    return {
        "user_ids": torch.tensor([sample.user_id for sample in samples], dtype=torch.long),
        "positive_item_ids": torch.tensor([sample.positive_item_id for sample in samples], dtype=torch.long),
        "negative_item_ids": torch.tensor([sample.negative_item_id for sample in samples], dtype=torch.long),
        "weights": torch.tensor([sample.weight for sample in samples], dtype=torch.float32),
    }


def set_seed(seed: int, device: torch.device | None = None) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if device is not None and device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def build_model(model_name: str, dataset: RatingsDataset, embedding_dim: int, hidden_dim: int) -> RecommenderModel:
    if model_name == "bpr_mf":
        return BPRMatrixFactorization(dataset.num_users, dataset.num_items, embedding_dim=embedding_dim)
    if model_name == "two_tower":
        return TwoTowerHistoryModel(dataset.num_items, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    if model_name == "neural_cf":
        return NeuralCollaborativeFiltering(
            dataset.num_users,
            dataset.num_items,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )
    raise ValueError(f"Unknown model: {model_name}")


def train_model(
    model: RecommenderModel,
    dataset: RatingsDataset,
    samples: List[PreferenceSample],
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> List[float]:
    history_tensor = dataset.history_tensor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loader = DataLoader(
        PreferenceDataset(samples),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_samples,
    )

    model.train()
    train_losses: List[float] = []
    for _ in range(epochs):
        total_loss = 0.0
        total_examples = 0
        for batch in loader:
            user_ids = batch["user_ids"].to(device)
            positive_item_ids = batch["positive_item_ids"].to(device)
            negative_item_ids = batch["negative_item_ids"].to(device)
            weights = batch["weights"].to(device)

            user_histories = history_tensor[user_ids]
            positive_scores = model.score(user_ids, positive_item_ids, user_histories)
            negative_scores = model.score(user_ids, negative_item_ids, user_histories)
            loss = -(F.logsigmoid(positive_scores - negative_scores) * weights).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size_actual = user_ids.size(0)
            total_loss += loss.detach().item() * batch_size_actual
            total_examples += batch_size_actual
        train_losses.append(total_loss / max(total_examples, 1))
    return train_losses


@torch.no_grad()
def rank_unseen_items(model: RecommenderModel, dataset: RatingsDataset, device: torch.device) -> Dict[int, List[int]]:
    model.eval()
    history_tensor = dataset.history_tensor().to(device)
    all_items = torch.arange(dataset.num_items, dtype=torch.long, device=device)
    ranked_items_by_user: Dict[int, List[int]] = {}
    for user_id in dataset.user_ids:
        user_ids = torch.full((dataset.num_items,), user_id, dtype=torch.long, device=device)
        user_history = history_tensor[user_id].unsqueeze(0).expand(dataset.num_items, -1)
        scores = model.score(user_ids, all_items, user_history)
        unseen_items = dataset.unseen_items(user_id)
        unseen_scores = sorted(
            ((item_id, float(scores[item_id].item())) for item_id in unseen_items),
            key=lambda pair: pair[1],
            reverse=True,
        )
        ranked_items_by_user[user_id] = [item_id for item_id, _ in unseen_scores]
    return ranked_items_by_user


def rank_unseen_items_by_popularity(dataset: RatingsDataset) -> Dict[int, List[int]]:
    popularity_scores = build_item_popularity(dataset)
    ranked_items_by_user: Dict[int, List[int]] = {}
    for user_id in dataset.user_ids:
        unseen_items = dataset.unseen_items(user_id)
        ranked_items = sorted(unseen_items, key=lambda item_id: float(popularity_scores[item_id]), reverse=True)
        ranked_items_by_user[user_id] = ranked_items
    return ranked_items_by_user


def run_experiment(
    model_name: str,
    dataset: RatingsDataset,
    samples: List[PreferenceSample],
    device: torch.device,
    epochs: int,
    batch_size: int,
    embedding_dim: int,
    hidden_dim: int,
    learning_rate: float,
    weight_decay: float,
) -> ExperimentResult:
    model = build_model(model_name, dataset, embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
    train_losses = train_model(
        model=model,
        dataset=dataset,
        samples=samples,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    ranked_items = rank_unseen_items(model, dataset, device)
    metrics = evaluate_rankings(dataset, ranked_items)
    return ExperimentResult(model=model_name, metrics=metrics, train_losses=train_losses)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train recommendation baselines on the synthetic ratings data.")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), default="bpr_mf")
    parser.add_argument("--compare-all", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_runtime_device(args.device)
    set_seed(args.seed, device)
    dataset = load_ratings_dataset(args.data_root)
    samples = build_pairwise_preferences(dataset)

    results: Dict[str, object] = {}
    popularity_rankings = rank_unseen_items_by_popularity(dataset)
    results["popularity"] = {"metrics": evaluate_rankings(dataset, popularity_rankings)}

    model_names = sorted(MODEL_REGISTRY) if args.compare_all else [args.model]
    for model_name in model_names:
        result = run_experiment(
            model_name=model_name,
            dataset=dataset,
            samples=samples,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
        )
        results[model_name] = {
            "metrics": result.metrics,
            "train_losses": result.train_losses,
        }
        metric_summary = result.metrics
        print(
            f"{model_name}: top1={metric_summary['top1_accuracy']:.4f} "
            f"ndcg@3={metric_summary['ndcg@3']:.4f} "
            f"pairwise={metric_summary['pairwise_accuracy']:.4f} "
            f"mrr={metric_summary['mrr']:.4f}"
        )

    print(
        f"popularity: top1={results['popularity']['metrics']['top1_accuracy']:.4f} "
        f"ndcg@3={results['popularity']['metrics']['ndcg@3']:.4f} "
        f"pairwise={results['popularity']['metrics']['pairwise_accuracy']:.4f} "
        f"mrr={results['popularity']['metrics']['mrr']:.4f}"
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
