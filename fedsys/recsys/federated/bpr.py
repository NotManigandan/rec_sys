from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .movielens import MovieLensFederatedDataset


NEURAL_EVAL_USER_CHUNK_SIZE = 128


@dataclass(frozen=True)
class FederatedShard:
    user_indices: tuple[int, ...]
    train_ratings_by_user: Dict[int, tuple[tuple[int, float], ...]]
    known_items_by_user: Dict[int, frozenset[int]]
    synthetic_user_count: int = 0

    @property
    def sample_count(self) -> int:
        return sum(len(ratings) for ratings in self.train_ratings_by_user.values())


@dataclass
class FederatedConfig:
    embedding_dim: int = 32
    hidden_dim: int = 1024
    mlp_layers: int = 4
    dropout: float = 0.1
    model_name: str = "mf_bpr"
    batch_size: int = 1024
    local_epochs: int = 1
    federated_rounds: int = 5
    learning_rate: float = 0.05
    weight_decay: float = 1e-4
    min_positive_rating: float = 4.0
    eval_every_rounds: int = 1
    eval_batch_size: int = 1024
    aggregation_method: str = "mean"
    clip_factor: float = 1.5
    trim_ratio: float = 0.25
    focus_top_k: int = 3
    focus_factor: float = 2.0


@dataclass
class ServerState:
    item_embeddings: torch.Tensor
    item_bias: torch.Tensor
    scorer_state: Dict[str, torch.Tensor] | None = None

    def clone(self) -> "ServerState":
        cloned_state = None
        if self.scorer_state is not None:
            cloned_state = {name: tensor.clone() for name, tensor in self.scorer_state.items()}
        return ServerState(self.item_embeddings.clone(), self.item_bias.clone(), cloned_state)

    def to(self, device: torch.device) -> "ServerState":
        scorer_state = None
        if self.scorer_state is not None:
            scorer_state = {name: tensor.to(device).clone() for name, tensor in self.scorer_state.items()}
        return ServerState(
            item_embeddings=self.item_embeddings.to(device).clone(),
            item_bias=self.item_bias.to(device).clone(),
            scorer_state=scorer_state,
        )


@dataclass(frozen=True)
class PreparedShardData:
    shard_id: int
    synthetic_user_count: int
    sample_count: int
    active_local_users: torch.Tensor
    positive_items: torch.Tensor
    positive_weights: torch.Tensor
    known_items: torch.Tensor


@dataclass
class ShardRuntime:
    shard_id: int
    synthetic_user_count: int
    sample_count: int
    device: torch.device
    active_local_users: torch.Tensor
    positive_items: torch.Tensor
    positive_weights: torch.Tensor
    known_items: torch.Tensor
    user_state: torch.Tensor


@dataclass
class ShardUpdate:
    shard_id: int
    weight: int
    server_state: ServerState
    mean_loss: float = 0.0
    last_loss: float = 0.0
    steps: int = 0

    def to(self, device: torch.device) -> "ShardUpdate":
        return ShardUpdate(
            shard_id=self.shard_id,
            weight=self.weight,
            server_state=self.server_state.to(device),
            mean_loss=self.mean_loss,
            last_loss=self.last_loss,
            steps=self.steps,
        )


def validate_model_name(model_name: str) -> str:
    if model_name not in {"mf_bpr", "neural_bpr"}:
        raise ValueError(f"Unsupported federated model: {model_name}")
    return model_name


def validate_aggregation_method(aggregation_method: str) -> str:
    if aggregation_method not in {
        "mean",
        "clip_mean",
        "clip_trimmed_mean",
        "focus_clip_mean",
        "focus_clip_trimmed_mean",
    }:
        raise ValueError(f"Unsupported aggregation method: {aggregation_method}")
    return aggregation_method


def _feature_interactions(user_vector: torch.Tensor, item_vector: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            user_vector,
            item_vector,
            user_vector * item_vector,
            torch.abs(user_vector - item_vector),
        ],
        dim=-1,
    )


class MLPScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, mlp_layers: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(max(mlp_layers, 1)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class LocalFederatedRanker(nn.Module):
    def __init__(
        self,
        config: FederatedConfig,
        user_embeddings: torch.Tensor,
        server_state: ServerState,
    ) -> None:
        super().__init__()
        self.config = config
        self.model_name = validate_model_name(config.model_name)
        self.user_embedding = nn.Embedding.from_pretrained(user_embeddings.clone(), freeze=False)
        self.item_embedding = nn.Embedding.from_pretrained(server_state.item_embeddings.clone(), freeze=False)
        self.item_bias = nn.Embedding.from_pretrained(server_state.item_bias.unsqueeze(-1).clone(), freeze=False)
        self.scorer: MLPScorer | None = None
        if self.model_name == "neural_bpr":
            self.scorer = MLPScorer(
                input_dim=config.embedding_dim * 4,
                hidden_dim=config.hidden_dim,
                mlp_layers=config.mlp_layers,
                dropout=config.dropout,
            )
            if server_state.scorer_state is not None:
                self.scorer.load_state_dict(server_state.scorer_state)

    def score(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        user_vector = self.user_embedding(user_indices)
        item_vector = self.item_embedding(item_indices)
        item_bias = self.item_bias(item_indices).squeeze(-1)
        if self.model_name == "mf_bpr":
            interaction = (user_vector * item_vector).sum(dim=-1)
            return interaction + item_bias
        assert self.scorer is not None
        features = _feature_interactions(user_vector, item_vector)
        return self.scorer(features) + item_bias

    def export_server_state(self) -> ServerState:
        scorer_state = None
        if self.scorer is not None:
            scorer_state = {name: tensor.detach() for name, tensor in self.scorer.state_dict().items()}
        return ServerState(
            item_embeddings=self.item_embedding.weight.detach(),
            item_bias=self.item_bias.weight.detach().squeeze(-1),
            scorer_state=scorer_state,
        )


def build_scorer_module(
    server_state: ServerState,
    config: FederatedConfig,
    device: torch.device,
) -> MLPScorer | None:
    model_name = validate_model_name(config.model_name)
    if model_name == "mf_bpr":
        return None
    scorer = MLPScorer(
        input_dim=config.embedding_dim * 4,
        hidden_dim=config.hidden_dim,
        mlp_layers=config.mlp_layers,
        dropout=config.dropout,
    ).to(device)
    if server_state.scorer_state is not None:
        scorer.load_state_dict(server_state.scorer_state)
    scorer.eval()
    return scorer


def score_candidate_items(
    user_embeddings: torch.Tensor,
    item_indices: torch.Tensor,
    server_state: ServerState,
    config: FederatedConfig,
    scorer_module: MLPScorer | None = None,
) -> torch.Tensor:
    model_name = validate_model_name(config.model_name)
    with torch.inference_mode():
        item_embeddings = server_state.item_embeddings[item_indices]
        item_bias = server_state.item_bias[item_indices]

        if user_embeddings.dim() == 1:
            if item_indices.dim() != 1:
                raise ValueError("1D user embeddings require 1D item indices.")
            if model_name == "mf_bpr":
                return torch.matmul(item_embeddings, user_embeddings) + item_bias
            if scorer_module is None:
                raise ValueError("A scorer module is required for neural_bpr scoring.")
            repeated_user = user_embeddings.unsqueeze(0).expand(item_embeddings.size(0), -1)
            return scorer_module(_feature_interactions(repeated_user, item_embeddings)) + item_bias

        if item_indices.dim() != 2:
            raise ValueError("Batched user embeddings require 2D item indices.")
        if model_name == "mf_bpr":
            return (item_embeddings * user_embeddings.unsqueeze(1)).sum(dim=-1) + item_bias
        if scorer_module is None:
            raise ValueError("A scorer module is required for neural_bpr scoring.")

        outputs: list[torch.Tensor] = []
        for start in range(0, user_embeddings.size(0), NEURAL_EVAL_USER_CHUNK_SIZE):
            end = min(start + NEURAL_EVAL_USER_CHUNK_SIZE, user_embeddings.size(0))
            chunk_users = user_embeddings[start:end]
            chunk_item_embeddings = item_embeddings[start:end]
            chunk_item_bias = item_bias[start:end]
            repeated_users = chunk_users.unsqueeze(1).expand(-1, chunk_item_embeddings.size(1), -1)
            features = _feature_interactions(repeated_users, chunk_item_embeddings)
            flat_scores = scorer_module(features.reshape(-1, features.size(-1)))
            outputs.append(
                flat_scores.reshape(chunk_item_embeddings.size(0), chunk_item_embeddings.size(1)) + chunk_item_bias
            )
        return torch.cat(outputs, dim=0)


def build_initial_server_state(
    num_items: int,
    embedding_dim: int,
    seed: int,
    config: FederatedConfig | None = None,
) -> ServerState:
    generator = torch.Generator().manual_seed(seed)
    item_embeddings = torch.randn((num_items, embedding_dim), generator=generator) * 0.05
    item_bias = torch.zeros(num_items, dtype=torch.float32)
    scorer_state = None
    if config is not None and validate_model_name(config.model_name) == "neural_bpr":
        scorer = MLPScorer(
            input_dim=config.embedding_dim * 4,
            hidden_dim=config.hidden_dim,
            mlp_layers=config.mlp_layers,
            dropout=config.dropout,
        )
        scorer_state = {name: tensor.detach().clone() for name, tensor in scorer.state_dict().items()}
    return ServerState(item_embeddings=item_embeddings, item_bias=item_bias, scorer_state=scorer_state)


def build_initial_user_states(
    shards: Dict[int, FederatedShard],
    embedding_dim: int,
    seed: int,
) -> Dict[int, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed + 17)
    user_states: Dict[int, torch.Tensor] = {}
    for shard_id, shard in shards.items():
        user_states[shard_id] = torch.randn((len(shard.user_indices), embedding_dim), generator=generator) * 0.05
    return user_states


def build_clean_shards(
    dataset: MovieLensFederatedDataset,
    shard_to_users: Dict[int, tuple[int, ...]],
) -> Dict[int, FederatedShard]:
    shards: Dict[int, FederatedShard] = {}
    for shard_id, user_indices in shard_to_users.items():
        train_ratings_by_user = {
            user_index: dataset.splits_by_user[user_index].train_ratings
            for user_index in user_indices
        }
        known_items_by_user = {
            user_index: dataset.splits_by_user[user_index].known_items
            for user_index in user_indices
        }
        shards[shard_id] = FederatedShard(
            user_indices=user_indices,
            train_ratings_by_user=train_ratings_by_user,
            known_items_by_user=known_items_by_user,
        )
    return shards


def _pad_rows(
    rows: list[list[int]] | list[list[float]],
    *,
    pad_value: int | float,
    dtype: torch.dtype,
) -> torch.Tensor:
    max_len = max(len(row) for row in rows)
    padded = []
    for row in rows:
        padded.append(row + [pad_value] * (max_len - len(row)))
    return torch.tensor(padded, dtype=dtype)


def prepare_shard_training_data(shard_id: int, shard: FederatedShard) -> PreparedShardData:
    active_local_users: list[int] = []
    positive_items_rows: list[list[int]] = []
    positive_weights_rows: list[list[float]] = []
    known_items_rows: list[list[int]] = []

    for local_index, user_index in enumerate(shard.user_indices):
        ratings = list(shard.train_ratings_by_user.get(user_index, ()))
        known_items = list(shard.known_items_by_user.get(user_index, frozenset()))
        if ratings:
            active_local_users.append(local_index)
        positive_items_rows.append([item_index for item_index, _ in ratings] or [0])
        positive_weights_rows.append([rating for _, rating in ratings] or [1.0])
        known_items_rows.append(known_items or [-1])

    if not active_local_users:
        raise ValueError(f"Shard {shard_id} has no trainable users.")

    return PreparedShardData(
        shard_id=shard_id,
        synthetic_user_count=shard.synthetic_user_count,
        sample_count=max(shard.sample_count, 1),
        active_local_users=torch.tensor(active_local_users, dtype=torch.long),
        positive_items=_pad_rows(positive_items_rows, pad_value=0, dtype=torch.long),
        positive_weights=_pad_rows(positive_weights_rows, pad_value=0.0, dtype=torch.float32),
        known_items=_pad_rows(known_items_rows, pad_value=-1, dtype=torch.long),
    )


def build_prepared_shards(shards: Dict[int, FederatedShard]) -> Dict[int, PreparedShardData]:
    return {
        shard_id: prepare_shard_training_data(shard_id, shard)
        for shard_id, shard in shards.items()
    }


def create_shard_runtime(
    prepared_shard: PreparedShardData,
    user_state: torch.Tensor,
    device: torch.device,
) -> ShardRuntime:
    return ShardRuntime(
        shard_id=prepared_shard.shard_id,
        synthetic_user_count=prepared_shard.synthetic_user_count,
        sample_count=prepared_shard.sample_count,
        device=device,
        active_local_users=prepared_shard.active_local_users.to(device),
        positive_items=prepared_shard.positive_items.to(device),
        positive_weights=prepared_shard.positive_weights.to(device),
        known_items=prepared_shard.known_items.to(device),
        user_state=user_state.to(device),
    )


def build_shard_runtimes(
    prepared_shards: Dict[int, PreparedShardData],
    user_states: Dict[int, torch.Tensor],
    shard_to_device: Dict[int, torch.device],
) -> Dict[int, ShardRuntime]:
    return {
        shard_id: create_shard_runtime(prepared_shards[shard_id], user_states[shard_id], shard_to_device[shard_id])
        for shard_id in prepared_shards
    }


def _torch_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = "cpu" if device.type == "cpu" else str(device)
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


def _sample_batch_tensorized(
    runtime: ShardRuntime,
    num_items: int,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sampled_active_positions = torch.randint(
        high=runtime.active_local_users.numel(),
        size=(batch_size,),
        generator=generator,
        device=runtime.device,
    )
    batch_users = runtime.active_local_users[sampled_active_positions]
    batch_positive_weights = runtime.positive_weights[batch_users]
    sampled_positive_positions = torch.multinomial(batch_positive_weights, num_samples=1, generator=generator).squeeze(1)
    batch_positive_items = runtime.positive_items[batch_users].gather(
        1,
        sampled_positive_positions.unsqueeze(1),
    ).squeeze(1)

    known_items = runtime.known_items[batch_users]
    batch_negative_items = torch.randint(
        high=num_items,
        size=(batch_size,),
        generator=generator,
        device=runtime.device,
    )
    invalid = (known_items == batch_negative_items.unsqueeze(1)).any(dim=1)
    while invalid.any():
        replacement = torch.randint(
            high=num_items,
            size=(int(invalid.sum().item()),),
            generator=generator,
            device=runtime.device,
        )
        batch_negative_items[invalid] = replacement
        invalid = (known_items == batch_negative_items.unsqueeze(1)).any(dim=1)

    return batch_users, batch_positive_items, batch_negative_items


def train_local_shard_runtime(
    runtime: ShardRuntime,
    num_items: int,
    server_state: ServerState,
    config: FederatedConfig,
    seed: int,
) -> tuple[ServerState, Dict[str, float | int]]:
    local_model = LocalFederatedRanker(
        config=config,
        user_embeddings=runtime.user_state.clone(),
        server_state=server_state,
    ).to(runtime.device)
    optimizer = torch.optim.AdamW(
        local_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    steps_per_epoch = max(1, math.ceil(runtime.sample_count / max(config.batch_size, 1)))
    generator = _torch_generator(runtime.device, seed)
    total_loss = 0.0
    last_loss = 0.0
    total_steps = 0

    for _ in range(config.local_epochs):
        for _ in range(steps_per_epoch):
            batch_users, batch_positive_items, batch_negative_items = _sample_batch_tensorized(
                runtime=runtime,
                num_items=num_items,
                batch_size=config.batch_size,
                generator=generator,
            )
            positive_scores = local_model.score(batch_users, batch_positive_items)
            negative_scores = local_model.score(batch_users, batch_negative_items)
            loss = -F.logsigmoid(positive_scores - negative_scores).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            current_loss = float(loss.detach().item())
            total_loss += current_loss
            last_loss = current_loss
            total_steps += 1

    runtime.user_state = local_model.user_embedding.weight.detach()
    mean_loss = total_loss / max(total_steps, 1)
    return local_model.export_server_state(), {
        "mean_loss": mean_loss,
        "last_loss": last_loss,
        "steps": total_steps,
    }


def train_local_shard(
    shard: FederatedShard,
    dataset: MovieLensFederatedDataset,
    server_state: ServerState,
    user_state: torch.Tensor,
    config: FederatedConfig,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, ServerState]:
    prepared_shard = prepare_shard_training_data(0, shard)
    runtime = create_shard_runtime(prepared_shard, user_state, device)
    local_server_state, _ = train_local_shard_runtime(
        runtime=runtime,
        num_items=dataset.num_items,
        server_state=server_state.to(device),
        config=config,
        seed=seed,
    )
    updated_user_state = runtime.user_state.detach().cpu()
    updated_server_state = ServerState(
        item_embeddings=local_server_state.item_embeddings.detach().cpu(),
        item_bias=local_server_state.item_bias.detach().cpu(),
        scorer_state=None
        if local_server_state.scorer_state is None
        else {name: tensor.detach().cpu() for name, tensor in local_server_state.scorer_state.items()},
    )
    return updated_user_state, updated_server_state


def _server_state_delta(updated_state: ServerState, base_state: ServerState) -> ServerState:
    scorer_state = None
    if updated_state.scorer_state is not None and base_state.scorer_state is not None:
        scorer_state = {
            name: updated_state.scorer_state[name] - base_state.scorer_state[name]
            for name in updated_state.scorer_state
        }
    return ServerState(
        item_embeddings=updated_state.item_embeddings - base_state.item_embeddings,
        item_bias=updated_state.item_bias - base_state.item_bias,
        scorer_state=scorer_state,
    )


def _scale_server_state(state: ServerState, scale: float) -> ServerState:
    scorer_state = None
    if state.scorer_state is not None:
        scorer_state = {name: tensor * scale for name, tensor in state.scorer_state.items()}
    return ServerState(
        item_embeddings=state.item_embeddings * scale,
        item_bias=state.item_bias * scale,
        scorer_state=scorer_state,
    )


def _add_server_state(base_state: ServerState, delta_state: ServerState) -> ServerState:
    scorer_state = None
    if base_state.scorer_state is not None:
        scorer_state = {
            name: base_state.scorer_state[name] + delta_state.scorer_state[name]
            for name in base_state.scorer_state
        }
    return ServerState(
        item_embeddings=base_state.item_embeddings + delta_state.item_embeddings,
        item_bias=base_state.item_bias + delta_state.item_bias,
        scorer_state=scorer_state,
    )


def _server_state_norm(state: ServerState) -> float:
    squared_norm = state.item_embeddings.pow(2).sum() + state.item_bias.pow(2).sum()
    if state.scorer_state is not None:
        for tensor in state.scorer_state.values():
            squared_norm = squared_norm + tensor.pow(2).sum()
    return float(torch.sqrt(squared_norm).item())


def _item_push_scores(state: ServerState) -> torch.Tensor:
    embedding_push = torch.linalg.vector_norm(state.item_embeddings, ord=2, dim=1)
    return embedding_push + state.item_bias.abs()


def _focus_score(state: ServerState, top_k: int) -> float:
    item_push = _item_push_scores(state)
    if item_push.numel() == 0:
        return 0.0
    total_push = float(item_push.sum().item())
    if total_push <= 1e-12:
        return 0.0
    focus_top_k = min(max(top_k, 1), int(item_push.numel()))
    top_push = float(torch.topk(item_push, k=focus_top_k).values.sum().item())
    return top_push / total_push


def _weighted_average_server_state(states: Sequence[tuple[int, ServerState]]) -> ServerState:
    if not states:
        raise ValueError("No states were provided for weighted averaging.")
    total_weight = float(sum(weight for weight, _ in states))
    if total_weight <= 0:
        raise ValueError("Aggregation weights must be positive.")
    aggregated_embeddings = torch.zeros_like(states[0][1].item_embeddings)
    aggregated_bias = torch.zeros_like(states[0][1].item_bias)
    aggregated_scorer_state = None
    if states[0][1].scorer_state is not None:
        aggregated_scorer_state = {
            name: torch.zeros_like(tensor)
            for name, tensor in states[0][1].scorer_state.items()
        }
    for weight, state in states:
        scale = weight / total_weight
        aggregated_embeddings += state.item_embeddings * scale
        aggregated_bias += state.item_bias * scale
        if aggregated_scorer_state is not None and state.scorer_state is not None:
            for name, tensor in state.scorer_state.items():
                aggregated_scorer_state[name] += tensor * scale
    return ServerState(
        item_embeddings=aggregated_embeddings,
        item_bias=aggregated_bias,
        scorer_state=aggregated_scorer_state,
    )


def _mean_server_state(states: Sequence[ServerState]) -> ServerState:
    return _weighted_average_server_state([(1, state) for state in states])


def _trimmed_mean_tensor(stacked: torch.Tensor, trim_count: int) -> torch.Tensor:
    if trim_count <= 0 or stacked.size(0) <= 2 * trim_count:
        return stacked.mean(dim=0)
    sorted_values, _ = torch.sort(stacked, dim=0)
    return sorted_values[trim_count : sorted_values.size(0) - trim_count].mean(dim=0)


def _trimmed_mean_server_state(states: Sequence[ServerState], trim_ratio: float) -> ServerState:
    if not states:
        raise ValueError("No states were provided for trimmed-mean aggregation.")
    trim_count = int(math.floor(len(states) * max(trim_ratio, 0.0)))
    item_embeddings = _trimmed_mean_tensor(torch.stack([state.item_embeddings for state in states]), trim_count)
    item_bias = _trimmed_mean_tensor(torch.stack([state.item_bias for state in states]), trim_count)
    scorer_state = None
    if states[0].scorer_state is not None:
        scorer_state = {}
        for name in states[0].scorer_state:
            scorer_state[name] = _trimmed_mean_tensor(
                torch.stack([state.scorer_state[name] for state in states if state.scorer_state is not None]),
                trim_count,
            )
    return ServerState(item_embeddings=item_embeddings, item_bias=item_bias, scorer_state=scorer_state)


def aggregate_server_states(
    shard_updates: Sequence[ShardUpdate],
    base_server_state: ServerState | None = None,
    config: FederatedConfig | None = None,
) -> tuple[ServerState, Dict[str, object]]:
    if not shard_updates:
        raise ValueError("No shard states were provided for aggregation.")
    if base_server_state is None or config is None:
        aggregated_state = _weighted_average_server_state(
            [(update.weight, update.server_state) for update in shard_updates]
        )
        diagnostics = {
            "method": "mean",
            "reference_delta_norm": None,
            "trim_ratio": 0.0,
            "trim_count": 0,
            "suppressed_shards": [],
            "shards": [
                {
                    "shard_id": update.shard_id,
                    "weight": update.weight,
                    "delta_norm": None,
                    "clip_scale": 1.0,
                }
                for update in shard_updates
            ],
        }
        return aggregated_state, diagnostics

    aggregation_method = validate_aggregation_method(config.aggregation_method)
    if aggregation_method == "mean":
        aggregated_state = _weighted_average_server_state(
            [(update.weight, update.server_state) for update in shard_updates]
        )
        diagnostics = {
            "method": aggregation_method,
            "reference_delta_norm": None,
            "trim_ratio": 0.0,
            "trim_count": 0,
            "suppressed_shards": [],
            "shards": [
                {
                    "shard_id": update.shard_id,
                    "weight": update.weight,
                    "delta_norm": None,
                    "clip_scale": 1.0,
                }
                for update in shard_updates
            ],
        }
        return aggregated_state, diagnostics

    raw_deltas = [_server_state_delta(update.server_state, base_server_state) for update in shard_updates]
    raw_norms = torch.tensor([_server_state_norm(delta) for delta in raw_deltas], dtype=torch.float32)
    reference_delta_norm = float(torch.median(raw_norms).item()) if raw_norms.numel() > 0 else 0.0
    clip_radius = max(reference_delta_norm * max(config.clip_factor, 0.0), 1e-12)
    uses_focus_clipping = aggregation_method in {"focus_clip_mean", "focus_clip_trimmed_mean"}
    uses_trimmed_mean = aggregation_method in {"clip_trimmed_mean", "focus_clip_trimmed_mean"}
    focus_scores = torch.tensor(
        [_focus_score(delta, config.focus_top_k) for delta in raw_deltas],
        dtype=torch.float32,
    )
    reference_focus_score = float(torch.median(focus_scores).item()) if focus_scores.numel() > 0 else 0.0
    focus_radius = None
    if uses_focus_clipping:
        focus_radius = min(max(reference_focus_score * max(config.focus_factor, 0.0), 1e-12), 1.0)

    clipped_deltas: list[ServerState] = []
    shard_diagnostics = []
    suppressed_shards: list[int] = []
    for update, raw_delta, raw_norm, focus_score in zip(
        shard_updates,
        raw_deltas,
        raw_norms.tolist(),
        focus_scores.tolist(),
    ):
        norm_scale = 1.0
        if raw_norm > clip_radius:
            norm_scale = clip_radius / raw_norm
        focus_scale = 1.0
        if uses_focus_clipping and focus_radius is not None and focus_score > focus_radius:
            focus_scale = focus_radius / focus_score
        clip_scale = min(norm_scale, focus_scale)
        suppression_reasons = []
        if norm_scale < 1.0:
            suppression_reasons.append("norm")
        if focus_scale < 1.0:
            suppression_reasons.append("focus")
        if suppression_reasons:
            suppressed_shards.append(update.shard_id)
        clipped_delta = _scale_server_state(raw_delta, clip_scale)
        clipped_deltas.append(clipped_delta)
        shard_diagnostics.append(
            {
                "shard_id": update.shard_id,
                "weight": update.weight,
                "delta_norm": float(raw_norm),
                "focus_score": float(focus_score),
                "norm_scale": float(norm_scale),
                "focus_scale": float(focus_scale),
                "clip_scale": float(clip_scale),
                "suppression_reasons": suppression_reasons,
            }
        )

    if not uses_trimmed_mean:
        aggregated_delta = _mean_server_state(clipped_deltas)
        aggregated_state = _add_server_state(base_server_state, aggregated_delta)
        trim_count = 0
    else:
        aggregated_delta = _trimmed_mean_server_state(clipped_deltas, config.trim_ratio)
        aggregated_state = _add_server_state(base_server_state, aggregated_delta)
        trim_count = int(math.floor(len(clipped_deltas) * max(config.trim_ratio, 0.0)))

    diagnostics = {
        "method": aggregation_method,
        "reference_delta_norm": reference_delta_norm,
        "clip_radius": clip_radius,
        "reference_focus_score": reference_focus_score if uses_focus_clipping else None,
        "focus_radius": focus_radius if uses_focus_clipping else None,
        "focus_top_k": config.focus_top_k if uses_focus_clipping else None,
        "trim_ratio": config.trim_ratio if uses_trimmed_mean else 0.0,
        "trim_count": trim_count if uses_trimmed_mean else 0,
        "weighting": "uniform",
        "suppressed_shards": suppressed_shards,
        "shards": shard_diagnostics,
    }
    return aggregated_state, diagnostics
