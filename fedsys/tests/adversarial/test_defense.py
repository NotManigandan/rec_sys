"""
tests/adversarial/test_defense.py
===================================
Unit tests for fedsys.adversarial.defense.aggregation

All tests run on CPU without any gRPC or coordinator process.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import io
from typing import Dict

import torch

from fedsys.adversarial.defense.aggregation import (
    RobustAggregator,
    DEFENSE_METHODS,
    _l2_norm,
    _focus_score,
    _trimmed_mean_tensor,
    _delta,
)
from fedsys.coordinator.aggregator import serialize_state_dict, deserialize_state_dict
from fedsys.logging_async import AsyncTelemetryLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_logger() -> AsyncTelemetryLogger:
    return AsyncTelemetryLogger(
        log_file=os.devnull,
        db_path=":memory:",
    ).start()


def _make_state(
    num_users: int = 20,
    num_items: int = 10,
    dim: int = 4,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {
        "user_embedding.weight": torch.randn(num_users, dim),
        "item_embedding.weight": torch.randn(num_items, dim),
        "user_bias.weight":      torch.randn(num_users, 1),
        "item_bias.weight":      torch.randn(num_items, 1),
    }


def _serialize(state):
    payload, _ = serialize_state_dict(state)
    return payload


# ---------------------------------------------------------------------------
# Tests for helpers
# ---------------------------------------------------------------------------

def test_l2_norm():
    d = {"w": torch.ones(10)}
    import math
    assert abs(_l2_norm(d) - math.sqrt(10)) < 1e-5


def test_delta():
    a = {"w": torch.tensor([3.0, 4.0])}
    b = {"w": torch.tensor([1.0, 1.0])}
    d = _delta(a, b)
    assert torch.allclose(d["w"], torch.tensor([2.0, 3.0]))


def test_trimmed_mean_tensor_basic():
    tensors = [torch.tensor([1.0, 10.0]), torch.tensor([2.0, 9.0]),
               torch.tensor([100.0, 8.0]), torch.tensor([3.0, 7.0])]
    result = _trimmed_mean_tensor(tensors, trim_fraction=0.25)
    # After trimming 1 from each side of 4, keep middle 2
    # Sorted col 0: [1, 2, 3, 100] -> trim [1] and [100] -> mean([2, 3]) = 2.5
    assert abs(result[0].item() - 2.5) < 0.01


def test_focus_score_zero_when_key_missing():
    d = {"other.weight": torch.randn(10, 4)}
    score = _focus_score(d, torch.randn(10, 4))
    assert score == 0.0


def test_focus_score_high_for_concentrated_update():
    # Concentrate update on just 1 item out of 100
    emb_delta = torch.zeros(100, 8)
    emb_delta[0] = torch.ones(8) * 10  # only item 0 has big delta
    d = {"item_embedding.weight": emb_delta}
    score = _focus_score(d, torch.zeros(100, 8), k_frac=0.05)
    # Top-5 items account for item 0; score should be very high
    assert score > 0.5


# ---------------------------------------------------------------------------
# Tests for RobustAggregator
# ---------------------------------------------------------------------------

def test_all_defense_methods_registered():
    assert "none" in DEFENSE_METHODS
    assert "clip_mean" in DEFENSE_METHODS
    assert "focus_clip_trimmed_mean" in DEFENSE_METHODS


def test_invalid_method_raises():
    logger = _dummy_logger()
    try:
        RobustAggregator(logger, method="bad_method")
        assert False, "Should have raised"
    except ValueError:
        pass
    finally:
        logger.stop()


def _run_aggregator_method(method: str):
    logger = _dummy_logger()
    agg = RobustAggregator(logger, method=method)

    global_state = _make_state(seed=99)
    updates = {
        "node_a": _serialize(_make_state(seed=1)),
        "node_b": _serialize(_make_state(seed=2)),
    }
    sample_counts = {"node_a": 100, "node_b": 120}

    new_state, serialized = agg.aggregate(
        updates, sample_counts, epoch=0,
        global_state=global_state,
    )
    logger.stop()
    assert isinstance(new_state, dict)
    assert "user_embedding.weight" in new_state
    assert isinstance(serialized, bytes)
    # Check shapes preserved
    for k, v in global_state.items():
        assert new_state[k].shape == v.shape


def test_none_method():        _run_aggregator_method("none")
def test_clip_mean():          _run_aggregator_method("clip_mean")
def test_clip_trimmed_mean():  _run_aggregator_method("clip_trimmed_mean")
def test_focus_clip_mean():    _run_aggregator_method("focus_clip_mean")
def test_focus_clip_trimmed(): _run_aggregator_method("focus_clip_trimmed_mean")


def test_clipping_reduces_large_update():
    """A very large update should be clipped to be within the threshold."""
    logger = _dummy_logger()
    agg = RobustAggregator(logger, method="clip_mean", clip_threshold=1.0)

    global_state = _make_state(seed=0)
    # Create a huge update (scale global by 1000)
    big_state = {k: v * 1000 for k, v in global_state.items()}
    normal_state = _make_state(seed=1)

    updates = {
        "malicious": _serialize(big_state),
        "clean":     _serialize(normal_state),
    }
    new_state, _ = agg.aggregate(
        updates, {"malicious": 100, "clean": 100},
        epoch=0, global_state=global_state,
    )
    logger.stop()

    # Item embedding delta for malicious should have been clipped
    delta_mal = new_state["item_embedding.weight"] - global_state["item_embedding.weight"]
    assert _l2_norm({"w": delta_mal}) < 10.0, "Clipping should prevent huge deltas"


if __name__ == "__main__":
    test_l2_norm()
    test_delta()
    test_trimmed_mean_tensor_basic()
    test_focus_score_zero_when_key_missing()
    test_focus_score_high_for_concentrated_update()
    test_all_defense_methods_registered()
    test_invalid_method_raises()
    test_none_method()
    test_clip_mean()
    test_clip_trimmed_mean()
    test_focus_clip_mean()
    test_focus_clip_trimmed()
    test_clipping_reduces_large_update()
    print("All defense tests passed.")
