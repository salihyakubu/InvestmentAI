"""In-memory prediction/outcome stores must stay bounded over a long soak.

Covers the retention caps added to ModelEvaluator, TradingFeedbackLoop, and
PredictionService: unbounded growth (~10-17k predictions/day) would OOM the
container mid-soak.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

import services.continuous_learning.evaluator as evaluator_module
import services.continuous_learning.feedback_loop as feedback_module
from core.events.base import InProcessEventBus
from services.continuous_learning.evaluator import ModelEvaluator
from services.continuous_learning.feedback_loop import TradingFeedbackLoop
from services.prediction.service import PredictionService

# ----------------------------------------------------------------------
# ModelEvaluator
# ----------------------------------------------------------------------


def test_evaluator_cap_evicts_predictions_and_outcomes_in_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluator_module, "_MAX_PREDICTIONS_PER_MODEL", 5)
    ev = ModelEvaluator()

    for i in range(7):
        ev.record_prediction(f"p{i}", "model-a", "AAPL", "long", 0.8)
        ev.record_outcome(f"p{i}", "long", 0.01)

    retained = [p["prediction_id"] for p in ev._predictions["model-a"]]
    assert retained == ["p2", "p3", "p4", "p5", "p6"]
    # Evicted predictions take their outcomes with them; retained ones keep theirs.
    assert set(ev._outcomes) == set(retained)


def test_evaluator_cap_is_per_model_and_tolerates_missing_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluator_module, "_MAX_PREDICTIONS_PER_MODEL", 2)
    ev = ModelEvaluator()

    # No outcome ever recorded for the evicted ids -> eviction must not raise.
    for i in range(4):
        ev.record_prediction(f"a{i}", "model-a", "AAPL", "long", 0.8)
    ev.record_prediction("b0", "model-b", "MSFT", "short", 0.7)
    ev.record_outcome("b0", "short", -0.01)

    assert len(ev._predictions["model-a"]) == 2
    assert len(ev._predictions["model-b"]) == 1
    # Other models' outcomes are untouched by model-a evictions.
    assert "b0" in ev._outcomes


def test_evaluator_flat_deadband_matches_outcome_resolver() -> None:
    # Resolver labels |actual_return| <= 0.0005 as flat; the evaluator must
    # score flat predictions against the same boundary (not the old 0.001).
    ev = ModelEvaluator()
    ev.record_prediction("p-in", "model-a", "AAPL", "flat", 0.9)
    ev.record_outcome("p-in", "flat", 0.0005)  # on the boundary -> match
    assert ev.evaluate_live_performance("model-a").directional_accuracy == 1.0

    ev2 = ModelEvaluator()
    ev2.record_prediction("p-out", "model-a", "AAPL", "flat", 0.9)
    ev2.record_outcome("p-out", "flat", 0.0008)  # inside old 0.001 band -> no match
    assert ev2.evaluate_live_performance("model-a").directional_accuracy == 0.0


# ----------------------------------------------------------------------
# TradingFeedbackLoop
# ----------------------------------------------------------------------


def test_feedback_metrics_pass_prunes_stale_and_orphaned_outcomes() -> None:
    loop = TradingFeedbackLoop()

    loop.register_prediction("fresh", "model-a")
    loop.register_prediction("stale", "model-a")
    loop.record_outcome("fresh", actual_return=0.02, trade_pnl=10.0)
    loop.record_outcome("stale", actual_return=-0.05, trade_pnl=-50.0)
    # Orphaned entry (e.g. keyed by order_id, never registered to a model).
    loop.record_outcome("orphan-order-id", actual_return=0.0, trade_pnl=0.0)

    old = datetime.now(UTC) - timedelta(days=31)
    loop._outcomes["stale"]["recorded_at"] = old
    loop._outcomes["orphan-order-id"]["recorded_at"] = old

    metrics = loop.compute_model_metrics("model-a")

    assert set(loop._outcomes) == {"fresh"}
    assert metrics["sample_size"] == 1
    assert metrics["total_pnl"] == 10.0


def test_feedback_model_predictions_deque_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feedback_module, "_MAX_PREDICTIONS_PER_MODEL", 3)
    loop = TradingFeedbackLoop()

    for i in range(5):
        loop.register_prediction(f"p{i}", "model-a")

    ids = loop._model_predictions["model-a"]
    assert ids.maxlen == 3
    assert list(ids) == ["p2", "p3", "p4"]


# ----------------------------------------------------------------------
# PredictionService
# ----------------------------------------------------------------------


def test_prediction_history_bounded_and_recent_unchanged() -> None:
    # No model server -> every predict() returns a flat record immediately.
    svc = PredictionService(event_bus=InProcessEventBus())

    for i in range(1100):
        svc.predict(f"S{i}", {"a": 1.0})

    assert svc._prediction_history.maxlen == 1000
    assert len(svc._prediction_history) == 1000

    recent = svc.recent_predictions
    assert len(recent) == 100
    assert recent[0].symbol == "S1000"
    assert recent[-1].symbol == "S1099"
