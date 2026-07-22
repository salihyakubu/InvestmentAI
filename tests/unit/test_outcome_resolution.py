"""Continuous learning: bounded outcome resolution, tracked-record hygiene,
order-event filtering, and the on-demand retrain trigger.

The service used to fabricate actuals, score session-close predictions against
the next session's first bar (overnight gaps as 5-minute outcomes), track
predictions by consumption time, feed the feedback loop raw market returns and
order NOTIONAL as P&L, and grow ``_tracked_predictions`` without bound. These
tests pin the fixes: t0/t5 bar lookups are window-bounded on the bar grid,
unresolvable records expire, resolved history is capped per model, tracking
uses event creation time (stale backlog skipped), feedback returns are
direction-signed, fills are observed but not recorded as P&L, and the drift
check gates at 1000 resolved records over a rolling 2000-record window.

Also pins the ensemble retrain contract: ``retrain()`` reports per-member
results under ``"members"`` and the evaluation cycle publishes one
``ModelRetrainedEvent`` per promoted member (skipped members publish nothing).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

import services.continuous_learning.service as cl_service
from config.settings import Settings
from core.events.base import Event, InProcessEventBus
from core.events.order_events import (
    OrderCreatedEvent,
    OrderFilledEvent,
    TradingControlEvent,
)
from core.events.streams import CONTROL
from core.events.system_events import ModelRetrainedEvent
from core.models.base import AsyncBase
from core.models.market_data import OHLCVRecord
from services.continuous_learning.drift_detector import DriftReport
from services.continuous_learning.evaluator import ModelEvaluator
from services.continuous_learning.feedback_loop import TradingFeedbackLoop
from services.continuous_learning.service import ContinuousLearningService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings() -> Settings:
    return Settings(
        trading_mode="paper",
        jwt_secret="test-secret-not-the-published-default-value",
    )


def _make_service(
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> tuple[ContinuousLearningService, InProcessEventBus, MagicMock, MagicMock]:
    bus = InProcessEventBus()
    evaluator = MagicMock(spec=ModelEvaluator)
    feedback = MagicMock(spec=TradingFeedbackLoop)
    svc = ContinuousLearningService(
        event_bus=bus,  # type: ignore[arg-type]
        settings=_settings(),
        evaluator=evaluator,
        retrainer=MagicMock(),
        drift_detector=MagicMock(),
        feedback_loop=feedback,
        session_factory=session_factory,
    )
    return svc, bus, evaluator, feedback


def _make_db() -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def _create_ohlcv_table(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(
            AsyncBase.metadata.create_all, tables=[OHLCVRecord.__table__]
        )


def _bar(t: datetime, close: str, symbol: str = "BTC/USDT") -> OHLCVRecord:
    price = Decimal(close)
    return OHLCVRecord(
        time=t,
        symbol=symbol,
        timeframe="1m",
        asset_class="crypto",
        open=price,
        high=price,
        low=price,
        close=price,
        volume=Decimal("1"),
        source="test",
        ingested_at=datetime.now(UTC),
    )


def _tracked(
    prediction_id: str,
    symbol: str,
    age: timedelta,
    predicted: str = "long",
) -> dict[str, Any]:
    return {
        "prediction_id": prediction_id,
        "symbol": symbol,
        "predicted": predicted,
        "confidence": 0.8,
        "timestamp": datetime.now(UTC) - age,
    }


def _resolved(i: int) -> dict[str, Any]:
    record = _tracked(f"p{i}", "BTC/USDT", age=timedelta(minutes=30))
    record["actual"] = "long"
    record["actual_return"] = 0.01
    return record


# ---------------------------------------------------------------------------
# (a) handle_order_filled only observes fills -- no notional-as-P&L
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_order_filled_ignores_order_created() -> None:
    svc, _bus, _evaluator, feedback = _make_service()

    await svc.handle_order_filled(
        OrderCreatedEvent(
            order_id="o1", symbol="BTC/USDT", side="buy",
            order_type="market", quantity=1.0, source_service="test",
        )
    )
    feedback.record_outcome.assert_not_called()


@pytest.mark.asyncio
async def test_handle_order_filled_does_not_record_notional_as_pnl() -> None:
    """fill_price * quantity is notional, not P&L, and order_id never matches
    a registered prediction_id -- fills must not reach the feedback loop."""
    svc, _bus, _evaluator, feedback = _make_service()

    await svc.handle_order_filled(
        OrderFilledEvent(
            order_id="o1", fill_price=100.0, fill_quantity=2.0,
            commission=0.5, source_service="test",
        )
    )
    feedback.record_outcome.assert_not_called()


# ---------------------------------------------------------------------------
# (b) outcome resolution against real ohlcv closes (bounded windows)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_outcomes_sets_real_actual_from_ohlcv() -> None:
    engine, factory = _make_db()
    await _create_ohlcv_table(engine)
    svc, _bus, evaluator, feedback = _make_service(session_factory=factory)

    record = _tracked("pred-1", "BTC/USDT", age=timedelta(minutes=10))
    svc._tracked_predictions["model-a"] = [record]

    t0 = record["timestamp"]
    async with factory() as session:
        session.add_all(
            [_bar(t0, "100"), _bar(t0 + timedelta(minutes=5), "101")]
        )
        await session.commit()

    await svc._resolve_outcomes_once()

    assert record["actual"] == "long"
    assert record["actual_return"] == pytest.approx(0.01)
    evaluator.record_outcome.assert_called_once_with(
        "pred-1", "long", pytest.approx(0.01)
    )
    feedback.record_outcome.assert_called_once_with(
        "pred-1", pytest.approx(0.01), trade_pnl=0.0
    )
    await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("close_t5", "expected"),
    [("99", "short"), ("100.01", "flat")],
)
async def test_resolve_outcomes_direction_thresholds(
    close_t5: str, expected: str
) -> None:
    engine, factory = _make_db()
    await _create_ohlcv_table(engine)
    svc, _bus, _evaluator, _feedback = _make_service(session_factory=factory)

    record = _tracked("pred-1", "BTC/USDT", age=timedelta(minutes=10))
    svc._tracked_predictions["model-a"] = [record]

    t0 = record["timestamp"]
    async with factory() as session:
        session.add_all(
            [_bar(t0, "100"), _bar(t0 + timedelta(minutes=5), close_t5)]
        )
        await session.commit()

    await svc._resolve_outcomes_once()

    assert record["actual"] == expected
    await engine.dispose()


@pytest.mark.asyncio
async def test_resolve_outcomes_skips_missing_bars_and_immature() -> None:
    engine, factory = _make_db()
    await _create_ohlcv_table(engine)
    svc, _bus, evaluator, feedback = _make_service(session_factory=factory)

    no_bars = _tracked("pred-nobars", "ETH/USDT", age=timedelta(minutes=10))
    too_young = _tracked("pred-young", "BTC/USDT", age=timedelta(minutes=1))
    svc._tracked_predictions["model-a"] = [no_bars, too_young]

    # Bars exist only for BTC/USDT around now (useless for both records).
    async with factory() as session:
        session.add(_bar(datetime.now(UTC), "100"))
        await session.commit()

    await svc._resolve_outcomes_once()

    assert "actual" not in no_bars
    assert "actual" not in too_young
    # Neither record is an hour old yet, so both stay tracked for retry.
    assert len(svc._tracked_predictions["model-a"]) == 2
    evaluator.record_outcome.assert_not_called()
    feedback.record_outcome.assert_not_called()
    await engine.dispose()


@pytest.mark.asyncio
async def test_overnight_gap_not_scored_as_five_minute_outcome() -> None:
    """A prediction near session close must NOT be scored against the next
    session's first bar: no bar in [t0+5m, t0+8m] -> unresolved."""
    engine, factory = _make_db()
    await _create_ohlcv_table(engine)
    svc, _bus, evaluator, feedback = _make_service(session_factory=factory)

    record = _tracked("pred-eod", "AAPL", age=timedelta(minutes=40))
    svc._tracked_predictions["model-a"] = [record]

    t0 = record["timestamp"]
    async with factory() as session:
        # Last bar of the session at t0; "next session" opens 30 min later.
        session.add_all(
            [
                _bar(t0, "100", symbol="AAPL"),
                _bar(t0 + timedelta(minutes=30), "110", symbol="AAPL"),
            ]
        )
        await session.commit()

    await svc._resolve_outcomes_once()

    assert "actual" not in record
    evaluator.record_outcome.assert_not_called()
    feedback.record_outcome.assert_not_called()
    await engine.dispose()


@pytest.mark.asyncio
async def test_stale_t0_bar_outside_lookback_window_not_used() -> None:
    """The t0 bar must be within 10 minutes of the prediction; an hours-old
    close is not a valid baseline."""
    engine, factory = _make_db()
    await _create_ohlcv_table(engine)
    svc, _bus, evaluator, _feedback = _make_service(session_factory=factory)

    record = _tracked("pred-1", "BTC/USDT", age=timedelta(minutes=10))
    svc._tracked_predictions["model-a"] = [record]

    ts = record["timestamp"]
    async with factory() as session:
        session.add_all(
            [
                _bar(ts - timedelta(minutes=20), "100"),
                _bar(ts + timedelta(minutes=5), "101"),
            ]
        )
        await session.commit()

    await svc._resolve_outcomes_once()

    assert "actual" not in record
    evaluator.record_outcome.assert_not_called()
    await engine.dispose()


# ---------------------------------------------------------------------------
# (c) tracked-record hygiene: expiry and resolved-history cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unresolvable_records_expire_after_an_hour() -> None:
    engine, factory = _make_db()
    await _create_ohlcv_table(engine)
    svc, _bus, evaluator, _feedback = _make_service(session_factory=factory)

    dead = _tracked("pred-dead", "AAPL", age=timedelta(minutes=70))
    legacy = _tracked("pred-legacy", "", age=timedelta(minutes=70))
    fresh = _tracked("pred-fresh", "AAPL", age=timedelta(minutes=10))
    svc._tracked_predictions["model-a"] = [dead, legacy, fresh]

    await svc._resolve_outcomes_once()

    # Expired unresolvable records are dropped; the fresh one is retried.
    kept = [r["prediction_id"] for r in svc._tracked_predictions["model-a"]]
    assert kept == ["pred-fresh"]
    evaluator.record_outcome.assert_not_called()
    await engine.dispose()


@pytest.mark.asyncio
async def test_resolved_records_pruned_to_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, factory = _make_db()
    await _create_ohlcv_table(engine)
    svc, _bus, _evaluator, _feedback = _make_service(session_factory=factory)
    monkeypatch.setattr(cl_service, "_MAX_RESOLVED_PER_MODEL", 5)

    resolved = [_resolved(i) for i in range(8)]
    pending = _tracked("pending", "BTC/USDT", age=timedelta(minutes=1))
    svc._tracked_predictions["model-a"] = [*resolved, pending]

    await svc._resolve_outcomes_once()

    kept = [r["prediction_id"] for r in svc._tracked_predictions["model-a"]]
    # Oldest 3 resolved dropped; newest 5 kept in order; unresolved retained.
    assert kept == ["p3", "p4", "p5", "p6", "p7", "pending"]
    await engine.dispose()


# ---------------------------------------------------------------------------
# (d) event-time tracking and the stale-backlog guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_prediction_tracks_symbol() -> None:
    """Outcome resolution needs the symbol; the tracked record must keep it."""
    svc, _bus, _evaluator, _feedback = _make_service()

    await svc.handle_prediction(
        Event(
            source_service="test",
            payload={
                "model_id": "model-a",
                "symbol": "BTC/USDT",
                "direction": "long",
                "confidence": 0.7,
            },
        )
    )

    (record,) = svc._tracked_predictions["model-a"]
    assert record["symbol"] == "BTC/USDT"
    assert record["predicted"] == "long"


@pytest.mark.asyncio
async def test_handle_prediction_uses_event_creation_time() -> None:
    """Backlog replay: t0 must come from when the prediction was MADE, not
    when the consumer got around to reading it."""
    svc, _bus, evaluator, feedback = _make_service()

    created = datetime.now(UTC) - timedelta(minutes=3)
    await svc.handle_prediction(
        Event(
            source_service="test",
            timestamp=created,
            payload={
                "model_id": "model-a",
                "symbol": "BTC/USDT",
                "direction": "long",
                "confidence": 0.7,
            },
        )
    )

    (record,) = svc._tracked_predictions["model-a"]
    assert record["timestamp"] == created
    evaluator.record_prediction.assert_called_once()
    feedback.register_prediction.assert_called_once()


@pytest.mark.asyncio
async def test_handle_prediction_skips_stale_backlog_event() -> None:
    """Events older than 2x the horizon have already missed their outcome
    window -- tracking them would score them against unrelated bars."""
    svc, _bus, evaluator, feedback = _make_service()

    await svc.handle_prediction(
        Event(
            source_service="test",
            timestamp=datetime.now(UTC) - timedelta(minutes=11),
            payload={
                "model_id": "model-a",
                "symbol": "BTC/USDT",
                "direction": "long",
                "confidence": 0.7,
            },
        )
    )

    assert svc._tracked_predictions == {}
    evaluator.record_prediction.assert_not_called()
    feedback.register_prediction.assert_not_called()


# ---------------------------------------------------------------------------
# (e) direction-signed feedback returns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feedback_gets_signed_return_for_short_prediction() -> None:
    """A correct short call on a 1% fall earns +1%; the raw market return
    (-1%) still goes to the evaluator for direction accuracy."""
    engine, factory = _make_db()
    await _create_ohlcv_table(engine)
    svc, _bus, evaluator, feedback = _make_service(session_factory=factory)

    record = _tracked(
        "pred-1", "BTC/USDT", age=timedelta(minutes=10), predicted="short"
    )
    svc._tracked_predictions["model-a"] = [record]

    t0 = record["timestamp"]
    async with factory() as session:
        session.add_all(
            [_bar(t0, "100"), _bar(t0 + timedelta(minutes=5), "99")]
        )
        await session.commit()

    await svc._resolve_outcomes_once()

    assert record["actual"] == "short"
    evaluator.record_outcome.assert_called_once_with(
        "pred-1", "short", pytest.approx(-0.01)
    )
    feedback.record_outcome.assert_called_once_with(
        "pred-1", pytest.approx(0.01), trade_pnl=0.0
    )
    await engine.dispose()


@pytest.mark.asyncio
async def test_feedback_gets_zero_return_for_flat_prediction() -> None:
    engine, factory = _make_db()
    await _create_ohlcv_table(engine)
    svc, _bus, _evaluator, feedback = _make_service(session_factory=factory)

    record = _tracked(
        "pred-1", "BTC/USDT", age=timedelta(minutes=10), predicted="flat"
    )
    svc._tracked_predictions["model-a"] = [record]

    t0 = record["timestamp"]
    async with factory() as session:
        session.add_all(
            [_bar(t0, "100"), _bar(t0 + timedelta(minutes=5), "101")]
        )
        await session.commit()

    await svc._resolve_outcomes_once()

    # A flat call takes no position; its realised return is 0 either way.
    feedback.record_outcome.assert_called_once_with(
        "pred-1", 0.0, trade_pnl=0.0
    )
    await engine.dispose()


# ---------------------------------------------------------------------------
# (f) drift gate, rolling window, and drift-event model_id
# ---------------------------------------------------------------------------


def _drift_ready_service() -> tuple[ContinuousLearningService, InProcessEventBus, MagicMock]:
    svc, bus, _evaluator, _feedback = _make_service()
    detector = MagicMock()
    detector.detect_accuracy_drift.return_value = DriftReport(
        is_drifting=False, drift_score=0.0, threshold=0.05
    )
    svc._model_drift_detector = detector  # type: ignore[assignment]
    svc._retrainer.should_retrain.return_value = False  # type: ignore[attr-defined]
    return svc, bus, detector


@pytest.mark.asyncio
async def test_drift_check_gated_below_1000_resolved_records() -> None:
    svc, _bus, detector = _drift_ready_service()

    svc._tracked_predictions["model-a"] = [_resolved(i) for i in range(999)]
    await svc._run_evaluation_cycle()
    detector.detect_accuracy_drift.assert_not_called()

    svc._tracked_predictions["model-a"].append(_resolved(999))
    await svc._run_evaluation_cycle()
    detector.detect_accuracy_drift.assert_called_once()
    kwargs = detector.detect_accuracy_drift.call_args.kwargs
    assert len(kwargs["recent_predictions"]) == 500
    assert len(kwargs["older_predictions"]) == 500


@pytest.mark.asyncio
async def test_drift_check_uses_rolling_2000_record_window() -> None:
    svc, _bus, detector = _drift_ready_service()

    svc._tracked_predictions["model-a"] = [_resolved(i) for i in range(2500)]
    await svc._run_evaluation_cycle()

    # Only the newest 2000 records enter the comparison, split at midpoint.
    kwargs = detector.detect_accuracy_drift.call_args.kwargs
    assert len(kwargs["recent_predictions"]) == 1000
    assert len(kwargs["older_predictions"]) == 1000


@pytest.mark.asyncio
async def test_drift_event_carries_model_id() -> None:
    svc, bus, detector = _drift_ready_service()
    detector.detect_accuracy_drift.return_value = DriftReport(
        is_drifting=True, drift_score=0.1, threshold=0.05
    )

    svc._tracked_predictions["model-a"] = [_resolved(i) for i in range(1000)]
    await svc._run_evaluation_cycle()

    drift_events = [
        e for _stream, e in bus.history if e.event_type == "DriftDetectedEvent"
    ]
    assert len(drift_events) == 1
    assert drift_events[0].model_id == "model-a"
    svc._retrainer.mark_drift.assert_called_once_with("model-a")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# (g) on-demand retrain trigger via the CONTROL stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_control_retrain_runs_evaluation_cycle() -> None:
    svc, _bus, _evaluator, _feedback = _make_service()
    spy = AsyncMock()
    svc._run_evaluation_cycle = spy  # type: ignore[method-assign]

    await svc.handle_control(
        TradingControlEvent(action="retrain", reason="manual", source_service="test")
    )
    spy.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_control_ignores_other_actions_and_events() -> None:
    svc, _bus, _evaluator, _feedback = _make_service()
    spy = AsyncMock()
    svc._run_evaluation_cycle = spy  # type: ignore[method-assign]

    for action in ("halt", "resume", "flatten"):
        await svc.handle_control(
            TradingControlEvent(action=action, source_service="test")
        )
    # A non-control event carrying a retrain-looking payload is also ignored.
    await svc.handle_control(
        Event(source_service="test", payload={"action": "retrain"})
    )
    spy.assert_not_awaited()


@pytest.mark.asyncio
async def test_control_stream_subscription_dispatches_retrain() -> None:
    """End-to-end over the bus: start() wires handle_control to CONTROL."""
    svc, bus, _evaluator, _feedback = _make_service()
    spy = AsyncMock()
    svc._run_evaluation_cycle = spy  # type: ignore[method-assign]

    await svc.start()
    try:
        # start() registers subscriptions inside freshly created tasks; yield
        # once so the in-process bus actually has the handler before publish.
        await asyncio.sleep(0)
        await bus.publish(
            CONTROL,
            TradingControlEvent(action="retrain", source_service="test"),
        )
        spy.assert_awaited_once()
    finally:
        await svc.stop()


# ---------------------------------------------------------------------------
# (d) evaluation cycle publishes one ModelRetrainedEvent per promoted member
# ---------------------------------------------------------------------------


def _retraining_service(
    model_id: str, retrain_result: dict[str, Any]
) -> tuple[ContinuousLearningService, AsyncMock]:
    """Service primed to retrain *model_id*, with a spy on event publication."""
    svc, _bus, _evaluator, _feedback = _make_service()
    svc._tracked_predictions[model_id] = []
    svc._retrainer.should_retrain.return_value = True
    svc._retrainer.retrain = AsyncMock(return_value=retrain_result)
    publish = AsyncMock()
    svc._event_bus.publish = publish  # type: ignore[method-assign]
    return svc, publish


def _retrained_events(publish: AsyncMock) -> list[ModelRetrainedEvent]:
    return [
        call.args[1]
        for call in publish.await_args_list
        if isinstance(call.args[1], ModelRetrainedEvent)
    ]


@pytest.mark.asyncio
async def test_evaluation_cycle_publishes_event_per_promoted_member() -> None:
    """An ensemble retrain reports per-member results; every promoted member
    gets its own ModelRetrainedEvent and skipped members publish nothing."""
    svc, publish = _retraining_service(
        "ensemble:xgboost,lightgbm,lstm",
        {
            "members": [
                {
                    "new_model_id": "xgb-new",
                    "version": 3,
                    "metrics": {"val_accuracy": 0.51},
                    "model_type": "xgboost",
                },
                {
                    "skipped": True,
                    "reason": "new_model_not_better",
                    "model_type": "lightgbm",
                },
                {
                    "new_model_id": "lstm-new",
                    "version": 2,
                    "metrics": {"val_accuracy": 0.48},
                    "model_type": "lstm",
                },
            ]
        },
    )

    await svc._run_evaluation_cycle()

    events = _retrained_events(publish)
    assert [(e.model_id, e.version) for e in events] == [
        ("xgb-new", 3),
        ("lstm-new", 2),
    ]
    assert events[0].metrics == {"val_accuracy": 0.51}


@pytest.mark.asyncio
async def test_evaluation_cycle_publishes_single_model_result() -> None:
    svc, publish = _retraining_service(
        "xgboost",
        {
            "new_model_id": "xgb-new",
            "version": 5,
            "metrics": {"val_accuracy": 0.6},
            "model_type": "xgboost",
        },
    )

    await svc._run_evaluation_cycle()

    events = _retrained_events(publish)
    assert [(e.model_id, e.version) for e in events] == [("xgb-new", 5)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "retrain_result",
    [
        {"skipped": True, "reason": "no_training_data"},
        {
            "skipped": True,
            "reason": "no_member_promoted",
            "members": [
                {
                    "skipped": True,
                    "reason": "new_model_not_better",
                    "model_type": "xgboost",
                },
                {
                    "skipped": True,
                    "reason": "new_model_not_better",
                    "model_type": "lightgbm",
                },
            ],
        },
    ],
)
async def test_evaluation_cycle_skipped_retrain_publishes_nothing(
    retrain_result: dict[str, Any],
) -> None:
    svc, publish = _retraining_service("ensemble:xgboost,lightgbm", retrain_result)

    await svc._run_evaluation_cycle()

    assert _retrained_events(publish) == []
