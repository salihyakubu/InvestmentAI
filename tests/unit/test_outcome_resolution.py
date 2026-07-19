"""Continuous learning: real outcome resolution, order-event filtering, and
the on-demand retrain trigger.

The service used to fabricate actuals (defaulting ``actual`` to the predicted
value) and processed every order-lifecycle event as a fill. These tests pin
the fixes: only ``OrderFilledEvent`` reaches the feedback loop, outcomes are
resolved from real 1m ohlcv closes, and ``TradingControlEvent(retrain)``
triggers an immediate evaluation cycle.
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

from config.settings import Settings
from core.events.base import Event, InProcessEventBus
from core.events.order_events import (
    OrderCreatedEvent,
    OrderFilledEvent,
    TradingControlEvent,
)
from core.events.streams import CONTROL
from core.models.base import AsyncBase
from core.models.market_data import OHLCVRecord
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


# ---------------------------------------------------------------------------
# (a) handle_order_filled only reacts to fills
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
async def test_handle_order_filled_records_fill() -> None:
    svc, _bus, _evaluator, feedback = _make_service()

    await svc.handle_order_filled(
        OrderFilledEvent(
            order_id="o1", fill_price=100.0, fill_quantity=2.0,
            commission=0.5, source_service="test",
        )
    )
    feedback.record_outcome.assert_called_once_with(
        prediction_id="o1", actual_return=0.0, trade_pnl=199.5
    )


# ---------------------------------------------------------------------------
# (b) outcome resolution against real ohlcv closes
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
    evaluator.record_outcome.assert_not_called()
    feedback.record_outcome.assert_not_called()
    await engine.dispose()


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


# ---------------------------------------------------------------------------
# (c) on-demand retrain trigger via the CONTROL stream
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
