"""Exploration paper-trading + real round-trip P&L attribution.

The learning period must exercise the trade path itself: when nothing clears
the full gate, the single best sub-gate signal opens a small, tagged,
auto-expiring EXPLORATION position (paper mode only), and closed round trips
feed REAL realised P&L into the feedback loop -- the "learn when to place and
when to exit" half of the loop. These tests pin: the paper-only hard gate, the
concurrent-position cap, hold-expiry exits, graduation to conviction, and
FIFO lot matching with commission-aware P&L.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from config.settings import Settings
from core.events.base import Event, InProcessEventBus
from core.events.market_events import PriceUpdateEvent
from core.events.order_events import OrderFilledEvent
from core.events.risk_events import RebalanceRequestEvent
from core.events.signal_events import PredictionReadyEvent
from core.events.streams import PREDICTIONS_READY, REBALANCE
from services.continuous_learning.evaluator import ModelEvaluator
from services.continuous_learning.feedback_loop import TradingFeedbackLoop
from services.continuous_learning.service import ContinuousLearningService
from services.portfolio.service import PortfolioOptimizerService

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Portfolio-side exploration
# ---------------------------------------------------------------------------


class RecordingBus(InProcessEventBus):
    def __init__(self) -> None:
        super().__init__()
        self.published: list[tuple[str, Event]] = []
        self.subscriptions: list[tuple[str, Any]] = []

    async def publish(self, stream: str, event: Event) -> str:
        self.published.append((stream, event))
        return await super().publish(stream, event)

    async def subscribe(self, **kwargs: Any) -> None:  # type: ignore[override]
        self.subscriptions.append((kwargs.get("stream", ""), kwargs.get("handler")))
        await super().subscribe(**kwargs)


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = dict(
        trading_mode="paper",
        jwt_secret="test-secret-not-the-published-default-value",
        exploration_hold_minutes=15,
    )
    values.update(overrides)
    return Settings(**values)


def _prediction(symbol: str, margin: float = 0.05) -> PredictionReadyEvent:
    p_long = 0.34 + margin / 2
    p_short = 0.34 - margin / 2
    return PredictionReadyEvent(
        symbol=symbol,
        direction="long",
        confidence=p_long,
        expected_return=0.001,
        model_id="ensemble:test",
        probabilities={"long": p_long, "short": p_short, "flat": 1 - p_long - p_short},
        source_service="test",
    )


def _price(symbol: str, price: float = 100.0) -> PriceUpdateEvent:
    return PriceUpdateEvent(
        symbol=symbol, price=price, volume=0.0, market_timestamp=datetime.now(UTC)
    )


def _rebalances(bus: RecordingBus) -> list[RebalanceRequestEvent]:
    return [e for s, e in bus.published if s == REBALANCE]  # type: ignore[misc]


async def _make_service(settings: Settings) -> tuple[PortfolioOptimizerService, RecordingBus]:
    bus = RecordingBus()
    svc = PortfolioOptimizerService(event_bus=bus, settings=settings)
    await svc.start()
    return svc, bus


async def test_flat_labelled_lean_qualifies_for_exploration() -> None:
    """The serving pipeline flattens sub-gate directions (abstention), so
    exploration must read the probability margin, not the label -- otherwise
    the signals it exists to learn from are invisible to it."""
    svc, bus = await _make_service(_settings())
    await bus.publish("market.prices", _price("DOT/USDT"))
    await bus.publish(
        PREDICTIONS_READY,
        PredictionReadyEvent(
            symbol="DOT/USDT",
            direction="flat",  # abstained label...
            confidence=0.55,
            expected_return=0.0005,
            model_id="ensemble:test",
            # ...over a clear long lean above the exploration floor.
            probabilities={"long": 0.30, "short": 0.215, "flat": 0.485},
            source_service="test",
        ),
    )
    events = _rebalances(bus)
    assert len(events) == 1
    assert events[0].exploration_symbols == ["DOT/USDT"]
    assert events[0].target_allocations == {
        "DOT/USDT": pytest.approx(svc.settings.exploration_weight)
    }


async def test_exploration_disabled_in_live_mode() -> None:
    svc, bus = await _make_service(
        _settings(trading_mode="live", jwt_secret="x" * 48)
    )
    await bus.publish("market.prices", _price("BTC/USDT"))
    await bus.publish(PREDICTIONS_READY, _prediction("BTC/USDT", margin=0.05))
    assert _rebalances(bus) == []


async def test_exploration_picks_single_best_and_caps_concurrency() -> None:
    svc, bus = await _make_service(_settings())
    # Steady state: the buffer holds fresh predictions for ALL symbols (they
    # arrive in per-minute batches, far more often than the 5-min cooldown).
    # Model it by blocking the cooldown while the batch lands, then releasing.
    svc._last_rebalance_at = time.monotonic()
    for sym, margin in (("A/USDT", 0.04), ("B/USDT", 0.06), ("C/USDT", 0.05)):
        await bus.publish("market.prices", _price(sym))
        await bus.publish(PREDICTIONS_READY, _prediction(sym, margin=margin))
    assert _rebalances(bus) == []  # cooldown held

    svc._last_rebalance_at = time.monotonic() - 1000.0
    await bus.publish(PREDICTIONS_READY, _prediction("A/USDT", margin=0.04))
    events = _rebalances(bus)
    assert len(events) == 1  # one entry per cycle: the BEST margin of the batch
    assert list(events[0].target_allocations) == ["B/USDT"]
    assert events[0].exploration_symbols == ["B/USDT"]

    # Force a second cycle past the cooldown: next-best enters; then the cap
    # (2 concurrent) blocks a third even after another cooldown.
    svc._last_rebalance_at = time.monotonic() - 1000.0
    await bus.publish(PREDICTIONS_READY, _prediction("C/USDT", margin=0.05))
    svc._last_rebalance_at = time.monotonic() - 1000.0
    await bus.publish(PREDICTIONS_READY, _prediction("A/USDT", margin=0.04))
    events = _rebalances(bus)
    assert len(events) == 2
    assert list(events[1].target_allocations) == ["C/USDT"]
    assert set(svc._exploration_positions) == {"B/USDT", "C/USDT"}


async def test_exploration_hold_expiry_publishes_exit() -> None:
    svc, bus = await _make_service(_settings())
    await bus.publish("market.prices", _price("A/USDT"))
    await bus.publish(PREDICTIONS_READY, _prediction("A/USDT", margin=0.05))
    assert set(svc._exploration_positions) == {"A/USDT"}

    # Age the position past the hold window and past the cooldown, then let
    # any prediction event drive the cycle (no new candidate qualifies).
    svc._exploration_positions["A/USDT"] = time.monotonic() - 16 * 60
    svc._last_rebalance_at = time.monotonic() - 1000.0
    await bus.publish(PREDICTIONS_READY, _prediction("A/USDT", margin=0.0))

    events = _rebalances(bus)
    exit_event = events[-1]
    assert exit_event.target_allocations == {"A/USDT": 0.0}
    assert exit_event.exploration_symbols == ["A/USDT"]
    assert svc._exploration_positions == {}


async def test_conviction_cycle_graduates_exploration_position() -> None:
    svc, bus = await _make_service(_settings())
    await bus.publish("market.prices", _price("A/USDT"))
    await bus.publish(PREDICTIONS_READY, _prediction("A/USDT", margin=0.05))
    assert set(svc._exploration_positions) == {"A/USDT"}

    svc._last_rebalance_at = time.monotonic() - 1000.0
    await bus.publish(PREDICTIONS_READY, _prediction("A/USDT", margin=0.30))
    events = _rebalances(bus)
    final = events[-1]
    assert final.exploration_symbols == []  # conviction, not exploration
    assert final.target_allocations["A/USDT"] > svc.settings.exploration_weight
    assert svc._exploration_positions == {}  # graduated


# ---------------------------------------------------------------------------
# CL-side round-trip P&L attribution
# ---------------------------------------------------------------------------


def _fill(
    symbol: str, side: str, price: float, qty: float,
    commission: float = 0.0, client_order_id: str = "",
) -> OrderFilledEvent:
    return OrderFilledEvent(
        order_id="x",
        symbol=symbol,
        side=side,
        fill_price=price,
        fill_quantity=qty,
        commission=commission,
        client_order_id=client_order_id,
        source_service="test",
    )


def _cl_service() -> tuple[ContinuousLearningService, MagicMock]:
    feedback = MagicMock(spec=TradingFeedbackLoop)
    svc = ContinuousLearningService(
        event_bus=InProcessEventBus(),  # type: ignore[arg-type]
        settings=_settings(),
        evaluator=MagicMock(spec=ModelEvaluator),
        retrainer=MagicMock(),
        drift_detector=MagicMock(),
        feedback_loop=feedback,
    )
    return svc, feedback


async def test_round_trip_pnl_feeds_feedback_loop() -> None:
    svc, feedback = _cl_service()
    svc._latest_prediction_by_symbol["BTC/USDT"] = "pred-1"

    await svc.handle_order_filled(
        _fill("BTC/USDT", "buy", 100.0, 2.0, commission=0.2, client_order_id="explore-e1-BTC/USDT")
    )
    await svc.handle_order_filled(
        _fill("BTC/USDT", "sell", 110.0, 2.0, commission=0.22)
    )

    feedback.record_outcome.assert_called_once()
    args, kwargs = feedback.record_outcome.call_args
    assert args[0] == "pred-1"
    assert args[1] == pytest.approx(0.10)  # realised return
    # P&L = (110-100)*2 - 0.22 - 0.2 = 19.58, commission-aware
    assert kwargs["trade_pnl"] == pytest.approx(19.58)
    assert svc._open_lots["BTC/USDT"] == []


async def test_partial_fifo_close_and_unmatched_sell() -> None:
    svc, feedback = _cl_service()
    svc._latest_prediction_by_symbol["ETH/USDT"] = "pred-a"
    await svc.handle_order_filled(_fill("ETH/USDT", "buy", 100.0, 1.0))
    svc._latest_prediction_by_symbol["ETH/USDT"] = "pred-b"
    await svc.handle_order_filled(_fill("ETH/USDT", "buy", 105.0, 1.0))

    # Sell 1.5: closes all of lot A (pred-a) and half of lot B (pred-b), FIFO.
    await svc.handle_order_filled(_fill("ETH/USDT", "sell", 110.0, 1.5))
    assert feedback.record_outcome.call_count == 2
    first, second = feedback.record_outcome.call_args_list
    assert first.args[0] == "pred-a" and first.args[1] == pytest.approx(0.10)
    assert second.args[0] == "pred-b" and second.args[1] == pytest.approx(110 / 105 - 1)
    assert svc._open_lots["ETH/USDT"][0]["quantity"] == pytest.approx(0.5)

    # A sell with no tracked entry fabricates nothing.
    feedback.reset_mock()
    await svc.handle_order_filled(_fill("SOL/USDT", "sell", 50.0, 1.0))
    feedback.record_outcome.assert_not_called()
