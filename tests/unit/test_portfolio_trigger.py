"""Unit tests for the autonomous rebalance trigger in PortfolioOptimizerService.

Covers the prediction -> optimisation -> RebalanceRequestEvent wire: correct
stream subscriptions, the qualification gates (edge margin / direction /
freshness), the rebalance cooldown, explicit weight-0.0 exits, and the
reference-price requirements the risk manager depends on for sizing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from config.settings import Settings
from core.events.base import Event, EventHandler, InProcessEventBus
from core.events.market_events import PriceUpdateEvent
from core.events.risk_events import RebalanceRequestEvent
from core.events.signal_events import PredictionReadyEvent
from core.events.streams import PREDICTIONS_READY, REBALANCE
from services.portfolio import service as portfolio_service_module
from services.portfolio.service import PortfolioOptimizerService


class RecordingBus(InProcessEventBus):
    """In-process bus that also records subscribe() calls for assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.subscriptions: list[tuple[str, str]] = []  # (stream, group)

    async def subscribe(
        self,
        stream: str,
        group: str,
        consumer: str,
        handler: EventHandler,
        **_kwargs: Any,
    ) -> None:
        self.subscriptions.append((stream, group))
        await super().subscribe(stream, group, consumer, handler)


class FakeClock:
    """Controllable stand-in for the service module's ``time`` import."""

    def __init__(self, start: float = 10_000.0) -> None:
        self.now = start

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(portfolio_service_module, "time", fake)
    return fake


@pytest.fixture
def bus() -> RecordingBus:
    return RecordingBus()


@pytest.fixture
def settings() -> Settings:
    # Wide per-position cap so one- and two-symbol targets keep material
    # weights after constraint clipping.
    return Settings(min_edge_margin=0.10, max_position_pct=0.5)


@pytest.fixture
async def service(
    bus: RecordingBus, settings: Settings, clock: FakeClock
) -> PortfolioOptimizerService:
    svc = PortfolioOptimizerService(event_bus=bus, settings=settings)
    await svc.start()
    return svc


# Default probability maps per direction, each with a decisive edge margin
# (|p(long) - p(short)| = 0.23) so direction/freshness/price tests are not
# entangled with the margin gate. Margin-specific tests pass explicit maps.
_DEFAULT_PROBABILITIES = {
    "long": {"long": 0.45, "flat": 0.33, "short": 0.22},
    "short": {"long": 0.22, "flat": 0.33, "short": 0.45},
    "flat": {"long": 0.25, "flat": 0.50, "short": 0.25},
}


def _prediction(
    symbol: str,
    direction: str = "long",
    confidence: float = 0.9,
    expected_return: float | None = 0.05,
    probabilities: dict[str, float] | None = None,
) -> PredictionReadyEvent:
    if probabilities is None:
        probabilities = _DEFAULT_PROBABILITIES[direction]
    return PredictionReadyEvent(
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        expected_return=expected_return,
        model_id="model-1",
        probabilities=probabilities,
    )


def _price(symbol: str, price: float) -> PriceUpdateEvent:
    return PriceUpdateEvent(
        symbol=symbol,
        price=price,
        volume=0.0,
        market_timestamp=datetime.now(UTC),
    )


def _rebalances(bus: RecordingBus) -> list[RebalanceRequestEvent]:
    events: list[RebalanceRequestEvent] = []
    for stream, event in bus.history:
        if stream == REBALANCE:
            assert isinstance(event, RebalanceRequestEvent)
            events.append(event)
    return events


async def test_subscribes_to_predictions_ready_and_market_prices(
    service: PortfolioOptimizerService, bus: RecordingBus
) -> None:
    """The optimizer must consume the stream predictions are published on."""
    assert PREDICTIONS_READY == "predictions.ready"
    assert ("predictions.ready", "portfolio_optimizer") in bus.subscriptions
    assert ("market.prices", "portfolio_optimizer") in bus.subscriptions
    # The legacy "predictions" stream never carried traffic; nothing should
    # subscribe to it here.
    assert all(stream != "predictions" for stream, _ in bus.subscriptions)


async def test_thin_edge_does_not_trigger(
    service: PortfolioOptimizerService, bus: RecordingBus
) -> None:
    """A long argmax whose p(long) - p(short) sits below min_edge_margin is
    prior-shaped noise, not signal, and must not trade."""
    await bus.publish("market.prices", _price("AAPL", 100.0))
    await bus.publish(
        PREDICTIONS_READY,
        _prediction(
            "AAPL",
            confidence=0.36,
            probabilities={"long": 0.36, "flat": 0.35, "short": 0.29},
        ),
    )
    assert _rebalances(bus) == []


async def test_high_confidence_without_edge_does_not_trigger(
    service: PortfolioOptimizerService, bus: RecordingBus
) -> None:
    """Absolute confidence is not the gate: even a high max-class probability
    fails to qualify when the long-short margin is thin."""
    await bus.publish("market.prices", _price("AAPL", 100.0))
    await bus.publish(
        PREDICTIONS_READY,
        _prediction(
            "AAPL",
            confidence=0.48,
            probabilities={"long": 0.48, "flat": 0.08, "short": 0.44},
        ),
    )
    assert _rebalances(bus) == []


async def test_edge_at_margin_triggers(
    service: PortfolioOptimizerService, bus: RecordingBus
) -> None:
    """p(long) - p(short) exactly at min_edge_margin qualifies (>= gate)."""
    await bus.publish("market.prices", _price("AAPL", 100.0))
    await bus.publish(
        PREDICTIONS_READY,
        _prediction(
            "AAPL",
            confidence=0.40,
            probabilities={"long": 0.40, "flat": 0.30, "short": 0.30},
        ),
    )
    assert len(_rebalances(bus)) == 1


async def test_missing_probabilities_fail_closed(
    service: PortfolioOptimizerService, bus: RecordingBus
) -> None:
    """An event without a probability map (pre-upgrade publisher) cannot
    demonstrate edge and must not trade, whatever its confidence claims."""
    await bus.publish("market.prices", _price("AAPL", 100.0))
    await bus.publish(
        PREDICTIONS_READY, _prediction("AAPL", confidence=0.99, probabilities={})
    )
    assert _rebalances(bus) == []


async def test_short_and_flat_do_not_trigger(
    service: PortfolioOptimizerService, bus: RecordingBus
) -> None:
    """Long-only: short/flat predictions never qualify, however confident."""
    await bus.publish("market.prices", _price("AAPL", 100.0))
    await bus.publish("market.prices", _price("MSFT", 50.0))
    await bus.publish(PREDICTIONS_READY, _prediction("AAPL", direction="short", confidence=0.99))
    await bus.publish(PREDICTIONS_READY, _prediction("MSFT", direction="flat", confidence=0.99))
    assert _rebalances(bus) == []


async def test_stale_prediction_does_not_trigger(
    service: PortfolioOptimizerService, bus: RecordingBus, clock: FakeClock
) -> None:
    """A qualifying prediction older than the freshness window is ignored."""
    # Buffered while unpriced, so it cannot fire on arrival.
    await bus.publish(PREDICTIONS_READY, _prediction("AAPL", confidence=0.9))
    assert _rebalances(bus) == []

    # By the time a price exists and the gate re-runs, AAPL has gone stale.
    clock.advance(200.0)
    await bus.publish("market.prices", _price("AAPL", 100.0))
    await bus.publish(PREDICTIONS_READY, _prediction("MSFT", direction="short", confidence=0.99))
    assert _rebalances(bus) == []


async def test_qualifying_predictions_publish_rebalance_with_prices(
    service: PortfolioOptimizerService, bus: RecordingBus
) -> None:
    # Buffer both symbols before any price is known (no publish yet), then
    # deliver prices and a refreshed prediction so one cycle sees both.
    await bus.publish(PREDICTIONS_READY, _prediction("AAPL", confidence=0.9))
    await bus.publish(PREDICTIONS_READY, _prediction("MSFT", confidence=0.8))
    assert _rebalances(bus) == []

    await bus.publish("market.prices", _price("AAPL", 100.0))
    await bus.publish("market.prices", _price("MSFT", 50.0))
    await bus.publish(PREDICTIONS_READY, _prediction("AAPL", confidence=0.9))

    events = _rebalances(bus)
    assert len(events) == 1
    event = events[0]
    assert event.source_service == "portfolio_optimizer"
    assert set(event.target_allocations) == {"AAPL", "MSFT"}
    # Normalised long-only weights within the per-position cap.
    for weight in event.target_allocations.values():
        assert 0.0 < weight <= 0.5 + 1e-6
    assert sum(event.target_allocations.values()) <= 1.0 + 1e-6
    # Risk cannot size a symbol without a positive reference price.
    assert event.reference_prices == {"AAPL": 100.0, "MSFT": 50.0}


async def test_cooldown_suppresses_second_trigger(
    service: PortfolioOptimizerService, bus: RecordingBus, clock: FakeClock
) -> None:
    await bus.publish("market.prices", _price("AAPL", 100.0))
    await bus.publish(PREDICTIONS_READY, _prediction("AAPL", confidence=0.9))
    assert len(_rebalances(bus)) == 1

    # Fresh qualifying prediction inside the cooldown window: no publish.
    clock.advance(100.0)
    await bus.publish(PREDICTIONS_READY, _prediction("AAPL", confidence=0.9))
    assert len(_rebalances(bus)) == 1

    # Once the cooldown elapses the next qualifying prediction fires again.
    clock.advance(250.0)
    await bus.publish(PREDICTIONS_READY, _prediction("AAPL", confidence=0.9))
    assert len(_rebalances(bus)) == 2


async def test_dropped_symbol_exits_with_zero_weight(
    service: PortfolioOptimizerService, bus: RecordingBus, clock: FakeClock
) -> None:
    """A previously held symbol absent from the new target must be published
    as an explicit weight-0.0 allocation with a reference price -- risk never
    touches symbols missing from target_allocations."""
    await bus.publish(PREDICTIONS_READY, _prediction("AAPL", confidence=0.9))
    await bus.publish(PREDICTIONS_READY, _prediction("MSFT", confidence=0.8))
    await bus.publish("market.prices", _price("AAPL", 100.0))
    await bus.publish("market.prices", _price("MSFT", 50.0))
    await bus.publish(PREDICTIONS_READY, _prediction("AAPL", confidence=0.9))

    first = _rebalances(bus)
    assert len(first) == 1
    assert set(first[0].target_allocations) == {"AAPL", "MSFT"}

    # Past the cooldown, both buffered predictions have gone stale; only AAPL
    # gets a fresh signal, so MSFT drops out of the target and must exit.
    clock.advance(400.0)
    await bus.publish(PREDICTIONS_READY, _prediction("AAPL", confidence=0.9))

    events = _rebalances(bus)
    assert len(events) == 2
    second = events[1]
    assert set(second.target_allocations) == {"AAPL", "MSFT"}
    assert second.target_allocations["MSFT"] == 0.0
    assert second.target_allocations["AAPL"] > 0.0
    # Exits need a reference price too, or risk skips sizing the sell.
    assert second.reference_prices["MSFT"] == 50.0
    # Held book now tracks only the non-zero target.
    assert service._last_target_symbols == {"AAPL"}


async def test_unpriced_target_symbol_is_omitted(
    service: PortfolioOptimizerService, bus: RecordingBus
) -> None:
    """Qualifying symbols with no known price are dropped from the payload;
    the priced remainder still publishes."""
    await bus.publish(PREDICTIONS_READY, _prediction("MSFT", confidence=0.8))
    await bus.publish("market.prices", _price("AAPL", 100.0))
    await bus.publish(PREDICTIONS_READY, _prediction("AAPL", confidence=0.9))

    events = _rebalances(bus)
    assert len(events) == 1
    assert set(events[0].target_allocations) == {"AAPL"}
    assert "MSFT" not in events[0].reference_prices


async def test_no_publish_when_no_prices_known(
    service: PortfolioOptimizerService, bus: RecordingBus
) -> None:
    """Without a single reference price the cycle is skipped entirely --
    an all-unpriced rebalance would be a guaranteed downstream no-op."""
    await bus.publish(PREDICTIONS_READY, _prediction("AAPL", confidence=0.95))
    await bus.publish(PREDICTIONS_READY, _prediction("MSFT", confidence=0.95))
    assert _rebalances(bus) == []


async def test_non_positive_prices_are_ignored(
    service: PortfolioOptimizerService, bus: RecordingBus
) -> None:
    """A zero/negative tick must not become a usable reference price."""
    await bus.publish("market.prices", _price("AAPL", 0.0))
    await bus.publish(PREDICTIONS_READY, _prediction("AAPL", confidence=0.9))
    assert _rebalances(bus) == []


async def test_pending_predictions_bounded_per_symbol(
    service: PortfolioOptimizerService, bus: RecordingBus
) -> None:
    """The buffer is keyed by symbol: repeated predictions replace, not grow."""
    for confidence in (0.1, 0.2, 0.3, 0.4, 0.5):
        await bus.publish(PREDICTIONS_READY, _prediction("AAPL", confidence=confidence))
    assert len(service._pending_predictions) == 1
    assert service._pending_predictions["AAPL"].confidence == 0.5


async def test_price_handler_ignores_foreign_events(
    service: PortfolioOptimizerService, bus: RecordingBus
) -> None:
    """Non-PriceUpdateEvent traffic on market.prices must not poison the
    price cache."""
    await bus.publish("market.prices", Event(payload={"symbol": "AAPL", "price": 123.0}))
    await bus.publish(PREDICTIONS_READY, _prediction("AAPL", confidence=0.9))
    assert _rebalances(bus) == []
