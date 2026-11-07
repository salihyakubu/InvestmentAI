"""Every held position has an owner and can leave the book (GO_LIVE.md 2026-11-07).

The live review found seven BLOCK verdicts sharing one root: no held position
had an exit. The held book was process memory (a restart orphaned it), no stop
was ever registered, the pre-trade rules ran on exits as well as entries (the
correlation rule rejected both), exits were sized from a dollar book divided
by a price (dust and paper shorts), and the correlation rule counted every
pair in the book regardless of the symbol proposed. These tests pin the five
registered behaviours on the in-process bus.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pytest

from config.settings import Settings
from core.events.base import Event, InProcessEventBus
from core.events.market_events import PriceUpdateEvent
from core.events.order_events import OrderFilledEvent
from core.events.risk_events import RebalanceRequestEvent
from core.events.signal_events import PredictionReadyEvent
from core.events.streams import LIQUIDATION, ORDERS, PREDICTIONS_READY, REBALANCE, RISK_APPROVED
from core.events.system_events import LiquidationTriggeredEvent
from services.execution.brokers.base import BrokerOrder
from services.execution.brokers.paper_broker import PaperBroker
from services.execution.service import ExecutionEngineService
from services.liquidation.service import LiquidationManagerService
from services.portfolio.service import PortfolioOptimizerService
from services.risk.service import RiskManagerService

pytestmark = pytest.mark.asyncio

_PRICES = "market.prices"


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = dict(
        trading_mode="paper",
        jwt_secret="test-secret-not-the-published-default-value",
        initial_capital=Decimal("100"),
        exploration_hold_minutes=15,
        stop_loss_pct=0.05,
    )
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def mock_settings() -> Settings:
    """Self-contained so the file also runs under ``--noconftest``."""
    return _settings(initial_capital=Decimal("10000"))


def _price(symbol: str, price: float) -> PriceUpdateEvent:
    return PriceUpdateEvent(symbol=symbol, price=price, volume=0.0, market_timestamp=datetime.now(UTC))


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


class _Capture:
    def __init__(self, bus: InProcessEventBus, stream: str, event_type: str | None = None) -> None:
        self.events: list[Event] = []
        self._type = event_type

        async def handler(event: Event) -> None:
            if self._type is None or event.event_type == self._type:
                self.events.append(event)

        self._handler = handler
        self._bus = bus
        self._stream = stream

    async def start(self) -> _Capture:
        await self._bus.subscribe(self._stream, "t", "t", self._handler)
        return self


# ---------------------------------------------------------------------------
# (1) Restored positions get an owner in the held book
# ---------------------------------------------------------------------------


async def test_restored_positions_get_an_owner_and_exit_on_the_hold_timer() -> None:
    bus = InProcessEventBus()
    rebalances = await _Capture(bus, REBALANCE).start()
    svc = PortfolioOptimizerService(event_bus=bus, settings=_settings())
    await svc.start()
    symbols = ["SOL/USDT", "ADA/USDT", "DOT/USDT"]

    # Seeded with an entry time already past the hold period, as a position
    # restored after a long outage would be.
    svc.seed_held_book(symbols, now=time.monotonic() - 16 * 60)

    assert svc.held_symbols == set(symbols)
    assert set(svc.exploration_positions) == set(symbols)  # count toward the cap

    # A fresh sub-gate signal in a held symbol is not a new entry...
    for symbol in symbols:
        await bus.publish(_PRICES, _price(symbol, 10.0))
    await bus.publish(PREDICTIONS_READY, _prediction("SOL/USDT"))

    # ...and the one rebalance published is the exit of all three, tagged.
    assert len(rebalances.events) == 1
    event = rebalances.events[0]
    assert isinstance(event, RebalanceRequestEvent)
    assert event.target_allocations == {s: 0.0 for s in symbols}
    assert set(event.exploration_symbols) == set(symbols)
    assert svc.held_symbols == set()
    assert svc.exploration_positions == {}


# ---------------------------------------------------------------------------
# (2) A stop is registered per fill and a breach exits the whole position
# ---------------------------------------------------------------------------


async def test_buy_fill_registers_a_stop_and_a_breach_exits_the_whole_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("services.execution.service._POLL_INTERVAL", 0.02)
    settings = _settings(initial_capital=Decimal("100000"))
    bus = InProcessEventBus()
    fills = await _Capture(bus, ORDERS, "OrderFilledEvent").start()
    liquidations = await _Capture(bus, LIQUIDATION).start()
    broker = PaperBroker(initial_cash=Decimal("100000"))
    broker.update_price("SOL/USDT", Decimal("100"))
    ex = ExecutionEngineService(event_bus=bus, settings=settings, brokers={"paper": broker})
    liq = LiquidationManagerService(event_bus=bus, settings=settings, position_provider=broker.get_positions)
    await ex.start()
    await liq.start()
    try:
        await ex.submit_order(
            symbol="SOL/USDT",
            side="buy",
            order_type="market",
            quantity=Decimal("0.0111"),
            reference_price=Decimal("100"),
        )
        assert len(fills.events) == 1
        entry = Decimal(str(getattr(fills.events[0], "fill_price")))

        stops = liq.stop_loss_manager.active_stops
        assert "SOL/USDT" in stops
        assert stops["SOL/USDT"].stop_price == entry * Decimal("0.95")

        # Above the stop: nothing happens.
        await bus.publish(_PRICES, _price("SOL/USDT", float(entry * Decimal("0.97"))))
        assert liquidations.events == []
        assert len(await broker.get_positions()) == 1

        # At the stop: one liquidation, the whole position closes through
        # the tracked path, tagged liq-, and the stop is gone.
        await bus.publish(_PRICES, _price("SOL/USDT", float(entry * Decimal("0.95"))))
        assert len(liquidations.events) == 1
        assert isinstance(liquidations.events[0], LiquidationTriggeredEvent)
        assert await broker.get_positions() == []
        closing = [f for f in fills.events if getattr(f, "side", "") == "sell"]
        assert len(closing) == 1
        assert str(getattr(closing[0], "client_order_id", "")).startswith("liq-")
        assert Decimal(str(getattr(closing[0], "fill_quantity"))) == Decimal("0.0111")
        assert "SOL/USDT" not in liq.stop_loss_manager.active_stops
    finally:
        await ex.stop()


async def test_boot_seeds_a_stop_for_every_restored_long() -> None:
    settings = _settings()
    broker = PaperBroker(initial_cash=Decimal("100"))
    broker.restore_state(
        cash=Decimal("90"),
        positions={"SOL/USDT": Decimal("0.0111"), "ETH/USDT": Decimal("-0.0000001")},
        avg_entry={"SOL/USDT": Decimal("90.22"), "ETH/USDT": Decimal("2577.81")},
        last_prices={"SOL/USDT": Decimal("90.24"), "ETH/USDT": Decimal("2245.80")},
    )
    liq = LiquidationManagerService(
        event_bus=InProcessEventBus(), settings=settings, position_provider=broker.get_positions
    )
    await liq.seed_stops(await broker.get_positions())
    stops = liq.stop_loss_manager.active_stops
    assert set(stops) == {"SOL/USDT"}  # a long gets a stop; a restored short cannot
    assert stops["SOL/USDT"].entry_price == Decimal("90.22")
    assert stops["SOL/USDT"].stop_price == Decimal("90.22") * Decimal("0.95")


# ---------------------------------------------------------------------------
# (3) Risk-reducing orders are exempt from the entry rules, not from the halt
# ---------------------------------------------------------------------------


def _correlated_book(svc: RiskManagerService, equity: float = 100.0) -> np.ndarray:
    """Two real positions with identical return series: one violation."""
    rng = np.random.default_rng(7)
    base = rng.normal(scale=1e-3, size=60)
    svc.update_equity(Decimal(str(equity)))
    svc.drawdown_monitor.update(equity)
    for symbol in ("A/USDT", "B/USDT"):
        svc.update_returns(symbol, list(base))
        svc.update_position(symbol, 5.0)
    return base


async def test_exit_is_approved_despite_violations_and_an_open_breaker_while_entry_is_rejected(
    mock_settings: Settings,
) -> None:
    svc = RiskManagerService(event_bus=InProcessEventBus(), settings=mock_settings)
    base = _correlated_book(svc)
    svc.update_returns("C/USDT", list(base))  # the entry is correlated with the book
    svc.update_daily_pnl(-0.08)  # trips the 7% breaker on the next check

    entry = svc.pre_trade_check({"symbol": "C/USDT", "target_weight": 0.05})
    assert not entry.approved
    assert any("MaxCorrelation" in r for r in entry.rejections)
    assert any("CircuitBreaker" in r for r in entry.rejections)

    exit_ = svc.pre_trade_check({"symbol": "A/USDT", "target_weight": 0.0}, risk_reducing=True)
    assert exit_.approved and exit_.rejections == []


async def test_rebalance_publishes_the_exit_and_not_the_entry(mock_settings: Settings) -> None:
    broker = PaperBroker(initial_cash=Decimal("100"))
    broker.restore_state(
        cash=Decimal("90"),
        positions={"A/USDT": Decimal("0.5"), "B/USDT": Decimal("0.5")},
        avg_entry={"A/USDT": Decimal("10"), "B/USDT": Decimal("10")},
        last_prices={"A/USDT": Decimal("10"), "B/USDT": Decimal("10")},
    )
    bus = InProcessEventBus()
    approved = await _Capture(bus, RISK_APPROVED).start()
    svc = RiskManagerService(event_bus=bus, settings=mock_settings, position_provider=broker.get_positions)
    base = _correlated_book(svc)
    svc.update_returns("C/USDT", list(base))
    await svc.start()

    await bus.publish(
        REBALANCE,
        RebalanceRequestEvent(
            target_allocations={"A/USDT": 0.0, "C/USDT": 0.05},
            reference_prices={"A/USDT": 11.0, "C/USDT": 20.0},
            source_service="portfolio_optimizer",
        ),
    )
    assert [getattr(e, "symbol", "") for e in approved.events] == ["A/USDT"]
    exit_order = approved.events[0]
    assert getattr(exit_order, "side", "") == "sell"
    assert Decimal(str(getattr(exit_order, "quantity"))) == Decimal("0.5")


async def test_a_halted_engine_refuses_a_rebalance_exit_but_executes_a_liquidation_exit(
    mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("services.execution.service._POLL_INTERVAL", 0.02)
    bus = InProcessEventBus()
    broker = PaperBroker(initial_cash=Decimal("100000"))
    broker.update_price("A/USDT", Decimal("10"))
    ex = ExecutionEngineService(event_bus=bus, settings=mock_settings, brokers={"paper": broker})
    await ex.start()
    try:
        await ex.submit_order(
            symbol="A/USDT",
            side="buy",
            order_type="market",
            quantity=Decimal("2"),
            reference_price=Decimal("10"),
        )
        ex.halt("operator")
        refused = await ex.submit_order(
            symbol="A/USDT",
            side="sell",
            order_type="market",
            quantity=Decimal("2"),
            client_order_id="rebal-x-A/USDT",
        )
        assert refused is None
        assert Decimal((await broker.get_positions())[0]["quantity"]) == Decimal("2")

        await bus.publish(
            LIQUIDATION,
            LiquidationTriggeredEvent(
                position_id="A/USDT",
                symbol="A/USDT",
                reason="stop",
                quantity=0.0,
                source_service="LiquidationManagerService",
            ),
        )
        assert await broker.get_positions() == []
    finally:
        await ex.stop()


# ---------------------------------------------------------------------------
# (4) Exits are sized from the broker's signed quantity; oversells are refused
# ---------------------------------------------------------------------------


async def test_exit_quantity_is_the_held_quantity_not_dollars_over_price(
    mock_settings: Settings,
) -> None:
    broker = PaperBroker(initial_cash=Decimal("100"))
    broker.restore_state(
        cash=Decimal("99"),
        positions={"SOL/USDT": Decimal("0.0111")},
        avg_entry={"SOL/USDT": Decimal("90.22")},
        last_prices={"SOL/USDT": Decimal("90.24")},
    )
    bus = InProcessEventBus()
    approved = await _Capture(bus, RISK_APPROVED).start()
    svc = RiskManagerService(event_bus=bus, settings=mock_settings, position_provider=broker.get_positions)
    svc.update_equity(Decimal("100"))
    svc.update_position("SOL/USDT", 1.0014)  # the dollar book at fill value
    await svc.start()
    # A moved reference price would size 1.0014 / 97 = 0.01032 from the dollar
    # book and leave 0.00078 SOL of dust; the signed quantity leaves nothing.
    await bus.publish(
        REBALANCE,
        RebalanceRequestEvent(
            target_allocations={"SOL/USDT": 0.0},
            reference_prices={"SOL/USDT": 97.0},
            source_service="portfolio_optimizer",
        ),
    )
    assert len(approved.events) == 1
    assert Decimal(str(getattr(approved.events[0], "quantity"))) == Decimal("0.0111")
    assert getattr(approved.events[0], "side", "") == "sell"


async def test_a_restored_short_is_closed_with_a_buy_of_its_size(mock_settings: Settings) -> None:
    broker = PaperBroker(initial_cash=Decimal("100"))
    broker.restore_state(
        cash=Decimal("100"),
        positions={"DOT/USDT": Decimal("-0.00117783")},
        avg_entry={"DOT/USDT": Decimal("0.92")},
        last_prices={"DOT/USDT": Decimal("0.91")},
    )
    bus = InProcessEventBus()
    approved = await _Capture(bus, RISK_APPROVED).start()
    svc = RiskManagerService(event_bus=bus, settings=mock_settings, position_provider=broker.get_positions)
    svc.update_equity(Decimal("100"))
    await svc.start()
    await bus.publish(
        REBALANCE,
        RebalanceRequestEvent(
            target_allocations={"DOT/USDT": 0.0},
            reference_prices={"DOT/USDT": 0.91},
            source_service="portfolio_optimizer",
        ),
    )
    assert len(approved.events) == 1
    assert getattr(approved.events[0], "side", "") == "buy"
    assert Decimal(str(getattr(approved.events[0], "quantity"))) == Decimal("0.00117783")


async def test_paper_broker_rejects_a_sell_larger_than_held_and_fills_the_exact_exit() -> None:
    broker = PaperBroker(initial_cash=Decimal("1000"))
    broker.update_price("SOL/USDT", Decimal("100"))
    await broker.submit_order(
        BrokerOrder(external_id="b1", symbol="SOL/USDT", side="buy", order_type="market", quantity=Decimal("0.0111"))
    )
    await broker.submit_order(
        BrokerOrder(external_id="s-over", symbol="SOL/USDT", side="sell", order_type="market", quantity=Decimal("0.02"))
    )
    over = await broker.get_order_status("s-over")
    assert over["status"] == "rejected"
    assert "insufficient_position" in str(over.get("reason", ""))
    held = await broker.get_positions()
    assert len(held) == 1 and Decimal(held[0]["quantity"]) == Decimal("0.0111")

    await broker.submit_order(
        BrokerOrder(
            external_id="s-exact", symbol="SOL/USDT", side="sell", order_type="market", quantity=Decimal("0.0111")
        )
    )
    assert (await broker.get_order_status("s-exact"))["status"] == "filled"
    assert await broker.get_positions() == []

    # Nothing held at all: a sell cannot open a short on a spot venue.
    await broker.submit_order(
        BrokerOrder(
            external_id="s-naked", symbol="ETH/USDT", side="sell", order_type="market", quantity=Decimal("0.001")
        )
    )
    broker.update_price("ETH/USDT", Decimal("2000"))
    assert (await broker.get_order_status("s-naked"))["status"] == "rejected"
    assert await broker.get_positions() == []


# ---------------------------------------------------------------------------
# (5) The correlation rule is symbol-aware and ignores dust
# ---------------------------------------------------------------------------


def test_correlation_rule_judges_the_proposed_symbol_against_real_holdings_only(
    mock_settings: Settings,
) -> None:
    svc = RiskManagerService(event_bus=InProcessEventBus(), settings=mock_settings)
    svc.update_equity(Decimal("100"))
    svc.drawdown_monitor.update(100.0)
    rng = np.random.default_rng(11)
    # Per-bar returns of ~0.1%, so VaR stays far inside its 3% limit and the
    # correlation rule is the only one that can speak.
    dust_series = list(rng.normal(scale=1e-3, size=60))
    real_series = list(rng.normal(scale=1e-3, size=60))

    # Four dust positions, pairwise identical: six violations in the old count.
    for symbol in ("ETH/USDT", "DOT/USDT", "SPY", "AMZN"):
        svc.update_returns(symbol, dust_series)
        svc.update_position(symbol, 0.002)
    # One real position.
    svc.update_returns("SOL/USDT", real_series)
    svc.update_position("SOL/USDT", 5.0)
    assert svc.check_portfolio_risk().correlation_violations >= 6  # the report still counts them

    # An uncorrelated entry passes.
    svc.update_returns("ADA/USDT", list(rng.normal(scale=1e-3, size=60)))
    assert svc.pre_trade_check({"symbol": "ADA/USDT", "target_weight": 0.05}).approved

    # An entry identical to the real holding is rejected.
    svc.update_returns("X/USDT", real_series)
    decision = svc.pre_trade_check({"symbol": "X/USDT", "target_weight": 0.05})
    assert not decision.approved
    assert any("MaxCorrelation" in r for r in decision.rejections)

    # An entry identical only to dust is not.
    svc.update_returns("Y/USDT", dust_series)
    assert svc.pre_trade_check({"symbol": "Y/USDT", "target_weight": 0.05}).approved


# ---------------------------------------------------------------------------
# Risk sign-off conditions (2026-11-07)
# ---------------------------------------------------------------------------


async def test_unpriced_restored_dust_does_not_freeze_exploration() -> None:
    """Equity dust restored into a crypto-only paper session never gets a
    price, so it can neither exit nor count toward the exploration cap."""
    bus = InProcessEventBus()
    rebalances = await _Capture(bus, REBALANCE).start()
    settings = _settings(active_symbols_crypto=["SOL/USDT", "ADA/USDT"], active_symbols_stocks=["SPY", "AMZN"])
    svc = PortfolioOptimizerService(event_bus=bus, settings=settings)
    await svc.start()
    svc.seed_held_book(["SPY", "AMZN", "NVDA"])  # NVDA is outside the universe
    assert svc.held_symbols == {"SPY", "AMZN", "NVDA"}
    assert set(svc.exploration_positions) == {"SPY", "AMZN"}

    # A priced sub-gate crypto signal still opens an exploration entry.
    await bus.publish(_PRICES, _price("SOL/USDT", 100.0))
    await bus.publish(PREDICTIONS_READY, _prediction("SOL/USDT"))
    assert len(rebalances.events) == 1
    event = rebalances.events[0]
    assert isinstance(event, RebalanceRequestEvent)
    assert event.target_allocations == {"SOL/USDT": settings.exploration_weight}


async def test_buy_fill_the_provider_has_not_seen_yet_still_gets_a_stop() -> None:
    settings = _settings()
    bus = InProcessEventBus()

    async def lagging_provider() -> list[dict[str, Any]]:
        return []  # the venue balance has not caught up with the fill

    liq = LiquidationManagerService(event_bus=bus, settings=settings, position_provider=lagging_provider)
    await liq.start()
    await bus.publish(
        ORDERS,
        OrderFilledEvent(
            order_id="o1",
            symbol="SOL/USDT",
            side="buy",
            fill_price=100.0,
            fill_quantity=0.0111,
            source_service="execution-engine",
        ),
    )
    stops = liq.stop_loss_manager.active_stops
    assert "SOL/USDT" in stops
    assert stops["SOL/USDT"].stop_price == Decimal("95.00")


def test_stop_loss_pct_outside_the_unit_interval_is_refused() -> None:
    with pytest.raises(ValueError):
        _settings(stop_loss_pct=5.0)
    with pytest.raises(ValueError):
        _settings(stop_loss_pct=-0.01)
    assert _settings(stop_loss_pct=0.0).stop_loss_pct == 0.0
