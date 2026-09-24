"""Kill switch + position reconciliation on the execution engine.

These are pre-live safety primitives: an emergency flatten/halt and a way to
detect when our books have drifted from the broker's.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from config.settings import Settings
from core.events.base import InProcessEventBus
from services.execution.brokers.paper_broker import PaperBroker
from services.execution.service import ExecutionEngineService


def _engine(mock_settings: Settings):
    broker = PaperBroker(initial_cash=Decimal("100000"))
    broker.update_price("AAPL", Decimal("100"))
    ex = ExecutionEngineService(
        event_bus=InProcessEventBus(), settings=mock_settings, brokers={"paper": broker}
    )
    return ex, broker


@pytest.mark.asyncio
async def test_emergency_flatten_closes_positions_and_halts(mock_settings: Settings) -> None:
    ex, broker = _engine(mock_settings)
    await ex.submit_order(
        symbol="AAPL", side="buy", order_type="market",
        quantity=Decimal("5"), reference_price=Decimal("100"),
    )
    assert any(p["symbol"] == "AAPL" for p in await broker.get_positions())

    summary = await ex.emergency_flatten()
    assert summary["halted"] is True
    assert ex.halted is True
    assert await broker.get_positions() == []  # position flattened

    # New orders are refused while halted.
    await ex.submit_order(
        symbol="AAPL", side="buy", order_type="market",
        quantity=Decimal("1"), reference_price=Decimal("100"),
    )
    assert await broker.get_positions() == []  # nothing opened


@pytest.mark.asyncio
async def test_resume_re_enables_trading(mock_settings: Settings) -> None:
    ex, broker = _engine(mock_settings)
    ex.halt()
    await ex.submit_order(
        symbol="AAPL", side="buy", order_type="market",
        quantity=Decimal("2"), reference_price=Decimal("100"),
    )
    assert await broker.get_positions() == []  # halted -> refused
    ex.resume()
    await ex.submit_order(
        symbol="AAPL", side="buy", order_type="market",
        quantity=Decimal("2"), reference_price=Decimal("100"),
    )
    assert any(p["symbol"] == "AAPL" for p in await broker.get_positions())


@pytest.mark.asyncio
async def test_reconcile_detects_and_clears_mismatch(mock_settings: Settings) -> None:
    ex, broker = _engine(mock_settings)
    await ex.submit_order(
        symbol="AAPL", side="buy", order_type="market",
        quantity=Decimal("5"), reference_price=Decimal("100"),
    )
    # Broker holds 5; our books claim 3 -> discrepancy of +2.
    disc = await ex.reconcile_positions({"AAPL": Decimal("3")})
    assert len(disc) == 1 and disc[0]["symbol"] == "AAPL"
    assert Decimal(disc[0]["diff"]) == Decimal("2")

    # Books agree with the broker -> no discrepancy.
    assert await ex.reconcile_positions({"AAPL": Decimal("5")}) == []


@pytest.mark.asyncio
async def test_flatten_orders_are_tracked_not_fired_straight_at_the_broker(
    mock_settings: Settings,
) -> None:
    """The kill switch used to call broker.submit_order directly, so its
    closing trades changed the book while the durable record learned
    nothing -- no Order row, no fill event, no ledger entry -- precisely
    during the emergency a reconstructable record exists for. Flatten
    orders must go through the tracked path like everything else."""
    ex, broker = _engine(mock_settings)
    await ex.submit_order(
        symbol="AAPL", side="buy", order_type="market",
        quantity=Decimal("5"), reference_price=Decimal("100"),
    )
    before = len(ex._order_manager._orders)

    summary = await ex.emergency_flatten()
    assert await broker.get_positions() == []
    assert summary["flattened"], "the flatten reported nothing closed"

    after = list(ex._order_manager._orders.values())
    assert len(after) > before, "flatten produced no tracked order"
    closing = [o for o in after if o.symbol == "AAPL" and o.side == "sell"]
    assert closing, "the closing trade was never registered with the order manager"


@pytest.mark.asyncio
async def test_the_halt_bypass_is_only_for_the_flatten_itself(
    mock_settings: Settings,
) -> None:
    """The narrow exemption must not become a general hole: ordinary orders
    are still refused while halted."""
    ex, broker = _engine(mock_settings)
    await ex.emergency_flatten()
    assert ex.halted is True

    refused = await ex.submit_order(
        symbol="AAPL", side="buy", order_type="market",
        quantity=Decimal("1"), reference_price=Decimal("100"),
    )
    assert refused is None
    assert await broker.get_positions() == []
