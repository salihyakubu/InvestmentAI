"""End-to-end pipeline test: rebalance request -> risk -> execution -> fill.

Exercises the full risk->execution wiring on the in-process event bus with the
paper broker, including the fill feeding back into the risk manager's state.
This is the path that was previously severed by stream-name mismatches and a
RiskApprovedEvent that carried no order parameters.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from config.settings import Settings
from core.events.base import Event, InProcessEventBus
from core.events.risk_events import RebalanceRequestEvent, RiskApprovedEvent
from core.events.streams import ORDERS, REBALANCE, RISK_APPROVED
from services.execution.brokers.paper_broker import PaperBroker
from services.execution.service import ExecutionEngineService
from services.risk.service import RiskManagerService


@pytest.mark.asyncio
async def test_rebalance_request_flows_to_broker_fill(
    mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A RebalanceRequestEvent should be sized, risk-checked, routed, and
    filled, with the fill feeding back into the risk manager's positions."""
    # Speed up the execution monitor's polling loop for the test.
    monkeypatch.setattr("services.execution.service._POLL_INTERVAL", 0.02)

    bus = InProcessEventBus()

    broker = PaperBroker(initial_cash=Decimal("10000"))
    broker.update_price("AAPL", Decimal("100"))  # seed the simulated market

    risk = RiskManagerService(event_bus=bus, settings=mock_settings)
    risk.update_equity(Decimal("10000"))
    risk.drawdown_monitor.update(10000.0)  # establish the drawdown peak

    execution = ExecutionEngineService(
        event_bus=bus, settings=mock_settings, brokers={"paper": broker}
    )

    await risk.start()
    await execution.start()
    try:
        # Target 5% of a $10k account in AAPL @ $100 => $500 => 5 shares.
        await bus.publish(
            REBALANCE,
            RebalanceRequestEvent(
                target_allocations={"AAPL": 0.05},
                reference_prices={"AAPL": 100.0},
                source_service="test",
            ),
        )

        # The order is created + submitted synchronously and the paper market
        # order fills inline, so the broker should already hold the position.
        positions = await broker.get_positions()
        aapl = next((p for p in positions if p["symbol"] == "AAPL"), None)
        assert aapl is not None, "expected an AAPL position in the paper broker"
        assert Decimal(aapl["quantity"]) == Decimal("5")

        # The background monitor polls the broker, emits OrderFilledEvent, which
        # feeds back into the risk manager. Wait for that to propagate.
        fed_back = False
        for _ in range(200):
            if risk._positions.get("AAPL", 0.0) > 0:
                fed_back = True
                break
            await asyncio.sleep(0.02)
        assert fed_back, "fill did not propagate back to the risk manager"
    finally:
        await execution.stop()


@pytest.mark.asyncio
async def test_rebalance_blocked_by_risk_emits_no_order(
    mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A target weight above the position limit is rejected by risk, so no
    order is created and the broker stays flat."""
    monkeypatch.setattr("services.execution.service._POLL_INTERVAL", 0.02)

    bus = InProcessEventBus()
    broker = PaperBroker(initial_cash=Decimal("10000"))
    broker.update_price("AAPL", Decimal("100"))

    risk = RiskManagerService(event_bus=bus, settings=mock_settings)
    risk.update_equity(Decimal("10000"))
    risk.drawdown_monitor.update(10000.0)

    execution = ExecutionEngineService(
        event_bus=bus, settings=mock_settings, brokers={"paper": broker}
    )
    await risk.start()
    await execution.start()
    try:
        # 50% in one name exceeds max_position_pct (10%) -> blocked by risk.
        await bus.publish(
            REBALANCE,
            RebalanceRequestEvent(
                target_allocations={"AAPL": 0.50},
                reference_prices={"AAPL": 100.0},
                source_service="test",
            ),
        )
        await asyncio.sleep(0.1)
        positions = await broker.get_positions()
    finally:
        await execution.stop()

    assert positions == [], "risk should have blocked the oversized order"


@pytest.mark.asyncio
async def test_execution_rejects_buy_exceeding_buying_power(
    mock_settings: Settings,
) -> None:
    """A buy whose notional exceeds buying power is rejected before reaching the
    broker -- no fill, and an OrderRejectedEvent is emitted."""
    bus = InProcessEventBus()
    broker = PaperBroker(initial_cash=Decimal("100"))
    broker.update_price("AAPL", Decimal("100"))

    rejected: list[Event] = []

    async def capture(event: Event) -> None:
        if event.event_type == "OrderRejectedEvent":
            rejected.append(event)

    await bus.subscribe(ORDERS, "t", "t", capture)

    execution = ExecutionEngineService(
        event_bus=bus, settings=mock_settings, brokers={"paper": broker}
    )
    # 5 shares @ $100 = $500 notional, but only $100 of buying power.
    await execution.submit_order(
        symbol="AAPL",
        side="buy",
        order_type="market",
        quantity=Decimal("5"),
        reference_price=Decimal("100"),
    )

    positions = await broker.get_positions()
    assert positions == [], "order should have been rejected, not filled"
    assert rejected, "expected an OrderRejectedEvent"
    assert "insufficient_buying_power" in rejected[-1].reason


@pytest.mark.asyncio
async def test_duplicate_risk_approved_is_submitted_once(
    mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same RiskApprovedEvent delivered twice yields a single order."""
    monkeypatch.setattr("services.execution.service._POLL_INTERVAL", 0.02)
    bus = InProcessEventBus()
    broker = PaperBroker(initial_cash=Decimal("10000"))
    broker.update_price("AAPL", Decimal("100"))

    execution = ExecutionEngineService(
        event_bus=bus, settings=mock_settings, brokers={"paper": broker}
    )
    await execution.start()
    try:
        approved = RiskApprovedEvent(
            order_id="dup-corr-1",
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=5.0,
            reference_price=100.0,
            source_service="test",
        )
        await bus.publish(RISK_APPROVED, approved)
        await bus.publish(RISK_APPROVED, approved)  # at-least-once redelivery
        await asyncio.sleep(0.1)
        positions = await broker.get_positions()
    finally:
        await execution.stop()

    aapl = next((p for p in positions if p["symbol"] == "AAPL"), None)
    assert aapl is not None
    # Deduped: 5 shares from one order, not 10 from two submissions.
    assert Decimal(aapl["quantity"]) == Decimal("5")
