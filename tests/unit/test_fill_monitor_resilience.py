"""One failed publish must not poison the fill monitor (GO_LIVE.md 2026-11-02).

Before the fix, the monitor recorded a fill, then published it; if the publish
raised (a Redis blip), the order stayed in ``_in_flight`` and the next poll
called ``record_fill`` on an already-FILLED order, which raised OrderError and
aborted the whole loop body -- every 2 s, for the life of the process. Every
order behind it never had its fill published: closes went missing from the
ledger with no restart involved.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from config.settings import Settings
from core.events.base import Event, InProcessEventBus
from core.events.streams import ORDERS
from services.execution.brokers.paper_broker import PaperBroker
from services.execution.service import ExecutionEngineService


class FlakyBus(InProcessEventBus):
    """Raises on the first OrderFilledEvent publish, then behaves."""

    def __init__(self) -> None:
        super().__init__()
        self.failures_left = 1

    async def publish(self, stream: str, event: Event) -> str:
        if event.event_type == "OrderFilledEvent" and self.failures_left > 0:
            self.failures_left -= 1
            raise RuntimeError("redis blip")
        return await super().publish(stream, event)


@pytest.mark.asyncio
async def test_failed_publish_neither_poisons_monitor_nor_starves_other_orders(
    mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("services.execution.service._POLL_INTERVAL", 0.02)
    bus = FlakyBus()
    fills: list[Event] = []

    async def capture(event: Event) -> None:
        if event.event_type == "OrderFilledEvent":
            fills.append(event)

    await bus.subscribe(ORDERS, "t", "t", capture)

    broker = PaperBroker(initial_cash=Decimal("100000"))
    broker.update_price("AAPL", Decimal("100"))
    broker.update_price("MSFT", Decimal("50"))
    ex = ExecutionEngineService(
        event_bus=bus, settings=mock_settings, brokers={"paper": broker}
    )
    # Both orders are in flight before the first poll; AAPL is polled first and
    # its publish fails, MSFT sits behind it in the in-flight map.
    a = await ex.submit_order(
        symbol="AAPL", side="buy", order_type="market",
        quantity=Decimal("5"), reference_price=Decimal("100"),
    )
    b = await ex.submit_order(
        symbol="MSFT", side="buy", order_type="market",
        quantity=Decimal("3"), reference_price=Decimal("50"),
    )
    assert a is not None and b is not None

    await ex.start()
    try:
        for _ in range(250):
            if len(fills) >= 2 and not ex._in_flight:
                break
            await asyncio.sleep(0.02)
    finally:
        await ex.stop()

    # Both fills published, each exactly once, in-flight drained.
    assert sorted(f.order_id for f in fills) == sorted([a.order_id, b.order_id])
    assert ex._in_flight == {}
    # The book was counted once despite the retry on AAPL.
    aapl = ex._fill_tracker.get_position("AAPL")
    assert aapl is not None and aapl.quantity == Decimal("5")
    assert ex._order_manager.get_order(a.order_id).filled_quantity == Decimal("5")
    assert bus.failures_left == 0  # the failure path was actually exercised
