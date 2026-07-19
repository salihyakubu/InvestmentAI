"""WebSocket fanout: event-bus streams must reach websocket clients."""

from __future__ import annotations

from typing import Any

import pytest

from api.websockets import fanout
from api.websockets.manager import CHANNEL_ALERTS, CHANNEL_ORDERS, CHANNEL_PRICES
from core.events.base import InProcessEventBus
from core.events.market_events import PriceUpdateEvent
from core.events.order_events import OrderFilledEvent
from core.events.streams import ORDERS


@pytest.fixture()
def broadcasts(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    sent: list[tuple[str, dict[str, Any]]] = []

    async def _capture(channel: str, data: dict[str, Any]) -> None:
        sent.append((channel, data))

    monkeypatch.setattr(fanout.manager, "broadcast", _capture)
    return sent


@pytest.mark.asyncio
async def test_price_events_reach_the_prices_channel(
    broadcasts: list[tuple[str, dict[str, Any]]],
) -> None:
    bus = InProcessEventBus()
    await fanout.start_fanout(bus)

    from datetime import UTC, datetime

    await bus.publish(
        fanout.MARKET_PRICES_STREAM,
        PriceUpdateEvent(
            symbol="BTC/USDT", price=65000.5, volume=1.2,
            market_timestamp=datetime.now(UTC), source_service="ingestion",
        ),
    )

    assert len(broadcasts) == 1
    channel, msg = broadcasts[0]
    assert channel == CHANNEL_PRICES
    # Browser routing key + payload fields the dashboard consumes.
    assert msg["channel"] == CHANNEL_PRICES
    assert msg["symbol"] == "BTC/USDT"
    assert msg["price"] == 65000.5


@pytest.mark.asyncio
async def test_order_events_reach_the_orders_channel(
    broadcasts: list[tuple[str, dict[str, Any]]],
) -> None:
    bus = InProcessEventBus()
    await fanout.start_fanout(bus)

    await bus.publish(
        ORDERS,
        OrderFilledEvent(
            order_id="o-1", fill_price=10.0, fill_quantity=2.0,
            symbol="AAPL", side="buy", source_service="execution",
        ),
    )

    orders_msgs = [m for c, m in broadcasts if c == CHANNEL_ORDERS]
    assert len(orders_msgs) == 1
    assert orders_msgs[0]["event_type"] == "OrderFilledEvent"
    assert orders_msgs[0]["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_broadcast_failure_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _boom(channel: str, data: dict[str, Any]) -> None:
        raise RuntimeError("client exploded")

    monkeypatch.setattr(fanout.manager, "broadcast", _boom)
    bus = InProcessEventBus()
    await fanout.start_fanout(bus)

    from datetime import UTC, datetime

    # Must be swallowed by the handler, never propagate into the consumer loop.
    await bus.publish(
        fanout.MARKET_PRICES_STREAM,
        PriceUpdateEvent(
            symbol="AAPL", price=1.0, volume=0.0,
            market_timestamp=datetime.now(UTC), source_service="ingestion",
        ),
    )
    assert CHANNEL_ALERTS  # reached without exception
