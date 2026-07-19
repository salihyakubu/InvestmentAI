"""Fan out event-bus streams to connected WebSocket clients.

The worker publishes market/order/risk events on Redis streams; the dashboard
holds WebSocket connections. This module bridges them: the api subscribes to
the relevant streams (reusing the EventBus's tracked background consumer loops)
and re-broadcasts each event on the matching WebSocket channel. Without it the
WebSocket layer accepts connections but never sends anything.
"""

from __future__ import annotations

import logging
from typing import Any

from api.websockets.manager import (
    CHANNEL_ALERTS,
    CHANNEL_ORDERS,
    CHANNEL_PRICES,
    manager,
)
from core.events.base import Event, EventBus
from core.events.streams import ORDERS, RISK_BREACHED

logger = logging.getLogger(__name__)

# Stream names owned by the data-ingestion service (not in core.events.streams
# because only ingestion produces them; keep the literals in sync).
MARKET_PRICES_STREAM = "market.prices"

# (stream, websocket channel) pairs this fanout bridges.
_ROUTES: list[tuple[str, str]] = [
    (MARKET_PRICES_STREAM, CHANNEL_PRICES),
    (ORDERS, CHANNEL_ORDERS),
    (RISK_BREACHED, CHANNEL_ALERTS),
]

_CONSUMER_GROUP = "ws-fanout"


def _event_to_message(channel: str, event: Event) -> dict[str, Any]:
    """Serialize an event for the browser: payload fields + routing channel."""
    data: dict[str, Any] = {"channel": channel, "event_type": event.event_type}
    for key, value in event.model_dump(mode="json").items():
        if key not in data:
            data[key] = value
    return data


def _make_handler(channel: str) -> Any:
    async def _handler(event: Event) -> None:
        try:
            await manager.broadcast(channel, _event_to_message(channel, event))
        except Exception:  # never let a bad client break the consumer loop
            logger.exception("ws fanout broadcast failed on channel %s", channel)

    return _handler


async def start_fanout(event_bus: EventBus) -> None:
    """Subscribe the fanout handlers; consumer loops run as EventBus tasks."""
    for stream, channel in _ROUTES:
        await event_bus.subscribe(
            stream=stream,
            group=_CONSUMER_GROUP,
            consumer=f"api-{channel}",
            handler=_make_handler(channel),
        )
    logger.info("WebSocket fanout subscribed: %s", [s for s, _ in _ROUTES])
