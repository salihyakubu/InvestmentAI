"""WebSocket connection manager for real-time streaming."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)

# Well-known channel names
CHANNEL_PRICES = "prices"
CHANNEL_ORDERS = "orders"
CHANNEL_ALERTS = "alerts"
CHANNEL_PORTFOLIO = "portfolio"

ALL_CHANNELS = {CHANNEL_PRICES, CHANNEL_ORDERS, CHANNEL_ALERTS, CHANNEL_PORTFOLIO}


class ConnectionManager:
    """Manage WebSocket connections grouped by channel.

    Each channel (prices, orders, alerts, portfolio) maintains its own
    set of connected clients. Messages can be broadcast to all clients
    on a given channel.
    """

    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, websocket: WebSocket, channel: str) -> None:
        """Accept a WebSocket connection and add it to *channel*."""
        await websocket.accept()
        self._connections[channel].add(websocket)
        logger.info(
            "WebSocket connected to channel=%s (total=%d)",
            channel,
            len(self._connections[channel]),
        )

    def disconnect(self, websocket: WebSocket, channel: str) -> None:
        """Remove a WebSocket connection from *channel*."""
        self._connections[channel].discard(websocket)
        logger.info(
            "WebSocket disconnected from channel=%s (remaining=%d)",
            channel,
            len(self._connections[channel]),
        )

    async def broadcast(self, channel: str, data: dict[str, Any]) -> None:
        """Send *data* as JSON to every client on *channel*.

        Broken connections are silently removed.
        """
        dead: list[WebSocket] = []
        message = json.dumps(data)

        for ws in self._connections.get(channel, set()):
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)

        for ws in dead:
            self._connections[channel].discard(ws)

    @property
    def active_connections(self) -> dict[str, int]:
        """Return a mapping of channel -> number of active connections."""
        return {ch: len(conns) for ch, conns in self._connections.items() if conns}


# Singleton manager instance shared across the application
manager = ConnectionManager()
