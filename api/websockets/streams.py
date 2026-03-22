"""WebSocket streaming endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.websockets.manager import (
    CHANNEL_ALERTS,
    CHANNEL_ORDERS,
    CHANNEL_PORTFOLIO,
    CHANNEL_PRICES,
    manager,
)

logger = logging.getLogger(__name__)

router = APIRouter()


async def _stream_loop(websocket: WebSocket, channel: str) -> None:
    """Common loop: connect, listen for client messages, handle disconnect."""
    await manager.connect(websocket, channel)
    try:
        while True:
            # Keep the connection alive; clients can send pings or
            # subscription filters as JSON messages.
            data = await websocket.receive_text()
            logger.debug("Received on %s: %s", channel, data)
    except WebSocketDisconnect:
        manager.disconnect(websocket, channel)


@router.websocket("/ws/prices")
async def ws_prices(websocket: WebSocket) -> None:
    """Real-time price stream.

    Clients receive OHLCV updates as they arrive from market data providers.
    """
    await _stream_loop(websocket, CHANNEL_PRICES)


@router.websocket("/ws/orders")
async def ws_orders(websocket: WebSocket) -> None:
    """Order status update stream.

    Clients receive notifications when orders change state
    (submitted, filled, cancelled, etc.).
    """
    await _stream_loop(websocket, CHANNEL_ORDERS)


@router.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket) -> None:
    """Risk alert stream.

    Clients receive real-time alerts for risk threshold breaches,
    circuit breaker activations, and other critical events.
    """
    await _stream_loop(websocket, CHANNEL_ALERTS)


@router.websocket("/ws/portfolio")
async def ws_portfolio(websocket: WebSocket) -> None:
    """Portfolio update stream.

    Clients receive real-time portfolio value updates, position changes,
    and P&L notifications.
    """
    await _stream_loop(websocket, CHANNEL_PORTFOLIO)
