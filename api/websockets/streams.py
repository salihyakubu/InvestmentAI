"""WebSocket streaming endpoints.

Every channel is authenticated: the client's FIRST message must be
``{"type": "auth", "token": "<jwt>"}`` within a few seconds, or the socket is
closed with 1008 (policy violation) and never joins a channel. The token
travels in a message, never in the URL -- the frontend used to append
``?token=<jwt>`` to the connect URL, which uvicorn and nginx both write to
their access logs, and the server never read it anyway (2026-11-02 review).
Channels are capped so an anonymous flood cannot starve the process that
also hosts the kill switch.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.middleware.auth import JWTAuth
from api.websockets.manager import (
    CHANNEL_ALERTS,
    CHANNEL_ORDERS,
    CHANNEL_PORTFOLIO,
    CHANNEL_PRICES,
    manager,
)
from config.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Seconds the client has to send its auth message after the handshake.
AUTH_TIMEOUT_SECONDS = 5.0
# Per-channel connection cap (the operator UI opens one socket per channel).
MAX_CONNECTIONS_PER_CHANNEL = 50

_WS_POLICY_VIOLATION = 1008
_WS_TRY_AGAIN_LATER = 1013


async def authenticate(websocket: WebSocket) -> dict[str, Any] | None:
    """Read and verify the first message; return the token payload or None."""
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=AUTH_TIMEOUT_SECONDS)
    except (TimeoutError, WebSocketDisconnect):
        return None
    try:
        message = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(message, dict) or message.get("type") != "auth":
        return None
    token = str(message.get("token") or "").strip()
    if not token:
        return None
    settings = get_settings()
    jwt_auth = JWTAuth(
        secret=settings.jwt_secret.get_secret_value(), algorithm=settings.jwt_algorithm
    )
    return jwt_auth.verify_token(token)


async def _stream_loop(websocket: WebSocket, channel: str) -> None:
    """Accept, authenticate, register, then idle until the client disconnects."""
    await websocket.accept()
    if manager.count(channel) >= MAX_CONNECTIONS_PER_CHANNEL:
        logger.warning("WebSocket channel %s full; refusing connection", channel)
        await websocket.close(code=_WS_TRY_AGAIN_LATER)
        return
    user = await authenticate(websocket)
    if user is None:
        logger.info("WebSocket on %s refused: no valid auth message", channel)
        await websocket.close(code=_WS_POLICY_VIOLATION)
        return
    manager.register(websocket, channel)
    try:
        await websocket.send_text(json.dumps({"type": "auth_ok", "channel": channel}))
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
