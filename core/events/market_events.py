"""Market-data related events."""

from __future__ import annotations

from datetime import datetime

from core.events.base import Event


class BarCloseEvent(Event):
    """Emitted when an OHLCV bar closes for a given symbol and timeframe."""

    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
    trade_count: int | None = None
    bar_time: datetime


class PriceUpdateEvent(Event):
    """Emitted on every real-time price tick or quote update."""

    symbol: str
    price: float
    volume: float
    market_timestamp: datetime
