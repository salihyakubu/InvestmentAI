"""Market data request/response schemas."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field

from core.enums import TimeFrame


class BarSchema(BaseModel):
    """Single OHLCV bar."""

    time: datetime
    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


class BarQueryParams(BaseModel):
    """Query parameters for historical bar requests."""

    symbol: str
    timeframe: TimeFrame = TimeFrame.D1
    start: datetime | None = None
    end: datetime | None = None
    limit: int = Field(default=100, ge=1, le=10000)


class LatestPriceResponse(BaseModel):
    """Latest price for a symbol."""

    symbol: str
    price: Decimal
    timestamp: datetime
