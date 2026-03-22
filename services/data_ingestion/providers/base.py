"""Abstract base class for market data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Coroutine

from core.enums import AssetClass


@dataclass(slots=True)
class RawBar:
    """Provider-agnostic OHLCV bar returned by every data provider."""

    time: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
    trade_count: int | None = None


# Callback signature for real-time bar delivery.
RealtimeCallback = Callable[[RawBar], Coroutine[Any, Any, None]]


class BaseDataProvider(ABC):
    """Contract that every market-data provider must implement.

    Providers are responsible for fetching raw bars from a single source
    (broker API, exchange, free data service) and converting them into
    :class:`RawBar` instances.  Normalisation, validation, and persistence
    are handled by the ingestion service layer above.
    """

    name: str
    asset_class: AssetClass

    # ------------------------------------------------------------------
    # Historical data
    # ------------------------------------------------------------------

    @abstractmethod
    async def fetch_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[RawBar]:
        """Return historical OHLCV bars for *symbol* over [start, end)."""
        ...

    # ------------------------------------------------------------------
    # Real-time streaming
    # ------------------------------------------------------------------

    @abstractmethod
    async def subscribe_realtime(
        self,
        symbols: list[str],
        callback: RealtimeCallback,
    ) -> None:
        """Start streaming real-time bars for *symbols*.

        Each closed bar is delivered to *callback* as a :class:`RawBar`.
        """
        ...

    @abstractmethod
    async def unsubscribe(self) -> None:
        """Gracefully disconnect from the real-time stream."""
        ...

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @abstractmethod
    async def health_check(self) -> bool:
        """Return ``True`` if the provider is reachable and functional."""
        ...
