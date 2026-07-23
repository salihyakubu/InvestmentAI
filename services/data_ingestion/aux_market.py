"""Auxiliary market-regime ingestion service.

Polls a handful of KEYLESS public sources on a slow interval and records
slow-moving market-regime metrics both durably (the ``aux_market_state`` table,
for training as-of replay) and in-memory (a :class:`LiveAuxProvider` snapshot,
for live serving):

- crypto funding rate -- Binance public futures ``/fapi/v1/premiumIndex``
  (``lastFundingRate``), per crypto symbol;
- Crypto Fear & Greed index -- ``https://api.alternative.me/fng/`` (global);
- VIX close -- Yahoo ``^VIX`` daily (global);
- SPY daily return -- Yahoo ``SPY`` daily, last two closes (global).

Every source is fetched best-effort and fully isolated: one source (or the DB
write) failing never stops the others and never kills the polling loop. No
source requires an API key.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from config.settings import Settings, get_settings
from core.enums import TimeFrame
from core.models.market_data import AuxMarketState
from services.data_ingestion.providers.yahoo_provider import YahooDataProvider
from services.feature_engineering.aux_features import (
    GLOBAL_SYMBOL,
    METRIC_FEAR_GREED,
    METRIC_FUNDING_RATE,
    METRIC_SPY_RETURN,
    METRIC_VIX_CLOSE,
    LiveAuxProvider,
)

logger = structlog.get_logger(__name__)

_POLL_INTERVAL_SECONDS = 300.0
_BINANCE_PREMIUM_INDEX = "https://fapi.binance.com/fapi/v1/premiumIndex"
_FNG_URL = "https://api.alternative.me/fng/"
_HTTP_TIMEOUT = 10.0


def _to_binance_futures_symbol(symbol: str) -> str:
    """Map a canonical crypto symbol to Binance's futures ticker.

    ``"BTC/USDT" -> "BTCUSDT"``. Also tolerates a ``"-"`` separator.
    """
    return symbol.replace("/", "").replace("-", "").upper()


class AuxMarketService:
    """Background poller for keyless market-regime metrics.

    Parameters
    ----------
    session_factory : async session factory used to upsert ``aux_market_state``.
    live_provider   : the shared :class:`LiveAuxProvider` snapshot to refresh.
    settings        : platform settings (for the active symbol lists).
    poll_interval   : seconds between polls (default 300).
    http_client_factory : builds an ``httpx.AsyncClient`` (injectable for tests).
    yahoo_provider  : Yahoo data provider for ^VIX / SPY (injectable for tests).
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        live_provider: LiveAuxProvider,
        settings: Settings | None = None,
        *,
        poll_interval: float = _POLL_INTERVAL_SECONDS,
        http_client_factory: Callable[[], httpx.AsyncClient] | None = None,
        yahoo_provider: YahooDataProvider | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._live = live_provider
        self._settings = settings or get_settings()
        self._poll_interval = poll_interval
        self._http_client_factory = http_client_factory or (
            lambda: httpx.AsyncClient(timeout=_HTTP_TIMEOUT)
        )
        self._yahoo = yahoo_provider or YahooDataProvider()
        self._running = False
        self._task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            logger.warning("aux_market.already_running")
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="aux-market-poll")
        logger.info("aux_market.started", poll_interval=self._poll_interval)

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("aux_market.stopped")

    async def _run_loop(self) -> None:
        """Poll immediately, then every ``poll_interval`` seconds until stopped.

        A failed poll iteration is logged and the loop continues -- a transient
        source or DB outage must never take the poller down.
        """
        while self._running:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("aux_market.poll_loop_error")
            try:
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                raise

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def poll_once(self) -> None:
        """Fetch every source once, each isolated so one failure never stops
        the others."""
        for source in (
            self._poll_funding,
            self._poll_fear_greed,
            self._poll_vix,
            self._poll_spy,
        ):
            try:
                await source()
            except Exception:
                logger.exception("aux_market.source_failed", source=source.__name__)

    async def _poll_funding(self) -> None:
        """Per-symbol crypto funding rate from Binance public futures."""
        symbols = self._settings.active_symbols_crypto
        if not symbols:
            return
        async with self._http_client_factory() as client:
            for symbol in symbols:
                try:
                    resp = await client.get(
                        _BINANCE_PREMIUM_INDEX,
                        params={"symbol": _to_binance_futures_symbol(symbol)},
                    )
                    resp.raise_for_status()
                    rate = float(resp.json()["lastFundingRate"])
                except Exception:
                    logger.warning("aux_market.funding_failed", symbol=symbol)
                    continue
                self._live.update_funding(symbol, rate)
                await self._store_metric(
                    datetime.now(UTC), METRIC_FUNDING_RATE, symbol, rate
                )

    async def _poll_fear_greed(self) -> None:
        """Global Crypto Fear & Greed index from alternative.me."""
        async with self._http_client_factory() as client:
            resp = await client.get(_FNG_URL)
            resp.raise_for_status()
            value = float(resp.json()["data"][0]["value"])
        self._live.update_fear_greed(value)
        await self._store_metric(
            datetime.now(UTC), METRIC_FEAR_GREED, GLOBAL_SYMBOL, value
        )

    async def _poll_vix(self) -> None:
        """Global VIX close from Yahoo ^VIX daily bars."""
        bars = await self._recent_daily_bars("^VIX")
        if not bars:
            return
        value = float(bars[-1].close)
        self._live.update_vix_close(value)
        await self._store_metric(
            datetime.now(UTC), METRIC_VIX_CLOSE, GLOBAL_SYMBOL, value
        )

    async def _poll_spy(self) -> None:
        """Global SPY daily return from the last two Yahoo SPY daily closes."""
        bars = await self._recent_daily_bars("SPY")
        if len(bars) < 2:
            return
        prev_close = float(bars[-2].close)
        if prev_close == 0.0:
            return
        value = float(bars[-1].close) / prev_close - 1.0
        self._live.update_spy_daily_return(value)
        await self._store_metric(
            datetime.now(UTC), METRIC_SPY_RETURN, GLOBAL_SYMBOL, value
        )

    async def _recent_daily_bars(self, symbol: str) -> list[Any]:
        """Fetch roughly the last 5 trading days of daily bars for *symbol*."""
        end = datetime.now(UTC)
        start = end - timedelta(days=7)
        return await self._yahoo.fetch_historical_bars(
            symbol, TimeFrame.D1, start, end
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _store_metric(
        self, ts: datetime, metric: str, symbol: str, value: float
    ) -> None:
        """Upsert one ``(time, metric, symbol)`` observation into
        ``aux_market_state``.

        Uses ``session.merge`` -- a portable, idempotent upsert on the primary
        key that behaves identically on PostgreSQL (production) and SQLite
        (tests); re-storing the same key is a harmless no-op. Best-effort: a DB
        error is logged and swallowed so it never interrupts the poll.
        """
        try:
            async with self._session_factory() as session:
                await session.merge(
                    AuxMarketState(time=ts, metric=metric, symbol=symbol, value=value)
                )
                await session.commit()
        except Exception:
            logger.exception(
                "aux_market.store_failed", metric=metric, symbol=symbol
            )
