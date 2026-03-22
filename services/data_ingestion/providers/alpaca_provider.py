"""Alpaca Markets data provider for US equities."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.enums import AssetClass, TimeFrame
from services.data_ingestion.providers.base import BaseDataProvider, RawBar, RealtimeCallback

logger = structlog.get_logger(__name__)

# Mapping from our TimeFrame enum values to Alpaca TimeFrame objects.
# Resolved lazily to avoid import-time failures when alpaca-py is absent.
_ALPACA_TF_MAP: dict[str, Any] | None = None


def _get_alpaca_tf_map() -> dict[str, Any]:
    global _ALPACA_TF_MAP
    if _ALPACA_TF_MAP is None:
        from alpaca.data.timeframe import TimeFrame as AlpacaTF, TimeFrameUnit

        _ALPACA_TF_MAP = {
            TimeFrame.M1: AlpacaTF(1, TimeFrameUnit.Minute),
            TimeFrame.M5: AlpacaTF(5, TimeFrameUnit.Minute),
            TimeFrame.M15: AlpacaTF(15, TimeFrameUnit.Minute),
            TimeFrame.H1: AlpacaTF(1, TimeFrameUnit.Hour),
            TimeFrame.H4: AlpacaTF(4, TimeFrameUnit.Hour),
            TimeFrame.D1: AlpacaTF(1, TimeFrameUnit.Day),
            TimeFrame.W1: AlpacaTF(1, TimeFrameUnit.Week),
        }
    return _ALPACA_TF_MAP


class AlpacaDataProvider(BaseDataProvider):
    """Fetches US equity data via the Alpaca Markets v2 API.

    Uses ``alpaca-py`` for both historical REST queries and real-time
    WebSocket streams.
    """

    name = "alpaca"
    asset_class = AssetClass.STOCK

    def __init__(self, api_key: str, secret_key: str) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._stream: Any | None = None
        self._stream_task: asyncio.Task[None] | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Historical
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def fetch_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[RawBar]:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest

        tf_map = _get_alpaca_tf_map()
        alpaca_tf = tf_map.get(timeframe)
        if alpaca_tf is None:
            raise ValueError(f"Unsupported timeframe for Alpaca: {timeframe}")

        client = StockHistoricalDataClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
        )

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=alpaca_tf,
            start=start.astimezone(timezone.utc),
            end=end.astimezone(timezone.utc),
        )

        # The SDK call is synchronous; run in executor to keep the loop free.
        loop = asyncio.get_running_loop()
        bar_set = await loop.run_in_executor(None, client.get_stock_bars, request)

        raw_bars: list[RawBar] = []
        bars = bar_set[symbol] if symbol in bar_set else []
        for bar in bars:
            raw_bars.append(
                RawBar(
                    time=bar.timestamp.replace(tzinfo=timezone.utc)
                    if bar.timestamp.tzinfo is None
                    else bar.timestamp,
                    symbol=symbol,
                    open=float(bar.open),
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                    volume=float(bar.volume),
                    vwap=float(bar.vwap) if bar.vwap is not None else None,
                    trade_count=int(bar.trade_count) if bar.trade_count is not None else None,
                )
            )

        logger.info(
            "alpaca.historical.fetched",
            symbol=symbol,
            timeframe=timeframe,
            bar_count=len(raw_bars),
        )
        return raw_bars

    # ------------------------------------------------------------------
    # Real-time
    # ------------------------------------------------------------------

    async def subscribe_realtime(
        self,
        symbols: list[str],
        callback: RealtimeCallback,
    ) -> None:
        from alpaca.data.live import StockDataStream

        self._stream = StockDataStream(
            api_key=self._api_key,
            secret_key=self._secret_key,
        )
        self._running = True

        async def _on_bar(bar: Any) -> None:
            raw = RawBar(
                time=bar.timestamp.replace(tzinfo=timezone.utc)
                if bar.timestamp.tzinfo is None
                else bar.timestamp,
                symbol=bar.symbol,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(bar.volume),
                vwap=float(bar.vwap) if bar.vwap is not None else None,
                trade_count=int(bar.trade_count) if bar.trade_count is not None else None,
            )
            try:
                await callback(raw)
            except Exception:
                logger.exception("alpaca.realtime.callback_error", symbol=bar.symbol)

        self._stream.subscribe_bars(_on_bar, *symbols)
        logger.info("alpaca.realtime.subscribed", symbols=symbols)

        # _run() blocks until the stream is stopped.
        self._stream_task = asyncio.create_task(self._run_stream())

    async def _run_stream(self) -> None:
        """Run the WebSocket event loop in a background task."""
        try:
            await asyncio.to_thread(self._stream.run)
        except asyncio.CancelledError:
            logger.info("alpaca.realtime.cancelled")
        except Exception:
            logger.exception("alpaca.realtime.stream_error")

    async def unsubscribe(self) -> None:
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                logger.warning("alpaca.realtime.stop_error", exc_info=True)
            self._stream = None
        if self._stream_task is not None:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
            self._stream_task = None
        logger.info("alpaca.realtime.unsubscribed")

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        try:
            from alpaca.data.historical import StockHistoricalDataClient

            client = StockHistoricalDataClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
            )
            # A lightweight call to verify credentials and connectivity.
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: client.get_stock_latest_bar("AAPL"))
            return True
        except Exception:
            logger.warning("alpaca.health_check.failed", exc_info=True)
            return False
