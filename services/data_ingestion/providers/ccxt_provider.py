"""CCXT-based cryptocurrency data provider (Binance by default)."""

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

# Mapping from our canonical TimeFrame values to CCXT/Binance timeframe strings.
_CCXT_TF_MAP: dict[str, str] = {
    TimeFrame.M1: "1m",
    TimeFrame.M5: "5m",
    TimeFrame.M15: "15m",
    TimeFrame.H1: "1h",
    TimeFrame.H4: "4h",
    TimeFrame.D1: "1d",
    TimeFrame.W1: "1w",
}


class CCXTDataProvider(BaseDataProvider):
    """Cryptocurrency OHLCV provider backed by CCXT (Binance).

    Uses ``ccxt.async_support.binance`` for both REST historical queries
    and WebSocket-based real-time candle streaming via ``watch_ohlcv``.
    """

    name = "ccxt_binance"
    asset_class = AssetClass.CRYPTO

    def __init__(
        self,
        api_key: str = "",
        secret_key: str = "",
        *,
        sandbox: bool = False,
    ) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._sandbox = sandbox
        self._exchange: Any | None = None
        self._running = False
        self._watch_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Exchange lifecycle
    # ------------------------------------------------------------------

    async def _get_exchange(self) -> Any:
        if self._exchange is None:
            import ccxt.async_support as ccxt_async

            config: dict[str, Any] = {
                "enableRateLimit": True,
            }
            if self._api_key:
                config["apiKey"] = self._api_key
            if self._secret_key:
                config["secret"] = self._secret_key
            self._exchange = ccxt_async.binance(config)
            if self._sandbox:
                self._exchange.set_sandbox_mode(True)
        return self._exchange

    async def _close_exchange(self) -> None:
        if self._exchange is not None:
            try:
                await self._exchange.close()
            except Exception:
                logger.warning("ccxt.exchange.close_error", exc_info=True)
            self._exchange = None

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
        exchange = await self._get_exchange()

        ccxt_tf = _CCXT_TF_MAP.get(timeframe)
        if ccxt_tf is None:
            raise ValueError(f"Unsupported timeframe for CCXT: {timeframe}")

        since_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        # CCXT symbol format uses "/" separator (e.g. BTC/USDT).
        ccxt_symbol = symbol.replace("-", "/") if "-" in symbol else symbol

        all_candles: list[list[Any]] = []
        current_since = since_ms

        while current_since < end_ms:
            candles = await exchange.fetch_ohlcv(
                ccxt_symbol,
                timeframe=ccxt_tf,
                since=current_since,
                limit=1000,
            )
            if not candles:
                break

            for c in candles:
                if c[0] < end_ms:
                    all_candles.append(c)

            # Advance cursor past the last candle timestamp.
            last_ts = candles[-1][0]
            if last_ts <= current_since:
                break
            current_since = last_ts + 1

        raw_bars: list[RawBar] = []
        for c in all_candles:
            # CCXT candle format: [timestamp_ms, open, high, low, close, volume]
            raw_bars.append(
                RawBar(
                    time=datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc),
                    symbol=symbol,
                    open=float(c[1]),
                    high=float(c[2]),
                    low=float(c[3]),
                    close=float(c[4]),
                    volume=float(c[5]),
                )
            )

        logger.info(
            "ccxt.historical.fetched",
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
        """Poll Binance REST API for the latest closed candles.

        Binance does not support ``watch_ohlcv`` via CCXT WebSocket, so we
        poll ``fetch_ohlcv`` every 60 seconds and deliver newly closed bars.
        """
        self._running = True

        ccxt_symbols = [
            s.replace("-", "/") if "-" in s else s for s in symbols
        ]

        async def _poll_loop() -> None:
            exchange = await self._get_exchange()
            # Track the last delivered candle timestamp per symbol.
            last_delivered: dict[str, int] = {}

            while self._running:
                try:
                    for ccxt_sym, original_sym in zip(ccxt_symbols, symbols):
                        candles = await exchange.fetch_ohlcv(
                            ccxt_sym, timeframe="1m", limit=2,
                        )
                        if not candles or len(candles) < 2:
                            continue

                        # The second-to-last candle is the most recently closed.
                        closed = candles[-2]
                        ts = closed[0]
                        prev_ts = last_delivered.get(original_sym)

                        if prev_ts is None or prev_ts < ts:
                            raw = RawBar(
                                time=datetime.fromtimestamp(
                                    ts / 1000, tz=timezone.utc
                                ),
                                symbol=original_sym,
                                open=float(closed[1]),
                                high=float(closed[2]),
                                low=float(closed[3]),
                                close=float(closed[4]),
                                volume=float(closed[5]),
                            )
                            try:
                                await callback(raw)
                            except Exception:
                                logger.exception(
                                    "ccxt.realtime.callback_error",
                                    symbol=original_sym,
                                )
                            last_delivered[original_sym] = ts

                    # Poll every 60 seconds (one candle period for 1m bars).
                    await asyncio.sleep(60)

                except asyncio.CancelledError:
                    break
                except Exception:
                    logger.exception("ccxt.realtime.poll_error")
                    await asyncio.sleep(5)

        self._watch_task = asyncio.create_task(_poll_loop())
        logger.info("ccxt.realtime.subscribed", symbols=symbols)

    async def unsubscribe(self) -> None:
        self._running = False
        if self._watch_task is not None:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
            self._watch_task = None
        await self._close_exchange()
        logger.info("ccxt.realtime.unsubscribed")

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        try:
            exchange = await self._get_exchange()
            await exchange.fetch_time()
            return True
        except Exception:
            logger.warning("ccxt.health_check.failed", exc_info=True)
            return False
