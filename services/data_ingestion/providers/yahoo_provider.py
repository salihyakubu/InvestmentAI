"""Yahoo Finance data provider (historical backfill only)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import structlog

from core.enums import AssetClass, TimeFrame
from services.data_ingestion.providers.base import BaseDataProvider, RawBar, RealtimeCallback

logger = structlog.get_logger(__name__)

# yfinance uses its own interval strings.
_YF_INTERVAL_MAP: dict[str, str] = {
    TimeFrame.M1: "1m",
    TimeFrame.M5: "5m",
    TimeFrame.M15: "15m",
    TimeFrame.H1: "1h",
    TimeFrame.H4: "1h",   # yfinance has no 4h; we'll resample from 1h.
    TimeFrame.D1: "1d",
    TimeFrame.W1: "1wk",
}


class YahooDataProvider(BaseDataProvider):
    """Free historical OHLCV data from Yahoo Finance via ``yfinance``.

    This provider is intended as a zero-cost backfill source for US
    equities and ETFs.  It does **not** support real-time streaming.
    """

    name = "yahoo"
    asset_class = AssetClass.STOCK

    # ------------------------------------------------------------------
    # Historical
    # ------------------------------------------------------------------

    async def fetch_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[RawBar]:
        import yfinance as yf

        yf_interval = _YF_INTERVAL_MAP.get(timeframe)
        if yf_interval is None:
            raise ValueError(f"Unsupported timeframe for Yahoo: {timeframe}")

        need_resample = timeframe == TimeFrame.H4

        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        loop = asyncio.get_running_loop()
        ticker = yf.Ticker(symbol)

        df = await loop.run_in_executor(
            None,
            lambda: ticker.history(
                interval=yf_interval,
                start=start_str,
                end=end_str,
                auto_adjust=True,
            ),
        )

        if df is None or df.empty:
            logger.info(
                "yahoo.historical.empty",
                symbol=symbol,
                timeframe=timeframe,
            )
            return []

        # Resample to 4-hour bars if needed.
        if need_resample:
            df = df.resample("4h").agg(
                {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum",
                }
            ).dropna()

        raw_bars: list[RawBar] = []
        for idx, row in df.iterrows():
            bar_time = idx.to_pydatetime()
            if bar_time.tzinfo is None:
                bar_time = bar_time.replace(tzinfo=timezone.utc)

            raw_bars.append(
                RawBar(
                    time=bar_time,
                    symbol=symbol,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                )
            )

        logger.info(
            "yahoo.historical.fetched",
            symbol=symbol,
            timeframe=timeframe,
            bar_count=len(raw_bars),
        )
        return raw_bars

    # ------------------------------------------------------------------
    # Real-time (not supported)
    # ------------------------------------------------------------------

    async def subscribe_realtime(
        self,
        symbols: list[str],
        callback: RealtimeCallback,
    ) -> None:
        raise NotImplementedError(
            "Yahoo Finance does not support real-time streaming. "
            "Use Alpaca or CCXT for live data."
        )

    async def unsubscribe(self) -> None:
        # Nothing to tear down for a REST-only provider.
        pass

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        try:
            import yfinance as yf

            loop = asyncio.get_running_loop()
            ticker = yf.Ticker("AAPL")
            info = await loop.run_in_executor(None, lambda: ticker.fast_info)
            return info is not None
        except Exception:
            logger.warning("yahoo.health_check.failed", exc_info=True)
            return False
