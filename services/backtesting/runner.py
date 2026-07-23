"""Background execution of backtest jobs.

The API router creates a ``backtest_jobs`` row and spawns
:func:`execute_job` as an asyncio task in the API process. Data fetching is
async (keyless daily bars: Yahoo for stocks, ccxt/Binance public for crypto,
chosen by symbol form); the CPU-bound engine runs in a thread so the event
loop stays live. All failures land in the row's ``error`` -- a job never
crashes the API.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import numpy as np
import structlog
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.enums import TimeFrame
from core.models.backtests import BacktestJob
from services.backtesting.edge import DEFAULT_COST_BPS, run_backtest

logger = structlog.get_logger(__name__)

# Jobs orphaned by an API restart would stay "running" forever; status reads
# mark anything running longer than this as failed.
STALE_RUNNING_AFTER_S = 30 * 60


def _is_crypto(symbol: str) -> bool:
    return "/" in symbol


async def fetch_series(
    symbols: list[str], start: datetime, end: datetime
) -> dict[str, tuple[list[datetime], np.ndarray]]:
    """Fetch daily close series per symbol (keyless providers)."""
    out: dict[str, tuple[list[datetime], np.ndarray]] = {}
    for symbol in symbols:
        if _is_crypto(symbol):
            from services.data_ingestion.providers.ccxt_provider import CCXTDataProvider

            provider = CCXTDataProvider()
            try:
                bars = await provider.fetch_historical_bars(
                    symbol, TimeFrame.D1, start, end
                )
            finally:
                close = getattr(provider, "_close_exchange", None)
                if close is not None:
                    await close()
        else:
            from services.data_ingestion.providers.yahoo_provider import (
                YahooDataProvider,
            )

            bars = await YahooDataProvider().fetch_historical_bars(
                symbol, TimeFrame.D1, start, end
            )
        if not bars:
            logger.warning("backtest.fetch_empty", symbol=symbol)
            continue
        dates = [b.time for b in bars]
        closes = np.array([float(b.close) for b in bars], dtype=float)
        out[symbol] = (dates, closes)
    return out


async def execute_job(
    job_id: Any,
    config: dict[str, Any],
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Run one backtest job end to end, recording status transitions."""

    async def _set(**values: Any) -> None:
        async with session_factory() as session:
            await session.execute(
                update(BacktestJob).where(BacktestJob.id == job_id).values(**values)
            )
            await session.commit()

    try:
        await _set(status="running", started_at=datetime.now(UTC))

        start = datetime.fromisoformat(config["start_date"]).replace(tzinfo=UTC)
        end = datetime.fromisoformat(config["end_date"]).replace(tzinfo=UTC)
        symbols = [str(s) for s in config["symbols"]]
        # The UI sends commission as a round-trip FRACTION (e.g. 0.001 = 10 bps).
        cost_bps = float(config.get("commission", 0.0)) * 1e4 or DEFAULT_COST_BPS
        initial_capital = float(config.get("initial_capital", 10_000.0))

        series = await fetch_series(symbols, start, end)
        if not series:
            raise ValueError("no price data for any requested symbol")

        result = await asyncio.to_thread(
            run_backtest, series, cost_bps=cost_bps, initial_capital=initial_capital
        )
        await _set(
            status="completed", result=result, finished_at=datetime.now(UTC)
        )
        logger.info(
            "backtest.completed",
            job_id=str(job_id),
            symbols=len(series),
            verdict=result.get("verdict", "")[:60],
        )
    except Exception as exc:
        logger.exception("backtest.failed", job_id=str(job_id))
        try:
            await _set(
                status="failed", error=str(exc)[:2000], finished_at=datetime.now(UTC)
            )
        except Exception:
            logger.exception("backtest.fail_record_failed", job_id=str(job_id))
