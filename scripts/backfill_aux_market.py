"""Backfill historical market-regime metrics into ``aux_market_state``.

The AuxMarketService collects regime metrics (funding rate, Fear & Greed, VIX,
SPY return) going forward, but a retrain needs history over its whole training
window with train/serve parity: each value stamped at the instant it first
became observable, so ``HistoricalAuxProvider`` can reproduce -- without
look-ahead -- what the live snapshot would have held at any past bar time.

Sources (all keyless):
- Funding rate: Binance ``/fapi/v1/fundingRate`` (realized 8h settlements),
  stamped at ``fundingTime``. NOTE: the live poller reads the *predicted*
  ``premiumIndex.lastFundingRate``; realized vs predicted differ slightly, a
  documented minor parity nuance the champion/challenger gate backstops.
- Fear & Greed: alternative.me full daily history, stamped at its timestamp.
- VIX / SPY: Yahoo daily bars, stamped at each bar's date (its close).

Idempotent (``session.merge`` on the PK); safe to re-run. Best-effort per
source. Usage::

    DATABASE_URL=... PYTHONPATH=. python scripts/backfill_aux_market.py --days 90
    ... --dry-run     # fetch + report counts, write nothing
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from core.enums import TimeFrame
from services.data_ingestion.aux_market import _to_binance_futures_symbol
from services.data_ingestion.providers.yahoo_provider import YahooDataProvider

_BINANCE_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
_FNG_URL = "https://api.alternative.me/fng/"

CRYPTO_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "DOT/USDT"]


async def _fetch_funding(symbol: str, start: datetime) -> list[tuple[datetime, float]]:
    """Paginate realized funding settlements for *symbol* since *start*."""
    out: list[tuple[datetime, float]] = []
    fut = _to_binance_futures_symbol(symbol)
    cursor = int(start.timestamp() * 1000)
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    async with httpx.AsyncClient(timeout=20.0) as client:
        while cursor < now_ms:
            resp = await client.get(
                _BINANCE_FUNDING,
                params={"symbol": fut, "startTime": cursor, "limit": 1000},
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            for row in batch:
                ts = datetime.fromtimestamp(row["fundingTime"] / 1000, UTC)
                out.append((ts, float(row["fundingRate"])))
            cursor = batch[-1]["fundingTime"] + 1
            if len(batch) < 1000:
                break
    return out


async def _fetch_fear_greed(start: datetime) -> list[tuple[datetime, float]]:
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(_FNG_URL, params={"limit": 0, "format": "json"})
        resp.raise_for_status()
        data = resp.json()["data"]
    out: list[tuple[datetime, float]] = []
    for row in data:
        ts = datetime.fromtimestamp(int(row["timestamp"]), UTC)
        if ts >= start:
            out.append((ts, float(row["value"])))
    return out


async def _fetch_daily(symbol: str, start: datetime) -> list[Any]:
    provider = YahooDataProvider()
    return await provider.fetch_historical_bars(
        symbol, TimeFrame.D1, start, datetime.now(UTC)
    )


async def _run(days: int, dry_run: bool) -> int:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        print("DATABASE_URL not set", file=sys.stderr)
        return 2
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)

    start = datetime.now(UTC) - timedelta(days=days)
    # (metric, symbol, [(time, value)]) tuples to persist.
    batches: list[tuple[str, str, list[tuple[datetime, float]]]] = []

    for sym in CRYPTO_SYMBOLS:
        try:
            rows = await _fetch_funding(sym, start)
            batches.append(("funding_rate", sym, rows))
            print(f"  funding_rate {sym}: {len(rows)} settlements")
        except Exception as exc:  # noqa: BLE001 -- best-effort per source
            print(f"  funding_rate {sym}: FAILED ({exc})")

    try:
        fng = await _fetch_fear_greed(start)
        batches.append(("fear_greed", "", fng))
        print(f"  fear_greed: {len(fng)} days")
    except Exception as exc:  # noqa: BLE001
        print(f"  fear_greed: FAILED ({exc})")

    for metric, sym in (("vix_close", "^VIX"), ("spy_daily_return", "SPY")):
        try:
            bars = await _fetch_daily(sym, start)
            if metric == "vix_close":
                rows = [(b.time, float(b.close)) for b in bars]
            else:  # spy_daily_return from consecutive closes
                rows = []
                for prev, cur in zip(bars, bars[1:], strict=False):
                    pc = float(prev.close)
                    if pc:
                        rows.append((cur.time, float(cur.close) / pc - 1.0))
            batches.append((metric, "", rows))
            print(f"  {metric}: {len(rows)} days")
        except Exception as exc:  # noqa: BLE001
            print(f"  {metric}: FAILED ({exc})")

    total = sum(len(r) for _, _, r in batches)
    if dry_run:
        print(f"DRY RUN: would upsert {total} rows; nothing written")
        return 0

    # Fast batched upsert via asyncpg (this script only runs against prod
    # Postgres): one executemany with ON CONFLICT DO NOTHING beats ~1.5k
    # sequential ORM merges over the remote proxy by orders of magnitude.
    import asyncpg

    records: list[tuple[datetime, str, str, float]] = []
    for metric, sym, rows in batches:
        for ts, value in rows:
            records.append(
                (ts if ts.tzinfo else ts.replace(tzinfo=UTC), metric, sym, value)
            )

    pg_url = url.replace("postgresql+asyncpg://", "postgres://", 1).replace(
        "postgresql://", "postgres://", 1
    )
    conn = await asyncpg.connect(pg_url, timeout=30)
    try:
        await conn.executemany(
            "INSERT INTO aux_market_state (time, metric, symbol, value) "
            "VALUES ($1, $2, $3, $4) ON CONFLICT (time, metric, symbol) DO NOTHING",
            records,
        )
    finally:
        await conn.close()

    print(f"DONE: upserted {len(records)} aux_market_state rows")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=90, help="days of history (default 90)")
    parser.add_argument("--dry-run", action="store_true", help="fetch + report; write nothing")
    args = parser.parse_args(argv[1:])
    if args.days < 1:
        print("--days must be >= 1", file=sys.stderr)
        return 2
    return asyncio.run(_run(args.days, args.dry_run))


if __name__ == "__main__":
    sys.exit(main(sys.argv))
