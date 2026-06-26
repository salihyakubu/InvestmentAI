"""Fetch real historical OHLCV into a CSV the edge harness consumes.

This is Stage-2 step 1 of the go-live runbook ("ingest real historical OHLCV").
It reuses the platform's own data providers and writes a CSV with the columns
``validate_model.py`` reads (it needs a ``close`` column).

    # US stock / ETF via Yahoo (keyless):
    PYTHONPATH=. python scripts/fetch_history.py AAPL --start 2015-01-01 --out aapl.csv
    # Crypto via Binance public klines (CCXT, keyless):
    PYTHONPATH=. python scripts/fetch_history.py BTC/USDT --start 2021-01-01 --out btc.csv

Then run the edge gate on it:

    PYTHONPATH=. python scripts/validate_model.py aapl.csv

Provider is auto-selected (crypto when the symbol contains '/', else Yahoo) and
can be forced with --provider. Output columns: time,open,high,low,close,volume.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from datetime import UTC, datetime

from core.enums import TimeFrame
from services.data_ingestion.providers.base import BaseDataProvider

# StrEnum -> {"1d": TimeFrame.D1, "1h": TimeFrame.H1, ...}
_TIMEFRAMES: dict[str, TimeFrame] = {str(tf): tf for tf in TimeFrame}


def _make_provider(name: str) -> BaseDataProvider:
    if name == "yahoo":
        from services.data_ingestion.providers.yahoo_provider import YahooDataProvider

        return YahooDataProvider()
    if name == "ccxt":
        from services.data_ingestion.providers.ccxt_provider import CCXTDataProvider

        # Public klines need no credentials.
        return CCXTDataProvider()
    raise ValueError(f"Unknown provider: {name!r} (use 'yahoo' or 'ccxt')")


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)


async def _run(args: argparse.Namespace) -> int:
    provider_name = args.provider or ("ccxt" if "/" in args.symbol else "yahoo")
    timeframe = _TIMEFRAMES.get(args.timeframe)
    if timeframe is None:
        print(f"Unsupported --timeframe {args.timeframe!r}; choose from {sorted(_TIMEFRAMES)}", file=sys.stderr)
        return 2

    start = _parse_date(args.start)
    end = _parse_date(args.end) if args.end else datetime.now(UTC)
    out = args.out or (args.symbol.replace("/", "-") + ".csv")

    provider = _make_provider(provider_name)
    print(f"fetching {args.symbol} [{args.timeframe}] {start:%Y-%m-%d}..{end:%Y-%m-%d} via {provider_name} ...")
    try:
        bars = await provider.fetch_historical_bars(args.symbol, timeframe, start, end)
    finally:
        # Best-effort cleanup for providers that hold an open client (CCXT).
        closer = getattr(provider, "_close_exchange", None)
        if closer is not None:
            try:
                await closer()
            except Exception:
                pass

    if not bars:
        print(f"No bars returned for {args.symbol}. Check the symbol / date range / provider.", file=sys.stderr)
        return 1

    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "open", "high", "low", "close", "volume"])
        for b in bars:
            writer.writerow([b.time.isoformat(), b.open, b.high, b.low, b.close, b.volume])

    print(f"wrote {len(bars)} bars to {out}  (first {bars[0].time:%Y-%m-%d}, last {bars[-1].time:%Y-%m-%d})")
    print(f"next: PYTHONPATH=. python scripts/validate_model.py {out}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Fetch historical OHLCV into a CSV for the edge harness.")
    parser.add_argument("symbol", help="e.g. AAPL, SPY, or BTC/USDT for crypto")
    parser.add_argument("--start", default="2015-01-01", help="YYYY-MM-DD (default 2015-01-01)")
    parser.add_argument("--end", default="", help="YYYY-MM-DD (default: today)")
    parser.add_argument("--timeframe", default="1d", help="1m,5m,15m,1h,4h,1d,1w (default 1d)")
    parser.add_argument("--provider", choices=["yahoo", "ccxt"], default="", help="default: auto by symbol")
    parser.add_argument("--out", default="", help="output CSV path (default: <symbol>.csv)")
    args = parser.parse_args(argv[1:])
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main(sys.argv))
