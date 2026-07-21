"""Idempotent deep-history backfill of the ``ohlcv`` table.

Fills months of historical 1-minute bars so cloud retraining can train on far
more than the few days live ingestion has accumulated:

    # 90 days of 1m history for the active crypto pairs into the production DB:
    PYTHONPATH=. python scripts/backfill_history.py --crypto-days 90 \\
        --batch 5000 --database-url-env DATABASE_URL

    # also backfill the active stocks (Yahoo caps 1m depth at ~7 days):
    PYTHONPATH=. python scripts/backfill_history.py --crypto-days 90 --stocks

    # see what would be inserted without writing anything:
    PYTHONPATH=. python scripts/backfill_history.py --crypto-days 90 --dry-run

Safety / idempotency contract:

- The ``ohlcv`` table's real uniqueness constraint is its composite PRIMARY KEY
  ``(time, symbol, timeframe)`` (core/models/market_data.py; migration
  0001_baseline creates it via ``metadata.create_all`` and the TimescaleDB
  hypertable conversion in 0002 preserves it -- ``create_hypertable`` requires
  the time column inside the PK). Inserts therefore use
  ``INSERT .. ON CONFLICT DO NOTHING`` on exactly that key -- the same dedupe
  the live ingestion service's ``_store_bars`` uses -- so re-running the script
  never duplicates a bar and never modifies an existing row.
- The script only ever INSERTs into ``ohlcv`` (plus one ``SELECT count(*)`` per
  symbol in ``--dry-run`` mode). It never UPDATEs or DELETEs, so it is safe to
  run against the production database while the worker is live.
- Every bar flows through the live pipeline's own ``normalize_bar`` +
  ``validate_bar``, so symbol normalisation ('BTC/USDT'-style crypto pairs,
  plain upper-case stock tickers), tz-aware UTC timestamps, and Decimal
  price/volume handling are byte-identical to what live ingestion writes.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from sqlalchemy import CursorResult, Insert, func, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from core.enums import AssetClass, TimeFrame
from core.models.market_data import OHLCVRecord
from services.data_ingestion.normalizer import NormalizationError, normalize_bar
from services.data_ingestion.providers.base import BaseDataProvider
from services.data_ingestion.validator import validate_bar

# Live ingestion cadence: this script backfills 1-minute bars only.
TIMEFRAME = TimeFrame.M1

# asyncpg caps a statement at 32767 bind parameters; an ohlcv row binds 13
# columns, so a single INSERT may carry at most 32767 // 13 = 2520 rows.
# Stay comfortably below that. --batch controls commit granularity; each batch
# is split into sub-statements of at most this many rows.
MAX_ROWS_PER_STATEMENT = 2000

SessionFactory = async_sessionmaker[AsyncSession]


@dataclass(slots=True)
class SymbolStats:
    """Per-symbol backfill accounting."""

    fetched: int = 0
    valid: int = 0
    inserted: int = 0
    skipped: int = 0
    error: str | None = None


def _resolve_database_url(env_name: str) -> str:
    """Read the DB URL from *env_name* and coerce it to the asyncpg driver.

    Mirrors ``Settings.async_database_url`` (Railway hands out plain
    ``postgresql://``/``postgres://`` URLs). sqlite+aiosqlite URLs pass through
    untouched so the script stays testable against sqlite.
    """
    url = os.environ.get(env_name, "")
    if not url:
        raise SystemExit(
            f"Environment variable {env_name!r} is not set -- refusing to guess a database. "
            f"Export it (e.g. the Railway DATABASE_URL) or point --database-url-env elsewhere."
        )
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


def _conflict_free_insert(dialect: str, rows: list[dict[str, Any]]) -> Insert:
    """INSERT .. ON CONFLICT DO NOTHING on the table's composite PK."""
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        return pg_insert(OHLCVRecord).values(rows).on_conflict_do_nothing(
            index_elements=["time", "symbol", "timeframe"],
        )
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        return sqlite_insert(OHLCVRecord).values(rows).on_conflict_do_nothing(
            index_elements=["time", "symbol", "timeframe"],
        )
    raise SystemExit(f"Unsupported database dialect for backfill: {dialect!r}")


def _normalize_and_validate(
    raw_bars: list[Any],
    asset_class: AssetClass,
    source: str,
) -> list[dict[str, Any]]:
    """Run raw provider bars through the live normalise/validate pipeline.

    Also dedupes on (time, symbol, timeframe) inside the fetch itself --
    provider pagination can occasionally return an overlapping candle.
    """
    rows: list[dict[str, Any]] = []
    seen: set[tuple[datetime, str, str]] = set()
    for raw in raw_bars:
        try:
            bar = normalize_bar(raw, asset_class, source, TIMEFRAME)
        except NormalizationError:
            continue
        is_valid, _ = validate_bar(bar)
        if not is_valid:
            continue
        key = (bar["time"], bar["symbol"], bar["timeframe"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(bar)
    return rows


async def _insert_rows(
    session_factory: SessionFactory,
    dialect: str,
    rows: list[dict[str, Any]],
    batch: int,
) -> int:
    """Insert *rows* in commit batches of *batch*; return rows actually inserted."""
    inserted = 0
    for start in range(0, len(rows), batch):
        chunk = rows[start : start + batch]
        async with session_factory() as session:
            for sub_start in range(0, len(chunk), MAX_ROWS_PER_STATEMENT):
                sub = chunk[sub_start : sub_start + MAX_ROWS_PER_STATEMENT]
                stmt = _conflict_free_insert(dialect, sub)
                result = cast(CursorResult[Any], await session.execute(stmt))
                inserted += max(result.rowcount or 0, 0)
            await session.commit()
    return inserted


async def _count_existing_in_range(
    session_factory: SessionFactory,
    rows: list[dict[str, Any]],
) -> int:
    """Count rows already stored for this symbol/timeframe inside the fetched span."""
    if not rows:
        return 0
    times = [r["time"] for r in rows]
    symbol = rows[0]["symbol"]
    async with session_factory() as session:
        result = await session.execute(
            select(func.count())
            .select_from(OHLCVRecord)
            .where(
                OHLCVRecord.symbol == symbol,
                OHLCVRecord.timeframe == str(TIMEFRAME),
                OHLCVRecord.time >= min(times),
                OHLCVRecord.time <= max(times),
            )
        )
        return int(result.scalar_one())


async def _backfill_symbol(
    session_factory: SessionFactory,
    dialect: str,
    provider: BaseDataProvider,
    symbol: str,
    asset_class: AssetClass,
    start: datetime,
    end: datetime,
    batch: int,
    dry_run: bool,
) -> SymbolStats:
    stats = SymbolStats()
    try:
        raw_bars = await provider.fetch_historical_bars(symbol, TIMEFRAME, start, end)
    except Exception as exc:  # a single symbol failing must not kill the run
        stats.error = f"fetch failed: {exc}"
        print(f"  {symbol}: FETCH FAILED ({exc}) -- skipping")
        return stats

    stats.fetched = len(raw_bars)
    rows = _normalize_and_validate(raw_bars, asset_class, provider.name)
    stats.valid = len(rows)

    if dry_run:
        existing = await _count_existing_in_range(session_factory, rows)
        would_insert = max(stats.valid - existing, 0)
        print(
            f"  {symbol}: fetched={stats.fetched} valid={stats.valid} "
            f"existing_in_range={existing} would_insert<={would_insert} [dry-run: nothing written]"
        )
        stats.skipped = stats.valid
        return stats

    stats.inserted = await _insert_rows(session_factory, dialect, rows, batch)
    stats.skipped = stats.valid - stats.inserted
    print(
        f"  {symbol}: fetched={stats.fetched} valid={stats.valid} "
        f"inserted={stats.inserted} skipped={stats.skipped} (already present)"
    )
    return stats


async def run_backfill(
    engine: AsyncEngine,
    *,
    crypto_symbols: list[str],
    stock_symbols: list[str],
    crypto_days: int,
    stock_days: int = 7,
    batch: int = 5000,
    dry_run: bool = False,
    crypto_provider: BaseDataProvider | None = None,
    stock_provider: BaseDataProvider | None = None,
) -> dict[str, SymbolStats]:
    """Backfill 1m bars for the given symbols. Returns per-symbol stats.

    Providers are injectable so tests can supply fakes; when omitted, the
    platform's own CCXT (Binance public klines, keyless) and Yahoo providers
    are used -- the same ones live ingestion uses for these asset classes.
    """
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")

    session_factory: SessionFactory = async_sessionmaker(engine, expire_on_commit=False)
    dialect = engine.dialect.name
    end = datetime.now(UTC)
    results: dict[str, SymbolStats] = {}

    owns_crypto_provider = False
    if crypto_symbols and crypto_provider is None:
        from services.data_ingestion.providers.ccxt_provider import CCXTDataProvider

        crypto_provider = CCXTDataProvider()  # public klines: no credentials needed
        owns_crypto_provider = True

    if stock_symbols and stock_provider is None:
        from services.data_ingestion.providers.yahoo_provider import YahooDataProvider

        stock_provider = YahooDataProvider()

    try:
        if crypto_symbols and crypto_provider is not None:
            print(f"Backfilling {len(crypto_symbols)} crypto pair(s), {crypto_days}d of 1m bars ...")
            for symbol in crypto_symbols:
                results[symbol] = await _backfill_symbol(
                    session_factory, dialect, crypto_provider, symbol,
                    AssetClass.CRYPTO, end - timedelta(days=crypto_days), end,
                    batch, dry_run,
                )

        if stock_symbols and stock_provider is not None:
            print(f"Backfilling {len(stock_symbols)} stock(s), {stock_days}d of 1m bars (Yahoo depth cap) ...")
            for symbol in stock_symbols:
                results[symbol] = await _backfill_symbol(
                    session_factory, dialect, stock_provider, symbol,
                    AssetClass.STOCK, end - timedelta(days=stock_days), end,
                    batch, dry_run,
                )
    finally:
        if owns_crypto_provider and crypto_provider is not None:
            close = getattr(crypto_provider, "_close_exchange", None)
            if close is not None:
                await close()

    return results


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--crypto-days", type=int, default=90,
                        help="days of 1m crypto history to backfill (default 90)")
    parser.add_argument("--stocks", action="store_true",
                        help="also backfill the active stock symbols via Yahoo")
    parser.add_argument("--stock-days", type=int, default=7,
                        help="days of 1m stock history (Yahoo caps at ~7; default 7)")
    parser.add_argument("--batch", type=int, default=5000,
                        help="rows per commit batch (default 5000)")
    parser.add_argument("--database-url-env", default="DATABASE_URL",
                        help="name of the env var holding the DB URL (default DATABASE_URL)")
    parser.add_argument("--dry-run", action="store_true",
                        help="fetch + report only; write nothing")
    parser.add_argument("--symbols", default="",
                        help="comma-separated subset of the active symbols to "
                             "backfill (e.g. 'ADA/USDT'); useful to finish one "
                             "symbol after a provider rate-limit without "
                             "re-fetching the rest")
    args = parser.parse_args(argv[1:])

    if args.crypto_days < 1:
        print("--crypto-days must be >= 1", file=sys.stderr)
        return 2
    if args.batch < 1:
        print("--batch must be >= 1", file=sys.stderr)
        return 2

    from config.settings import get_settings

    settings = get_settings()
    crypto_symbols = list(settings.active_symbols_crypto)
    stock_symbols = list(settings.active_symbols_stocks) if args.stocks else []
    if args.symbols:
        wanted = {s.strip() for s in args.symbols.split(",") if s.strip()}
        unknown = wanted - set(crypto_symbols) - set(settings.active_symbols_stocks)
        if unknown:
            print(f"--symbols not in the active universe: {sorted(unknown)}", file=sys.stderr)
            return 2
        crypto_symbols = [s for s in crypto_symbols if s in wanted]
        stock_symbols = [s for s in settings.active_symbols_stocks if s in wanted]

    url = _resolve_database_url(args.database_url_env)

    async def _run() -> dict[str, SymbolStats]:
        engine = create_async_engine(url, echo=False)
        try:
            return await run_backfill(
                engine,
                crypto_symbols=crypto_symbols,
                stock_symbols=stock_symbols,
                crypto_days=args.crypto_days,
                stock_days=args.stock_days,
                batch=args.batch,
                dry_run=args.dry_run,
            )
        finally:
            await engine.dispose()

    mode = "DRY-RUN (no writes)" if args.dry_run else "insert-only (ON CONFLICT DO NOTHING)"
    print(f"ohlcv history backfill -- mode: {mode}")
    results = asyncio.run(_run())

    total_fetched = sum(s.fetched for s in results.values())
    total_inserted = sum(s.inserted for s in results.values())
    total_skipped = sum(s.skipped for s in results.values())
    failures = {sym: s.error for sym, s in results.items() if s.error}

    print(
        f"\nDONE: symbols={len(results)} fetched={total_fetched} "
        f"inserted={total_inserted} skipped={total_skipped}"
    )
    if failures:
        print(f"FAILED symbols ({len(failures)}): {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
