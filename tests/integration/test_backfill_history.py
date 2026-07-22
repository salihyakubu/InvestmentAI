"""Idempotency + normalisation tests for scripts/backfill_history.py.

Runs the backfill against an in-memory sqlite ``ohlcv`` table (the same
composite-PK schema production has) with fake providers -- no network. Covers:

- insert a batch, re-run, no duplicates and no data loss (ON CONFLICT DO
  NOTHING on the (time, symbol, timeframe) PK);
- partial-overlap re-runs only add the genuinely new bars;
- symbol normalisation matches live ingestion ('BTC-USDT' -> 'BTC/USDT',
  'aapl' -> 'AAPL'), timeframe='1m', tz-aware UTC handling, Decimal columns;
- --dry-run writes nothing;
- invalid bars are dropped, intra-fetch duplicate timestamps deduped;
- multi-batch commits still insert everything exactly once.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from core.enums import AssetClass
from core.models.base import AsyncBase
from core.models.market_data import OHLCVRecord
from scripts.backfill_history import run_backfill
from services.data_ingestion.providers.base import BaseDataProvider, RawBar, RealtimeCallback

# A fixed, minute-aligned base in the past (the validator rejects future bars).
BASE = datetime.now(UTC).replace(second=0, microsecond=0) - timedelta(days=2)


class FakeProvider(BaseDataProvider):
    """Deterministic in-memory provider; returns preset bars per requested symbol."""

    def __init__(self, name: str, asset_class: AssetClass, bars: dict[str, list[RawBar]]) -> None:
        self.name = name
        self.asset_class = asset_class
        self._bars = bars
        self.calls: list[tuple[str, str, datetime, datetime]] = []

    async def fetch_historical_bars(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> list[RawBar]:
        self.calls.append((symbol, timeframe, start, end))
        return list(self._bars.get(symbol, []))

    async def subscribe_realtime(self, symbols: list[str], callback: RealtimeCallback) -> None:
        raise NotImplementedError

    async def unsubscribe(self) -> None:
        pass

    async def health_check(self) -> bool:
        return True


def _bar(minute: int, *, symbol: str, price: float = 100.0) -> RawBar:
    p = price + minute * 0.25
    return RawBar(
        time=BASE + timedelta(minutes=minute),
        symbol=symbol,
        open=p,
        high=p * 1.001,
        low=p * 0.999,
        close=p * 1.0005,
        volume=1000.0 + minute,
    )


async def _make_engine() -> AsyncEngine:
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.run_sync(AsyncBase.metadata.create_all, tables=[OHLCVRecord.__table__])
    return engine


async def _all_rows(engine: AsyncEngine) -> list[OHLCVRecord]:
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        result = await session.execute(select(OHLCVRecord).order_by(OHLCVRecord.time))
        return list(result.scalars().all())


async def _count(engine: AsyncEngine) -> int:
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        result = await session.execute(select(func.count()).select_from(OHLCVRecord))
        return int(result.scalar_one())


async def test_rerun_is_idempotent_no_duplicates_no_data_loss() -> None:
    engine = await _make_engine()
    try:
        bars = {"BTC/USDT": [_bar(i, symbol="BTC-USDT") for i in range(25)]}
        provider = FakeProvider("fake_ccxt", AssetClass.CRYPTO, bars)

        first = await run_backfill(
            engine, crypto_symbols=["BTC/USDT"], stock_symbols=[],
            crypto_days=1, batch=10, crypto_provider=provider,
        )
        assert first["BTC/USDT"].fetched == 25
        assert first["BTC/USDT"].inserted == 25
        assert first["BTC/USDT"].skipped == 0
        assert await _count(engine) == 25

        # Re-run with the identical fetch: nothing inserted, nothing lost.
        second = await run_backfill(
            engine, crypto_symbols=["BTC/USDT"], stock_symbols=[],
            crypto_days=1, batch=10, crypto_provider=provider,
        )
        assert second["BTC/USDT"].inserted == 0
        assert second["BTC/USDT"].skipped == 25
        assert await _count(engine) == 25

        # Existing rows are untouched (no update/delete): closes still intact.
        rows = await _all_rows(engine)
        assert rows[0].close == Decimal("100.05")  # 100.0 * 1.0005, Numeric(20, 8)
    finally:
        await engine.dispose()


async def test_overlapping_rerun_inserts_only_new_bars() -> None:
    engine = await _make_engine()
    try:
        provider1 = FakeProvider(
            "fake_ccxt", AssetClass.CRYPTO,
            {"ETH/USDT": [_bar(i, symbol="ETH/USDT") for i in range(10)]},
        )
        await run_backfill(
            engine, crypto_symbols=["ETH/USDT"], stock_symbols=[],
            crypto_days=1, crypto_provider=provider1,
        )
        assert await _count(engine) == 10

        # Second fetch overlaps bars 5..9 and adds 10..14.
        provider2 = FakeProvider(
            "fake_ccxt", AssetClass.CRYPTO,
            {"ETH/USDT": [_bar(i, symbol="ETH/USDT") for i in range(5, 15)]},
        )
        stats = await run_backfill(
            engine, crypto_symbols=["ETH/USDT"], stock_symbols=[],
            crypto_days=1, crypto_provider=provider2,
        )
        assert stats["ETH/USDT"].inserted == 5
        assert stats["ETH/USDT"].skipped == 5
        assert await _count(engine) == 15
    finally:
        await engine.dispose()


async def test_symbol_normalization_matches_live_ingestion() -> None:
    engine = await _make_engine()
    try:
        crypto = FakeProvider(
            "fake_ccxt", AssetClass.CRYPTO,
            # Provider hands back dash-separated pairs; storage must be 'BTC/USDT'.
            {"BTC/USDT": [_bar(0, symbol="btc-usdt"), _bar(1, symbol="BTC-USDT")]},
        )
        stock = FakeProvider(
            "fake_yahoo", AssetClass.STOCK,
            # Lower-case ticker in, plain upper-case ticker stored.
            {"AAPL": [_bar(0, symbol="aapl", price=200.0)]},
        )
        await run_backfill(
            engine, crypto_symbols=["BTC/USDT"], stock_symbols=["AAPL"],
            crypto_days=1, crypto_provider=crypto, stock_provider=stock,
        )

        rows = await _all_rows(engine)
        symbols = {r.symbol for r in rows}
        assert symbols == {"BTC/USDT", "AAPL"}
        for r in rows:
            assert r.timeframe == "1m"
            assert isinstance(r.open, Decimal)
            assert r.asset_class == ("crypto" if r.symbol == "BTC/USDT" else "stock")
            assert r.source == ("fake_ccxt" if r.symbol == "BTC/USDT" else "fake_yahoo")
        # sqlite stores naive UTC; the wall-clock instant must match BASE exactly.
        stored_times = {r.time.replace(tzinfo=UTC) for r in rows if r.symbol == "BTC/USDT"}
        assert stored_times == {BASE, BASE + timedelta(minutes=1)}
    finally:
        await engine.dispose()


async def test_dry_run_writes_nothing() -> None:
    engine = await _make_engine()
    try:
        provider = FakeProvider(
            "fake_ccxt", AssetClass.CRYPTO,
            {"BTC/USDT": [_bar(i, symbol="BTC/USDT") for i in range(8)]},
        )
        stats = await run_backfill(
            engine, crypto_symbols=["BTC/USDT"], stock_symbols=[],
            crypto_days=1, dry_run=True, crypto_provider=provider,
        )
        assert stats["BTC/USDT"].fetched == 8
        assert stats["BTC/USDT"].inserted == 0
        assert await _count(engine) == 0
    finally:
        await engine.dispose()


async def test_invalid_and_duplicate_bars_are_dropped() -> None:
    engine = await _make_engine()
    try:
        good = _bar(0, symbol="BTC/USDT")
        dup = _bar(0, symbol="BTC/USDT")  # same timestamp: intra-fetch duplicate
        negative = RawBar(
            time=BASE + timedelta(minutes=1), symbol="BTC/USDT",
            open=-1.0, high=1.0, low=0.5, close=0.9, volume=10.0,
        )
        inconsistent = RawBar(
            time=BASE + timedelta(minutes=2), symbol="BTC/USDT",
            open=100.0, high=90.0, low=95.0, close=100.0, volume=10.0,  # high < low
        )
        provider = FakeProvider(
            "fake_ccxt", AssetClass.CRYPTO,
            {"BTC/USDT": [good, dup, negative, inconsistent]},
        )
        stats = await run_backfill(
            engine, crypto_symbols=["BTC/USDT"], stock_symbols=[],
            crypto_days=1, crypto_provider=provider,
        )
        assert stats["BTC/USDT"].fetched == 4
        assert stats["BTC/USDT"].valid == 1
        assert stats["BTC/USDT"].inserted == 1
        assert await _count(engine) == 1
    finally:
        await engine.dispose()


async def test_multi_batch_run_inserts_everything_once() -> None:
    engine = await _make_engine()
    try:
        n = 23  # not a multiple of the batch size on purpose
        provider = FakeProvider(
            "fake_ccxt", AssetClass.CRYPTO,
            {"SOL/USDT": [_bar(i, symbol="SOL/USDT") for i in range(n)]},
        )
        stats = await run_backfill(
            engine, crypto_symbols=["SOL/USDT"], stock_symbols=[],
            crypto_days=1, batch=7, crypto_provider=provider,
        )
        assert stats["SOL/USDT"].inserted == n
        assert await _count(engine) == n
    finally:
        await engine.dispose()


async def test_fetch_failure_is_isolated_per_symbol() -> None:
    engine = await _make_engine()
    try:
        class ExplodingProvider(FakeProvider):
            async def fetch_historical_bars(
                self, symbol: str, timeframe: str, start: datetime, end: datetime
            ) -> list[RawBar]:
                if symbol == "BAD/USDT":
                    raise ConnectionError("exchange down")
                return await super().fetch_historical_bars(symbol, timeframe, start, end)

        provider = ExplodingProvider(
            "fake_ccxt", AssetClass.CRYPTO,
            {"BTC/USDT": [_bar(0, symbol="BTC/USDT")]},
        )
        stats = await run_backfill(
            engine, crypto_symbols=["BAD/USDT", "BTC/USDT"], stock_symbols=[],
            crypto_days=1, crypto_provider=provider,
        )
        assert stats["BAD/USDT"].error is not None
        assert stats["BAD/USDT"].inserted == 0
        assert stats["BTC/USDT"].inserted == 1
        assert await _count(engine) == 1
    finally:
        await engine.dispose()


async def test_rejects_nonpositive_batch() -> None:
    engine = await _make_engine()
    try:
        provider = FakeProvider("fake_ccxt", AssetClass.CRYPTO, {})
        with pytest.raises(ValueError, match="batch"):
            await run_backfill(
                engine, crypto_symbols=["BTC/USDT"], stock_symbols=[],
                crypto_days=1, batch=0, crypto_provider=provider,
            )
    finally:
        await engine.dispose()
