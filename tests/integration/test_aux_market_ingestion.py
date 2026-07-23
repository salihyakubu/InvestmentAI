"""AuxMarketService end-to-end against a real (sqlite) aux_market_state table:
a single poll persists every source's metric, and a HistoricalAuxProvider built
from those rows reproduces them via as-of lookup.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from config.settings import Settings
from core.models.base import AsyncBase
from core.models.market_data import AuxMarketState
from services.data_ingestion.aux_market import AuxMarketService
from services.feature_engineering.aux_features import HistoricalAuxProvider, LiveAuxProvider

_BASE = datetime(2024, 1, 1, tzinfo=UTC)


def _make_factory() -> tuple[Any, async_sessionmaker[Any]]:
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def _create_tables(engine: Any) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(
            AsyncBase.metadata.create_all, tables=[AuxMarketState.__table__]
        )


class _SuccessHTTPClient:
    async def __aenter__(self) -> _SuccessHTTPClient:
        return self

    async def __aexit__(self, *_: Any) -> bool:
        return False

    async def get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        if "premiumIndex" in url:
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"lastFundingRate": "0.00012"},
            )
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"data": [{"value": "55"}]},
        )


class _FakeYahoo:
    async def fetch_historical_bars(
        self, symbol: str, timeframe: str, start: Any, end: Any
    ) -> list[Any]:
        if symbol == "^VIX":
            return [SimpleNamespace(close=20.0, time=_BASE)]
        if symbol == "SPY":
            return [
                SimpleNamespace(close=100.0, time=_BASE),
                SimpleNamespace(close=101.0, time=_BASE + timedelta(days=1)),
            ]
        return []


@pytest.mark.asyncio
async def test_poll_once_persists_all_sources_and_asof_reads_back() -> None:
    engine, factory = _make_factory()
    await _create_tables(engine)

    svc = AuxMarketService(
        session_factory=factory,
        live_provider=LiveAuxProvider(),
        settings=Settings(active_symbols_crypto=["BTC/USDT", "ETH/USDT"]),
        http_client_factory=_SuccessHTTPClient,  # type: ignore[arg-type]
        yahoo_provider=_FakeYahoo(),  # type: ignore[arg-type]
    )
    await svc.poll_once()

    async with factory() as session:
        rows = list((await session.execute(select(AuxMarketState))).scalars().all())

    by_key = {(r.metric, r.symbol): r.value for r in rows}
    assert by_key[("funding_rate", "BTC/USDT")] == pytest.approx(0.00012)
    assert by_key[("funding_rate", "ETH/USDT")] == pytest.approx(0.00012)
    assert by_key[("fear_greed", "")] == 55.0
    assert by_key[("vix_close", "")] == 20.0
    assert by_key[("spy_daily_return", "")] == pytest.approx(0.01)

    # Rebuild the historical provider from the persisted rows and read as-of a
    # time after everything was written (derived from stored times so tz-awareness
    # matches regardless of the sqlite round-trip).
    tuples = [(r.time, r.metric, r.symbol, r.value) for r in rows]
    ts = max(r.time for r in rows) + timedelta(seconds=1)
    provider = HistoricalAuxProvider(tuples)

    btc = provider.features_asof("BTC/USDT", ts)
    assert btc["aux_funding_rate"] == pytest.approx(0.00012)
    assert btc["aux_fear_greed"] == 55.0
    assert btc["aux_vix_close"] == 20.0
    assert btc["aux_spy_daily_return"] == pytest.approx(0.01)
    # A stock symbol shares the globals but has no funding row.
    assert provider.features_asof("AAPL", ts)["aux_funding_rate"] == 0.0

    await engine.dispose()
