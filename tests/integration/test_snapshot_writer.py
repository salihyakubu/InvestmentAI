"""PortfolioSnapshotWriter: periodic portfolio_snapshots rows so the dashboard
equity curve has data to render during the paper soak.
"""

from __future__ import annotations

import asyncio
import contextlib
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.pool import StaticPool

from core.models.base import AsyncBase
from core.models.portfolio import PortfolioSnapshot
from services.persistence.snapshot_writer import PortfolioSnapshotWriter


@compiles(JSONB, "sqlite")
def _jsonb_as_json(type_: JSONB, compiler: Any, **kw: Any) -> str:
    """sqlite can't render JSONB (the snapshots table has an ``allocations``
    column); render it as plain JSON for the in-memory test schema."""
    return "JSON"


def _make_factory():
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def _create_tables(engine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(
            AsyncBase.metadata.create_all, tables=[PortfolioSnapshot.__table__]
        )


async def _all_snapshots(factory) -> list[PortfolioSnapshot]:
    async with factory() as session:
        result = await session.execute(
            select(PortfolioSnapshot).order_by(PortfolioSnapshot.time)
        )
        return list(result.scalars().all())


@pytest.mark.asyncio
async def test_write_once_inserts_snapshot_row() -> None:
    engine, factory = _make_factory()
    await _create_tables(engine)

    writer = PortfolioSnapshotWriter(session_factory=factory, trading_mode="paper")
    await writer.write_once(
        equity=100500.25,
        cash=Decimal("50000"),
        positions_value=50500.25,
        unrealized_pnl=Decimal("500.25"),
        position_count=2,
    )

    rows = await _all_snapshots(factory)
    assert len(rows) == 1
    snap = rows[0]
    assert snap.trading_mode == "paper"
    assert snap.total_equity == Decimal("100500.25")
    assert snap.cash == Decimal("50000")
    assert snap.positions_value == Decimal("50500.25")
    assert snap.unrealized_pnl == Decimal("500.25")
    assert snap.realized_pnl == Decimal("0")
    assert snap.position_count == 2
    assert snap.time is not None
    await engine.dispose()


@pytest.mark.asyncio
async def test_run_loop_writes_snapshots_and_cancels_cleanly() -> None:
    engine, factory = _make_factory()
    await _create_tables(engine)

    async def account_provider() -> dict[str, Any]:
        return {"equity": "101000.00", "cash": "50000.00", "positions_value": "51000.00"}

    async def positions_provider() -> list[dict[str, Any]]:
        return [
            {"symbol": "AAPL", "quantity": "10", "market_value": "51000.00",
             "unrealized_pnl": "1000"},
        ]

    writer = PortfolioSnapshotWriter(session_factory=factory, trading_mode="paper")
    task = asyncio.create_task(
        writer.run(account_provider, positions_provider, interval_seconds=0.01)
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    rows = await _all_snapshots(factory)
    assert len(rows) >= 1
    snap = rows[0]
    assert snap.trading_mode == "paper"
    assert snap.total_equity == Decimal("101000")
    assert snap.positions_value == Decimal("51000")
    assert snap.unrealized_pnl == Decimal("1000")
    assert snap.position_count == 1
    await engine.dispose()


@pytest.mark.asyncio
async def test_run_skips_iteration_when_account_is_none() -> None:
    engine, factory = _make_factory()
    await _create_tables(engine)
    calls = {"n": 0}

    async def account_provider() -> dict[str, Any] | None:
        calls["n"] += 1
        return None

    writer = PortfolioSnapshotWriter(session_factory=factory, trading_mode="paper")
    task = asyncio.create_task(writer.run(account_provider, interval_seconds=0.01))
    await asyncio.sleep(0.05)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    # The loop kept polling (no crash) but never wrote a row.
    assert calls["n"] >= 2
    assert await _all_snapshots(factory) == []
    await engine.dispose()
