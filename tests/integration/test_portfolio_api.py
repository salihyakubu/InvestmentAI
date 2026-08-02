"""The dashboard's KPI contract, pinned at the wire.

The six summary tiles rendered fields the API never sent, and the frontend
normaliser turned every one of them into a confident 0.00 -- a losing account
showed a 0.00% return and a 0.0% win rate. These tests pin what actually
crosses the wire: real derived metrics when the history supports them, nulls
(never zeros) when it does not, and a chronological equity series.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.pool import NullPool

import api.routers.portfolio as portfolio_router
from api.dependencies import get_current_user, get_db
from api.main import app
from core.enums import OrderStatus
from core.models.base import AsyncBase
from core.models.market_data import OHLCVRecord
from core.models.orders import Fill, Order
from core.models.portfolio import PortfolioSnapshot


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):  # noqa: ANN001, ANN003, ANN201, ARG001
    return "JSON"


_TABLES = [
    PortfolioSnapshot.__table__,
    Order.__table__,
    Fill.__table__,
    OHLCVRecord.__table__,
]


@pytest.fixture
def client():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}", poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _setup() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(AsyncBase.metadata.create_all, tables=_TABLES)

    asyncio.run(_setup())

    async def _override_db():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_user] = lambda: {"user_id": "t", "role": "admin"}
    # Metrics are cached per trading mode for a minute; each test needs a
    # clean slate or it would read the previous test's account.
    portfolio_router._metrics_cache.clear()
    try:
        with TestClient(app) as c:
            yield c, factory
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_current_user, None)
        portfolio_router._metrics_cache.clear()
        asyncio.run(engine.dispose())
        os.unlink(path)


async def _seed(factory, equities: list[tuple[datetime, str]], fills: list[dict[str, Any]]):
    async with factory() as session:
        for when, equity in equities:
            session.add(
                PortfolioSnapshot(
                    time=when,
                    trading_mode="paper",
                    total_equity=Decimal(equity),
                    cash=Decimal(equity),
                    positions_value=Decimal("0"),
                    unrealized_pnl=Decimal("0"),
                    realized_pnl=Decimal("0"),
                    position_count=0,
                )
            )
        for spec in fills:
            order = Order(
                external_id=spec["tag"],
                symbol=spec["symbol"],
                asset_class="crypto",
                side=spec["side"],
                order_type="market",
                quantity=Decimal(spec["qty"]),
                status=OrderStatus.FILLED.value,
                trading_mode="paper",
                filled_at=spec["at"],
                created_at=spec["at"],
                updated_at=spec["at"],
            )
            session.add(order)
            await session.flush()
            session.add(
                Fill(
                    order_id=order.id,
                    price=Decimal(spec["price"]),
                    quantity=Decimal(spec["qty"]),
                    commission=Decimal("0"),
                    filled_at=spec["at"],
                    created_at=spec["at"],
                )
            )
        await session.commit()


def test_summary_carries_every_field_the_dashboard_renders(client) -> None:
    c, factory = client
    now = datetime.now(UTC)
    asyncio.run(
        _seed(
            factory,
            [
                (now - timedelta(days=2), "100.00"),
                (now - timedelta(days=1), "99.00"),
                (now, "98.85"),
            ],
            [
                {"tag": "explore-1", "symbol": "BTC/USDT", "side": "buy",
                 "qty": "1", "price": "100", "at": now - timedelta(hours=3)},
                {"tag": "explore-2", "symbol": "BTC/USDT", "side": "sell",
                 "qty": "1", "price": "110", "at": now - timedelta(hours=2)},
            ],
        )
    )

    body = c.get("/api/v1/portfolio/summary").json()
    for field in (
        "daily_pnl", "daily_pnl_pct", "total_return", "total_return_pct",
        "sharpe_ratio", "max_drawdown", "win_rate", "closed_trades",
        "position_count",
    ):
        assert field in body, f"{field} missing from the summary contract"

    # Real numbers, not zeros: 100.00 -> 98.85 is -1.15%.
    assert body["total_return_pct"] == pytest.approx(-1.15)
    assert float(body["total_return"]) == pytest.approx(-1.15)
    assert body["max_drawdown"] == pytest.approx(0.0115)
    # One closed round trip, and it won.
    assert body["closed_trades"] == 1
    assert body["win_rate"] == pytest.approx(1.0)


def test_metrics_the_history_cannot_support_are_null_not_zero(client) -> None:
    """The honesty gate: three days of soak cannot yield a Sharpe ratio."""
    c, factory = client
    now = datetime.now(UTC)
    asyncio.run(
        _seed(
            factory,
            [(now - timedelta(days=i), f"{100 - i * 0.1:.2f}") for i in range(3, 0, -1)],
            [],
        )
    )
    body = c.get("/api/v1/portfolio/summary").json()
    assert body["sharpe_ratio"] is None
    assert body["win_rate"] is None  # no closed round trips
    assert body["closed_trades"] == 0
    # But the metrics it CAN support are present and non-null.
    assert body["total_return_pct"] is not None


def test_summary_without_any_history_is_all_nulls(client) -> None:
    c, _ = client
    body = c.get("/api/v1/portfolio/summary").json()
    assert body["sharpe_ratio"] is None
    assert body["max_drawdown"] is None
    assert body["win_rate"] is None
    assert body["total_return_pct"] is None
    assert float(body["total_equity"]) == 0.0


def test_snapshots_are_chronological(client) -> None:
    """Served newest-first, the equity curve rendered mirrored in time and the
    Risk page measured drawdown against future equity."""
    c, factory = client
    now = datetime.now(UTC)
    asyncio.run(
        _seed(
            factory,
            [
                (now - timedelta(hours=3), "100.00"),
                (now - timedelta(hours=2), "99.00"),
                (now - timedelta(hours=1), "101.00"),
            ],
            [],
        )
    )
    rows = c.get("/api/v1/portfolio/snapshots").json()
    times = [r["time"] for r in rows]
    assert times == sorted(times)
    assert [r["total_equity"] for r in rows] == [100.0, 99.0, 101.0]


def test_snapshot_window_is_expressed_in_days_and_downsampled(client) -> None:
    c, factory = client
    now = datetime.now(UTC)
    # 400 five-minute snapshots ~ 33 hours: more than any row cap would show.
    asyncio.run(
        _seed(
            factory,
            [(now - timedelta(minutes=5 * i), "100.00") for i in range(400, 0, -1)],
            [],
        )
    )
    rows = c.get("/api/v1/portfolio/snapshots", params={"max_points": 50}).json()
    assert len(rows) <= 50
    times = [r["time"] for r in rows]
    assert times == sorted(times)
    # Downsampling must keep the newest observation: the curve has to end now.
    newest = c.get("/api/v1/portfolio/snapshots", params={"max_points": 5000}).json()
    assert rows[-1]["time"] == newest[-1]["time"]


def test_old_snapshots_fall_outside_the_default_window(client) -> None:
    c, factory = client
    now = datetime.now(UTC)
    asyncio.run(
        _seed(
            factory,
            [(now - timedelta(days=200), "50.00"), (now, "100.00")],
            [],
        )
    )
    rows = c.get("/api/v1/portfolio/snapshots").json()
    assert [r["total_equity"] for r in rows] == [100.0]
    wide = c.get("/api/v1/portfolio/snapshots", params={"days": 365}).json()
    assert [r["total_equity"] for r in wide] == [50.0, 100.0]


def test_benchmark_starts_at_account_equity_and_tracks_holdings(client) -> None:
    """The do-nothing benchmark: normalised to the account's equity at
    inception, moving only with the held symbols' prices."""
    from api.routers.portfolio import BENCHMARK_INCEPTION

    c, factory = client
    t0 = BENCHMARK_INCEPTION

    async def _seed() -> None:
        async with factory() as session:
            for hours, equity in ((0, "100.00"), (6, "99.50"), (12, "99.00")):
                session.add(
                    PortfolioSnapshot(
                        time=t0 + timedelta(hours=hours),
                        trading_mode="paper",
                        total_equity=Decimal(equity),
                        cash=Decimal(equity),
                        positions_value=Decimal("0"),
                        unrealized_pnl=Decimal("0"),
                        realized_pnl=Decimal("0"),
                        position_count=0,
                    )
                )
            # Two benchmark symbols with bars: both +10% by hour 12.
            for symbol, base in (("BTC/USDT", 50000.0), ("ETH/USDT", 2000.0)):
                for hours, mult in ((-1, 1.0), (6, 1.05), (12, 1.10)):
                    when = t0 + timedelta(hours=hours)
                    price = Decimal(str(base * mult))
                    session.add(
                        OHLCVRecord(
                            time=when, symbol=symbol, timeframe="1m",
                            open=price, high=price, low=price, close=price,
                            volume=Decimal("1"), source="test",
                            asset_class="crypto", ingested_at=when,
                        )
                    )
            await session.commit()

    asyncio.run(_seed())
    rows = c.get("/api/v1/portfolio/benchmark", params={"days": 30}).json()
    assert len(rows) == 3
    # Starts at the account's inception equity ($100), NOT at the asset price.
    assert rows[0]["benchmark_equity"] == pytest.approx(100.0)
    # Both symbols +10% -> benchmark +10%, regardless of what the account did.
    assert rows[-1]["benchmark_equity"] == pytest.approx(110.0)
    times = [r["time"] for r in rows]
    assert times == sorted(times)


def test_benchmark_refuses_to_exist_on_one_symbol(client) -> None:
    """A single-symbol 'benchmark' is not the registered one; serve nothing
    rather than something that looks like the benchmark and is not."""
    from api.routers.portfolio import BENCHMARK_INCEPTION

    c, factory = client
    t0 = BENCHMARK_INCEPTION

    async def _seed() -> None:
        async with factory() as session:
            session.add(
                PortfolioSnapshot(
                    time=t0 + timedelta(hours=1), trading_mode="paper",
                    total_equity=Decimal("100"), cash=Decimal("100"),
                    positions_value=Decimal("0"), unrealized_pnl=Decimal("0"),
                    realized_pnl=Decimal("0"), position_count=0,
                )
            )
            when = t0
            session.add(
                OHLCVRecord(
                    time=when, symbol="BTC/USDT", timeframe="1m",
                    open=Decimal("50000"), high=Decimal("50000"),
                    low=Decimal("50000"), close=Decimal("50000"),
                    volume=Decimal("1"), source="test",
                    asset_class="crypto", ingested_at=when,
                )
            )
            await session.commit()

    asyncio.run(_seed())
    assert c.get("/api/v1/portfolio/benchmark").json() == []
