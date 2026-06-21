"""DB-layer tests for the money-path ORM models (orders + fills).

Exercises real ORM persistence on an in-memory SQLite database: insert, query,
the order->fills relationship, Decimal round-trip, and status updates. The
JSONB-bearing tables (audit/portfolio/risk/users/predictions) are PostgreSQL-only
and excluded here; this covers the order/fill path that actually moves money.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from core.enums import OrderSide, OrderStatus, OrderType
from core.models.base import AsyncBase
from core.models.orders import Fill, Order


def _make_session_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    return engine


async def _create_money_tables(engine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(
            AsyncBase.metadata.create_all,
            tables=[Order.__table__, Fill.__table__],
        )


@pytest.mark.asyncio
async def test_order_and_fill_round_trip() -> None:
    engine = _make_session_factory()
    await _create_money_tables(engine)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC)
    try:
        async with session_factory() as s:
            order = Order(
                symbol="AAPL",
                asset_class="stock",
                side=OrderSide.BUY.value,
                order_type=OrderType.MARKET.value,
                quantity=Decimal("10"),
                status=OrderStatus.PENDING.value,
                trading_mode="paper",
                created_at=now,
                updated_at=now,
            )
            s.add(order)
            await s.commit()
            await s.refresh(order)
            order_id = order.id
            assert order_id is not None

            s.add(
                Fill(
                    order_id=order_id,
                    price=Decimal("150.25"),
                    quantity=Decimal("10"),
                    commission=Decimal("0"),
                    filled_at=now,
                    created_at=now,
                )
            )
            await s.commit()

        # Fresh session: read back and verify persistence + relationship.
        async with session_factory() as s:
            fetched = await s.get(Order, order_id)
            assert fetched is not None
            assert fetched.symbol == "AAPL"
            assert fetched.quantity == Decimal("10")  # Decimal preserved, not float

            fills = (
                await s.execute(select(Fill).where(Fill.order_id == order_id))
            ).scalars().all()
            assert len(fills) == 1
            assert fills[0].price == Decimal("150.25")
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_order_status_update_persists() -> None:
    engine = _make_session_factory()
    await _create_money_tables(engine)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC)
    try:
        async with session_factory() as s:
            order = Order(
                symbol="BTC/USDT",
                asset_class="crypto",
                side=OrderSide.SELL.value,
                order_type=OrderType.MARKET.value,
                quantity=Decimal("1"),
                status=OrderStatus.PENDING.value,
                trading_mode="paper",
                created_at=now,
                updated_at=now,
            )
            s.add(order)
            await s.commit()
            await s.refresh(order)
            order_id = order.id

            order.status = OrderStatus.FILLED.value
            await s.commit()

        async with session_factory() as s:
            fetched = await s.get(Order, order_id)
            assert fetched.status == OrderStatus.FILLED.value
    finally:
        await engine.dispose()
