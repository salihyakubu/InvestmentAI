"""OrderPersistenceService: execution fill/reject events update the DB order row
and record fills, keyed by client_order_id (the originating DB order id).
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from core.enums import OrderStatus
from core.events.base import InProcessEventBus
from core.events.order_events import OrderFilledEvent, OrderRejectedEvent
from core.models.base import AsyncBase
from core.models.orders import Fill, Order
from services.persistence.order_sync import OrderPersistenceService


def _make_factory():
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def _create_tables(engine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(
            AsyncBase.metadata.create_all, tables=[Order.__table__, Fill.__table__]
        )


async def _seed_pending_order(factory):
    now = datetime.now(UTC)
    async with factory() as session:
        order = Order(
            symbol="AAPL", asset_class="stock", side="buy", order_type="market",
            quantity=Decimal("5"), status=OrderStatus.PENDING.value, trading_mode="paper",
            created_at=now, updated_at=now,
        )
        session.add(order)
        await session.commit()
        await session.refresh(order)
        return order.id


@pytest.mark.asyncio
async def test_fill_event_persists_status_and_fill() -> None:
    engine, factory = _make_factory()
    await _create_tables(engine)
    order_id = await _seed_pending_order(factory)

    svc = OrderPersistenceService(event_bus=InProcessEventBus(), session_factory=factory)
    await svc.handle_event(
        OrderFilledEvent(
            order_id="exec-1", client_order_id=str(order_id),
            fill_price=100.0, fill_quantity=5.0, commission=0.0,
            symbol="AAPL", side="buy", source_service="execution",
        )
    )

    async with factory() as session:
        order = await session.get(Order, order_id)
        assert order.status == OrderStatus.FILLED.value
        assert order.filled_at is not None
        fills = (
            await session.execute(select(Fill).where(Fill.order_id == order_id))
        ).scalars().all()
        assert len(fills) == 1
        assert fills[0].price == Decimal("100")
        assert fills[0].quantity == Decimal("5")
    await engine.dispose()


@pytest.mark.asyncio
async def test_rejected_event_persists_status() -> None:
    engine, factory = _make_factory()
    await _create_tables(engine)
    order_id = await _seed_pending_order(factory)

    svc = OrderPersistenceService(event_bus=InProcessEventBus(), session_factory=factory)
    await svc.handle_event(
        OrderRejectedEvent(
            order_id="exec-1", reason="insufficient_buying_power",
            client_order_id=str(order_id), source_service="execution",
        )
    )

    async with factory() as session:
        order = await session.get(Order, order_id)
        assert order.status == OrderStatus.REJECTED.value
    await engine.dispose()


@pytest.mark.asyncio
async def test_non_db_correlation_id_is_ignored() -> None:
    engine, factory = _make_factory()
    await _create_tables(engine)
    svc = OrderPersistenceService(event_bus=InProcessEventBus(), session_factory=factory)

    # A rebalance correlation id is not a UUID -> safely ignored (no error).
    await svc.handle_event(
        OrderFilledEvent(
            order_id="exec-1", client_order_id="rebal-xyz-AAPL",
            fill_price=1.0, fill_quantity=1.0, symbol="AAPL", side="buy",
            source_service="execution",
        )
    )
    # No client_order_id at all -> also ignored.
    await svc.handle_event(
        OrderFilledEvent(
            order_id="exec-2", fill_price=1.0, fill_quantity=1.0,
            symbol="AAPL", side="buy", source_service="execution",
        )
    )
    await engine.dispose()  # reaching here without raising is the assertion
