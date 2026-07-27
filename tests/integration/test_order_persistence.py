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

@pytest.mark.asyncio
async def test_autonomous_fill_inserts_completed_order_row() -> None:
    """Fills carrying a rebalance/exploration correlation id (not a DB order
    id) must INSERT a completed row -- before this, every autonomous trade was
    invisible to the orders table and the audit trail."""
    engine, factory = _make_factory()
    await _create_tables(engine)

    svc = OrderPersistenceService(
        event_bus=InProcessEventBus(), session_factory=factory, trading_mode="paper"
    )
    await svc.handle_event(
        OrderFilledEvent(
            order_id="exec-1",
            symbol="SOL/USDT",
            side="buy",
            fill_price=76.08,
            fill_quantity=0.0394,
            commission=0.01,
            client_order_id="explore-abc123-SOL/USDT",
            source_service="execution-engine",
        )
    )

    async with factory() as session:
        orders = (await session.execute(select(Order))).scalars().all()
        fills = (await session.execute(select(Fill))).scalars().all()
    assert len(orders) == 1 and len(fills) == 1
    order = orders[0]
    assert order.symbol == "SOL/USDT"
    assert order.status == "filled"
    assert order.asset_class == "crypto"
    assert order.trading_mode == "paper"
    assert order.external_id == "explore-abc123-SOL/USDT"  # audit tag preserved
    assert fills[0].order_id == order.id
    await engine.dispose()

