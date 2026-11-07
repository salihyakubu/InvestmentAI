"""Order state survives the process (GO_LIVE.md 2026-11-02, build item 5).

Before this change the order manager and the in-flight map were process
memory: an order submitted and filled by the paper broker, then lost to a
restart before the 2 s poll, left the broker's book changed and the ledger
ignorant -- the mechanism behind the ~0.6% of closes missing since July. These
tests pin the new contract on SQLite + the in-process bus: a row before the
venue is called, exactly-once reconciliation after a restart, restart-safe
idempotency for redelivered approvals, fill dedupe, and a loud record for an
order nobody can account for.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.pool import StaticPool

from config.settings import Settings
from core.enums import OrderStatus
from core.events.base import Event, InProcessEventBus
from core.events.order_events import OrderFilledEvent
from core.events.risk_events import RiskApprovedEvent
from core.events.streams import ORDERS
from core.models.base import AsyncBase
from core.models.orders import Fill, Order
from core.models.portfolio import PortfolioSnapshot
from core.models.positions import Position
from core.models.safety import SafetyState
from services.execution.brokers.base import BrokerOrder
from services.execution.brokers.paper_broker import PaperBroker
from services.execution.service import ExecutionEngineService
from services.persistence.broker_state import BrokerStateStore
from services.persistence.order_sync import OrderPersistenceService

pytestmark = pytest.mark.asyncio

_TABLES = [
    Order.__table__,
    Fill.__table__,
    PortfolioSnapshot.__table__,
    Position.__table__,
    SafetyState.__table__,
]


@compiles(JSONB, "sqlite")
def _jsonb_as_json(type_: JSONB, compiler: Any, **kw: Any) -> str:
    return "JSON"


async def _factory():
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.run_sync(AsyncBase.metadata.create_all, tables=_TABLES)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


def _broker() -> PaperBroker:
    broker = PaperBroker(initial_cash=Decimal("100000"))
    broker.update_price("AAPL", Decimal("100"))
    return broker


async def _die_before_publish(*_: Any, **__: Any) -> None:
    raise RuntimeError("process killed between broker fill and publish")


async def _fill_rows(factory) -> list[Fill]:
    async with factory() as session:
        return list((await session.execute(select(Fill))).scalars().all())


class _RowCheckingBroker(PaperBroker):
    """Records whether the durable row existed when the venue was called."""

    def __init__(self, factory, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._factory = factory
        self.row_present_at_submit: list[bool] = []

    async def submit_order(self, order: BrokerOrder) -> str:
        async with self._factory() as session:
            row = await session.get(Order, uuid.UUID(order.external_id))
            self.row_present_at_submit.append(row is not None and row.broker_name == "paper")
        return await super().submit_order(order)


async def test_order_row_exists_before_the_venue_is_called(mock_settings: Settings) -> None:
    engine, factory = await _factory()
    broker = _RowCheckingBroker(factory, initial_cash=Decimal("100000"))
    broker.update_price("AAPL", Decimal("100"))
    ex = ExecutionEngineService(
        event_bus=InProcessEventBus(), settings=mock_settings,
        brokers={"paper": broker}, session_factory=factory,
    )
    order = await ex.submit_order(
        symbol="AAPL", side="buy", order_type="market", quantity=Decimal("5"),
        reference_price=Decimal("100"), client_order_id="rebal-evt-AAPL",
    )
    assert order is not None
    assert broker.row_present_at_submit == [True]
    async with factory() as session:
        row = await session.get(Order, uuid.UUID(order.order_id))
        assert row is not None
        assert row.status == OrderStatus.SUBMITTED.value  # no persistence consumer here
        assert row.correlation_id == "rebal-evt-AAPL"
        assert row.external_id == order.order_id  # paper venue echoes the client id
        assert row.submitted_at is not None
    await engine.dispose()


async def test_restart_between_fill_and_publish_is_reconciled_exactly_once(
    mock_settings: Settings,
) -> None:
    engine, factory = await _factory()
    bus = InProcessEventBus()
    broker = _broker()

    # Engine A dies between the broker filling and the fill being published.
    ex_a = ExecutionEngineService(
        event_bus=bus, settings=mock_settings, brokers={"paper": broker}, session_factory=factory
    )
    ex_a._record_and_publish_fill = _die_before_publish  # type: ignore[method-assign]
    order = await ex_a.submit_order(
        symbol="AAPL", side="buy", order_type="market", quantity=Decimal("5"),
        reference_price=Decimal("100"),
    )
    assert order is not None
    assert [Decimal(p["quantity"]) for p in await broker.get_positions()] == [Decimal("5")]
    assert await _fill_rows(factory) == []  # ledger knows nothing yet
    del ex_a

    # Engine B boots on the same durable store; the venue still knows the order.
    fills: list[Event] = []

    async def capture(event: Event) -> None:
        if event.event_type == "OrderFilledEvent":
            fills.append(event)

    await bus.subscribe(ORDERS, "t", "t", capture)
    persistence = OrderPersistenceService(event_bus=bus, session_factory=factory)
    await persistence.start()
    ex_b = ExecutionEngineService(
        event_bus=bus, settings=mock_settings, brokers={"paper": broker}, session_factory=factory
    )
    summary = await ex_b.reconcile_on_start()
    assert summary == {"rows": 1, "filled": 1, "open": 0, "closed": 0, "lost": 0}
    assert len(fills) == 1 and fills[0].order_id == order.order_id

    rows = await _fill_rows(factory)
    assert len(rows) == 1 and rows[0].quantity == Decimal("5")
    assert rows[0].external_fill_id  # the venue fill id travelled with the event
    async with factory() as session:
        assert (await session.get(Order, uuid.UUID(order.order_id))).status == "filled"

    # Nothing non-terminal is left, so a second boot changes nothing.
    assert (await ex_b.reconcile_on_start())["rows"] == 0
    assert len(await _fill_rows(factory)) == 1
    await engine.dispose()


async def test_redelivered_risk_approval_after_restart_submits_once(
    mock_settings: Settings,
) -> None:
    engine, factory = await _factory()
    bus = InProcessEventBus()
    broker = _broker()
    event = RiskApprovedEvent(
        order_id="rebal-evt-1-AAPL", symbol="AAPL", side="buy", order_type="market",
        quantity=5.0, reference_price=100.0, source_service="risk_manager",
    )
    ex_a = ExecutionEngineService(
        event_bus=bus, settings=mock_settings, brokers={"paper": broker}, session_factory=factory
    )
    await ex_a._handle_risk_approved(event)
    assert [Decimal(p["quantity"]) for p in await broker.get_positions()] == [Decimal("5")]

    # XAUTOCLAIM redelivers the same approval to a fresh process after a crash.
    ex_b = ExecutionEngineService(
        event_bus=bus, settings=mock_settings, brokers={"paper": broker}, session_factory=factory
    )
    await ex_b._handle_risk_approved(event)
    held = [Decimal(p["quantity"]) for p in await broker.get_positions()]
    assert held == [Decimal("5")]  # not 10
    await engine.dispose()


async def test_duplicate_fill_event_persists_one_fill_row(mock_settings: Settings) -> None:
    engine, factory = await _factory()
    row_id = uuid.uuid4()
    async with factory() as session:
        from datetime import UTC, datetime

        now = datetime.now(UTC)
        session.add(
            Order(
                id=row_id, symbol="AAPL", asset_class="stock", side="buy", order_type="market",
                quantity=Decimal("5"), status=OrderStatus.SUBMITTED.value, trading_mode="paper",
                created_at=now, updated_at=now,
            )
        )
        await session.commit()
    svc = OrderPersistenceService(event_bus=InProcessEventBus(), session_factory=factory)
    event = OrderFilledEvent(
        order_id=str(row_id), symbol="AAPL", side="buy", fill_price=100.0, fill_quantity=5.0,
        fill_id="venue-fill-1", source_service="execution-engine",
    )
    await svc.handle_event(event)
    await svc.handle_event(event)  # at-least-once delivery
    rows = await _fill_rows(factory)
    assert len(rows) == 1 and rows[0].external_fill_id == "venue-fill-1"
    await engine.dispose()


async def test_order_the_restarted_venue_cannot_account_for_is_expired_loudly(
    mock_settings: Settings,
) -> None:
    engine, factory = await _factory()
    bus = InProcessEventBus()
    ex_a = ExecutionEngineService(
        event_bus=bus, settings=mock_settings, brokers={"paper": _broker()}, session_factory=factory
    )
    ex_a._record_and_publish_fill = _die_before_publish  # type: ignore[method-assign]
    order = await ex_a.submit_order(
        symbol="AAPL", side="buy", order_type="market", quantity=Decimal("5"),
        reference_price=Decimal("100"),
    )
    assert order is not None

    cancelled: list[Event] = []

    async def capture(event: Event) -> None:
        if event.event_type == "OrderCancelledEvent":
            cancelled.append(event)

    await bus.subscribe(ORDERS, "t", "t", capture)
    # A fresh paper broker: the simulated venue has no memory across a restart.
    ex_b = ExecutionEngineService(
        event_bus=bus, settings=mock_settings, brokers={"paper": _broker()}, session_factory=factory
    )
    summary = await ex_b.reconcile_on_start()
    assert summary["lost"] == 1 and summary["filled"] == 0
    assert len(cancelled) == 1 and getattr(cancelled[0], "reason", "") == "lost_on_restart"
    async with factory() as session:
        assert (await session.get(Order, uuid.UUID(order.order_id))).status == "expired"
    assert await _fill_rows(factory) == []  # never a fabricated fill
    await engine.dispose()


async def test_paper_broker_tracks_realised_pnl_per_position() -> None:
    broker = PaperBroker(initial_cash=Decimal("100000"))
    broker.apply_external_fill("AAPL", "buy", Decimal("100"), Decimal("10"))
    assert broker.realized_by_symbol()["AAPL"] == Decimal("0")
    broker.apply_external_fill("AAPL", "sell", Decimal("110"), Decimal("4"))
    assert broker.realized_by_symbol()["AAPL"] == Decimal("40")
    broker.apply_external_fill("AAPL", "sell", Decimal("90"), Decimal("6"))
    assert broker._positions["AAPL"] == Decimal("0")
    assert broker.realized_by_symbol()["AAPL"] == Decimal("-20")  # kept after going flat
    broker.apply_external_fill("AAPL", "buy", Decimal("50"), Decimal("2"))
    assert broker.realized_by_symbol()["AAPL"] == Decimal("0")  # a new position starts fresh


async def test_checkpoint_writes_realised_pnl_on_open_and_closed_rows() -> None:
    engine, factory = await _factory()
    store = BrokerStateStore(session_factory=factory, trading_mode="paper")
    open_book = [{
        "symbol": "AAPL", "quantity": "6", "avg_entry_price": "100", "current_price": "110",
        "market_value": "660", "unrealized_pnl": "60", "realized_pnl": "40",
    }]
    async with factory() as session:
        await store.checkpoint(open_book, session, realized_by_symbol={"AAPL": Decimal("40")})
        await session.commit()
    async with factory() as session:
        row = (await session.execute(select(Position))).scalar_one()
        assert row.realized_pnl == Decimal("40") and row.closed_at is None

    async with factory() as session:
        await store.checkpoint([], session, realized_by_symbol={"AAPL": Decimal("-20")})
        await session.commit()
    async with factory() as session:
        row = (await session.execute(select(Position))).scalar_one()
        assert row.closed_at is not None
        assert row.realized_pnl == Decimal("-20")  # the broker's exact figure, not 0.00
    await engine.dispose()
