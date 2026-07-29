"""Paper-broker account state must survive a restart.

The broker holds cash, positions and cost basis in process memory. Before the
checkpoint/restore pair existed, every deploy silently rebased equity to
``initial_capital`` and orphaned the open book -- observed live on
2026-07-27T20:40:43Z, when a redeploy moved equity 98.8588 -> 100.0000 and
erased five days of soak P&L. These tests pin the restart contract: an exact
restore from a checkpoint, replay of fills that landed after it, continuity
when no checkpoint exists, and isolation between trading modes.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.pool import StaticPool

from core.enums import OrderStatus
from core.models.base import AsyncBase
from core.models.orders import Fill, Order
from core.models.portfolio import PortfolioSnapshot
from core.models.positions import Position
from services.execution.brokers.base import BrokerOrder
from services.execution.brokers.paper_broker import PaperBroker
from services.persistence.broker_state import BrokerStateStore
from services.persistence.snapshot_writer import PortfolioSnapshotWriter

pytestmark = pytest.mark.asyncio

_TABLES = [
    PortfolioSnapshot.__table__,
    Position.__table__,
    Order.__table__,
    Fill.__table__,
]


@compiles(JSONB, "sqlite")
def _jsonb_as_json(type_: JSONB, compiler: Any, **kw: Any) -> str:
    return "JSON"


async def _factory():
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.run_sync(AsyncBase.metadata.create_all, tables=_TABLES)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def _fresh_broker(cash: str = "100") -> PaperBroker:
    return PaperBroker(initial_cash=Decimal(cash))


async def _buy(broker: PaperBroker, symbol: str, price: str, qty: str) -> None:
    broker.update_price(symbol, Decimal(price))
    await broker.submit_order(
        BrokerOrder(
            external_id=str(uuid.uuid4()),
            symbol=symbol,
            side="buy",
            quantity=Decimal(qty),
            order_type="market",
        )
    )


async def _checkpoint(factory, broker: PaperBroker, mode: str = "paper") -> None:
    """Run one snapshot tick: equity row + position rows, one transaction."""
    store = BrokerStateStore(session_factory=factory, trading_mode=mode)
    writer = PortfolioSnapshotWriter(
        session_factory=factory, trading_mode=mode, state_store=store
    )
    account = await broker.get_account()
    positions = await broker.get_positions()
    await writer.write_once(
        equity=Decimal(account["equity"]),
        cash=Decimal(account["cash"]),
        positions_value=Decimal(account["positions_value"]),
        unrealized_pnl=Decimal(account["unrealized_pnl"]),
        realized_pnl=Decimal(account["realized_pnl"]),
        position_count=len(positions),
        positions=positions,
    )


async def test_restart_restores_cash_and_book_exactly() -> None:
    engine, factory = await _factory()
    broker = await _fresh_broker()
    await _buy(broker, "BTC/USDT", "100", "0.2")  # ~$20 of a $100 account
    await _checkpoint(factory, broker)

    before = await broker.get_account()

    # The restart: a brand-new broker at initial capital, as the worker builds it.
    restarted = await _fresh_broker()
    store = BrokerStateStore(session_factory=factory, trading_mode="paper")
    assert await store.restore(restarted) == "restored"

    after = await restarted.get_account()
    assert Decimal(after["cash"]) == Decimal(before["cash"])
    assert Decimal(after["equity"]) == pytest.approx(Decimal(before["equity"]))
    positions = await restarted.get_positions()
    assert len(positions) == 1
    assert positions[0]["symbol"] == "BTC/USDT"
    assert Decimal(positions[0]["quantity"]) == Decimal("0.2")
    # Cost basis survives, so unrealised P&L stays attributable after a restart.
    assert Decimal(positions[0]["avg_entry_price"]) > 0
    await engine.dispose()


async def test_restore_replays_fills_that_landed_after_the_checkpoint() -> None:
    """An unclean shutdown must not lose the trades in the checkpoint gap."""
    engine, factory = await _factory()
    broker = await _fresh_broker()
    await _buy(broker, "BTC/USDT", "100", "0.2")
    await _checkpoint(factory, broker)

    # A fill persisted by order_sync after the checkpoint was taken.
    async with factory() as session:
        order = Order(
            external_id="explore-late",
            symbol="ETH/USDT",
            asset_class="crypto",
            side="buy",
            order_type="market",
            quantity=Decimal("2"),
            status=OrderStatus.FILLED.value,
            trading_mode="paper",
            filled_at=datetime.now(UTC) + timedelta(minutes=1),
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        session.add(order)
        await session.flush()
        session.add(
            Fill(
                order_id=order.id,
                price=Decimal("10"),
                quantity=Decimal("2"),
                commission=Decimal("0"),
                filled_at=datetime.now(UTC) + timedelta(minutes=1),
                created_at=datetime.now(UTC),
            )
        )
        await session.commit()

    restarted = await _fresh_broker()
    store = BrokerStateStore(session_factory=factory, trading_mode="paper")
    assert await store.restore(restarted) == "restored"

    positions = {p["symbol"]: p for p in await restarted.get_positions()}
    assert set(positions) == {"BTC/USDT", "ETH/USDT"}
    assert Decimal(positions["ETH/USDT"]["quantity"]) == Decimal("2")
    # ~$20 spent before the checkpoint (at a slipped price), exactly $20
    # replayed after it.
    cash = Decimal((await restarted.get_account())["cash"])
    assert cash == pytest.approx(Decimal("60"), abs=Decimal("0.05"))
    await engine.dispose()


async def test_rebase_carries_equity_forward_when_no_book_was_checkpointed() -> None:
    """Equity history but no position rows (the pre-durability state): equity
    must carry forward, not snap back to initial capital."""
    engine, factory = await _factory()
    async with factory() as session:
        session.add(
            PortfolioSnapshot(
                time=datetime.now(UTC),
                trading_mode="paper",
                total_equity=Decimal("98.8588"),
                cash=Decimal("94.01"),
                positions_value=Decimal("4.8488"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                position_count=3,
            )
        )
        await session.commit()

    broker = await _fresh_broker()
    store = BrokerStateStore(session_factory=factory, trading_mode="paper")
    assert await store.restore(broker) == "rebased"

    account = await broker.get_account()
    assert Decimal(account["equity"]) == Decimal("98.8588")
    assert Decimal(account["equity"]) != Decimal("100")  # the bug this fixes
    await engine.dispose()


async def test_no_history_leaves_initial_capital_untouched() -> None:
    engine, factory = await _factory()
    broker = await _fresh_broker()
    store = BrokerStateStore(session_factory=factory, trading_mode="paper")
    assert await store.restore(broker) == "fresh"
    assert Decimal((await broker.get_account())["cash"]) == Decimal("100")
    await engine.dispose()


async def test_checkpoint_closes_rows_for_exited_positions() -> None:
    engine, factory = await _factory()
    broker = await _fresh_broker()
    await _buy(broker, "BTC/USDT", "100", "0.2")
    await _checkpoint(factory, broker)

    await broker.submit_order(
        BrokerOrder(
            external_id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side="sell",
            quantity=Decimal("0.2"),
            order_type="market",
        )
    )
    await _checkpoint(factory, broker)

    async with factory() as session:
        rows = list((await session.execute(select(Position))).scalars().all())
    assert len(rows) == 1
    assert rows[0].closed_at is not None

    restarted = await _fresh_broker()
    store = BrokerStateStore(session_factory=factory, trading_mode="paper")
    # A genuinely flat book is an exact restore, not a rebase: the snapshot
    # itself reports position_count == 0, so there is nothing unaccounted for.
    assert await store.restore(restarted) == "restored"
    assert await restarted.get_positions() == []
    await engine.dispose()


async def test_a_live_book_is_never_restored_into_the_paper_broker() -> None:
    engine, factory = await _factory()
    broker = await _fresh_broker()
    await _buy(broker, "BTC/USDT", "100", "0.2")
    await _checkpoint(factory, broker, mode="live")

    paper_store = BrokerStateStore(session_factory=factory, trading_mode="paper")
    restarted = await _fresh_broker()
    assert await paper_store.restore(restarted) == "fresh"
    assert Decimal((await restarted.get_account())["cash"]) == Decimal("100")
    await engine.dispose()


async def test_cost_basis_and_realised_pnl_track_a_round_trip() -> None:
    """Both feed the snapshot's P&L columns, which were hardcoded to zero."""
    broker = await _fresh_broker("1000")
    broker.update_price("SOL/USDT", Decimal("100"))
    await broker.submit_order(
        BrokerOrder(
            external_id=str(uuid.uuid4()),
            symbol="SOL/USDT",
            side="buy",
            quantity=Decimal("1"),
            order_type="market",
        )
    )
    entry = Decimal((await broker.get_positions())[0]["avg_entry_price"])
    assert entry > 0

    broker.update_price("SOL/USDT", Decimal("110"))
    account = await broker.get_account()
    # Marked up ~10 before any close; realised is still zero.
    assert Decimal(account["unrealized_pnl"]) > Decimal("8")
    assert Decimal(account["realized_pnl"]) == Decimal("0")

    await broker.submit_order(
        BrokerOrder(
            external_id=str(uuid.uuid4()),
            symbol="SOL/USDT",
            side="sell",
            quantity=Decimal("1"),
            order_type="market",
        )
    )
    closed = await broker.get_account()
    assert Decimal(closed["realized_pnl"]) > Decimal("8")
    assert await broker.get_positions() == []


async def test_restored_positions_are_marked_at_a_real_price_immediately() -> None:
    """Without seeded prices the first post-restart snapshot would value the
    whole book at zero and register as a false drawdown."""
    engine, factory = await _factory()
    broker = await _fresh_broker()
    await _buy(broker, "BTC/USDT", "100", "0.2")
    await _checkpoint(factory, broker)

    restarted = await _fresh_broker()
    store = BrokerStateStore(session_factory=factory, trading_mode="paper")
    await store.restore(restarted)

    # No price tick has arrived yet on the fresh process.
    account = await restarted.get_account()
    assert Decimal(account["positions_value"]) > Decimal("19")
    assert Decimal(account["equity"]) > Decimal("99")
    await engine.dispose()


async def test_unpriced_position_is_not_valued_at_zero() -> None:
    broker = await _fresh_broker()
    broker.restore_state(
        cash=Decimal("50"),
        positions={"XYZ": Decimal("2")},
        last_prices={"XYZ": Decimal("25")},
    )
    account = await broker.get_account()
    assert Decimal(account["equity"]) == Decimal("100")


async def test_uuid_client_order_ids_are_untouched_by_replay() -> None:
    """Replay is keyed off persisted fills, so a manual API order (whose
    client_order_id is a DB uuid) replays exactly once, like any other."""
    engine, factory = await _factory()
    broker = await _fresh_broker()
    await _checkpoint(factory, broker)

    async with factory() as session:
        order = Order(
            external_id=str(uuid.uuid4()),
            symbol="AAPL",
            asset_class="stock",
            side="buy",
            order_type="market",
            quantity=Decimal("1"),
            status=OrderStatus.FILLED.value,
            trading_mode="paper",
            filled_at=datetime.now(UTC) + timedelta(minutes=1),
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        session.add(order)
        await session.flush()
        session.add(
            Fill(
                order_id=order.id,
                price=Decimal("10"),
                quantity=Decimal("1"),
                commission=Decimal("0"),
                filled_at=datetime.now(UTC) + timedelta(minutes=1),
                created_at=datetime.now(UTC),
            )
        )
        await session.commit()

    restarted = await _fresh_broker()
    store = BrokerStateStore(session_factory=factory, trading_mode="paper")
    await store.restore(restarted)
    assert Decimal((await restarted.get_account())["cash"]) == Decimal("90")
    await engine.dispose()
