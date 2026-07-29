"""Durable paper-broker account state: checkpoint on a tick, restore at boot.

The paper broker keeps cash, positions and cost basis in process memory, so
before this existed every worker restart silently rebased equity to
``initial_capital`` and orphaned the open book -- five days of soak P&L
vanished on a deploy, and the ``positions`` table stayed empty because nothing
ever wrote it.

The checkpoint is the pair (latest ``portfolio_snapshots`` row, open
``positions`` rows) written in the same transaction on the same tick, so cash
and the book are always consistent with each other. Restore reads that pair
and replays any fills that landed after it, which bounds the loss from an
unclean shutdown to nothing rather than to one snapshot interval.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Protocol

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.models.orders import Fill, Order
from core.models.portfolio import PortfolioSnapshot
from core.models.positions import Position

logger = structlog.get_logger(__name__)


class RestorableBroker(Protocol):
    """The slice of the paper broker this store drives."""

    def restore_state(
        self,
        *,
        cash: Decimal,
        positions: dict[str, Decimal] | None = ...,
        avg_entry: dict[str, Decimal] | None = ...,
        last_prices: dict[str, Decimal] | None = ...,
        realized_pnl: Decimal | None = ...,
    ) -> None: ...

    def apply_external_fill(
        self, symbol: str, side: str, price: Decimal, quantity: Decimal
    ) -> None: ...

    async def get_positions(self) -> list[dict[str, Any]]: ...


def _asset_class(symbol: str) -> str:
    return "crypto" if "/" in symbol else "stock"


class BrokerStateStore:
    """Persist and restore simulated broker account state."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        trading_mode: str,
    ) -> None:
        self._session_factory = session_factory
        self._trading_mode = trading_mode

    # ------------------------------------------------------------------
    # Restore
    # ------------------------------------------------------------------

    async def restore(self, broker: RestorableBroker) -> str:
        """Rebuild *broker* state from the last checkpoint.

        Returns the outcome: ``fresh`` (no history -- keep initial capital),
        ``rebased`` (equity history but no position checkpoint, so equity is
        carried forward onto a flat book) or ``restored`` (exact).
        """
        async with self._session_factory() as session:
            snapshot = await self._latest_snapshot(session)
            if snapshot is None:
                logger.info("broker_state_fresh", trading_mode=self._trading_mode)
                return "fresh"

            rows = (
                await session.execute(
                    select(Position).where(
                        Position.closed_at.is_(None),
                        Position.trading_mode == self._trading_mode,
                    )
                )
            ).scalars().all()

            if not rows and snapshot.position_count > 0:
                # The snapshot reports a book that has no checkpoint rows: it
                # predates the checkpoint writer. Carry equity forward onto a
                # flat book rather than resetting to initial capital or
                # dropping the position value -- both would misstate the soak.
                # Fills are NOT replayed: there is no consistent book to
                # replay them onto.
                broker.restore_state(
                    cash=Decimal(str(snapshot.total_equity)),
                    realized_pnl=Decimal(str(snapshot.realized_pnl)),
                )
                logger.warning(
                    "broker_state_rebased",
                    equity=str(snapshot.total_equity),
                    checkpoint_at=snapshot.time.isoformat(),
                )
                return "rebased"

            positions: dict[str, Decimal] = {}
            avg_entry: dict[str, Decimal] = {}
            last_prices: dict[str, Decimal] = {}
            for row in rows:
                signed = row.quantity if row.side == "long" else -row.quantity
                positions[row.symbol] = signed
                if row.avg_entry_price:
                    avg_entry[row.symbol] = Decimal(str(row.avg_entry_price))
                price = row.current_price or row.avg_entry_price
                if price:
                    last_prices[row.symbol] = Decimal(str(price))

            broker.restore_state(
                cash=Decimal(str(snapshot.cash)),
                positions=positions,
                avg_entry=avg_entry,
                last_prices=last_prices,
                realized_pnl=Decimal(str(snapshot.realized_pnl)),
            )
            replayed = await self._replay_fills_after(session, broker, snapshot.time)

        logger.info(
            "broker_state_restored",
            cash=str(snapshot.cash),
            positions=len(positions),
            replayed_fills=replayed,
            checkpoint_at=snapshot.time.isoformat(),
        )
        return "restored"

    async def _latest_snapshot(self, session: AsyncSession) -> PortfolioSnapshot | None:
        result = await session.execute(
            select(PortfolioSnapshot)
            .where(PortfolioSnapshot.trading_mode == self._trading_mode)
            .order_by(PortfolioSnapshot.time.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def _replay_fills_after(
        self, session: AsyncSession, broker: RestorableBroker, since: datetime
    ) -> int:
        """Apply fills that landed after the checkpoint (unclean-shutdown gap)."""
        rows = (
            await session.execute(
                select(Order.symbol, Order.side, Fill.price, Fill.quantity)
                .join(Fill, Fill.order_id == Order.id)
                .where(
                    Fill.filled_at > since,
                    Order.trading_mode == self._trading_mode,
                )
                .order_by(Fill.filled_at)
            )
        ).all()
        for symbol, side, price, quantity in rows:
            broker.apply_external_fill(
                symbol=symbol,
                side=side,
                price=Decimal(str(price)),
                quantity=Decimal(str(quantity)),
            )
        return len(rows)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    async def checkpoint(
        self, positions: list[dict[str, Any]], session: AsyncSession
    ) -> None:
        """Write the open book into ``positions`` within the caller's session.

        Takes the caller's session so the position rows commit in the same
        transaction as the equity snapshot they pair with.
        """
        now = datetime.now(UTC)
        held = {
            p["symbol"]: p for p in positions if Decimal(str(p["quantity"])) != 0
        }

        existing = (
            await session.execute(
                select(Position).where(
                    Position.closed_at.is_(None),
                    Position.trading_mode == self._trading_mode,
                )
            )
        ).scalars().all()

        seen: set[str] = set()
        for row in existing:
            live = held.get(row.symbol)
            if live is None:
                row.closed_at = now
                row.updated_at = now
                continue
            seen.add(row.symbol)
            quantity = Decimal(str(live["quantity"]))
            row.side = "long" if quantity > 0 else "short"
            row.quantity = abs(quantity)
            row.avg_entry_price = Decimal(str(live.get("avg_entry_price", 0) or 0))
            row.current_price = Decimal(str(live.get("current_price", 0) or 0)) or None
            row.unrealized_pnl = Decimal(str(live.get("unrealized_pnl", 0) or 0))
            row.updated_at = now

        for symbol, live in held.items():
            if symbol in seen:
                continue
            quantity = Decimal(str(live["quantity"]))
            session.add(
                Position(
                    symbol=symbol,
                    asset_class=_asset_class(symbol),
                    trading_mode=self._trading_mode,
                    side="long" if quantity > 0 else "short",
                    quantity=abs(quantity),
                    avg_entry_price=Decimal(str(live.get("avg_entry_price", 0) or 0)),
                    current_price=Decimal(str(live.get("current_price", 0) or 0)) or None,
                    unrealized_pnl=Decimal(str(live.get("unrealized_pnl", 0) or 0)),
                    opened_at=now,
                    updated_at=now,
                )
            )
