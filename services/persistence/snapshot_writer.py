"""Periodically persist portfolio equity snapshots to the database.

Nothing else in the platform writes the ``portfolio_snapshots`` hypertable,
yet the dashboard's equity curve (``/api/portfolio`` endpoints) reads from it
-- so without a writer the curve stays empty during the paper soak. This
service polls the broker account (``broker.get_account()``) and optionally
the open positions (``broker.get_positions()``) on an interval and inserts
one ``PortfolioSnapshot`` row per tick, keyed by ``(time, trading_mode)``.
Reporting only; risk and execution track account state independently.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.models.portfolio import PortfolioSnapshot
from services.persistence.broker_state import BrokerStateStore

logger = structlog.get_logger(__name__)


class PortfolioSnapshotWriter:
    """Write periodic ``portfolio_snapshots`` rows for the dashboard equity curve."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        trading_mode: str,
        state_store: BrokerStateStore | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._trading_mode = trading_mode
        self._state_store = state_store

    async def write_once(
        self,
        *,
        equity: Decimal | float,
        cash: Decimal | float,
        positions_value: Decimal | float,
        unrealized_pnl: Decimal | float = 0,
        realized_pnl: Decimal | float = 0,
        position_count: int = 0,
        positions: list[dict[str, Any]] | None = None,
    ) -> None:
        """Insert a snapshot row, and the open book alongside it.

        The position checkpoint shares this transaction so a restore always
        reads a cash balance and a book that were true at the same instant.
        """
        snapshot = PortfolioSnapshot(
            time=datetime.now(UTC),
            trading_mode=self._trading_mode,
            total_equity=Decimal(str(equity)),
            cash=Decimal(str(cash)),
            positions_value=Decimal(str(positions_value)),
            unrealized_pnl=Decimal(str(unrealized_pnl)),
            realized_pnl=Decimal(str(realized_pnl)),
            position_count=position_count,
        )
        async with self._session_factory() as session:
            session.add(snapshot)
            if self._state_store is not None and positions is not None:
                await self._state_store.checkpoint(positions, session)
            await session.commit()
        logger.debug(
            "portfolio_snapshot_written",
            trading_mode=self._trading_mode,
            equity=str(snapshot.total_equity),
        )

    async def run(
        self,
        account_provider: Callable[[], Awaitable[dict[str, Any] | None]],
        positions_provider: Callable[[], Awaitable[list[dict[str, Any]]]] | None = None,
        interval_seconds: float = 300.0,
    ) -> None:
        """Background loop: periodically pull the broker account via
        *account_provider* and persist a snapshot, so the equity curve
        accumulates real history. Runs until cancelled; a failed iteration
        is logged and the loop continues.

        The first write happens immediately: sleeping first left a hole of one
        full interval after every restart, and a worker crash-looping faster
        than the interval wrote no snapshots at all.
        """
        first = True
        while True:
            try:
                if not first:
                    await asyncio.sleep(interval_seconds)
                first = False
                account = await account_provider()
                if account is None:
                    continue
                positions_value = Decimal(str(account.get("positions_value", 0) or 0))
                unrealized_pnl = Decimal(str(account.get("unrealized_pnl", 0) or 0))
                realized_pnl = Decimal(str(account.get("realized_pnl", 0) or 0))
                position_count = 0
                positions: list[dict[str, Any]] | None = None
                if positions_provider is not None:
                    positions = await positions_provider()
                    positions_value = Decimal(
                        str(sum(float(p["market_value"]) for p in positions))
                    )
                    unrealized_pnl = Decimal(
                        str(sum(float(p.get("unrealized_pnl", 0) or 0) for p in positions))
                    )
                    position_count = len(positions)
                await self.write_once(
                    equity=Decimal(str(account["equity"])),
                    cash=Decimal(str(account["cash"])),
                    positions_value=positions_value,
                    unrealized_pnl=unrealized_pnl,
                    realized_pnl=realized_pnl,
                    position_count=position_count,
                    positions=positions,
                )
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Portfolio snapshot iteration failed")
