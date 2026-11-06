"""Persist and restore the safety flags a restart must never silently reset.

One ``safety_state`` row per trading mode: the execution engine's halt flag,
the circuit breaker's state, and the risk engine's daily baseline and peak.
Written on every change and on every account-sync tick; read by the worker
before any service starts (GO_LIVE.md 2026-11-02).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.models.safety import SafetyState

logger = structlog.get_logger(__name__)


class SafetyStateStore:
    """Upsert-style access to the single ``safety_state`` row of a trading mode."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        trading_mode: str,
    ) -> None:
        self._session_factory = session_factory
        self._trading_mode = trading_mode

    async def load(self) -> SafetyState | None:
        """Return the stored row, or ``None`` when nothing has been saved yet."""
        async with self._session_factory() as session:
            return await session.get(SafetyState, self._trading_mode)

    async def save_halt(self, halted: bool, reason: str | None = None) -> None:
        await self._update(halted=halted, halt_reason=(reason or None) if halted else None)

    async def save_risk(
        self,
        *,
        breaker_state: str,
        breaker_tripped_at: datetime | None,
        breaker_reason: str | None,
        day_start_equity: Decimal | float | None,
        peak_equity: Decimal | float | None,
        session_date: date,
    ) -> None:
        await self._update(
            breaker_state=breaker_state,
            breaker_tripped_at=breaker_tripped_at,
            breaker_reason=(breaker_reason or None),
            day_start_equity=(
                Decimal(str(day_start_equity)) if day_start_equity is not None else None
            ),
            peak_equity=Decimal(str(peak_equity)) if peak_equity is not None else None,
            session_date=session_date,
        )

    async def _update(self, **fields: Any) -> None:
        now = datetime.now(UTC)
        async with self._session_factory() as session:
            row = await session.get(SafetyState, self._trading_mode)
            if row is None:
                row = SafetyState(trading_mode=self._trading_mode, updated_at=now)
                session.add(row)
            for key, value in fields.items():
                setattr(row, key, value)
            row.updated_at = now
            await session.commit()
        logger.debug("safety_state_saved", trading_mode=self._trading_mode, **{
            k: (str(v) if v is not None else None) for k, v in fields.items()
        })
