"""Durable safety state: the few flags a restart must never silently reset.

Until 2026-11-02 the execution engine's halt flag, the circuit breaker's state
and the risk engine's daily baseline and all-time peak lived only in process
memory. A deploy after an operator halt resumed trading; a deploy after a
losing morning re-based the daily P&L to initial capital (a $92 account tripped
the breaker on every restart for the rest of the UTC day; a $120 account could
lose 10% intraday and read +8%). One row per trading mode, written on change
and on every account-sync tick, read before any service starts.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import Boolean, Date, DateTime, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from core.models.base import AsyncBase


class SafetyState(AsyncBase):
    """Maps to the ``safety_state`` table (one row per trading mode)."""

    __tablename__ = "safety_state"

    trading_mode: Mapped[str] = mapped_column(String(20), primary_key=True)
    halted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false", default=False
    )
    halt_reason: Mapped[str | None] = mapped_column(String(200), nullable=True)
    breaker_state: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="closed", default="closed"
    )
    breaker_tripped_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    breaker_reason: Mapped[str | None] = mapped_column(String(500), nullable=True)
    day_start_equity: Mapped[Decimal | None] = mapped_column(
        Numeric(20, 8), nullable=True
    )
    peak_equity: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    # UTC date the daily baseline belongs to; a restore on a later date
    # re-bases from the first account sync instead of trusting a stale baseline.
    session_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<SafetyState mode={self.trading_mode!r} halted={self.halted} "
            f"breaker={self.breaker_state!r} day_start={self.day_start_equity}>"
        )
