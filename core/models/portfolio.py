"""Portfolio snapshot ORM model (TimescaleDB hypertable)."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import DateTime, Float, Integer, Numeric, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from core.models.base import AsyncBase


class PortfolioSnapshot(AsyncBase):
    """Maps to the ``portfolio_snapshots`` TimescaleDB hypertable."""

    __tablename__ = "portfolio_snapshots"

    time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, nullable=False
    )
    trading_mode: Mapped[str] = mapped_column(
        String(20), primary_key=True, nullable=False
    )

    total_equity: Mapped[Decimal] = mapped_column(
        Numeric(20, 8), nullable=False
    )
    cash: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    positions_value: Mapped[Decimal] = mapped_column(
        Numeric(20, 8), nullable=False
    )
    unrealized_pnl: Mapped[Decimal] = mapped_column(
        Numeric(20, 8), nullable=False
    )
    realized_pnl: Mapped[Decimal] = mapped_column(
        Numeric(20, 8), nullable=False
    )
    daily_return_pct: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    max_drawdown_pct: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    sharpe_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    position_count: Mapped[int] = mapped_column(Integer, nullable=False)
    allocations: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )

    def __repr__(self) -> str:
        return (
            f"<PortfolioSnapshot time={self.time} equity={self.total_equity} "
            f"mode={self.trading_mode!r}>"
        )
