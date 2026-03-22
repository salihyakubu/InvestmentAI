"""Position ORM model."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, Float, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from core.models.base import AsyncBase, UUIDMixin


class Position(UUIDMixin, AsyncBase):
    """Maps to the ``positions`` table."""

    __tablename__ = "positions"

    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    asset_class: Mapped[str] = mapped_column(String(10), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    avg_entry_price: Mapped[Decimal] = mapped_column(
        Numeric(20, 8), nullable=False
    )
    current_price: Mapped[Decimal | None] = mapped_column(
        Numeric(20, 8), nullable=True
    )
    unrealized_pnl: Mapped[Decimal | None] = mapped_column(
        Numeric(20, 8), nullable=True
    )
    realized_pnl: Mapped[Decimal] = mapped_column(
        Numeric(20, 8), nullable=False, server_default="0"
    )
    stop_loss_price: Mapped[Decimal | None] = mapped_column(
        Numeric(20, 8), nullable=True
    )
    trailing_stop_pct: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    highest_price: Mapped[Decimal | None] = mapped_column(
        Numeric(20, 8), nullable=True
    )
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    closed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    @property
    def is_open(self) -> bool:
        """Return True if the position has not been closed."""
        return self.closed_at is None

    def __repr__(self) -> str:
        status = "OPEN" if self.is_open else "CLOSED"
        return (
            f"<Position id={self.id} symbol={self.symbol!r} "
            f"side={self.side!r} qty={self.quantity} [{status}]>"
        )
