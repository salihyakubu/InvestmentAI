"""Walk-forward research observations (ORM).

One row per resolved (factor, stamp): the cross-sectional information
coefficient of a registered factor, computed ONCE when the stamp's forward
window closes and never recomputed.

The append-only discipline is what makes the record trustworthy. Recomputing
history from today's listing set would silently drop contracts delisted since
the stamp was resolved -- reintroducing, inside the adjudication record
itself, the exact survivorship bias PR #64 was built to eliminate. Rows are
inserted at resolution time with the universe as it existed then, and are
immutable thereafter.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from core.models.base import AsyncBase


class FactorWatchObservation(AsyncBase):
    """Maps to the ``factor_watch`` table."""

    __tablename__ = "factor_watch"

    # The 8h stamp this IC belongs to (start of the factor's decision bar).
    time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, nullable=False
    )
    factor: Mapped[str] = mapped_column(
        String(60), primary_key=True, nullable=False
    )

    horizon_stamps: Mapped[int] = mapped_column(Integer, nullable=False)
    ic: Mapped[float] = mapped_column(Float, nullable=False)
    n_symbols: Mapped[int] = mapped_column(Integer, nullable=False)
    resolved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<FactorWatch {self.factor} @ {self.time:%Y-%m-%d %H:%M} "
            f"ic={self.ic:+.4f} n={self.n_symbols}>"
        )
