"""Backtest job ORM model.

Backtests run as background jobs in the API process (minutes-scale: keyless
daily-bar fetches plus per-symbol model training). The row is the job's source
of truth: the UI polls ``status`` and fetches ``result`` when completed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from core.models.base import AsyncBase, UUIDMixin


class BacktestJob(UUIDMixin, AsyncBase):
    """Maps to the ``backtest_jobs`` table."""

    __tablename__ = "backtest_jobs"

    # queued -> running -> completed | failed
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="queued")
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    def __repr__(self) -> str:
        return f"<BacktestJob id={self.id} status={self.status!r}>"
