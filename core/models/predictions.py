"""Prediction ORM model."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from core.models.base import AsyncBase, UUIDMixin


class Prediction(UUIDMixin, AsyncBase):
    """Maps to the ``predictions`` table.

    Uses a simple ``id`` primary key (not init.sql's composite ``(id, time)``)
    so that ``orders.prediction_id`` can reference it with a valid foreign key.
    This trades the TimescaleDB hypertable partitioning on predictions for FK
    integrity; prediction volume is low enough that partitioning is optional.
    """

    __tablename__ = "predictions"

    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    model_id: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    direction: Mapped[str] = mapped_column(String(10), nullable=False)  # SignalDirection value
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    expected_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    horizon_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    features_used: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    explanation: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction id={self.id} symbol={self.symbol!r} "
            f"direction={self.direction!r}>"
        )
