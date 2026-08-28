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
    # none_as_null: an absent feature vector must be SQL NULL, not JSON
    # 'null' -- IS NOT NULL is the feature_rows_persisted counter, and JSON
    # nulls inflated it ~5x until the 2026-08-09 fix (migration 0009).
    features_used: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB(none_as_null=True), nullable=True
    )
    # Raw pre-calibration p(flat) (migration 0010): any future calibration
    # layer transforms the served probabilities, and phase 1 proved the raw
    # series must never be unrecoverable (GO_LIVE 2026-08-28).
    p_flat_raw: Mapped[float | None] = mapped_column(Float, nullable=True)
    explanation: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # The prediction event's id, which the continuous-learning tracker uses as
    # its prediction key. Indexed so the outcome resolver can UPDATE this row
    # by event_id and startup rehydration can reload resolved rows efficiently.
    event_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    # Realised outcome written back by the outcome resolver (5-minute forward
    # move classified with the +/-5bp deadband). NULL until resolved. These
    # columns make learning state durable: the continuous-learning service
    # rehydrates recent resolved rows on startup so drift statistics (which
    # need >=1000 resolved outcomes) survive deploys instead of resetting.
    actual_direction: Mapped[str | None] = mapped_column(String(10), nullable=True)
    actual_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction id={self.id} symbol={self.symbol!r} "
            f"direction={self.direction!r}>"
        )
