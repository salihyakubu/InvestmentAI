"""Risk metrics ORM model (TimescaleDB hypertable)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from core.models.base import AsyncBase


class RiskMetric(AsyncBase):
    """Maps to the ``risk_metrics`` TimescaleDB hypertable."""

    __tablename__ = "risk_metrics"

    time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, nullable=False
    )

    var_95: Mapped[float | None] = mapped_column(Float, nullable=True)
    var_99: Mapped[float | None] = mapped_column(Float, nullable=True)
    cvar_95: Mapped[float | None] = mapped_column(Float, nullable=True)
    cvar_99: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_drawdown: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    beta: Mapped[float | None] = mapped_column(Float, nullable=True)
    correlation_max: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    concentration_max: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    circuit_breaker_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    details: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )

    def __repr__(self) -> str:
        return (
            f"<RiskMetric time={self.time} var95={self.var_95} "
            f"breaker={self.circuit_breaker_active}>"
        )
