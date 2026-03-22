"""Audit log ORM model (TimescaleDB hypertable)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, String
from sqlalchemy.dialects.postgresql import INET, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from core.models.base import AsyncBase, UUIDMixin


class AuditLog(UUIDMixin, AsyncBase):
    """Maps to the ``audit_logs`` TimescaleDB hypertable."""

    __tablename__ = "audit_logs"

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    service: Mapped[str] = mapped_column(String(50), nullable=False)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    entity_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    decision_trace: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    ip_address: Mapped[str | None] = mapped_column(INET, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<AuditLog id={self.id} service={self.service!r} "
            f"action={self.action!r} at={self.timestamp}>"
        )
