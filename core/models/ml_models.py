"""ML model metadata ORM model."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from core.models.base import AsyncBase, UUIDMixin


class ModelMetadata(UUIDMixin, AsyncBase):
    """Maps to the ``model_metadata`` table."""

    __tablename__ = "model_metadata"
    __table_args__ = (
        UniqueConstraint("model_name", "version", name="uq_model_name_version"),
    )

    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    hyperparameters: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False
    )
    training_metrics: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    validation_metrics: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    feature_importance: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    artifact_path: Mapped[str] = mapped_column(String(500), nullable=False)
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    training_data_start: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    training_data_end: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<ModelMetadata name={self.model_name!r} v{self.version} "
            f"active={self.is_active}>"
        )
